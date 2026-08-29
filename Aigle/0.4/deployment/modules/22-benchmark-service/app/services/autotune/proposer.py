"""LLM proposer for the optimization loop (AUTOTUNE Phase B3 — Phase 2).

Replaces the Step-1 RandomProposer: given the search space (the action space)
and the leaderboard history (config -> score), an LLM proposes the next config
to try (with a one-line reason), or signals convergence (stop).

Design: the proposer is a *single structured decision*, not a multi-step agent,
so it is one LLM call over the same Module 07 -> Ollama path the judge uses. The
LLM call is injected as ``complete`` — the one swap point — so a smolagents-backed
implementation can replace it later without touching the prompt / parse / clamp /
retry logic or the orchestrator wiring.

Robustness (AUTOTUNE §8): the raw reply is parsed with ``json_repair``, which
tolerates the usual LLM JSON quirks — prose/markdown around it, trailing commas,
single quotes, and (crucially) truncated output missing its closing braces. A
parse/validation failure is retried once, then falls back to a random sample so
the loop never stalls. Whatever is proposed is passed through
``clamp_to_search_space`` (the code-side safety whitelist).
"""
from __future__ import annotations

import json
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import json_repair
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.experiment import Plan, SearchDimension
from app.services import judge
from app.services.judge import ContentPolicyBlockedError
from app.services.autotune.search_space import clamp_to_search_space, sample_random

logger = logging.getLogger(__name__)

# The single swap point: async (prompt) -> raw completion text.
CompleteFn = Callable[[str], Awaitable[str]]

# (config, stop, reason)
Decision = Tuple[Dict[str, Any], bool, Optional[str]]


class Proposal(BaseModel):
    """Structured proposer output."""

    stop: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


def _build_prompt(search_space: Dict[str, SearchDimension],
                  history: List[Dict[str, Any]],
                  extra_note: str = "") -> str:
    space_lines = []
    for name, dim in search_space.items():
        if dim.type == "categorical":
            space_lines.append(f"- {name}: one of {list(dim.choices)}")
        else:
            rng = f"[{dim.min}, {dim.max}]" + (" (log scale)" if dim.log else "")
            space_lines.append(f"- {name}: {dim.type} in {rng}")

    hist_lines = []
    for h in history:
        raw = h.get("config_override") or h.get("config") or {}
        cfg = {k: v for k, v in raw.items() if k not in ("model_path", "_reason")}
        line = f"- score={h.get('aggregate_score')} | config={json.dumps(cfg)}"
        reason = raw.get("_reason") or h.get("reason")
        if reason:
            # Feed back the rationale so the model can see whether its past
            # hypotheses paid off (and avoid repeating failed reasoning).
            line += f" | tried because: {reason}"
        hist_lines.append(line)
    hist_text = "\n".join(hist_lines) if hist_lines else "(no experiments yet)"

    return (
        "You are tuning hyperparameters to MAXIMIZE a score in [0, 1].\n\n"
        "Tunable parameters (use ONLY these, stay within range):\n"
        f"{chr(10).join(space_lines)}\n\n"
        "Past experiments (config -> score), best first:\n"
        f"{hist_text}\n\n"
        "Propose the next configuration to beat the best score, and give a one-\n"
        "sentence reason.\n"
        "Rules:\n"
        "- Do NOT repeat any configuration already listed above; every proposal\n"
        "  must change at least one parameter.\n"
        "- Explore the WHOLE space — vary parameters that have barely moved across\n"
        "  past experiments, not only the one you have been sweeping.\n"
        "- Do not re-run a config just to 'confirm stability'; assume each score is\n"
        "  reliable. If no genuinely new configuration is worth trying, set\n"
        "  stop=true instead of repeating one.\n"
        f"{extra_note}"
        'Reply with ONLY a JSON object, no prose, no markdown:\n'
        '{"config": {"<param>": <value>, ...}, "stop": false, "reason": "<why>"}'
    )


# ── Dedup: never let the proposer burn budget re-running an evaluated config ──
def _config_key(config: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """A hashable, order-independent key for a config (floats rounded so two
    numerically-identical proposals compare equal)."""
    def norm(v: Any) -> Any:
        return round(v, 10) if isinstance(v, float) else v
    return tuple(sorted((k, norm(v)) for k, v in config.items()))


def _tried_keys(history: List[Dict[str, Any]]) -> set:
    """Keys of every config already evaluated (from the leaderboard history)."""
    keys = set()
    for h in history:
        raw = h.get("config_override") or h.get("config") or {}
        cfg = {k: v for k, v in raw.items() if k not in ("model_path", "_reason")}
        if cfg:
            keys.add(_config_key(cfg))
    return keys


def _parse_proposal(raw: str) -> Proposal:
    """Repair + parse + validate a Proposal from a raw LLM reply.

    ``json_repair`` handles prose/markdown, trailing commas, single quotes, and
    truncated output; it returns "" on hopeless input, which we reject so the
    caller falls back. Raises if the result is not a valid Proposal object.
    """
    data = json_repair.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM reply did not contain a JSON object")
    return Proposal.model_validate(data)


class LLMProposer:
    """Phase-2 proposer: an LLM reads the leaderboard and proposes the next config."""

    def __init__(self, model: Optional[str] = None, complete: Optional[CompleteFn] = None,
                 rng: Optional[random.Random] = None) -> None:
        s = get_settings()
        self._model = model or s.proposer_model or s.judge_model
        self._complete: CompleteFn = complete or self._default_complete
        self._rng = rng or random.Random()

    async def _default_complete(self, prompt: str) -> str:
        s = get_settings()
        return await judge.complete(prompt, model=self._model,
                                    max_length=s.proposer_max_tokens,
                                    timeout=s.proposer_timeout)

    async def propose(self, plan: Plan, history: List[Dict[str, Any]]) -> Decision:
        tried = _tried_keys(history)
        prompt = _build_prompt(plan.search_space, history)
        nudged = False  # have we already told it "that one's a duplicate, try another"?
        blocked = False  # set True on a guardrail block, so the fallback reason says why
        # attempts = parse retries + one dedup nudge before we give up to random.
        for attempt in (1, 2, 3):
            try:
                raw = await self._complete(prompt)
                decision = _parse_proposal(raw)
                if decision.stop:
                    logger.info("LLMProposer: stop — %s", decision.reason)
                    return {}, True, decision.reason
                config = clamp_to_search_space(decision.config, plan.search_space)
                if not config:
                    raise ValueError("proposal had no valid knobs")
                if _config_key(config) in tried:
                    # Re-proposed an already-evaluated config (the "re-run to confirm"
                    # loop that wastes budget). Nudge once to explore something new; if
                    # it still repeats, it has genuinely converged → stop.
                    if not nudged:
                        nudged = True
                        prompt = _build_prompt(
                            plan.search_space, history,
                            extra_note="- You just proposed a configuration IDENTICAL to one already "
                                       "tried above. Propose a DIFFERENT one (change at least one "
                                       "parameter, e.g. lora_r/lora_dropout/warmup_ratio), or stop=true.\n")
                        logger.info("LLMProposer: duplicate config %s — nudging to explore", config)
                        continue
                    logger.info("LLMProposer: converged (still repeating %s after nudge) — stopping", config)
                    return {}, True, decision.reason or "converged: no untried configuration proposed"
                logger.info("LLMProposer: proposed %s — %s", config, decision.reason)
                return config, False, decision.reason
            except ContentPolicyBlockedError as exc:
                # Permanent -- retrying the identical prompt against the same
                # policy can never succeed, so stop immediately instead of
                # burning the remaining attempts (each a full inference call).
                logger.warning("LLMProposer: prompt blocked by guardrail policy, "
                               "not retrying — %s", exc)
                blocked = True
                break
            except Exception as exc:  # noqa: BLE001 — bad LLM output; retry then fall back
                logger.warning("LLMProposer attempt %d failed: %s", attempt, exc)

        # Never stall the loop: fall back to a random sample (AUTOTUNE §8).
        reason = "random fallback (prompt blocked by guardrail policy)" if blocked \
            else "random fallback (LLM output unusable)"
        logger.warning("LLMProposer: falling back to a random sample")
        return sample_random(plan.search_space, self._rng), False, reason
