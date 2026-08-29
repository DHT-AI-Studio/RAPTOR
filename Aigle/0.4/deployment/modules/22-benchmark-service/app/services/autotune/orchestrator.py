"""Optimization orchestrator loop (AUTOTUNE Phase B — the core).

Pure Python control loop. The *loop and the hard budget live here* (deterministic,
crash-safe, guaranteed to stop); only the per-iteration *decision* ("which config
next?") is delegated to a proposer. Step 1 uses a random-search proposer to
validate the whole machine end-to-end; Step 2 swaps in an LLM proposer behind
the same ``Proposer`` interface (see proposer.py) — the loop does not change.
(An LLM's single structured call; a smolagents-backed proposer can drop in via
the same interface later.)

Each iteration:
    1. read leaderboard (durable (config → score) history)
    2. proposer picks the next config point (clamped to the search space)
    3. train_and_wait  → a fine-tuned checkpoint (Module 16)
    4. evaluate        → aggregate_score (in-process Module 22 run)
    5. record + update best + early-stop bookkeeping

All durable state is in PostgreSQL (experiments + benchmark_runs), so a crash
mid-experiment is recovered by re-reading the leaderboard and remaining budget
(AUTOTUNE §B7) — nothing depends on in-memory or agent state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

from app.core.config import get_settings
from app.core.db import db
from app.models.experiment import Budget, ExperimentStatus, Plan
from app.services import run_manager
from app.services.autotune import experiment_store, training_client
from app.services.autotune.search_space import clamp_to_search_space, sample_random
from app.services.autotune.training_client import TrainingError

logger = logging.getLogger(__name__)


# ── In-flight candidate (proposed config + LLM reason, before it finishes) ──
# The leaderboard only shows a candidate AFTER it is scored; this exposes the one
# currently training so a UI can announce "trying X because Y" up front. Best-effort
# Redis, cleared when the loop ends.
def _current_key(exp_id: str) -> str:
    return f"autotune:current:{exp_id}"


async def _set_current_candidate(exp_id: str, iteration: int,
                                 config: Dict[str, Any], reason: Optional[str]) -> None:
    try:
        await db.redis.set(_current_key(exp_id),
                           json.dumps({"iteration": iteration, "config": config, "reason": reason}),
                           ex=3600)
    except Exception:  # noqa: BLE001 — display-only, never break the loop
        pass


async def _set_current_job(exp_id: str, job_id: str) -> None:
    """Record the training job id for the in-flight candidate (merged into its entry)."""
    try:
        raw = await db.redis.get(_current_key(exp_id))
        data = json.loads(raw) if raw else {}
        data["job_id"] = job_id
        await db.redis.set(_current_key(exp_id), json.dumps(data), ex=3600)
    except Exception:  # noqa: BLE001
        pass


async def _clear_current_candidate(exp_id: str) -> None:
    try:
        await db.redis.delete(_current_key(exp_id))
    except Exception:  # noqa: BLE001
        pass


async def get_current_candidate(exp_id: str) -> Optional[Dict[str, Any]]:
    """The candidate currently training (config + reason), or None."""
    try:
        raw = await db.redis.get(_current_key(exp_id))
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        return None

# Keep strong refs to background loop tasks so they aren't garbage-collected.
_TASKS: set[asyncio.Task] = set()


# ── Proposer interface (Step 2 drops in a smolagents agent here) ─────────

# (next_config, stop, reason)
Decision = Tuple[Dict[str, Any], bool, Optional[str]]


class Proposer(Protocol):
    async def propose(self, plan: Plan, history: List[Dict[str, Any]]) -> Decision:
        """Return (next_config, stop, reason). stop=True means the proposer converged."""
        ...


class RandomProposer:
    """Step-1 proposer: uniform random search over the declared space."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()

    async def propose(self, plan: Plan, history: List[Dict[str, Any]]) -> Decision:
        return sample_random(plan.search_space, self._rng), False, None


def _default_proposer() -> Proposer:
    """Pick the proposer per settings: 'llm' (default) falls back to random on
    bad LLM output; 'random' forces pure random search (Step-1 behaviour)."""
    if get_settings().proposer == "random":
        return RandomProposer()
    from app.services.autotune.proposer import LLMProposer  # lazy: avoids import cost when unused
    return LLMProposer()


# ── Launch / stop ───────────────────────────────────────────────────────

def launch(exp_id: str, proposer: Optional[Proposer] = None) -> None:
    """Start (or resume) an experiment loop as a background task."""
    task = asyncio.create_task(run_experiment(exp_id, proposer))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def request_stop(exp_id: str) -> bool:
    """Ask a running experiment to stop after the current iteration."""
    exp = await experiment_store.get_experiment(exp_id)
    if exp is None or exp["status"] not in ("running", "awaiting_confirm", "planning"):
        return False
    await experiment_store.update_status(exp_id, ExperimentStatus.stopped)
    return True


async def resume_running() -> int:
    """Re-launch experiments left 'running' by a crash. Returns how many."""
    running = await experiment_store.list_running()
    for r in running:
        logger.info("Resuming experiment %s after restart", r["experiment_id"])
        launch(r["experiment_id"])
    return len(running)


# ── The loop ────────────────────────────────────────────────────────────

async def run_experiment(exp_id: str, proposer: Optional[Proposer] = None) -> None:
    """Run the optimization loop for one experiment until budget/early-stop/stop."""
    proposer = proposer or _default_proposer()
    try:
        exp = await experiment_store.get_experiment(exp_id)
        if exp is None:
            logger.error("run_experiment: experiment %s not found", exp_id)
            return
        if exp["plan"] is None:
            await experiment_store.update_status(exp_id, ExperimentStatus.failed,
                                                 error="experiment has no plan")
            return

        plan = Plan.model_validate(exp["plan"])
        budget = Budget.model_validate(exp["budget"])
        schema_id = plan.eval_schema_id

        await experiment_store.update_status(exp_id, ExperimentStatus.running)

        # Crash recovery: rebuild best + iteration count from durable state.
        best_score, best_run_id, best_config = await _current_best(schema_id)
        iters = exp["iterations_done"] or 0
        no_improve = 0  # reset the patience window on (re)start

        # Persist the starting best so the experiment row always reflects the
        # current leaderboard top — otherwise, if no iteration beats a pre-existing
        # baseline, best would misleadingly report null.
        if best_score is not None and exp.get("best_run_id") is None:
            await experiment_store.update_best(exp_id, best_run_id, best_score, best_config)

        logger.info("Experiment %s loop start: iters=%d best=%s budget=%s",
                    exp_id, iters, best_score, budget.model_dump())

        while iters < budget.max_experiments and no_improve < budget.early_stop_patience:
            # Cooperative stop: customer may have flipped status to 'stopped'.
            current = await experiment_store.get_experiment(exp_id)
            if current is None or current["status"] == ExperimentStatus.stopped.value:
                logger.info("Experiment %s stopped by request", exp_id)
                return

            history = await run_manager.leaderboard(schema_id, limit=10)
            knobs, stop, reason = await proposer.propose(plan, history)
            if stop:
                logger.info("Experiment %s: proposer converged — %s", exp_id, reason)
                break
            knobs = clamp_to_search_space(knobs, plan.search_space)
            # Announce the in-flight candidate (config + reason) before training, so
            # a UI can show "trying X because Y" up front rather than after scoring.
            await _set_current_candidate(exp_id, iters + 1, knobs, reason)

            async def _record_job(job_id: str) -> None:
                await _set_current_job(exp_id, job_id)

            score, run_id = await _train_and_evaluate(
                plan, knobs, schema_id, budget.minutes_per_experiment, reason, _record_job)
            iters += 1
            await experiment_store.set_iterations(exp_id, iters)

            if score is not None and (best_score is None or score > best_score):
                best_score, best_run_id, best_config = score, run_id, knobs
                await experiment_store.update_best(exp_id, best_run_id, best_score, best_config)
                no_improve = 0
                logger.info("Experiment %s iter %d: NEW BEST %.4f (%s)", exp_id, iters, score, knobs)
            else:
                no_improve += 1
                logger.info("Experiment %s iter %d: score=%s no_improve=%d",
                            exp_id, iters, score, no_improve)

        # Held-out validation: score the best config on the untouched held-out
        # split. dev best vs held-out reveals whether the loop overfit the eval.
        holdout_schema_id = exp.get("holdout_schema_id")
        if holdout_schema_id and best_run_id:
            heldout = await _heldout_eval(holdout_schema_id, best_run_id, best_config)
            await experiment_store.set_holdout_score(exp_id, heldout)
            logger.info("Experiment %s held-out score=%s (dev best=%s)", exp_id, heldout, best_score)

        await experiment_store.update_status(exp_id, ExperimentStatus.completed)
        logger.info("Experiment %s completed: %d iters, best=%s", exp_id, iters, best_score)

    except Exception as exc:  # noqa: BLE001 — a failed experiment must not crash the worker
        logger.exception("Experiment %s failed", exp_id)
        try:
            await experiment_store.update_status(exp_id, ExperimentStatus.failed, error=str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("Experiment %s: failed to persist failure state", exp_id)
    finally:
        # No candidate is in flight once the loop ends (done/stopped/failed).
        await _clear_current_candidate(exp_id)


async def _train_and_evaluate(plan: Plan, knobs: Dict[str, Any], schema_id: str,
                              minutes_per_experiment: float,
                              reason: Optional[str] = None,
                              on_job_id: Optional[Callable[[str], Awaitable[None]]] = None
                              ) -> Tuple[Optional[float], Optional[str]]:
    """One candidate: train (Module 16) → evaluate (in-process). Never raises.

    A training or eval failure yields score=None so the loop records the miss and
    moves on instead of aborting the whole experiment (AUTOTUNE §B4/§8).
    """
    try:
        model_path = await training_client.train_and_wait(
            plan, knobs, minutes_per_experiment, on_job_id=on_job_id)
    except TrainingError as exc:
        logger.warning("Candidate training failed (%s): %s", knobs, exc)
        return None, None
    return await _evaluate(schema_id, model_path, knobs, reason)


async def _evaluate(schema_id: str, model_path: str, knobs: Dict[str, Any],
                    reason: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    """Score a checkpoint on the eval schema, in-process (no HTTP round-trip).

    config_override carries model_path (used by the local_infer pipeline), this
    iteration's training knobs, and the proposer's reason (``_reason``) — so the
    resulting benchmark_runs row is the (config → score → why) history the
    proposer/leaderboard reads back.
    """
    config_override = {**knobs, "model_path": model_path}
    if reason:
        config_override["_reason"] = reason
    created = await run_manager.create_run(schema_id, config_override)
    run_id = created["run_id"]
    await run_manager.execute_run(run_id)  # in-process state machine, await to completion
    record = await run_manager.get_run(run_id)
    if record is None or record["status"] != "completed":
        return None, run_id
    return record["aggregate_score"], run_id


async def _heldout_eval(holdout_schema_id: str, best_run_id: str,
                        best_config: Optional[Dict[str, Any]]) -> Optional[float]:
    """Re-score the best config's checkpoint on the held-out split (anti-overfit).

    Reuses the best run's model_path so the exact winning checkpoint is graded on
    cases the optimizer never saw. Returns the held-out score, or None.
    """
    best_run = await run_manager.get_run(best_run_id)
    if best_run is None:
        return None
    model_path = (best_run.get("config_override") or {}).get("model_path")
    if not model_path:
        return None
    score, _ = await _evaluate(holdout_schema_id, model_path, best_config or {})
    return score


async def _current_best(schema_id: str) -> Tuple[Optional[float], Optional[str], Optional[Dict[str, Any]]]:
    """Read the current best (score, run_id, config) from the leaderboard."""
    board = await run_manager.leaderboard(schema_id, limit=1)
    if not board:
        return None, None, None
    top = board[0]
    cfg = top.get("config_override") or {}
    knobs = {k: v for k, v in cfg.items() if k not in ("model_path", "_reason")}
    return top["aggregate_score"], top["run_id"], knobs
