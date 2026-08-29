"""Guard-model classification orchestrator.

The single place that knows "ask every active guard model and combine
their verdicts." All model-specific prompt/parsing detail is delegated to
adapters registered in app/adapters/ — this module never branches on model
name or family. Used by app/routers/check_guard.py (HTTP layer, shapes the
FastAPI response), app/routers/check_policy_llm.py (threads a Policy
through), and app/services/checker.py (GB-4, in-process reuse — no HTTP
round-trip, same as before this refactor).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import app.adapters  # noqa: F401 — triggers self-registration of built-in adapters
from app.adapters.base import CATEGORY_NAMES, ChatMessage, GuardRequest, PolicyContent
from app.adapters.ollama_client import call as ollama_call
from app.adapters.registry import get_adapter
from app.core.config import get_settings

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelVerdict:
    model: str
    safe: bool
    categories: list[str]
    category_names: dict[str, str]
    raw: str


@dataclass(frozen=True)
class CombinedVerdict:
    safe: bool
    categories: list[str]
    category_names: dict[str, str]
    raw: str
    conflict: bool | None = None
    results: list[ModelVerdict] | None = None


def _to_model_verdict(model: str, verdict) -> ModelVerdict:
    names = {c: CATEGORY_NAMES.get(c, "Unknown") for c in verdict.categories}
    return ModelVerdict(model=model, safe=verdict.safe, categories=verdict.categories,
                         category_names=names, raw=verdict.raw)


async def _classify_one(model: str, content: str, role: str, *, policy: PolicyContent | None) -> ModelVerdict:
    adapter = get_adapter(model)
    _logger.info("guard(%s) ← role=%s content=%s", model, role, content[:200])
    request = adapter.build_request(content, role, policy=policy)
    raw = await ollama_call(model, request)
    verdict = adapter.parse(raw, used_policy_prompt=policy is not None)
    _logger.info("guard(%s) → raw | %s", model, raw)
    _logger.info("guard(%s) → safe=%s categories=%s", model, verdict.safe, verdict.categories)
    return _to_model_verdict(model, verdict)


async def _classify_one_conversation(model: str, messages: list[ChatMessage], *, policy: PolicyContent | None) -> ModelVerdict:
    adapter = get_adapter(model)
    last = messages[-1]
    _logger.info("guard(%s)(conv) ← role=%s content=%s", model, last.role, last.content[:200])
    request = adapter.build_conversation_request(messages, policy=policy)
    raw = await ollama_call(model, request, label="(conv)")
    verdict = adapter.parse(raw, used_policy_prompt=policy is not None)
    _logger.info("guard(%s)(conv) → raw | %s", model, raw)
    _logger.info("guard(%s)(conv) → safe=%s categories=%s", model, verdict.safe, verdict.categories)
    return _to_model_verdict(model, verdict)


def combine(results: list[ModelVerdict]) -> CombinedVerdict:
    """Merge per-model results into a single CombinedVerdict.

    Single-model: conflict and results are omitted (None).
    Multi-model:
      - safe = False if ANY model says unsafe (conservative)
      - conflict = True when models disagree
      - categories/raw come from the highest-priority model's adapter

    Shared by classify()/classify_conversation() below and by
    app/routers/check_policy_llm.py + app/routers/debug_policy_check_llm.py,
    which call it directly on classify_per_model()'s per-model results to
    shape their responses the same way as /guard/check/*.
    """
    if len(results) == 1:
        r = results[0]
        return CombinedVerdict(safe=r.safe, categories=r.categories, category_names=r.category_names, raw=r.raw)

    safe_flags = [r.safe for r in results]
    combined_safe = all(safe_flags)
    conflict = len(set(safe_flags)) > 1

    primary = min(results, key=lambda r: get_adapter(r.model).priority)
    return CombinedVerdict(safe=combined_safe, categories=primary.categories,
                            category_names=primary.category_names, raw=primary.raw,
                            conflict=conflict, results=results)


def _resolve_policy(model: str, policy: PolicyContent | None,
                     policy_by_family: dict[str, PolicyContent] | None) -> PolicyContent | None:
    """Pick the policy for this specific model: its adapter family's entry in
    `policy_by_family` if given, else the plain `policy` fallback."""
    if policy_by_family is None:
        return policy
    return policy_by_family.get(get_adapter(model).family, policy)


async def classify_per_model(content: str, role: str = "user", *, policy: PolicyContent | None = None,
                              policy_by_family: dict[str, PolicyContent] | None = None) -> list[ModelVerdict]:
    """Run every active guard model over one message, each with its own
    resolved policy, with no combination — used when callers report each
    model's result separately (/policy/check/llm/*, where each model can be
    given a different policy prompt)."""
    settings = get_settings()
    return list(await asyncio.gather(*[
        _classify_one(m, content, role, policy=_resolve_policy(m, policy, policy_by_family))
        for m in settings.active_models
    ]))


async def classify_conversation_per_model(messages: list[ChatMessage], *, policy: PolicyContent | None = None,
                                           policy_by_family: dict[str, PolicyContent] | None = None) -> list[ModelVerdict]:
    """Conversation-mode counterpart to classify_per_model."""
    settings = get_settings()
    return list(await asyncio.gather(*[
        _classify_one_conversation(m, messages, policy=_resolve_policy(m, policy, policy_by_family))
        for m in settings.active_models
    ]))


async def classify(content: str, role: str = "user", *, policy: PolicyContent | None = None) -> CombinedVerdict:
    """Run the active guard model(s) over one message and return a combined verdict."""
    return combine(await classify_per_model(content, role, policy=policy))


async def classify_conversation(messages: list[ChatMessage], *, policy: PolicyContent | None = None) -> CombinedVerdict:
    """Run the active guard model(s) over a full conversation (last-turn verdict)."""
    return combine(await classify_conversation_per_model(messages, policy=policy))


async def classify_raw(content: str, role: str = "user") -> list[ModelVerdict]:
    """Per-model raw outputs with no combination/verdict judgement — used by /guard/check/raw."""
    return await classify_per_model(content, role, policy=None)


# ── Debug variants: also capture the exact prompt text sent to each model ──────
#
# Deliberately separate from classify_per_model/classify_conversation_per_model
# above (not refactored to share code with them) so app/routers/debug_policy_check_llm.py
# can be deleted without touching anything the non-debug paths depend on.

def _render_prompt_text(request: GuardRequest) -> str:
    """Human-readable rendering of exactly what was sent to Ollama."""
    if request.endpoint == "generate":
        return request.payload.get("prompt", "")
    messages = request.payload.get("messages", [])
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


async def _classify_one_with_prompt(model: str, content: str, role: str, *,
                                     policy: PolicyContent | None) -> tuple[ModelVerdict, str]:
    adapter = get_adapter(model)
    request = adapter.build_request(content, role, policy=policy)
    prompt_text = _render_prompt_text(request)
    raw = await ollama_call(model, request)
    verdict = adapter.parse(raw, used_policy_prompt=policy is not None)
    return _to_model_verdict(model, verdict), prompt_text


async def _classify_one_conversation_with_prompt(model: str, messages: list[ChatMessage], *,
                                                   policy: PolicyContent | None) -> tuple[ModelVerdict, str]:
    adapter = get_adapter(model)
    request = adapter.build_conversation_request(messages, policy=policy)
    prompt_text = _render_prompt_text(request)
    raw = await ollama_call(model, request, label="(conv)")
    verdict = adapter.parse(raw, used_policy_prompt=policy is not None)
    return _to_model_verdict(model, verdict), prompt_text


async def classify_per_model_with_prompt(content: str, role: str = "user", *, policy: PolicyContent | None = None,
                                          policy_by_family: dict[str, PolicyContent] | None = None,
                                          ) -> list[tuple[ModelVerdict, str]]:
    """Same dispatch as classify_per_model, but also returns the rendered
    prompt text sent to each model — for debug endpoints only."""
    settings = get_settings()
    return list(await asyncio.gather(*[
        _classify_one_with_prompt(m, content, role, policy=_resolve_policy(m, policy, policy_by_family))
        for m in settings.active_models
    ]))


async def classify_conversation_per_model_with_prompt(messages: list[ChatMessage], *, policy: PolicyContent | None = None,
                                                        policy_by_family: dict[str, PolicyContent] | None = None,
                                                        ) -> list[tuple[ModelVerdict, str]]:
    """Conversation-mode counterpart to classify_per_model_with_prompt."""
    settings = get_settings()
    return list(await asyncio.gather(*[
        _classify_one_conversation_with_prompt(m, messages, policy=_resolve_policy(m, policy, policy_by_family))
        for m in settings.active_models
    ]))
