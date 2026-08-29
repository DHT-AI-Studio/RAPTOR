"""Debug variant of app/routers/check_policy_llm.py — functionally identical
(same policy resolution, same per-model dispatch, same audit logging), same
top-level CheckResponse shape as /guard/check/* and /policy/check/llm/*
(safe/categories/category_names/raw/conflict), except `results` carries one
extra field per model, `prompt` (the exact, human-readable prompt actually
sent to that model) and — unlike the other two endpoint families — is always
populated, never null, so `prompt` is never dropped even with a single
active guard model. Mounted at prefix "/debug" in app/main.py, so these land
at /debug/policy/check/llm/{input,output,conversation}.

Deliberately self-contained: this file duplicates check_policy_llm.py's
three small private helpers (_original_policy_content/_policy_by_family/
_unsafe_category) rather than importing them, and app/services/guard_classifier.py's
*_with_prompt functions are separate, additive functions that don't touch
the ones check_policy_llm.py calls. This whole file (plus app/models/debug.py
and its two lines in app/main.py) can be deleted with zero impact on
/policy/check/llm/* or anything else.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.adapters.base import ChatMessage, PolicyContent
from app.adapters.standard_policy import parse_standard_policies
from app.engine import policy_engine
from app.engine.models import PolicyDetail
from app.engine.policy_targets import COLUMN_BY_ADAPTER_FAMILY
from app.models.debug import DebugCheckResponse, DebugSingleModelResult
from app.models.guard import ConversationRequest, MessageRequest
from app.services import audit_log, guard_classifier
from app.services.guard_classifier import ModelVerdict
from app.services.state import is_policy_check_enabled

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Check (LLM-judged, debug)"])

# Global guardrail switch off, or the narrower policy-check switch off →
# pass immediately, no policy lookup/Ollama call
_DISABLED_RESPONSE = DebugCheckResponse(safe=True, categories=[], category_names={}, raw="guardrail disabled")


def _original_policy_content(policy: PolicyDetail) -> PolicyContent:
    """Same as check_policy_llm.py's helper of the same name — kept as a
    separate copy here deliberately, see module docstring."""
    return PolicyContent(raw=policy.raw_content, standard_policies=parse_standard_policies(policy.raw_content))


def _policy_by_family(policy: PolicyDetail, original: PolicyContent) -> dict[str, PolicyContent]:
    """Same as check_policy_llm.py's helper of the same name — kept as a
    separate copy here deliberately, see module docstring."""
    return {
        family: PolicyContent(raw=getattr(policy, column)) if getattr(policy, column) else original
        for family, column in COLUMN_BY_ADAPTER_FAMILY.items()
    }


def _unsafe_category(results: list[ModelVerdict]) -> str | None:
    """Same as check_policy_llm.py's helper of the same name — kept as a
    separate copy here deliberately, see module docstring."""
    codes = sorted({c for r in results if not r.safe for c in r.categories})
    return ",".join(codes) if codes else None


def _to_debug_response(results: list[tuple[ModelVerdict, str]]) -> DebugCheckResponse:
    """Same top-level shape as /guard/check/*'s CheckResponse (combine() over
    the per-model verdicts), but `results` is always populated — even with a
    single active guard model — so each model's `prompt` is never dropped."""
    verdicts = [v for v, _ in results]
    combined = guard_classifier.combine(verdicts)
    return DebugCheckResponse(
        safe=combined.safe, categories=combined.categories, category_names=combined.category_names,
        raw=combined.raw, conflict=combined.conflict,
        results=[
            DebugSingleModelResult(model=v.model, safe=v.safe, categories=v.categories,
                                    category_names=v.category_names, raw=v.raw, prompt=prompt)
            for v, prompt in results
        ],
    )


@router.post("/policy/check/llm/input", response_model=DebugCheckResponse,
             summary="[debug] Check user input (LLM-judged) — includes the prompt sent to each model")
async def debug_check_input_llm(body: MessageRequest) -> DebugCheckResponse:
    if not await is_policy_check_enabled():
        return _DISABLED_RESPONSE
    policy = await policy_engine._get_active_policy()
    _logger.info("debug check/llm/input policy=%s content=%s", policy.name, body.content[:100])
    original = _original_policy_content(policy)
    results = await guard_classifier.classify_per_model_with_prompt(
        body.content, role="user",
        policy=original, policy_by_family=_policy_by_family(policy, original),
    )
    verdicts = [v for v, _ in results]
    if not all(v.safe for v in verdicts):
        audit_log.record_violation(
            policy_id=policy.id, module=None, direction="input",
            category=_unsafe_category(verdicts), action_taken="block", request_id=None, content=None,
        )
    return _to_debug_response(results)


@router.post("/policy/check/llm/output", response_model=DebugCheckResponse,
             summary="[debug] Check AI response (LLM-judged) — includes the prompt sent to each model")
async def debug_check_output_llm(body: MessageRequest) -> DebugCheckResponse:
    if not await is_policy_check_enabled():
        return _DISABLED_RESPONSE
    policy = await policy_engine._get_active_policy()
    _logger.info("debug check/llm/output policy=%s content=%s", policy.name, body.content[:100])
    original = _original_policy_content(policy)
    results = await guard_classifier.classify_per_model_with_prompt(
        body.content, role="assistant",
        policy=original, policy_by_family=_policy_by_family(policy, original),
    )
    verdicts = [v for v, _ in results]
    if not all(v.safe for v in verdicts):
        audit_log.record_violation(
            policy_id=policy.id, module=None, direction="output",
            category=_unsafe_category(verdicts), action_taken="block", request_id=None, content=None,
        )
    return _to_debug_response(results)


@router.post("/policy/check/llm/conversation", response_model=DebugCheckResponse,
             summary="[debug] Check full conversation (LLM-judged) — includes the prompt sent to each model")
async def debug_check_conversation_llm(body: ConversationRequest) -> DebugCheckResponse:
    if not await is_policy_check_enabled():
        return _DISABLED_RESPONSE
    policy = await policy_engine._get_active_policy()
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    _logger.info("debug check/llm/conversation policy=%s messages=%d", policy.name, len(body.messages))
    original = _original_policy_content(policy)
    results = await guard_classifier.classify_conversation_per_model_with_prompt(
        messages, policy=original, policy_by_family=_policy_by_family(policy, original),
    )
    verdicts = [v for v, _ in results]
    if not all(v.safe for v in verdicts):
        direction = "input" if messages[-1].role == "user" else "output"
        audit_log.record_violation(
            policy_id=policy.id, module=None, direction=direction,
            category=_unsafe_category(verdicts), action_taken="block", request_id=None, content=None,
        )
    return _to_debug_response(results)
