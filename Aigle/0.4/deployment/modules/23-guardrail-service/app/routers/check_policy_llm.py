"""Content safety endpoints — LLM-judged variant.

Same request/response shapes as app/routers/check_guard.py — this module
reuses app/services/guard_classifier.py's multi-model orchestration and its
combine() step, so the response is the same CheckResponse shape as
/guard/check/*. Unlike check_guard.py, each active guard model here is given
its OWN policy prompt: the active policy's per-model override (llama-guard /
granite-guardian / gpt-oss-safeguard column — see app/engine/models.py's
PolicyDetail) if set, falling back to the shared "original" content
otherwise. Each model is still classified independently (classify_per_model,
not classify) since different models may be judged against genuinely
different prompts — only the final response shaping goes through the same
combine()/combined_to_response() as check_guard.py.

This exists alongside check_policy.py, not instead of it — check_policy.py's
deterministic Python rule-matching remains the audit-friendly default path;
these endpoints are the model-judged alternative.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.adapters.base import ChatMessage, PolicyContent
from app.adapters.standard_policy import parse_standard_policies
from app.engine import policy_engine
from app.engine.models import PolicyDetail
from app.engine.policy_targets import COLUMN_BY_ADAPTER_FAMILY
from app.models.guard import CheckResponse, ConversationRequest, MessageRequest, combined_to_response
from app.services import audit_log, guard_classifier
from app.services.guard_classifier import ModelVerdict
from app.services.state import is_policy_check_enabled

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Check (LLM-judged)"])

# Global guardrail switch off, or the narrower policy-check switch off →
# pass immediately, no policy lookup/Ollama call
_DISABLED_RESPONSE = CheckResponse(safe=True, categories=[], category_names={}, raw="guardrail disabled")


def _original_policy_content(policy: PolicyDetail) -> PolicyContent:
    """The shared "original" content, with a best-effort attempt to parse it
    as the Standard Policy Format (app.adapters.standard_policy) — if it
    parses, every family without its own override converts it into that
    model's own official prompt structure; if not (legacy free-text/YAML),
    it's used verbatim exactly as before."""
    return PolicyContent(raw=policy.raw_content, standard_policies=parse_standard_policies(policy.raw_content))


def _policy_by_family(policy: PolicyDetail, original: PolicyContent) -> dict[str, PolicyContent]:
    """Map each guard-adapter family (app.engine.policy_targets.COLUMN_BY_ADAPTER_FAMILY
    — the single source of truth, also used by the upload dropdown) to its
    policy content: the model-specific override if set on the active policy
    (always verbatim, never re-parsed as standard format), otherwise the
    shared "original" content."""
    return {
        family: PolicyContent(raw=getattr(policy, column)) if getattr(policy, column) else original
        for family, column in COLUMN_BY_ADAPTER_FAMILY.items()
    }


def _unsafe_category(results: list[ModelVerdict]) -> str | None:
    codes = sorted({c for r in results if not r.safe for c in r.categories})
    return ",".join(codes) if codes else None


@router.post("/check/llm/input", response_model=CheckResponse, summary="Check user input (LLM-judged)")
async def check_input_llm(body: MessageRequest) -> CheckResponse:
    """
    Classify **user input**, giving each active guard model its own policy
    prompt (model-specific override on the active policy, or the shared
    "original" content when no override is set).
    """
    if not await is_policy_check_enabled():
        return _DISABLED_RESPONSE
    policy = await policy_engine._get_active_policy()
    _logger.info("check/llm/input policy=%s content=%s", policy.name, body.content[:100])
    original = _original_policy_content(policy)
    results = await guard_classifier.classify_per_model(
        body.content, role="user",
        policy=original, policy_by_family=_policy_by_family(policy, original),
    )
    if not all(r.safe for r in results):
        audit_log.record_violation(
            policy_id=policy.id, module=None, direction="input",
            category=_unsafe_category(results), action_taken="block", request_id=None, content=None,
        )
    return combined_to_response(guard_classifier.combine(results))


@router.post("/check/llm/output", response_model=CheckResponse, summary="Check AI response (LLM-judged)")
async def check_output_llm(body: MessageRequest) -> CheckResponse:
    """Classify **AI-generated output** the same way as `/check/llm/input`."""
    if not await is_policy_check_enabled():
        return _DISABLED_RESPONSE
    policy = await policy_engine._get_active_policy()
    _logger.info("check/llm/output policy=%s content=%s", policy.name, body.content[:100])
    original = _original_policy_content(policy)
    results = await guard_classifier.classify_per_model(
        body.content, role="assistant",
        policy=original, policy_by_family=_policy_by_family(policy, original),
    )
    if not all(r.safe for r in results):
        audit_log.record_violation(
            policy_id=policy.id, module=None, direction="output",
            category=_unsafe_category(results), action_taken="block", request_id=None, content=None,
        )
    return combined_to_response(guard_classifier.combine(results))


@router.post("/check/llm/conversation", response_model=CheckResponse,
             summary="Check full conversation (LLM-judged)")
async def check_conversation_llm(body: ConversationRequest) -> CheckResponse:
    """Classify a **complete multi-turn conversation**, the same way as `/guard/check/conversation`."""
    if not await is_policy_check_enabled():
        return _DISABLED_RESPONSE
    policy = await policy_engine._get_active_policy()
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    _logger.info("check/llm/conversation policy=%s messages=%d", policy.name, len(body.messages))
    original = _original_policy_content(policy)
    results = await guard_classifier.classify_conversation_per_model(
        messages, policy=original, policy_by_family=_policy_by_family(policy, original),
    )
    if not all(r.safe for r in results):
        direction = "input" if messages[-1].role == "user" else "output"
        audit_log.record_violation(
            policy_id=policy.id, module=None, direction=direction,
            category=_unsafe_category(results), action_taken="block", request_id=None, content=None,
        )
    return combined_to_response(guard_classifier.combine(results))
