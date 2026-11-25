"""Policy-aware guard check endpoints — `/api/v0.2/check/*`.

The policy-aware counterpart of the deprecated `/guard/check/*`
(app/routers/check_guard.py): same request and `CheckResponse` shape, but the
active administrator policy is applied. Callers integrate with
`POST /guardrail/policy/decide` (the same check, answered as
allow | redact | warn | block); this shape stays for consumers that need
`CheckResponse`, and `/api/v0.2/check/conversation` is the successor of
`/guard/check/conversation` (decide takes a single message).

Each endpoint auto-selects its behaviour per request:

  * **Active policy exists** (and the narrower `guardrail:policy_enabled`
    switch is on) → per-model, policy-injected guard prompts, the same as
    app/routers/check_policy_llm.py's `/policy/check/llm/*`.
  * **No active policy** (or policy-checking is switched off) → the fixed
    guard prompts, the same as `/guard/check/*`.

The checking itself lives in app/services/policy_check_service.py (regex
pass → active policy → guard model(s) → GB-16 conflict resolution → rule
engine → audit), shared with `POST /guardrail/policy/decide` so the two can
never disagree; this file only shapes its outcome into `CheckResponse`.

**Final `safe` verdict**: the combined guard verdict is *not* what decides
the response's `safe` field — a rule-engine-shaped active policy re-evaluates
it (`policy_check_service.final_verdict()`) and that decision replaces the
guard verdict's `safe` for the response and the audit trail. The other
fields (`categories`, `raw`, ...) are still the combined guard verdict's.

The `/debug/check/*` variants are functionally identical, but use
`app.services.guard_classifier`'s `*_with_prompt` variants and return
`DebugCheckResponse` (see app/models/debug.py) — the exact, human-readable
prompt actually sent to each model, in addition to its verdict.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.adapters.base import ChatMessage
from app.models.debug import DebugCheckResponse, DebugSingleModelResult
from app.models.guard import (
    CheckResponse,
    ConversationRequest,
    MessageRequest,
    combined_to_response,
)
from app.services import guard_classifier, policy_check_service
from app.services.guard_classifier import ModelVerdict
from app.services.policy_check_service import PolicyCheckOutcome

router = APIRouter(tags=["Guard check (policy-aware)"])

# Global guardrail switch off → pass immediately, no regex/policy lookup/Ollama call
_DISABLED_RESPONSE = CheckResponse(safe=True, categories=[], category_names={}, raw="guardrail disabled")
_DEBUG_DISABLED_RESPONSE = DebugCheckResponse(safe=True, categories=[], category_names={}, raw="guardrail disabled")


# ── Shared helpers ──────────────────────────────────────────────────────────────

def _to_response(outcome: PolicyCheckOutcome) -> CheckResponse:
    if outcome.disabled:
        return _DISABLED_RESPONSE
    return combined_to_response(outcome.combined).model_copy(update={"safe": outcome.safe})


def _to_debug_response(results: list[tuple[ModelVerdict, str]]) -> DebugCheckResponse:
    """Same as debug_policy_check_llm.py's helper of the same name."""
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


def _to_debug(outcome: PolicyCheckOutcome) -> DebugCheckResponse:
    if outcome.disabled:
        return _DEBUG_DISABLED_RESPONSE
    return _to_debug_response(outcome.results).model_copy(update={"safe": outcome.safe})


def _messages(body: ConversationRequest) -> list[ChatMessage]:
    return [ChatMessage(role=m.role, content=m.content) for m in body.messages]


# ── /api/v0.2/check/* ────────────────────────────────────────────────────────────

@router.post("/check/input", response_model=CheckResponse,
             summary="Check user input against the active policy (CheckResponse shape)")
async def check_input(body: MessageRequest) -> CheckResponse:
    return _to_response(await policy_check_service.evaluate_message(
        body.content, "input", module=body.module, request_id=body.request_id))


@router.post("/check/output", response_model=CheckResponse,
             summary="Check an AI response against the active policy (CheckResponse shape)")
async def check_output(body: MessageRequest) -> CheckResponse:
    return _to_response(await policy_check_service.evaluate_message(
        body.content, "output", module=body.module, request_id=body.request_id))


@router.post("/check/conversation", response_model=CheckResponse,
             summary="Check a full conversation against the active policy (replaces /guard/check/conversation)")
async def check_conversation(body: ConversationRequest) -> CheckResponse:
    return _to_response(await policy_check_service.evaluate_conversation(_messages(body)))


# ── /api/v0.2/debug/check/* ──────────────────────────────────────────────────────

@router.post("/debug/check/input", response_model=DebugCheckResponse,
             summary="[debug] Same as /check/input — response includes the prompt sent to each model")
async def debug_check_input(body: MessageRequest) -> DebugCheckResponse:
    return _to_debug(await policy_check_service.evaluate_message(
        body.content, "input", module=body.module, request_id=body.request_id, with_prompts=True))


@router.post("/debug/check/output", response_model=DebugCheckResponse,
             summary="[debug] Same as /check/output — response includes the prompt sent to each model")
async def debug_check_output(body: MessageRequest) -> DebugCheckResponse:
    return _to_debug(await policy_check_service.evaluate_message(
        body.content, "output", module=body.module, request_id=body.request_id, with_prompts=True))


@router.post("/debug/check/conversation", response_model=DebugCheckResponse,
             summary="[debug] Same as /check/conversation — response includes the prompt sent to each model")
async def debug_check_conversation(body: ConversationRequest) -> DebugCheckResponse:
    return _to_debug(await policy_check_service.evaluate_conversation(_messages(body), with_prompts=True))
