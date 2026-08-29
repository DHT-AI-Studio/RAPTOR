from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from core.dependencies import get_current_user
from models.memory import (
    CompactEvaluateRequest,
    CompactEvaluation,
    CompactRequest,
    CompactResponse,
    SessionCompactEvaluateRequest,
    SummaryFrame,
)
from services.compact_memory import CompactMemoryService

router = APIRouter(prefix="/memory", tags=["Compact"])


def get_compact_service() -> CompactMemoryService:
    return CompactMemoryService()


@router.post("/compact/evaluate", response_model=CompactEvaluation)
async def evaluate_compact(
    body: CompactEvaluateRequest,
    user_id: str = Depends(get_current_user),
    svc: CompactMemoryService = Depends(get_compact_service),
) -> CompactEvaluation:
    """Estimate token budget without writing anything.

    Without `session_id`: sizes only the supplied `messages` list (legacy mode).
    With `session_id`: aggregates that session's archived turns + the user's
    long-term facts + multimedia snippets — the real combined context
    compactdesign.md §7/§8.1 describes, equivalent to calling
    POST /memory/sessions/{session_id}/compact/evaluate directly.
    """
    if body.session_id:
        return await svc.evaluate_session(
            user_id, body.session_id, body.context_window, body.extra_tokens, body.max_tokens
        )
    return svc.evaluate(body.messages, body.context_window, body.extra_tokens, body.max_tokens)


@router.post("/sessions/{session_id}/compact/evaluate", response_model=CompactEvaluation)
async def evaluate_session_compact(
    session_id: str,
    body: SessionCompactEvaluateRequest,
    user_id: str = Depends(get_current_user),
    svc: CompactMemoryService = Depends(get_compact_service),
) -> CompactEvaluation:
    """Estimate the real combined-context token budget for one session (session
    turns + long-term facts + multimedia snippets), without writing anything."""
    return await svc.evaluate_session(
        user_id, session_id, body.context_window, body.extra_tokens, body.max_tokens
    )


@router.post("/sessions/{session_id}/compact", response_model=CompactResponse)
async def compact_session(
    session_id: str,
    body: CompactRequest,
    user_id: str = Depends(get_current_user),
    svc: CompactMemoryService = Depends(get_compact_service),
) -> CompactResponse:
    """Compact one session using stored session memory or LLM fallback."""
    return await svc.compact_session(
        user_id=user_id,
        session_id=session_id,
        context_window=body.context_window,
        trigger=body.trigger,
        max_tokens=body.max_tokens,
        last_summarized_frame_id=body.last_summarized_frame_id,
        custom_instructions=body.custom_instructions,
        dry_run=body.dry_run,
    )


@router.get("/sessions/{session_id}/summaries", response_model=list[SummaryFrame])
async def get_session_summaries(
    session_id: str,
    user_id: str = Depends(get_current_user),
    svc: CompactMemoryService = Depends(get_compact_service),
) -> list[SummaryFrame]:
    """List compact summary frames for a session."""
    return await svc.get_session_summaries(user_id, session_id)


@router.delete(
    "/sessions/{session_id}/summaries/{summary_id}",
    status_code=204,
    response_class=Response,
)
async def delete_session_summary(
    session_id: str,
    summary_id: str,
    user_id: str = Depends(get_current_user),
    svc: CompactMemoryService = Depends(get_compact_service),
) -> Response:
    """Delete one summary frame (soft-delete via meta-index)."""
    found = await svc.delete_summary(user_id, session_id, summary_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Summary not found")
    return Response(status_code=204)
