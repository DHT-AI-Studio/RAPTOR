"""GD-1 / VIE01-246: internal erasure endpoints.

Identity comes from headers Module 13 injects after its own JWT + role
check -- same trust pattern modules 04/25/26 already use for X-User-ID.
This router is never reachable from outside the raptor network.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from app.models.job import ErasureSubmitResponse, JobStatus
from app.services.erasure_orchestrator import ErasureOrchestrator
from app.services.legal_hold_state import active_hold

logger = logging.getLogger("privacy_compliance.erasure_router")
router = APIRouter(prefix="/erasure", tags=["erasure"])


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    return x_user_id


@router.post("", response_model=ErasureSubmitResponse)
async def submit_erasure(
    request: Request,
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_requested_by: str = Header(None, alias="X-Requested-By"),
):
    """Submit an erasure job for x_user_id. Idempotent: an already-erased
    (or in-flight, or deferred) user returns the existing job instead of
    starting a second, overlapping one.

    PRIV-X4 / VIE01-358: under an active legal hold the request is recorded
    as a deferred job and answered DEFERRED; it runs once the hold ends.
    Every answer lists in exception_fields what an erasure keeps by law."""
    user_id = _require_user(x_user_id)
    requested_by = x_requested_by or user_id
    orchestrator: ErasureOrchestrator = request.app.state.orchestrator
    retained = await orchestrator.retained_records()

    existing = await orchestrator.find_active_or_completed_job(user_id)
    if existing is not None and existing.status == JobStatus.DEFERRED:
        return await _deferred(orchestrator, existing.job_id, user_id, retained)
    if existing is not None and existing.status in (
        JobStatus.PENDING, JobStatus.IN_PROGRESS, JobStatus.COMPLETED, JobStatus.PARTIAL_FAILURE,
    ):
        if existing.status == JobStatus.COMPLETED:
            return ErasureSubmitResponse(job_id=existing.job_id, status="already_erased", exception_fields=retained)
        return ErasureSubmitResponse(
            job_id=existing.job_id, status=existing.status.value,
            estimated_completion_at=existing.estimated_completion_at, exception_fields=retained,
        )

    if await active_hold(orchestrator.pool, user_id) is not None:
        job = await orchestrator.create_job(user_id, requested_by, status=JobStatus.DEFERRED)
        return await _deferred(orchestrator, job.job_id, user_id, retained)

    job = await orchestrator.create_job(user_id, requested_by)
    start_erasure(request.app, job.job_id)
    return ErasureSubmitResponse(
        job_id=job.job_id, status="pending", estimated_completion_at=job.estimated_completion_at,
        exception_fields=retained,
    )


async def _deferred(
    orchestrator: ErasureOrchestrator, job_id: str, user_id: str, retained: list[str],
) -> ErasureSubmitResponse:
    """The ticket's {"status": "DEFERRED", "reason": "legal_hold", "hold_until": ...}.
    hold_until is null once the hold has ended but the daily expiry job
    hasn't run the erasure yet."""
    hold = await active_hold(orchestrator.pool, user_id)
    return ErasureSubmitResponse(
        job_id=job_id, status="DEFERRED", reason="legal_hold",
        hold_until=hold.hold_until if hold else None, exception_fields=retained,
    )


def start_erasure(app, job_id: str) -> None:
    """Run one erasure job in the background, tracked until it finishes."""
    orchestrator: ErasureOrchestrator = app.state.orchestrator
    app.state.background_tasks.add(job_id)
    asyncio.create_task(_run_and_untrack(app, orchestrator, job_id))


async def _run_and_untrack(app, orchestrator: ErasureOrchestrator, job_id: str) -> None:
    try:
        await orchestrator.run(job_id)
    finally:
        app.state.background_tasks.discard(job_id)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    orchestrator: ErasureOrchestrator = request.app.state.orchestrator
    job = await orchestrator.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request):
    orchestrator: ErasureOrchestrator = request.app.state.orchestrator
    job = await orchestrator.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PARTIAL_FAILURE:
        raise HTTPException(status_code=409, detail=f"Job is {job.status.value}, not eligible for retry")
    # PRIV-X4: a retry deletes data too, so a legal hold stops it as well --
    # decided under the lock placing a hold takes.
    hold, claimed = await orchestrator.claim_retry(job_id, job.user_id)
    if hold is not None:
        raise HTTPException(
            status_code=409,
            detail=f"The user is under a legal hold until {hold.hold_until.isoformat()}; retry once it ends",
        )
    if not claimed:
        raise HTTPException(status_code=409, detail="Job is already being retried")
    return await orchestrator.retry(job_id)
