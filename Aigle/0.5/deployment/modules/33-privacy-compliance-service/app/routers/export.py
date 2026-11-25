"""GD-2 / VIE01-247: internal export endpoints.

Identity comes from headers Module 13 injects after its own JWT check --
same trust pattern erasure.py already uses. This router is never reachable
from outside the raptor network.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from app.models.job import ExportSubmitResponse
from app.services.export_orchestrator import ExportOrchestrator

logger = logging.getLogger("privacy_compliance.export_router")
router = APIRouter(prefix="/export", tags=["export"])


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    return x_user_id


@router.post("", response_model=ExportSubmitResponse)
async def submit_export(
    request: Request,
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_requested_by: str = Header(None, alias="X-Requested-By"),
):
    """Submit an export job for x_user_id. Always starts a fresh job --
    unlike erasure, an export has no "already done" terminal state worth
    deduplicating against; the data may have changed since the last export."""
    user_id = _require_user(x_user_id)
    requested_by = x_requested_by or user_id
    orchestrator: ExportOrchestrator = request.app.state.export_orchestrator

    job = await orchestrator.create_job(user_id, requested_by)
    request.app.state.background_tasks.add(job.job_id)
    asyncio.create_task(_run_and_untrack(request.app, orchestrator, job.job_id))
    return ExportSubmitResponse(job_id=job.job_id, status="pending")


async def _run_and_untrack(app, orchestrator: ExportOrchestrator, job_id: str) -> None:
    try:
        await orchestrator.run(job_id)
    finally:
        app.state.background_tasks.discard(job_id)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    orchestrator: ExportOrchestrator = request.app.state.export_orchestrator
    job = await orchestrator.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
