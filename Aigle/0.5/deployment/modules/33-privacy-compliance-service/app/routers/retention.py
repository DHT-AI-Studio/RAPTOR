"""GD-3 / VIE01-248: retention policy endpoints.

The gdpr-admin role check happens in Module 13's gateway (same as
erasure.py's admin routes) -- this router only trusts X-Internal-Token,
since Module 33 has no session of its own to check a JWT role against.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Header, Request

from app.models.retention import (
    RetentionDryRunResponse,
    RetentionPolicyListResponse,
    RetentionPolicyUpdate,
)
from app.services.retention_engine import RetentionEngine

router = APIRouter(prefix="/retention", tags=["retention"])


@router.get("/policy", response_model=RetentionPolicyListResponse)
async def get_policy(request: Request):
    engine: RetentionEngine = request.app.state.retention_engine
    return RetentionPolicyListResponse(policies=await engine.get_policies())


@router.put("/policy", response_model=RetentionPolicyListResponse)
async def update_policy(
    updates: list[RetentionPolicyUpdate],
    request: Request,
    x_requested_by: str = Header(None, alias="X-Requested-By"),
):
    engine: RetentionEngine = request.app.state.retention_engine
    policies = await engine.update_policies(
        [u.model_dump(exclude_unset=True) for u in updates],
        updated_by=x_requested_by or "unknown",
    )
    return RetentionPolicyListResponse(policies=policies)


@router.get("/policy/dry-run", response_model=RetentionDryRunResponse)
async def dry_run(request: Request):
    engine: RetentionEngine = request.app.state.retention_engine
    counts = await engine.dry_run()
    return RetentionDryRunResponse(counts=counts, as_of=datetime.now(timezone.utc))
