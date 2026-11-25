"""GD-1 / VIE01-246: internal erasure endpoint, called only by Module 33.

Distinct from DELETE /memory (management.py) -- that endpoint deletes the
*caller's own* memory via JWT (Depends(get_current_user)). This one takes
an explicit user_id and authenticates the caller (Module 33) via a shared
secret instead, since an admin-initiated erasure has no session of the
erased user's own to derive a JWT from.
"""
from __future__ import annotations

import json

from core.config import settings
from core.dependencies import get_redis
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from redis.asyncio import Redis

from routers.management import ManagementService, get_management_service

router = APIRouter(prefix="/internal/gdpr", tags=["gdpr"])


def require_internal_token(x_internal_token: str = Header(None, alias="X-Internal-Token")) -> None:
    if not settings.gdpr_internal_token or x_internal_token != settings.gdpr_internal_token:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Token")


@router.delete("/user/{user_id}", dependencies=[Depends(require_internal_token)])
async def erase_user(
    user_id: str,
    redis: Redis = Depends(get_redis),
    svc: ManagementService = Depends(get_management_service),
):
    found = await svc.delete_all(user_id, redis)
    return {"user_id": user_id, "deleted": found}


@router.delete("/memory/user/{user_id}", dependencies=[Depends(require_internal_token)])
async def erase_user_memory(
    user_id: str,
    redis: Redis = Depends(get_redis),
    svc: ManagementService = Depends(get_management_service),
):
    """GD-5 / VIE01-250: memory consent withdrawal -- a *partial* erasure
    (this module only), called by Module 33's consent cascade, not part of
    an Art. 17 full-account erasure. Everything this module stores is
    conversation memory, so today it deletes exactly what DELETE
    /internal/gdpr/user/{user_id} does; kept as its own route so the two
    legal bases can diverge without Module 33 changing."""
    found = await svc.delete_all(user_id, redis)
    return {"user_id": user_id, "deleted": found, "scope": "memory"}


@router.get("/user/{user_id}/export", dependencies=[Depends(require_internal_token)])
async def export_user(
    user_id: str,
    svc: ManagementService = Depends(get_management_service),
):
    """GD-2 / VIE01-247: JSON-serializable content for the export ZIP's
    memory/ section (sessions + long_term). Reuses export_generator's
    logic wholesale -- it already produces valid JSON once fully
    assembled, so parsing its output back into a dict is simpler and
    less error-prone than re-deriving the same session-shard-merging
    logic in a second, non-streaming code path."""
    raw = b"".join([chunk async for chunk in svc.export_generator(user_id)])
    return json.loads(raw)


@router.get("/user/{user_id}/summary", dependencies=[Depends(require_internal_token)])
async def summarize_user(
    user_id: str,
    svc: ManagementService = Depends(get_management_service),
):
    """GD-4 / VIE01-249: count-only summary for Module 33's DSAR assembler
    -- deliberately not /export: a DSAR only needs how much memory exists,
    and paying GD-2's full timeline read for that would blow the ticket's
    10-second DSAR budget on any real account."""
    return await svc.get_gdpr_summary(user_id)


@router.delete("/memory/sessions/expired",dependencies=[Depends(require_internal_token)])
async def expire_sessions(
    older_than_days: int = Query(..., ge=1),
    dry_run: bool = Query(False, description="GD-3: count only, don't delete -- powers Module 33's retention dry-run endpoint"),
    redis: Redis = Depends(get_redis),
    svc: ManagementService = Depends(get_management_service),
):
    """GD-3 / VIE01-248: called daily by Module 33's retention engine --
    system-wide sweep across every user's storage, not one user's own
    memory like DELETE /memory or DELETE /internal/gdpr/user/{user_id}."""
    count = await svc.expire_sessions(older_than_days, redis, dry_run=dry_run)
    return {"sessions_deleted": count, "dry_run": dry_run}
