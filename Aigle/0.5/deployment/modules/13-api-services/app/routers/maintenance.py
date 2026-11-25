"""VIE01-369: internal admin endpoints -- write-freeze maintenance mode.

Never reachable from outside the raptor network (X-Internal-Token, on the
router). Called by scripts/nfs-freeze-archive.sh around the window it
needs 04/26 to stop writing, not by any end user.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.middlewares.maintenance_mode import (
    DEFAULT_TTL_SECONDS,
    disable,
    enable,
    is_enabled,
)

router = APIRouter(prefix="/admin/maintenance", tags=["maintenance"], include_in_schema=False)


@router.post("/enable")
async def enable_maintenance(request: Request, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Starts (or renews) the write-freeze. ttl_seconds is a safety net, not
    a scheduled duration -- the caller is still expected to call /disable
    itself once its own archive step is done; this just bounds how long a
    crashed caller can leave writes frozen for."""
    await enable(request.app.state.redis_client, ttl_seconds)
    return {"enabled": True, "ttl_seconds": ttl_seconds}


@router.post("/disable")
async def disable_maintenance(request: Request):
    await disable(request.app.state.redis_client)
    return {"enabled": False}


@router.get("/status")
async def maintenance_status(request: Request):
    return {"enabled": await is_enabled(request.app.state.redis_client)}
