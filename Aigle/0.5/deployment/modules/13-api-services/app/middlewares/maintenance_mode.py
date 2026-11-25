"""VIE01-369: write-freeze maintenance mode.

Lets a host-level backup script (scripts/nfs-freeze-archive.sh) stop new
writes from reaching 04/26 through the gateway's own synchronous HTTP
path, without touching any container's lifecycle -- no docker stop/start,
so there's no "container didn't come back up" failure mode to worry
about. The flag lives in Redis (shared across however many gateway
processes/replicas there are, unlike an in-process bool), set with a TTL
so a crashed caller that never disables it can't leave the gateway
write-frozen forever.

Scoped to /asset and /memory only -- confirmed (grep for
asset_management_url/memory_service_url across every router) that no
other router talks to modules 04/26 at all, so freezing e.g. /chat,
/training, /gdpr or /personal-db during a 01 (NFS) backup would block
writes that were never at risk in the first place; those write to 03's
Postgres, which has its own MVCC-consistent pg_dump and needs no
request-level freeze. Registered after LegacyApiAliasMiddleware's rewrite
(see main.py's middleware-order comment) so this sees the canonical path
and covers /api/0.3/{asset,memory}/* too, not just /api/{v}/*.

Does not stop everything that can write under NFS_BASE_PATH -- SeaweedFS's
own background vacuum/balance/erasure-coding worker writes independently
of any request through this gateway. See docs/_internal/backup-restore-v0.5.md's
"01" section.
"""
from __future__ import annotations

import logging
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.middlewares.processing_restriction import WRITE_METHODS

_logger = logging.getLogger(__name__)

REDIS_KEY = "maintenance:write_freeze"
DEFAULT_TTL_SECONDS = 600
_FROZEN_GROUPS = re.compile(r"^/api/[^/]+/(?:asset|memory)(?:/|$)")


async def enable(redis, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    await redis.set(REDIS_KEY, "1", ex=ttl_seconds)


async def disable(redis) -> None:
    await redis.delete(REDIS_KEY)


async def is_enabled(redis) -> bool:
    try:
        return bool(await redis.exists(REDIS_KEY))
    except Exception as exc:  # noqa: BLE001 -- a broken/incomplete redis_client must never itself take down every write
        _logger.warning(f"Maintenance-mode Redis check failed: {exc} — allowing the write")
        return False  # fail open, same convention as the rate limiter


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method.upper() not in WRITE_METHODS or not _FROZEN_GROUPS.match(request.url.path):
            return await call_next(request)
        redis = getattr(request.app.state, "redis_client", None)
        if redis is None or not await is_enabled(redis):
            return await call_next(request)
        return JSONResponse(status_code=503, content={"reason": "maintenance_write_freeze"})
