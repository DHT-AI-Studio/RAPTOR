"""VIE01-369: write-freeze maintenance mode, enforced at the data owner itself.

Module 13's own MaintenanceModeMiddleware only sees requests that go
through the gateway -- but 21 (agent-protocol), 33 (privacy-compliance),
and 09-12's Kafka-driven processing workers all call this service directly
(http://raptor-asset-management:8000/...), bypassing 13 entirely. A
gateway-side freeze can't see those calls at all, so the freeze has to be
enforced here too, where every write -- no matter which service sent it --
has to land eventually.

Shares the exact same Redis key as 13's and 26's middleware
(REDIS_KEY = "maintenance:write_freeze") and the same Redis instance
(settings.redis_host -- raptor-redis-standalone). Deliberately a new,
dedicated connection, not object_store._redis: that one is disabled
whenever PRESIGN_URL_CACHE_ENABLED=false, which has nothing to do with
this and would silently defeat the freeze check if reused.
"""
from __future__ import annotations

import logging
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_logger = logging.getLogger(__name__)

REDIS_KEY = "maintenance:write_freeze"
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# POST-shaped reads, confirmed by reading each handler: /presign-urls-batch
# and /scan never touch LakeFS/Postgres state, just resolve/compare existing data.
_READ_ONLY_POSTS = re.compile(r"^/(?:presign-urls-batch|scan)/?$")


async def is_enabled(redis) -> bool:
    try:
        return bool(await redis.exists(REDIS_KEY))
    except Exception as exc:  # noqa: BLE001 -- a broken/incomplete redis_client must never itself take down every write
        _logger.warning(f"Maintenance-mode Redis check failed: {exc} — allowing the write")
        return False  # fail open, same convention 13's/26's middleware use


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method.upper() not in WRITE_METHODS or _READ_ONLY_POSTS.match(request.url.path):
            return await call_next(request)
        redis = getattr(request.app.state, "redis_client", None)
        if redis is None or not await is_enabled(redis):
            return await call_next(request)
        return JSONResponse(status_code=503, content={"reason": "maintenance_write_freeze"})
