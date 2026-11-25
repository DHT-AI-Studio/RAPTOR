"""VIE01-369: write-freeze maintenance mode, enforced at the data owner itself.

Module 13's own MaintenanceModeMiddleware only sees requests that go
through the gateway -- but 15 (chat), 33 (privacy-compliance), and 34
(realtime-audio) all call this service directly
(http://raptor-memory-service:8026/...), bypassing 13 entirely. A
gateway-side freeze can't see those calls at all, so the freeze has to be
enforced here too, where every write -- no matter which service sent it --
has to land eventually.

Shares the exact same Redis key as 13's middleware
(REDIS_KEY = "maintenance:write_freeze") and the same Redis instance
(settings.redis_host, confirmed the same raptor-redis-standalone both
already connect to) -- one enable/disable call from
scripts/nfs-freeze-archive.sh freezes both layers at once, no separate
coordination needed.
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
# Every route in this service writes under NFS_BASE_PATH/memvid_storage except
# these -- POST-shaped reads (search/evaluate), confirmed by reading each
# router: global_search, search_multimedia, and the two compact "evaluate"
# (dry-run) endpoints never touch storage.
_READ_ONLY_POSTS = re.compile(
    r"^/memory/(?:search|multimedia/search|compact/evaluate|sessions/[^/]+/compact/evaluate)/?$"
)


async def is_enabled(redis) -> bool:
    try:
        return bool(await redis.exists(REDIS_KEY))
    except Exception as exc:  # noqa: BLE001 -- a broken/incomplete redis_client must never itself take down every write
        _logger.warning(f"Maintenance-mode Redis check failed: {exc} — allowing the write")
        return False  # fail open, same convention 13's middleware uses


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method.upper() not in WRITE_METHODS or _READ_ONLY_POSTS.match(request.url.path):
            return await call_next(request)
        redis = getattr(request.app.state, "redis_client", None)
        if redis is None or not await is_enabled(redis):
            return await call_next(request)
        return JSONResponse(status_code=503, content={"reason": "maintenance_write_freeze"})
