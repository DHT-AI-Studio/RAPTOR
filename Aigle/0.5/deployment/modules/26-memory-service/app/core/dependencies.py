from __future__ import annotations

import logging
import re

from fastapi import Header, HTTPException, Request, status
from redis.asyncio import Redis

_logger = logging.getLogger(__name__)

# user_id is interpolated directly into filesystem paths as f"user_{user_id}"
# (see app/core/paths.py) — restrict it to a safe charset so a malformed or
# malicious header can never contain "/", "..", or other path-traversal
# sequences that would let one user's request touch another user's directory.
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def get_redis(request: Request) -> Redis:
    redis = getattr(request.app.state, "redis_client", None)
    if not redis:
        raise RuntimeError("Redis not initialised")
    return redis


async def get_current_user(x_user_id: str = Header(..., alias="X-User-ID")) -> str:
    """Resolve user_id from the X-User-ID header.

    Authentication (JWT / Keycloak) is handled by Module 13 (API Gateway); this
    service trusts that only the gateway can reach it and injects this header
    after verifying the caller — the same convention used by Module 04 (Object
    Storage). This service does not re-verify a token or accept a
    client-supplied user_id anywhere else.

    The value is still validated against a safe charset before use: it is
    interpolated directly into on-disk paths, so an unvalidated value would be
    a directory-traversal vector regardless of how much the gateway is
    trusted.
    """
    if not x_user_id.strip():
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-User-ID header")
    if not _SAFE_USER_ID.match(x_user_id):
        _logger.warning("Rejected X-User-ID with unsafe characters: %r", x_user_id)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid X-User-ID")
    return x_user_id
