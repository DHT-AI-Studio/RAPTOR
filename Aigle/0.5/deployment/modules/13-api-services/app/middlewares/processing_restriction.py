"""PRIV-X3 / VIE01-357: restriction of processing (GDPR Art. 18).

If the caller's Keycloak attribute gdpr_processing_restricted is true (set
by POST /gdpr/restrict, removed only by a gdpr-admin's /gdpr/restrict/lift),
every write endpoint answers 451 Unavailable For Legal Reasons with
{"reason": "processing_restricted"}; read endpoints remain available.

Module 13 has no Keycloak admin access, so the attribute is read through
Module 33 (GET /restrict/status, which reads it from Keycloak) -- once per
write request; reads never trigger the lookup.

"Write" means POST/PUT/PATCH/DELETE, except:
- /gdpr/*: a restricted user must still be able to export, erase, object,
  read or change consent -- restriction never blocks the user's own rights;
- /sso/*: login and logout;
- POST routes that only read: /search/*, /asset/scan,
  /memory/multimedia/search, /memory/compact/evaluate, /temporal/verify.
Everything else that can create or change data is blocked -- including chat,
A2A and MCP calls, which store conversation memory.
"""
from __future__ import annotations

import logging
import re

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import get_settings
from app.core.security import verify_jwt

_logger = logging.getLogger(__name__)

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_ALLOWED_GROUPS = re.compile(r"^/api/[^/]+/(?:gdpr|sso|search)(?:/|$)")
_READ_ONLY_POSTS = re.compile(
    r"^/api/[^/]+/(?:asset/scan|memory/multimedia/search|memory/compact/evaluate|temporal/verify|temporal/parse)/?$")
_STATUS_TIMEOUT = 5.0


def is_restricted_write(method: str, path: str) -> bool:
    """Whether a request of this method and path is blocked for a restricted user."""
    return (
        method.upper() in WRITE_METHODS
        and not _ALLOWED_GROUPS.match(path)
        and not _READ_ONLY_POSTS.match(path)
    )


async def is_restricted(client: httpx.AsyncClient, user_id: str) -> bool:
    """The user's Keycloak attribute gdpr_processing_restricted, via Module 33.
    Raises httpx.HTTPError if it can't be read."""
    settings = get_settings()
    resp = await client.get(
        f"{settings.privacy_compliance_url}/restrict/status",
        headers={"X-Internal-Token": settings.gdpr_internal_token, "X-User-ID": user_id},
        timeout=_STATUS_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["restricted"] is True


class ProcessingRestrictionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not is_restricted_write(request.method, request.url.path):
            return await call_next(request)
        try:
            user_id = verify_jwt(request, settings=get_settings()).get("sub")
        except HTTPException:
            user_id = None
        if not user_id:
            return await call_next(request)

        try:
            restricted = await is_restricted(request.app.state.http_client, user_id)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            _logger.error("Processing-restriction check failed for user=%s -- refusing the write: %s", user_id, exc)
            return JSONResponse(status_code=503, content={"reason": "processing_restriction_check_unavailable"})
        if restricted:
            return JSONResponse(status_code=451, content={"reason": "processing_restricted"})
        return await call_next(request)
