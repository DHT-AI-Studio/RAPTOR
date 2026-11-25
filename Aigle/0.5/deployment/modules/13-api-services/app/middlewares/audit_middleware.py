"""GD-7 / VIE01-252: data access audit log (GDPR Art. 30).

Every authenticated request to a personal-data route --
/api/{v}/personal-db/*, /memory/*, /benchmark/*, /gdpr/*, /search/* -- is
recorded once its response is ready: whose data (user_id), who asked
(accessor_id, the JWT sub), endpoint, method, data categories, purpose,
time and a SHA-256 of the caller's IP. The record is sent to Module 33
(which owns gdpr_access_log, like every gdpr_* table) from a background
task, so auditing never adds latency to the request or fails it.

Recorded only when the request was authenticated and succeeded (status
< 400): a 401/403/404 never reached anyone's data. Identity is the payload
get_current_user() verified and left on request.state.user -- this
middleware never decodes a token itself.

Same BaseHTTPMiddleware pattern as GB-19's GuardrailMiddleware
(guardrail.py), registered inside LegacyApiAliasMiddleware so it only ever
sees canonical /api/{v}/ paths.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# The ticket's five path groups; /personal/* is this gateway's /personal-db/*.
# `moderation` added for CM-5 / VIE01-408 -- an admin reviewing a flag reaches the
# actual uploaded asset (asset_url), not just metadata; that access was previously
# invisible to both this audit trail and the data subject's own GET /gdpr/audit-log/my.
# `compliance` added for CM-7 / VIE01-410: the compliance report lists many people's
# (hashed) GDPR records and access-log rows -- an admin reading them is an access too.
_AUDITED = re.compile(r"^/api/[^/]+/(personal-db|memory|benchmark|gdpr|search|moderation|compliance)(?:/|$)")
# Admin routes acting on another user's data: /gdpr/{erasure|export|dsar}/{user_id}
# (not .../jobs/{job_id}, whose subject is resolved from the job, not the path).
_ADMIN_SUBJECT = re.compile(r"^/api/[^/]+/gdpr/(?:erasure|export|dsar)/(?!jobs(?:/|$))([^/]+)/?$")

# What a DSAR / export / erasure covers -- the GD-4 report's data categories.
ALL_CATEGORIES = ["identity", "uploaded_files", "indexed_content", "memory", "benchmark"]

# /gdpr/<sub-route> -> (data_categories, purpose)
_GDPR_ROUTES: dict[str, tuple[list[str], str]] = {
    "dsar": (ALL_CATEGORIES, "right_of_access"),
    "audit-log": (["audit_log"], "right_of_access"),
    "export": (ALL_CATEGORIES, "data_portability"),
    "erasure": (ALL_CATEGORIES, "right_to_erasure"),
    "consent": (["consent"], "consent_management"),
    "retention": ([], "retention_management"),
    # PRIV-X3 / VIE01-357
    "rectify": (["identity"], "rectification"),
    "restrict": ([], "restriction_of_processing"),
    "object": (["consent"], "objection"),
    # V05-10 / VIE01-364: the operator job list
    "jobs": (["gdpr_records"], "gdpr_administration"),
}
# Everything else: the user using the product, which reads their data to serve them.
_SERVICE_ROUTES: dict[str, list[str]] = {
    "personal-db": ["indexed_content"],
    "search": ["indexed_content"],
    "memory": ["memory"],
    "benchmark": ["benchmark"],
}
SERVICE_PROVISION = "service_provision"


def classify(path: str) -> tuple[list[str], str]:
    """(data_categories, purpose) for an audited path, inferred from the path alone."""
    parts = path.strip("/").split("/")          # ["api", v, group, sub, ...]
    group = parts[2] if len(parts) > 2 else ""
    if group == "gdpr":
        sub = parts[3] if len(parts) > 3 else ""
        categories, purpose = _GDPR_ROUTES.get(sub, ([], "gdpr_administration"))
        return list(categories), purpose
    if group == "moderation":
        # Admin review, not the uploader's own use -- the flagged asset itself
        # (uploaded_files), never just a policy/system-config change.
        return ["uploaded_files"], "content_moderation"
    if group == "compliance":
        return ["audit_log", "gdpr_records"], "compliance_reporting"
    return list(_SERVICE_ROUTES.get(group, [])), SERVICE_PROVISION


def subject_of(request: Request, accessor_id: str) -> str | None:
    """Whose data was accessed: the {user_id} of an admin route, the user an
    admin filtered the audit log or job list on, or else the caller
    themselves. None for routes about no single person (an unfiltered admin
    listing, retention policy, job summary)."""
    path = request.url.path
    if match := _ADMIN_SUBJECT.match(path):
        return match.group(1)
    if re.search(r"/gdpr/(?:audit-log|jobs)/?$", path):
        return request.query_params.get("user_id")
    if re.search(r"/gdpr/jobs/summary/?$", path):
        return None
    if re.search(r"/gdpr/retention(?:/|$)", path):
        return None
    # PRIV-X3: an admin lifting someone's restriction names them in the body,
    # which this middleware never reads -- no subject rather than the admin.
    if re.search(r"/gdpr/restrict/lift/?$", path):
        return None
    # PRIV-X4: likewise a legal hold names its user in the body, and the hold
    # listing covers many users.
    if re.search(r"/gdpr/admin/legal-holds?/?$", path):
        return None
    # CM-5: a flag_id in the path is opaque -- whose content it is only shows up in
    # Module 23's response body, which this middleware never reads (same reasoning
    # as PRIV-X3/X4 above). An explicit branch_id filter on the list endpoint is the
    # one case the path/query *can* name a subject for, same as /gdpr/audit-log's
    # user_id query param.
    if re.search(r"/moderation(?:/|$)", path):
        return request.query_params.get("branch_id")
    # CM-7: the compliance report covers everyone in its period, not one person.
    if re.search(r"/compliance(?:/|$)", path):
        return None
    return accessor_id


def hash_ip(request: Request) -> str | None:
    """SHA-256 of X-Forwarded-For, or of the socket address with no proxy in front."""
    raw = request.headers.get("x-forwarded-for") or (request.client.host if request.client else None)
    return hashlib.sha256(raw.strip().encode("utf-8")).hexdigest() if raw and raw.strip() else None


class AuditLogClient:
    """Sends access records to Module 33. write() never raises."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def write(self, record: dict[str, Any]) -> None:
        settings = get_settings()
        if not settings.gdpr_internal_token:
            logger.warning("[audit] GATEWAY_GDPR_INTERNAL_TOKEN not set -- access record dropped: %s %s",
                           record["method"], record["endpoint"])
            return
        try:
            resp = await self._http().post(
                f"{settings.privacy_compliance_url}/internal/gdpr/access-log",
                json=record, headers={"X-Internal-Token": settings.gdpr_internal_token},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("[audit] access record not written (%s %s by %s): %s",
                         record["method"], record["endpoint"], record["accessor_id"], exc)


audit_log = AuditLogClient()
# Strong references to in-flight writes -- asyncio keeps only weak ones, and
# a garbage-collected task would silently drop a record.
_pending: set[asyncio.Task] = set()


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not _AUDITED.match(request.url.path):
            return await call_next(request)

        accessed_at = datetime.now(timezone.utc)
        response = await call_next(request)

        user = getattr(request.state, "user", None)
        if not user or not user.get("sub") or response.status_code >= 400:
            return response

        accessor_id = user["sub"]
        categories, purpose = classify(request.url.path)
        record = {
            "user_id": subject_of(request, accessor_id),
            "accessor_id": accessor_id,
            "endpoint": request.url.path,
            "method": request.method,
            "data_categories": categories,
            "purpose": purpose,
            "ip_hash": hash_ip(request),
            "timestamp": accessed_at.isoformat(),
        }
        task = asyncio.create_task(audit_log.write(record))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
        return response
