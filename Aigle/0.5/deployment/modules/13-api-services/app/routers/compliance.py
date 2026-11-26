"""CM-7 / VIE01-410: compliance audit report -- public gateway surface.

GET /api/{v}/compliance/report checks the caller's `compliance-auditor`
Keycloak role, then proxies to Module 33's internal GET /reports/compliance
with X-Internal-Token (same trust pattern as gdpr.py's admin routes). The
report itself -- Module 23 guardrail violations and moderation flags, Module
33 GDPR jobs and rights requests, the GD-7 access log, ids hashed -- is built
by Module 33; the CSV download passes through unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from app.api.dependencies import get_current_user, get_http_client
from app.core.config import Settings, get_settings

router = APIRouter(tags=["Compliance"])

ROLE = "compliance-auditor"
# Module 33's response headers worth keeping: the CSV's file name, and which
# sections were cut short or couldn't be read.
_PASS_HEADERS = ("content-disposition", "x-report-unavailable-sections", "x-report-truncated-sections")


def _is_compliance_auditor(payload: dict[str, Any]) -> bool:
    return ROLE in payload.get("realm_access", {}).get("roles", [])


@router.get(
    "/report",
    summary="Compliance audit report for a period (compliance-auditor only)",
    description=(
        "One report for `from`..`to` (ISO 8601, both inclusive; no zone = UTC) covering Module 23 guardrail "
        "violations, moderation flags, GDPR jobs (erasure, export, DSAR, consent withdrawal), GDPR rights "
        "requests (rectification, restriction, objection, consent, legal hold) and the data access log.\n\n"
        "- `format=json` (default): per section the exact `total` for the period, a `summary` (counts by "
        "category / action / type / status / day) and up to `max_rows` event rows, oldest first "
        "(`truncated: true` when there were more).\n"
        "- `format=csv`: one row per event (columns `section, occurred_at, record_id, actor_hash, "
        "subject_hash, kind, category, outcome, status, module, details`), as a file download.\n\n"
        "Every person id is a SHA-256 hash (the same as the audit log's `actor_id_hash`); no content, file names or "
        "free text. Headers `X-Report-Truncated-Sections` / `X-Report-Unavailable-Sections` list the sections "
        "cut short or whose table doesn't exist yet. Requires the `compliance-auditor` role."
    ),
)
async def compliance_report(
    from_: datetime = Query(..., alias="from", description="Start of the period, e.g. 2026-11-01T00:00:00Z (inclusive)"),
    to: datetime = Query(..., description="End of the period, e.g. 2026-11-30T23:59:59Z (inclusive)"),
    format_: Literal["json", "csv"] = Query("json", alias="format", description="json (default) or csv"),
    max_rows: int | None = Query(None, ge=1, description="Event rows per section (default and maximum 50000)"),
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_compliance_auditor(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Requires the {ROLE} role")
    # Declared above so Swagger shows them; Module 33 validates the period itself.
    params: dict[str, Any] = {"from": from_.isoformat(), "to": to.isoformat(), "format": format_}
    if max_rows is not None:
        params["max_rows"] = max_rows
    try:
        resp = await client.request(
            "GET", f"{settings.privacy_compliance_url}/reports/compliance", params=params,
            headers={"X-Internal-Token": settings.gdpr_internal_token},
            timeout=settings.compliance_report_timeout,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code,
                            detail=f"Privacy service error: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Could not connect to privacy compliance service.") from exc
    headers = {k: resp.headers[k] for k in _PASS_HEADERS if k in resp.headers}
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"), headers=headers)
