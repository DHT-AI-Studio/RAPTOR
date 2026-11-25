"""CM-7 / VIE01-410: GET /reports/compliance -- the compliance audit report
(app/services/compliance_report.py) for a period, as JSON or CSV.

Internal like every route here: Module 13 checks the caller's
compliance-auditor role, then calls this with X-Internal-Token.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from app.core.config import settings
from app.routers._listing import check_window
from app.services import compliance_report

logger = logging.getLogger("privacy_compliance.reports_router")
router = APIRouter(prefix="/reports", tags=["reports"])


async def _guardrails_connection() -> asyncpg.Connection:
    """Module 23's `guardrails` database, same server and credentials -- the
    same direct-SQL pattern as the DSAR assembler and the retention engine."""
    return await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.guardrails_postgres_db,
    )


@router.get("/compliance")
async def compliance(
    request: Request,
    from_: datetime = Query(..., alias="from", description="Start of the period (ISO 8601; no zone = UTC), inclusive"),
    to: datetime = Query(..., description="End of the period (ISO 8601; no zone = UTC), inclusive"),
    format_: Literal["json", "csv"] = Query("json", alias="format"),
    max_rows: int | None = Query(None, ge=1, description="Event rows per section (default and maximum: "
                                                         "GD_COMPLIANCE_REPORT_MAX_ROWS)"),
):
    check_window(compliance_report.utc(from_), compliance_report.utc(to))
    limit = settings.compliance_report_max_rows
    if max_rows is not None:
        if max_rows > limit:
            raise HTTPException(status_code=422, detail=f"max_rows must be at most {limit}")
        limit = max_rows
    try:
        guardrails = await _guardrails_connection()
    except (OSError, asyncpg.PostgresError) as exc:
        logger.error("guardrails database unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="guardrails database unavailable") from exc
    try:
        async with request.app.state.pool.acquire() as conn:
            report = await compliance_report.build_report(conn, guardrails, from_, to, limit)
    finally:
        await guardrails.close()

    headers = {
        "X-Report-Unavailable-Sections": ",".join(report["unavailable_sections"]),
        "X-Report-Truncated-Sections": ",".join(n for n, s in report["sections"].items() if s["truncated"]),
    }
    if format_ == "csv":
        period = report["period"]
        name = f"compliance-report_{period['from'][:10]}_{period['to'][:10]}.csv"
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
        return Response(compliance_report.to_csv(report), media_type="text/csv; charset=utf-8", headers=headers)
    return JSONResponse(report, headers=headers)
