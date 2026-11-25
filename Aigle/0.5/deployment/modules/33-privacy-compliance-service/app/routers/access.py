"""GD-4 / VIE01-249: internal DSAR (Right of Access) endpoints, and
GD-7 / VIE01-252: the data access audit log.

Identity comes from headers Module 13 injects after its own JWT (+ gdpr-admin
role, for another user's DSAR) check -- same trust pattern erasure.py and
export.py already use. This router is never reachable from outside the
raptor network.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.models.dsar import DSARPendingResponse
from app.routers._listing import (
    MAX_PAGE_SIZE,
    check_window,
    fetch_page,
    page_args,
    where_clause,
)
from app.services.dsar_assembler import DSARAssembler

logger = logging.getLogger("privacy_compliance.access_router")
router = APIRouter(prefix="/dsar", tags=["access"])


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    return x_user_id


@router.get("")
async def get_dsar(
    request: Request,
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_requested_by: str = Header(None, alias="X-Requested-By"),
):
    """The DSAR report itself (200) for a normal account, or
    {job_id, status: "pending"} (202) for one with more than
    GD_DSAR_ASYNC_THRESHOLD indexed items, or whose assembly overran
    GD_DSAR_SYNC_TIMEOUT_SECONDS -- then poll GET /dsar/jobs/{job_id}."""
    user_id = _require_user(x_user_id)
    assembler: DSARAssembler = request.app.state.dsar_assembler
    result = await assembler.request(user_id, x_requested_by or user_id)
    if isinstance(result, DSARPendingResponse):
        return JSONResponse(status_code=202, content=result.model_dump())
    return result


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    assembler: DSARAssembler = request.app.state.dsar_assembler
    job = await assembler.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── GD-7 / VIE01-252: data access audit log (GDPR Art. 30) ──────────────────
#
# Module 13's audit middleware sends one record per request to a
# personal-data route (POST /internal/gdpr/access-log, in the background);
# this service owns gdpr_access_log, like every other gdpr_* table. Rows
# expire under GD-3's gdpr_access_log retention policy (365 days).

audit_router = APIRouter(tags=["audit"])


class AccessRecord(BaseModel):
    user_id: str | None = None
    accessor_id: str
    endpoint: str
    method: str
    data_categories: list[str] = Field(default_factory=list)
    purpose: str | None = None
    ip_hash: str | None = None
    timestamp: datetime | None = None  # when the request arrived; now() if omitted


def hash_accessor(accessor_id: str) -> str:
    return hashlib.sha256(accessor_id.encode("utf-8")).hexdigest()


def _pool(request: Request):
    return request.app.state.pool


@audit_router.post("/internal/gdpr/access-log", status_code=204, response_model=None)
async def write_access_record(record: AccessRecord, request: Request) -> None:
    async with _pool(request).acquire() as conn:
        await conn.execute(
            """
            INSERT INTO gdpr_access_log
                (user_id, accessor_id, endpoint, method, data_categories, purpose, ip_hash, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8, now()))
            """,
            record.user_id, record.accessor_id, record.endpoint, record.method,
            record.data_categories, record.purpose, record.ip_hash, record.timestamp,
        )


@audit_router.get("/audit-log")
async def list_access_log(
    request: Request,
    user_id: str | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=MAX_PAGE_SIZE),
):
    """Admin view (Module 13 enforces gdpr-admin): every record, newest
    first, optionally narrowed to one data subject and/or a time window."""
    check_window(from_, to)
    where, args = where_clause([
        ("user_id = ${n}", user_id), ("timestamp >= ${n}", from_), ("timestamp <= ${n}", to),
    ])
    async with _pool(request).acquire() as conn:
        total, rows = await fetch_page(
            conn, table="gdpr_access_log", columns="*", where=where, args=args,
            order_by="timestamp DESC", page=page, page_size=page_size,
        )
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [{**dict(r), "id": str(r["id"])} for r in rows],
    }


@audit_router.get("/audit-log/my")
async def my_access_log(
    request: Request,
    x_user_id: str = Header(None, alias="X-User-ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=MAX_PAGE_SIZE),
):
    """Who accessed the caller's own data. Another person's accessor_id is
    shown only as its SHA-256 -- the data subject sees that someone else
    accessed it (and can tell repeat accessors apart), not who."""
    user_id = _require_user(x_user_id)
    limit, offset = page_args(page, page_size)
    async with _pool(request).acquire() as conn:
        total = await conn.fetchval("SELECT count(*) FROM gdpr_access_log WHERE user_id = $1", user_id)
        rows = await conn.fetch(
            "SELECT accessor_id, endpoint, timestamp FROM gdpr_access_log WHERE user_id = $1 "
            "ORDER BY timestamp DESC LIMIT $2 OFFSET $3",
            user_id, limit, offset,
        )
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [
            {
                "accessor_id": r["accessor_id"] if r["accessor_id"] == user_id else hash_accessor(r["accessor_id"]),
                "accessed_by_self": r["accessor_id"] == user_id,
                "endpoint": r["endpoint"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ],
    }
