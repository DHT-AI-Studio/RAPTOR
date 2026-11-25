"""V05-10 / VIE01-364: operator view of every GDPR job.

gdpr_erasure_jobs holds all four job types (erasure, export, dsar,
consent_withdrawal). Before this, an operator could only open one job by
its id; these two routes let them find the ones that need attention --
partial failures (each with the stores still failing), and jobs pending or
in progress for longer than they should be. Module 13 enforces gdpr-admin.

Only job metadata is returned: never an export's download URLs (presigned
links to the user's data) nor a DSAR job's stored report.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.models.job import JobStatus, JobType, StoreStatus
from app.routers._listing import MAX_PAGE_SIZE, check_window, fetch_page, where_clause

router = APIRouter(prefix="/jobs", tags=["jobs"])

_COLUMNS = "job_id, type, user_id, requested_by, requested_at, status, store_statuses, completed_at, notes"
_OPEN = (JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value)


def _filters(
    type_: JobType | None, status: JobStatus | None, user_id: str | None,
    from_: datetime | None, to: datetime | None, stuck_hours: float | None,
) -> tuple[str, list]:
    """WHERE clause + args; every value parameterized, column names fixed.

    stuck_hours means "still pending/in progress after N hours". With a
    status, that status must itself be pending or in_progress (the two
    filters then intersect); any other status can never be stuck, so it is
    rejected rather than silently matching nothing."""
    open_statuses = None
    if stuck_hours is not None:
        if status is not None and status.value not in _OPEN:
            raise HTTPException(
                status_code=422,
                detail=f"stuck_hours only applies to {' / '.join(_OPEN)} jobs, not status={status.value}",
            )
        # Still open past the threshold -- the escalation view.
        open_statuses = list(_OPEN) if status is None else None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=stuck_hours) if stuck_hours is not None else None
    return where_clause([
        ("type = ${n}", type_.value if type_ else None),
        ("status = ${n}", status.value if status else None),
        ("user_id = ${n}", user_id),
        ("requested_at >= ${n}", from_),
        ("requested_at <= ${n}", to),
        ("status = ANY(${n})", open_statuses),
        ("requested_at < ${n}", cutoff),
    ])


def _item(row) -> dict:
    statuses = row["store_statuses"]
    if isinstance(statuses, str):
        statuses = json.loads(statuses)
    statuses = statuses or {}
    return {
        "job_id": str(row["job_id"]),
        "type": row["type"],
        "user_id": row["user_id"],
        "requested_by": row["requested_by"],
        "requested_at": row["requested_at"],
        "status": row["status"],
        "store_statuses": statuses,
        "failed_stores": sorted(k for k, v in statuses.items() if v == StoreStatus.FAILED.value),
        "completed_at": row["completed_at"],
        "notes": row["notes"],
    }


@router.get("")
async def list_jobs(
    request: Request,
    type_: JobType | None = Query(None, alias="type"),
    status: JobStatus | None = None,
    user_id: str | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
    stuck_hours: float | None = Query(
        None, gt=0,
        description="Only jobs still pending/in progress after this many hours; with `status`, it must be pending or in_progress",
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=MAX_PAGE_SIZE),
):
    """Every job, newest first, narrowed by any of the filters."""
    check_window(from_, to)
    where, args = _filters(type_, status, user_id, from_, to, stuck_hours)
    async with request.app.state.pool.acquire() as conn:
        total, rows = await fetch_page(
            conn, table="gdpr_erasure_jobs", columns=_COLUMNS, where=where, args=args,
            order_by="requested_at DESC", page=page, page_size=page_size,
        )
    return {"total": total, "page": page, "page_size": page_size, "items": [_item(r) for r in rows]}


@router.get("/summary")
async def summary(
    request: Request,
    stuck_hours: float = Query(24, gt=0, description="A pending/in-progress job older than this counts as stuck"),
):
    """Job counts by type and status, plus how many are stuck -- what an
    operator checks first."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=stuck_hours)
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT type, status, count(*) AS n, "
            "count(*) FILTER (WHERE status = ANY($1) AND requested_at < $2) AS stuck "
            "FROM gdpr_erasure_jobs GROUP BY type, status",
            list(_OPEN), cutoff,
        )
    by_type: dict[str, dict[str, int]] = {}
    stuck: dict[str, int] = {}
    for r in rows:
        by_type.setdefault(r["type"], {})[r["status"]] = r["n"]
        if r["stuck"]:
            stuck[r["type"]] = stuck.get(r["type"], 0) + r["stuck"]
    partial = sum(counts.get(JobStatus.PARTIAL_FAILURE.value, 0) for counts in by_type.values())
    return {"by_type": by_type, "partial_failure": partial, "stuck": stuck, "stuck_hours": stuck_hours}
