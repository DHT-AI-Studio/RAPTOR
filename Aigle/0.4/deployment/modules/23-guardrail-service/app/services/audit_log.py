"""Audit log for guardrail violations (GB-6) — Postgres-backed, fire-and-forget writes.

`record_violation()` schedules the insert as a background task and returns
immediately, so a slow/unavailable Postgres never adds latency to a guardrail
check response. A module-level set of in-flight tasks keeps a strong reference
to each one (asyncio only guarantees a task runs to completion while something
holds a reference to it — otherwise it can be garbage-collected mid-flight).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import asyncpg

from app.db.connection import get_pool

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


async def _write(policy_id: Optional[UUID], module: Optional[str], direction: str, category: Optional[str],
                  action_taken: str, request_id: Optional[str], content: Optional[str]) -> None:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO guardrail_violations
                    (policy_id, module, direction, category, action_taken, request_id, content)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                policy_id, module, direction, category, action_taken, request_id, content,
            )
    except Exception as exc:                                    # never let a lost audit row surface
        logger.error("[audit_log] failed to write violation: %s", exc)


def record_violation(policy_id: Optional[UUID], module: Optional[str], direction: str, category: Optional[str],
                      action_taken: str, request_id: Optional[str], content: Optional[str] = None) -> None:
    """Schedule the write and return immediately — does not block the caller.

    policy_id is None for a violation from the raw guard-model group
    (app/routers/check_guard.py) -- there's no policy involved in that path
    at all. The guardrail_violations.policy_id column is already nullable
    (ON DELETE SET NULL, no NOT NULL constraint) for exactly this reason."""
    task = asyncio.create_task(
        _write(policy_id, module, direction, category, action_taken, request_id, content)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def list_violations(
    page: int = 1, page_size: int = 50,
    module: Optional[str] = None, direction: Optional[str] = None, category: Optional[str] = None,
    from_ts: Optional[datetime] = None, to_ts: Optional[datetime] = None,
) -> tuple[list[asyncpg.Record], int]:
    pool = await get_pool()
    clauses: list[str] = []
    params: list = []

    def _eq(col: str, value) -> None:
        params.append(value)
        clauses.append(f"{col} = ${len(params)}")

    if module is not None:
        _eq("module", module)
    if direction is not None:
        _eq("direction", direction)
    if category is not None:
        _eq("category", category)
    if from_ts is not None:
        params.append(from_ts)
        clauses.append(f"created_at >= ${len(params)}")
    if to_ts is not None:
        params.append(to_ts)
        clauses.append(f"created_at <= ${len(params)}")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    offset = (page - 1) * page_size

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT count(*) FROM guardrail_violations {where}", *params)
        rows = await conn.fetch(
            f"SELECT id, policy_id, module, direction, category, action_taken, request_id, created_at "
            f"FROM guardrail_violations {where} "
            f"ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}",
            *params, page_size, offset,
        )
    return list(rows), total


async def summary_last_24h() -> tuple[datetime, datetime, list[asyncpg.Record]]:
    pool = await get_pool()
    until = datetime.now(timezone.utc)
    since = until - timedelta(hours=24)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT category, action_taken, count(*) AS count FROM guardrail_violations "
            "WHERE created_at >= $1 AND created_at <= $2 "
            "GROUP BY category, action_taken ORDER BY count DESC",
            since, until,
        )
    return since, until, list(rows)
