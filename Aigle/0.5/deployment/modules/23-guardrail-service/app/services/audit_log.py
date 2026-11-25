"""Audit log for guardrail violations (GB-6) — Postgres-backed, fire-and-forget writes.

`record_violation()` schedules the insert as a background task and returns
immediately, so a slow/unavailable Postgres never adds latency to a guardrail
check response. A module-level set of in-flight tasks keeps a strong reference
to each one (asyncio only guarantees a task runs to completion while something
holds a reference to it — otherwise it can be garbage-collected mid-flight).

CF-4 (VIE01-234): every write is a CF-4 `audit_events` row (shared
raptor_audit library, event_type "guardrail.decision") plus the
`guardrail_violations` row that /guardrail/violations reads, inserted in one
transaction and linked by `event_id`. No raw text is kept: the screened
content is stored only as its SHA-256, in `audit_events.content_hash`
(guardrail_violations has no content column -- GRD-X4 / VIE01-354), and a
user id only as `actor_id_hash`. Writes stay fail-open (operational audit): a failure logs a
warning and never reaches the caller.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import asyncpg
from raptor_audit import AuditEvent, build_event, insert_event

from app.core.config import get_settings
from app.db.connection import get_pool

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()

SOURCE_MODULE = 23
SOURCE_SERVICE = "guardrail-service"

# action_taken -> CF-4 outcome. Anything else (warn, flag, review, ...) has no
# CF-4 outcome of its own; it is recorded as "allow" with the original
# decision kept in metadata["decision"].
_OUTCOMES = {"block": "block", "redact": "redact"}


def build_guardrail_event(policy_id: UUID | None, module: str | None, direction: str, category: str | None,
                          action_taken: str, request_id: str | None, content: str | None, *,
                          user_id: str | None = None, session_id: str | None = None,
                          policy_version: str | None = None) -> AuditEvent:
    """The CF-4 audit event for one guardrail decision; `content` and
    `user_id` go in only as SHA-256 hashes. Kept for GUARDRAIL_AUDIT_RETENTION_DAYS
    (default 90, GRD-X4 / VIE01-354)."""
    metadata = {
        "decision": action_taken, "direction": direction, "category": category,
        "caller_module": module, "policy_id": str(policy_id) if policy_id is not None else None,
    }
    if session_id is not None:
        metadata["session_id"] = session_id
    return build_event(
        source_module=SOURCE_MODULE, source_service=SOURCE_SERVICE, event_type="guardrail.decision",
        outcome=_OUTCOMES.get(action_taken, "allow"), actor_id=user_id,
        resource_id=request_id or session_id, content=content,
        policy_version=policy_version, metadata=metadata,
        retention_days=get_settings().guardrail_audit_retention_days,
    )


async def _write(policy_id: UUID | None, module: str | None, direction: str, category: str | None,
                  action_taken: str, request_id: str | None, content: str | None, *,
                  user_id: str | None = None, session_id: str | None = None,
                  policy_version: str | None = None) -> None:
    try:
        event = build_guardrail_event(
            policy_id, module, direction, category, action_taken, request_id, content,
            user_id=user_id, session_id=session_id, policy_version=policy_version,
        )
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await insert_event(conn, event)
            await conn.execute(
                """
                INSERT INTO guardrail_violations
                    (policy_id, module, direction, category, action_taken, request_id, event_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                policy_id, module, direction, category, action_taken, request_id, event.event_id,
            )
    except Exception as exc:  # noqa: BLE001 -- never let a lost audit row surface; asyncpg raises both PostgresError and InterfaceError
        logger.warning("[audit_log] failed to write violation: %s", exc)


def record_violation(policy_id: UUID | None, module: str | None, direction: str, category: str | None,
                      action_taken: str, request_id: str | None, content: str | None = None, *,
                      user_id: str | None = None, session_id: str | None = None,
                      policy_version: str | None = None) -> None:
    """Schedule the write and return immediately — does not block the caller.

    policy_id is None for a violation from the raw guard-model group
    (app/routers/check_guard.py) -- there's no policy involved in that path
    at all. The guardrail_violations.policy_id column is already nullable
    (ON DELETE SET NULL, no NOT NULL constraint) for exactly this reason.

    `content` and `user_id` are hashed before anything is stored (CF-4)."""
    task = asyncio.create_task(
        _write(policy_id, module, direction, category, action_taken, request_id, content,
               user_id=user_id, session_id=session_id, policy_version=policy_version)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def list_violations(
    page: int = 1, page_size: int = 50,
    module: str | None = None, direction: str | None = None, category: str | None = None,
    from_ts: datetime | None = None, to_ts: datetime | None = None,
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
        total = await conn.fetchval(f"SELECT count(*) FROM guardrail_violations {where}", *params)  # nosec B608 -- where/clauses built above from fixed column names, all values parameterized
        rows = await conn.fetch(
            f"SELECT id, policy_id, module, direction, category, action_taken, request_id, event_id, created_at "
            f"FROM guardrail_violations {where} "
            f"ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}",  # nosec B608 -- same where, same guarantee
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
