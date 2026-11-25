"""CM-5 (VIE01-408): the automated flagging system's flag store.

Unlike app/services/audit_log.py's fire-and-forget violation log, flag creation
is awaited and returns the created row -- the caller (module 10/11/12) gets the
flag id back, and the row + its CF-4 audit event are written in one transaction
so the two never disagree about whether a flag was actually created.

Every lifecycle transition (flagged -> cleared/removed) gets its own
CF-4 audit_events row (VIE01-408 AC: "lifecycle transitions audited"); only the
*creation* event is linked via moderation_flags.event_id (same one-row-per-event
convention as guardrail_violations) -- later transitions are correlated by
resource_id (the content id) and timestamp, like any other audit_events query.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

import asyncpg
from raptor_audit import build_event, insert_event

from app.core.config import get_settings
from app.db.connection import get_pool
from app.models.moderation_flag import ModerationFlagCreate, ModerationFlagRecord
from app.services.moderation_kafka import publish_flagged

logger = logging.getLogger(__name__)

SOURCE_MODULE = 23
SOURCE_SERVICE = "guardrail-service"


def _resource_id(flag: ModerationFlagCreate) -> str:
    """The most specific thing being flagged -- a moment/chunk within the asset when
    one is named, else the whole asset_path (always present, unlike the other two)."""
    return flag.moment_id or flag.chunk_id or flag.asset_path


async def create_flag(flag: ModerationFlagCreate) -> ModerationFlagRecord:
    """Insert a new 'flagged' row, write its CF-4 audit event, and publish the
    moderation.flagged Kafka event. Raises on a Postgres failure (unlike the
    violation log, callers here need to know whether the flag actually
    landed); the Kafka publish is best-effort and never raises (see
    moderation_kafka.publish_flagged)."""
    settings = get_settings()
    event = build_event(
        source_module=SOURCE_MODULE, source_service=SOURCE_SERVICE, event_type="moderation.flag",
        outcome="allow", resource_id=_resource_id(flag), policy_version=flag.policy_version,
        metadata={"status": "flagged", "category": flag.category, "model": flag.model,
                  "asset_path": flag.asset_path, "moment_id": flag.moment_id, "chunk_id": flag.chunk_id},
        retention_days=settings.guardrail_audit_retention_days,
    )
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await insert_event(conn, event)
        row = await conn.fetchrow(
            """
            INSERT INTO moderation_flags
                (branch_id, version_id, asset_path, moment_id, chunk_id, category, score, model, policy_version, event_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id, branch_id, version_id, asset_path, moment_id, chunk_id, category, score, model,
                      policy_version, status, event_id, created_at, updated_at
            """,
            flag.branch_id, flag.version_id, flag.asset_path, flag.moment_id, flag.chunk_id, flag.category,
            flag.score, flag.model, flag.policy_version, event.event_id,
        )
    record = ModerationFlagRecord(**dict(row))
    await publish_flagged(record.model_dump(mode="json"))
    return record


async def get_flag(flag_id: UUID) -> ModerationFlagRecord | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, branch_id, version_id, asset_path, moment_id, chunk_id, category, score, model, policy_version,
                   status, event_id, created_at, updated_at
            FROM moderation_flags WHERE id = $1
            """,
            flag_id,
        )
    return ModerationFlagRecord(**dict(row)) if row else None


async def update_status(flag_id: UUID, new_status: str, note: str | None) -> ModerationFlagRecord | None:
    """Transition a flag's status; records the transition as its own CF-4
    audit event (not linked via moderation_flags.event_id -- that column is
    reserved for the creation event). Returns None if flag_id doesn't exist.

    Also publishes the updated row to the same moderation.flagged topic
    create_flag() uses (CM-5 AC: Module 25's search exclusion needs to react
    to a flag being cleared/removed, not just created). A consumer
    keys off `status` on every event, create or transition alike -- there is
    no separate "created" vs "updated" event type to distinguish."""
    settings = get_settings()
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        row = await conn.fetchrow(
            """
            UPDATE moderation_flags SET status = $2, updated_at = now()
            WHERE id = $1
            RETURNING id, branch_id, version_id, asset_path, moment_id, chunk_id, category, score, model,
                      policy_version, status, event_id, created_at, updated_at
            """,
            flag_id, new_status,
        )
        if row is None:
            return None
        record = ModerationFlagRecord(**dict(row))
        event = build_event(
            source_module=SOURCE_MODULE, source_service=SOURCE_SERVICE, event_type="moderation.flag",
            outcome="allow", resource_id=_resource_id_from_record(record), policy_version=record.policy_version,
            metadata={"status": new_status, "flag_id": str(flag_id), "note": note},
            retention_days=settings.guardrail_audit_retention_days,
        )
        await insert_event(conn, event)
    await publish_flagged(record.model_dump(mode="json"))
    return record


def _resource_id_from_record(record: ModerationFlagRecord) -> str:
    return record.moment_id or record.chunk_id or record.asset_path


async def list_flags(
    page: int = 1, page_size: int = 50,
    status: str | None = None, category: str | None = None, branch_id: str | None = None,
    version_id: str | None = None,
    asset_path: str | None = None, moment_id: str | None = None, chunk_id: str | None = None,
    from_ts: datetime | None = None, to_ts: datetime | None = None,
) -> tuple[list[asyncpg.Record], int]:
    pool = await get_pool()
    clauses: list[str] = []
    params: list = []

    def _eq(col: str, value) -> None:
        params.append(value)
        clauses.append(f"{col} = ${len(params)}")

    if status is not None:
        _eq("status", status)
    if category is not None:
        _eq("category", category)
    if branch_id is not None:
        _eq("branch_id", branch_id)
    if version_id is not None:
        _eq("version_id", version_id)
    if asset_path is not None:
        _eq("asset_path", asset_path)
    if moment_id is not None:
        _eq("moment_id", moment_id)
    if chunk_id is not None:
        _eq("chunk_id", chunk_id)
    if from_ts is not None:
        params.append(from_ts)
        clauses.append(f"created_at >= ${len(params)}")
    if to_ts is not None:
        params.append(to_ts)
        clauses.append(f"created_at <= ${len(params)}")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    offset = (page - 1) * page_size

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT count(*) FROM moderation_flags {where}", *params)  # nosec B608 -- where/clauses built above from fixed column names, all values parameterized
        rows = await conn.fetch(
            f"SELECT id, branch_id, version_id, asset_path, moment_id, chunk_id, category, score, model, "
            f"policy_version, status, event_id, created_at, updated_at "
            f"FROM moderation_flags {where} "
            f"ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}",  # nosec B608 -- same where, same guarantee
            *params, page_size, offset,
        )
    return list(rows), total
