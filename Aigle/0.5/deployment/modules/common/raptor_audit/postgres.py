"""PostgreSQL sink for CF-4 audit events (the `audit_events` table).

Jira CF-4 allows a PostgreSQL table instead of the Kafka topic for
synchronous callers; Kafka (`acks=all`) is a follow-up. The adopting module
runs `AUDIT_EVENTS_DDL` at startup (idempotent) and passes in its own
asyncpg pool/connection, so this package has no driver dependency.

Delivery modes (Jira): operational audit is fail-open — a failed write logs a
warning and the caller carries on; erasure-confirmation audit is fail-closed —
`emit(..., fail_closed=True)` raises `AuditWriteError` so the caller can
refuse to confirm.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .event import AuditEvent

logger = logging.getLogger(__name__)

AUDIT_EVENTS_DDL = """
-- CF-4 (VIE01-234): normalized, hash-only audit events (raptor_audit).
CREATE TABLE IF NOT EXISTS audit_events (
    event_id        UUID PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    source_module   INTEGER NOT NULL,
    source_service  TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    actor_id_hash   TEXT,
    resource_id     TEXT,
    outcome         TEXT NOT NULL,
    policy_version  TEXT,
    content_hash    TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    retention_days  INTEGER NOT NULL DEFAULT 365
);

CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp     ON audit_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_event_type    ON audit_events (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_source_module ON audit_events (source_module);
"""

_INSERT = """
INSERT INTO audit_events
    (event_id, timestamp, source_module, source_service, event_type, actor_id_hash,
     resource_id, outcome, policy_version, content_hash, metadata, retention_days)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::text::jsonb, $12)
"""


class AuditWriteError(RuntimeError):
    """A fail-closed audit write did not reach the sink."""


async def insert_event(conn: Any, event: AuditEvent) -> None:
    """INSERT one event on `conn` (an asyncpg connection), so a caller can put
    it in the same transaction as its own rows. `metadata` is sent as JSON
    *text* and cast server-side (`$11::text::jsonb`): the parameter is typed
    text, so a pool-level jsonb codec (module 23 registers one) never
    re-encodes it."""
    await conn.execute(
        _INSERT,
        event.event_id, event.timestamp, event.source_module, event.source_service, event.event_type,
        event.actor_id_hash, event.resource_id, event.outcome, event.policy_version, event.content_hash,
        json.dumps(event.metadata), event.retention_days,
    )


class PostgresAuditWriter:
    """Writes events through `get_pool()` (an async callable returning an
    asyncpg pool)."""

    def __init__(self, get_pool: Callable[[], Awaitable[Any]]) -> None:
        self._get_pool = get_pool

    async def emit(self, event: AuditEvent, *, fail_closed: bool = False) -> None:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await insert_event(conn, event)
        except Exception as exc:  # the driver raises several unrelated types; the mode decides what to do
            if fail_closed:
                raise AuditWriteError(f"audit event {event.event_id} not written: {exc}") from exc
            logger.warning("[raptor_audit] audit event %s not written: %s", event.event_id, exc)
