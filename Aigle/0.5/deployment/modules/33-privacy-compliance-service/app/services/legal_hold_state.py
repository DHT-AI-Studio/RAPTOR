"""PRIV-X4 / VIE01-358: whether a user is under a legal hold now.

Its own module, importing no other service, because the erasure orchestrator
asks it and services/legal_hold.py depends on ConsentStore, which imports the
orchestrator.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import asyncpg

from app.models.legal_hold import LegalHold

# Active: not released, and hold_until (a UTC date) not reached yet.
ACTIVE_CONDITION = "released_at IS NULL AND hold_until > $1"

# Run at start-up (app/main.py), so a deployment updated without applying
# 03-database's 010_gdpr_legal_hold.sql by hand doesn't fail every erasure on
# a missing table. The same statements as that file and 002_gdpr.sql
# (tests/unit/test_legal_hold_ddl.py keeps them equal).
LEGAL_HOLDS_DDL = """
CREATE TABLE IF NOT EXISTS gdpr_legal_holds (
    id          BIGSERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    reason      VARCHAR(255) NOT NULL,
    authority   VARCHAR(255) NOT NULL,
    placed_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    hold_until  DATE         NOT NULL,
    released_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gdpr_legal_holds_open
    ON gdpr_legal_holds (user_id) WHERE released_at IS NULL;
"""


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


async def lock_user(conn: asyncpg.Connection, user_id: str) -> None:
    """Serialise, for one user, placing a hold against an erasure starting:
    both take this lock (released when the transaction ends) before they
    check and write, so a hold is never accepted while the erasure it should
    stop goes ahead."""
    await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('gdpr_legal_hold:' || $1, 0))", user_id)


async def active_hold(conn: asyncpg.Connection | asyncpg.Pool, user_id: str) -> LegalHold | None:
    """The user's hold in force now, if any."""
    row = await conn.fetchrow(
        f"SELECT * FROM gdpr_legal_holds WHERE {ACTIVE_CONDITION} AND user_id = $2",  # nosec B608 -- ACTIVE_CONDITION is a constant
        today_utc(), user_id,
    )
    return LegalHold(**dict(row)) if row else None
