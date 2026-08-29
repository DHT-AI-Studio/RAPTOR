"""Deletion audit trail (VIE01-189) — PostgreSQL `personal_db.personal_db_audit`.

Dropping a user's ArcadeDB database is irreversible and leaves nothing behind to
inspect afterwards, so the audit row is the only record that the deletion ever
happened. That makes the write mandatory: `record_deletion` raises if it cannot
persist, and the caller refuses the delete rather than performing it unrecorded
(`PD_AUDIT_REQUIRED=0` opts out for local runs with no Module 03).

The DDL lives in Module 03 (`init/postgresql/001_init.sql`); this module only
writes. Module 25's Kafka consumer will share the same pool for VIE01-190's
`personal_index_events` deduplication table.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import asyncpg

from app.core.config import settings

logger = logging.getLogger("personal_db.audit")

_pool: Optional[asyncpg.Pool] = None
_pool_loop: Optional[asyncio.AbstractEventLoop] = None


async def get_pool() -> asyncpg.Pool:
    """Lazily open the shared connection pool.

    The pool is rebuilt if the running event loop is not the one it was created
    on. asyncpg binds its connections to a loop, so a pool cached from an earlier
    loop raises "attached to a different loop" on first use — which is exactly
    what happens across pytest-asyncio tests, each of which gets a fresh loop.
    In the service there is only ever one loop and this check never fires.
    """
    global _pool, _pool_loop
    loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is not loop:
        _pool = None                       # abandon it; its loop is gone
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=5)
        _pool_loop = loop
    return _pool


async def close_pool() -> None:
    global _pool, _pool_loop
    if _pool is not None:
        await _pool.close()
        _pool = None
        _pool_loop = None


async def record_deletion(user_id: str, database: str, record_counts: Dict[str, Any]) -> None:
    """Write one `action='delete'` row, or raise.

    Called *before* the drop: the counts describe what the database held, and
    once ArcadeDB has dropped it there is no way to recover them. Writing first
    also means a database is never destroyed without a corresponding row — the
    failure mode is a row for a delete that then failed, which is auditable, and
    not a delete with no row, which is not.
    """
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO personal_db_audit (user_id, database, action, record_counts)
        VALUES ($1, $2, 'delete', $3::jsonb)
        """,
        user_id, database, json.dumps(record_counts),
    )
    logger.info("[audit] recorded deletion user_id=%s database=%s", user_id, database)
