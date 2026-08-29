"""Postgres connection pool management — module-level singleton."""
from __future__ import annotations

import json
import logging

import asyncpg

from app.db.schema import DDL

_logger = logging.getLogger(__name__)
_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog",
    )


async def init_db(host: str, port: int, user: str, password: str, database: str) -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        host=host, port=port, user=user, password=password, database=database,
        min_size=1, max_size=10, init=_init_connection,
    )
    async with _pool.acquire() as conn:
        await conn.execute(DDL)
    _logger.info("PostgreSQL pool initialized: %s@%s:%s/%s", user, host, port, database)


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Postgres pool not initialized — call init_db() at startup")
    return _pool


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        _logger.info("PostgreSQL pool closed")
