"""Redis client management — module-level singleton, dedicated instance."""
from __future__ import annotations

import logging

from redis.asyncio import Redis

_logger = logging.getLogger(__name__)
_client: Redis | None = None


async def init_redis(host: str, port: int, password: str, db: int) -> None:
    global _client
    _client = Redis(host=host, port=port, password=password or None, db=db, decode_responses=True)
    await _client.ping()
    _logger.info("Redis client initialized: %s:%s/%s", host, port, db)


async def get_redis() -> Redis:
    if _client is None:
        raise RuntimeError("Redis client not initialized — call init_redis() at startup")
    return _client


async def close_redis() -> None:
    global _client
    if _client:
        await _client.close()
        _client = None
        _logger.info("Redis client closed")
