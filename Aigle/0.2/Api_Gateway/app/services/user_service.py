"""User management service for gateway credentials backed by PostgreSQL."""
from __future__ import annotations

import logging
from typing import Optional

import asyncpg
from fastapi import HTTPException, status

_logger = logging.getLogger(__name__)


class UserService:
    """Manages gateway user accounts stored in PostgreSQL."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 5,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def initialise(self) -> None:
        return None

        """Initialise the connection pool and database schema."""

        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=self._min_size,
                    max_size=self._max_size,
                )
                await self._ensure_schema()
                _logger.info("User service connected to PostgreSQL")
            except Exception as exc:  # pragma: no cover - initialisation failure should raise
                _logger.error("Failed to initialise user service", exc_info=exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to connect to user database.",
                ) from exc

    async def close(self) -> None:
        """Close the underlying connection pool."""

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def is_ready(self) -> bool:
        return self._pool is not None

    async def _ensure_schema(self) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gateway_users (
                    username TEXT PRIMARY KEY,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("UserService has not been initialised")
        return self._pool

    async def get_user_password(self, username: str) -> str | None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            record = await connection.fetchrow(
                "SELECT hashed_password FROM gateway_users WHERE username = $1",
                username,
            )
        return record["hashed_password"] if record else None

    async def create_user(self, username: str, hashed_password: str) -> None:
        username = username.strip()
        if not username:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username cannot be empty")

        pool = self._require_pool()
        try:
            async with pool.acquire() as connection:
                await connection.execute(
                    "INSERT INTO gateway_users (username, hashed_password) VALUES ($1, $2)",
                    username,
                    hashed_password,
                )
        except asyncpg.exceptions.UniqueViolationError as exc:
            _logger.warning("Attempt to register duplicate username", extra={"username": username})
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists") from exc
        except asyncpg.PostgresError as exc:  # pragma: no cover - surfaced at runtime
            _logger.error("Failed to create user", exc_info=exc, extra={"username": username})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not persist user account.",
            ) from exc
        else:
            _logger.info("Registered new gateway user", extra={"username": username})

