"""PostgreSQL storage for session recording metadata."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS rt_audio_recordings (
    id          UUID PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    session_id  VARCHAR(64)  NOT NULL UNIQUE,
    fid         VARCHAR(255) NOT NULL,
    duration_s  INTEGER      NOT NULL,
    size_bytes  BIGINT       NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ  NOT NULL,
    erased_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_rt_audio_recordings_user_id ON rt_audio_recordings (user_id);
CREATE INDEX IF NOT EXISTS idx_rt_audio_recordings_expires_at ON rt_audio_recordings (expires_at) WHERE erased_at IS NULL;
"""


class NoDatabase(Exception):
    """The realtime_audio database does not exist: recording was never set up, nothing is stored."""


@dataclass(frozen=True)
class Recording:
    id: str
    user_id: str
    session_id: str
    fid: str
    duration_s: int
    size_bytes: int
    created_at: datetime | None = None
    expires_at: datetime | None = None


def _recording(r) -> Recording:
    return Recording(str(r["id"]), r["user_id"], r["session_id"], r["fid"], r["duration_s"], r["size_bytes"],
                     r["created_at"], r["expires_at"])


class RecordingsDb:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None

    async def pool(self):
        if self._pool is None:
            import asyncpg

            try:
                self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4, command_timeout=10)
            except asyncpg.InvalidCatalogNameError as exc:
                raise NoDatabase(str(exc)) from exc
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def insert(self, user_id: str, session_id: str, fid: str, duration_s: int, size_bytes: int,
                     retention_days: int) -> bool:
        """False when the session already has a recording (a redelivered finalize)."""
        now = datetime.now(timezone.utc)
        row = await (await self.pool()).fetchrow(
            """INSERT INTO rt_audio_recordings (id, user_id, session_id, fid, duration_s, size_bytes, created_at, expires_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT (session_id) DO NOTHING RETURNING id""",
            uuid.uuid4(), user_id, session_id, fid, duration_s, size_bytes, now, now + timedelta(days=retention_days))
        return row is not None

    async def live_for_user(self, user_id: str) -> list[Recording]:
        rows = await (await self.pool()).fetch(
            "SELECT id, user_id, session_id, fid, duration_s, size_bytes, created_at, expires_at FROM rt_audio_recordings "
            "WHERE user_id = $1 AND erased_at IS NULL", user_id)
        return [_recording(r) for r in rows]

    async def expired(self, limit: int = 100) -> list[Recording]:
        rows = await (await self.pool()).fetch(
            "SELECT id, user_id, session_id, fid, duration_s, size_bytes, created_at, expires_at FROM rt_audio_recordings "
            "WHERE erased_at IS NULL AND expires_at < now() ORDER BY expires_at LIMIT $1", limit)
        return [_recording(r) for r in rows]

    async def mark_erased(self, ids: list[str]) -> None:
        if ids:
            await (await self.pool()).execute(
                "UPDATE rt_audio_recordings SET erased_at = now() WHERE id = ANY($1::uuid[])", ids)
