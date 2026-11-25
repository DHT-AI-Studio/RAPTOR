"""Expiry of inline sources (temporal plan §5.8 / P21).

Content sent inline to the gateway's `temporal/ingest` becomes an asset that expires after 7 days. Its
`expires_at` is registered here, under the tenant, as soon as the asset exists -- before the source is
indexed -- in a `SourceExpiry` document: {version_id, branch_id, expires_at (UTC ms), state}.

**Logical expiry** is the source's archive: a sweeper (`ExpirySweeper`, every
PD_TEMPORAL_EXPIRY_SWEEP_SECONDS) archives every source whose `expires_at` has passed through the same
path as Module 04's archive (indexer.set_status_by_version -> P14 on_status_change). Archived Chunks,
Sources and facts are already left out of every query, and the source's pending temporal jobs are
cancelled. The sweeper re-checks expired sources on every pass, so content indexed after its expiry
(a late worker) is archived again too.

**No resurrection**: reactivating an expired source is refused (indexer.set_status_by_version keeps it
archived), and P14's generation checks stop late jobs from writing.

**Physical deletion** stays with Module 04: the asset is uploaded with archive / destroy dates at the
same expiry, and its destroy calls DELETE /personal/index/{version_id}, which marks the record
`deleted`. States: pending -> expired -> deleted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient, db_name_for

logger = logging.getLogger("personal_db.source_expiry")

STATES = ("pending", "expired", "deleted")


def expiry_ddl() -> list[str]:
    return [
        "CREATE DOCUMENT TYPE SourceExpiry IF NOT EXISTS",
        "CREATE PROPERTY SourceExpiry.version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY SourceExpiry.branch_id IF NOT EXISTS STRING",      # the sweeper needs the tenant
        "CREATE PROPERTY SourceExpiry.expires_at IF NOT EXISTS LONG",       # UTC epoch ms
        "CREATE PROPERTY SourceExpiry.state IF NOT EXISTS STRING",          # pending | expired | deleted
        "CREATE PROPERTY SourceExpiry.updated_at IF NOT EXISTS LONG",
        "CREATE INDEX IF NOT EXISTS ON SourceExpiry (version_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON SourceExpiry (expires_at) NOTUNIQUE",
    ]


_ready: set[str] = set()


async def _ensure(client: ArcadeDBClient, db: str) -> None:
    if db not in _ready:
        for stmt in expiry_ddl():
            await client.command(db, stmt, ignore_exists=True)
        _ready.add(db)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def register(client: ArcadeDBClient, branch_id: str, version_id: str, expires_at_ms: int) -> dict[str, Any]:
    """Record (or move) a source's expiry. A source already expired or deleted keeps its state: an expiry
    is never lifted by registering it again."""
    db = db_name_for(branch_id)
    await _ensure(client, db)
    current = await get(client, branch_id, version_id)
    if current and current["state"] != "pending":
        return current
    await client.command(
        db, "UPDATE SourceExpiry SET version_id = :v, branch_id = :b, expires_at = :e, state = 'pending', "
            "updated_at = :now UPSERT WHERE version_id = :v",
        params={"v": version_id, "b": branch_id, "e": expires_at_ms, "now": _now_ms()})
    return await get(client, branch_id, version_id)


async def get(client: ArcadeDBClient, branch_id: str, version_id: str) -> dict[str, Any] | None:
    db = db_name_for(branch_id)
    await _ensure(client, db)
    rows = await client.query(db, "SELECT version_id, expires_at, state FROM SourceExpiry WHERE version_id = :v",
                              params={"v": version_id})
    if not rows:
        return None
    r = rows[0]
    return {"version_id": r["version_id"], "expires_at": r.get("expires_at"), "state": r.get("state"),
            "expired": r.get("state") != "pending" or (r.get("expires_at") or 0) <= _now_ms()}


async def is_expired(client: ArcadeDBClient, db: str, version_id: str) -> bool:
    """Whether this source's expiry has passed (or it was already expired / deleted)."""
    try:
        rows = await client.query(db, "SELECT expires_at, state FROM SourceExpiry WHERE version_id = :v",
                                  params={"v": version_id})
    except Exception:  # noqa: BLE001 -- a database with no SourceExpiry type has no inline sources
        return False
    return bool(rows) and (rows[0].get("state") != "pending" or (rows[0].get("expires_at") or 0) <= _now_ms())


async def mark_deleted(client: ArcadeDBClient, db: str, version_id: str) -> None:
    """Physical deletion has removed the source (Module 04's destroy)."""
    try:
        await client.command(db, "UPDATE SourceExpiry SET state = 'deleted', updated_at = :now WHERE version_id = :v",
                             params={"v": version_id, "now": _now_ms()})
    except Exception as exc:  # noqa: BLE001 -- no SourceExpiry type: nothing was registered
        logger.debug("[expiry] nothing to mark deleted in %s: %s", db, exc)


async def _still_visible(client: ArcadeDBClient, db: str, version_id: str) -> bool:
    """Any Chunk, Source or fact of the version that is not archived."""
    for sql in ("SELECT count(*) AS n FROM Chunk WHERE version_id = :v AND (status IS NULL OR status <> 'archived')",
                "SELECT count(*) AS n FROM Source WHERE version_id = :v AND (status IS NULL OR status <> 'archived')",
                ("SELECT count(*) AS n FROM TemporalFact WHERE source_version_id = :v "
                 "AND (status IS NULL OR status <> 'archived')")):
        rows = await client.query(db, sql, params={"v": version_id})
        if rows and rows[0].get("n"):
            return True
    return False


async def sweep_database(client: ArcadeDBClient, db: str) -> int:
    """Archive every expired source of one tenant still visible. Returns how many were archived."""
    from app.services.indexer import (
        set_status_by_version,  # imported lazily: indexer imports this module
    )
    rows = await client.query(
        db, "SELECT version_id, branch_id, state FROM SourceExpiry WHERE state <> 'deleted' AND expires_at <= :now",
        params={"now": _now_ms()})
    archived = 0
    for r in rows:
        if await _still_visible(client, db, r["version_id"]):
            await set_status_by_version(client, r["branch_id"], r["version_id"], "archived")
            archived += 1
        if r.get("state") == "pending":
            await client.command(db, "UPDATE SourceExpiry SET state = 'expired', updated_at = :now WHERE version_id = :v",
                                 params={"v": r["version_id"], "now": _now_ms()})
            logger.info("[expiry] %s expired in %s", r["version_id"], db)
    return archived


async def sweep(client: ArcadeDBClient) -> int:
    total = 0
    for db in await client.list_databases():
        if not db.startswith("user_"):
            continue
        try:
            total += await sweep_database(client, db)
        except Exception as exc:  # noqa: BLE001 -- a database from before P21 has no SourceExpiry type
            logger.debug("[expiry] skip %s: %s", db, exc)
    return total


class ExpirySweeper:
    def __init__(self, client: ArcadeDBClient) -> None:
        self.client = client
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            try:
                archived = await sweep(self.client)
                if archived:
                    logger.info("[expiry] archived %d expired sources", archived)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- ArcadeDB down etc.; the next pass catches up
                logger.warning("[expiry] sweep failed: %s", exc)
            await asyncio.sleep(settings.temporal_expiry_sweep_seconds)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
