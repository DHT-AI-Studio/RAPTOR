"""Event identity and consumer deduplication for `personal-index-requests` (VIE01-190).

`event_id = sha256(asset_path + version_id + user_id)` is the contract with the
publishers in Modules 09-12 — the same three values always produce the same id,
so a worker that republishes after a retry produces a message the consumer can
recognise as already handled.

Two dedup layers exist and neither replaces the other:

  * `personal_index_events` (here, PostgreSQL) — whole-message, permanent. Stops a
    replayed *event*: same asset version, same user.
  * `personal:indexed:{chunk_id}` (Redis, 7-day TTL) — per chunk. Stops a single
    chunk arriving twice inside *different* events, and expires so long-lived
    keys do not accumulate.

The PG layer survives restarts and TTL expiry; the Redis layer is finer-grained
and cheap. Dropping either one lets a class of duplicate through.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import asyncpg

from app.services.audit import get_pool

logger = logging.getLogger("personal_db.index_events")


def compute_event_id(asset_path: Optional[str], version_id: Optional[str], user_id: str) -> str:
    """Derive the deduplication id. Field order is part of the contract with 09-12.

    The parts are joined with a separator that cannot appear in a sha256 input
    boundary ambiguously — without it, ("ab", "c") and ("a", "bc") would hash the
    same, so two different assets could collide into one event_id and the second
    would be silently dropped as a duplicate.
    """
    raw = "\x1f".join([asset_path or "", version_id or "", user_id or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def claim_event(
    event_id: str,
    user_id: str,
    asset_path: Optional[str] = None,
    version_id: Optional[str] = None,
    source_module: Optional[str] = None,
) -> bool:
    """Try to claim an event for processing.

    Returns True if this is the first time we have seen it (caller should
    process), False if it was already handled (caller should acknowledge and drop
    without reprocessing).

    The claim is the INSERT itself: the primary key makes the check and the
    record a single atomic step, so two consumer instances racing on the same
    message cannot both win. A separate SELECT-then-INSERT would leave that gap.
    """
    pool = await get_pool()
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO personal_index_events
                (event_id, user_id, asset_path, version_id, source_module, status)
            VALUES ($1, $2, $3, $4, $5, 'processed')
            ON CONFLICT (event_id) DO UPDATE
                SET status = 'processed', error = NULL,
                    processed_at = CURRENT_TIMESTAMP
                WHERE personal_index_events.status = 'failed'
            RETURNING event_id
            """,
            event_id, user_id, asset_path, version_id, source_module,
        )
    except asyncpg.PostgresError as exc:
        # Dedup is an optimisation, not a correctness guarantee — the per-chunk
        # Redis layer still protects the data. Failing the message here would
        # stall the pipeline on a PostgreSQL blip, so process it instead.
        logger.warning("[dedup] claim failed for %s (%s) — processing anyway", event_id, exc)
        return True

    # A row comes back on a fresh insert, and on an update of a previously failed
    # event. A conflict against an already-'processed' row is filtered out by the
    # WHERE clause and returns nothing — that is the duplicate case.
    claimed = row is not None
    if not claimed:
        logger.info("[dedup] event %s already processed — acknowledging and dropping", event_id)
    return claimed


async def mark_event_failed(event_id: str, error: str) -> None:
    """Record that an event exhausted its attempts and went to the DLQ.

    The claim row stays, so the failure is queryable, but flipping it to 'failed'
    means a later replay of the same event can claim it again — a DLQ'd message
    would otherwise be shut out permanently by its own claim.
    """
    try:
        pool = await get_pool()
        await pool.execute(
            "UPDATE personal_index_events SET status = 'failed', error = $2 WHERE event_id = $1",
            event_id, error[:2000],
        )
    except Exception as exc:
        logger.warning("[dedup] could not mark %s failed: %s", event_id, exc)
