"""CM-5 / VIE01-408: consumes Module 23's moderation.flagged Kafka topic (fired
on flag creation and on every status transition -- see 23's moderation_flags.py)
and keeps Chunk/Source.review_status in sync, so search can exclude flagged
content by default without calling Module 23 per result (searcher.py's
_filters()/_source_filters()).

Every event carries the flag's *current* status; there is no separate
"created" vs "updated" event type, so applying one is always just: set
review_status to event["status"], idempotently. asset_path + version_id are
always present on every event (CM-5's ModerationFlagCreate/Record require
both, on every flag, regardless of content type -- so an admin reviewing one
can always see the real asset, not just an opaque id), but chunk_id/moment_id
-- an optional finer-grained sub-reference, at most one of the two -- take
priority here when present, since they name the exact piece of content that
was actually flagged, not just the asset it belongs to:

  * chunk_id or moment_id -> Chunk.chunk_id = that value. moment_id is already
    the exact chunk_id Module 11's video moments get in this database (same
    hash as kafka_consumer._stable_chunk_id(); see that module's own
    kafka_handler.py comment on _report_moment_flags()).
  * neither set (the flag is about the whole asset, e.g. an image) ->
    Source (asset_path, version_id).

Deliberately simpler than kafka_consumer.run_consumer()'s DLQ-based retry: a
stuck review_status sync is a degraded search filter, not a lost upload, so
after kafka_max_attempts this logs and moves on (commits) rather than leaving
the partition stalled -- the next event for the same content (if any) still
corrects it, and nothing here is the only copy of anything.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient, db_name_for

logger = logging.getLogger("personal_db.moderation_flag_consumer")


async def _apply(client: ArcadeDBClient, event: dict[str, Any]) -> None:
    branch_id = event.get("branch_id")
    status = event.get("status")
    if not branch_id or not status:
        logger.warning("[moderation-flag-consumer] event missing branch_id/status, skipping: %s", event)
        return

    db = db_name_for(branch_id)
    if not await client.database_exists(db):
        logger.warning("[moderation-flag-consumer] database for branch %s not initialized, skipping", branch_id)
        return

    chunk_id = event.get("chunk_id") or event.get("moment_id")
    asset_path = event.get("asset_path")
    if chunk_id:
        await client.command(
            db, "UPDATE Chunk SET review_status = :s WHERE chunk_id = :c",
            params={"s": status, "c": chunk_id},
        )
    elif asset_path:
        version_id = event.get("version_id")
        if not version_id:
            logger.warning("[moderation-flag-consumer] asset_path %r has no version_id, skipping", asset_path)
            return
        await client.command(
            db, "UPDATE Source SET review_status = :s WHERE asset_path = :a AND version_id = :v",
            params={"s": status, "a": asset_path, "v": version_id},
        )
    else:
        logger.warning("[moderation-flag-consumer] event has no content reference, skipping: %s", event)


async def run_moderation_flag_consumer(client: ArcadeDBClient) -> None:
    """Subscribe to moderation.flagged and apply events indefinitely. Designed
    to run as a background asyncio.Task alongside run_consumer(); responds to
    asyncio.CancelledError for clean shutdown."""
    consumer = AIOKafkaConsumer(
        settings.moderation_flag_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id=settings.moderation_flag_group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode()),
    )
    await consumer.start()
    logger.info("[moderation-flag-consumer] started -- topic=%s group=%s",
                settings.moderation_flag_topic, settings.moderation_flag_group_id)
    try:
        async for msg in consumer:
            for attempt in range(1, settings.kafka_max_attempts + 1):
                try:
                    await _apply(client, msg.value)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- retry loop logs each attempt, eventually gives up and moves on
                    logger.error("[moderation-flag-consumer] attempt %d/%d failed for %s: %s",
                                 attempt, settings.kafka_max_attempts, msg.value, exc)
            await consumer.commit()
    finally:
        await consumer.stop()
        logger.info("[moderation-flag-consumer] stopped")
