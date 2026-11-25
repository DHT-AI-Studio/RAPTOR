"""Finalize pending recordings and retry orphan cleanup and retention."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time

import httpx
from redis.asyncio import Redis

from app.blobstore import SeaweedFS
from app.config import get_settings
from app.recording import RecordingService
from app.recordings_db import RecordingsDb
from app.session import ACTIVE_KEY, _dec

logger = logging.getLogger("app.recording_consumer")

GROUP = "rt-audio-recording"
ALIVE = pathlib.Path("/tmp/recording-consumer.alive")  # nosec B108 -- liveness marker for the compose healthcheck: only its mtime is read


async def sweep(service: RecordingService, redis: Redis, retention_every: int, count: list[int],
                idle_s: float = 300.0, heartbeat_s: float = 10.0) -> None:
    # Sessions with a fresh heartbeat; a crashed instance's entries go stale after 3 heartbeats.
    fresh = await redis.zrangebyscore(ACTIVE_KEY, time.time() - 3 * heartbeat_s, "+inf")
    active = {_dec(m) for m in fresh}
    for session_id in await service.orphans(active, idle_s):
        try:
            logger.info("session %s ended without session_ended: %s", session_id, await service.finalize(session_id))
        except Exception:
            logger.exception("orphaned recording %s not finalized", session_id)
    count[0] += 1
    if count[0] % retention_every == 1:
        try:
            expired = await service.enforce_retention()
            if expired:
                logger.info("retention: %d recordings past expires_at deleted", expired)
        except Exception:
            logger.exception("retention pass failed")
    ALIVE.touch()


async def main() -> None:
    from aiokafka import AIOKafkaConsumer

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.recording_enabled:
        logger.info("RT_RECORDING_ENABLED=false: nothing to finalize")
        while True:
            ALIVE.touch()
            await asyncio.sleep(60)
    if not settings.postgres_dsn:
        raise SystemExit("set RT_POSTGRES_DSN or POSTGRES_HOST/USER/PASSWORD (database RT_POSTGRES_DB=realtime_audio)")
    redis = Redis.from_url(settings.redis_url, socket_timeout=5, socket_connect_timeout=5)
    http = httpx.AsyncClient()
    db = RecordingsDb(settings.postgres_dsn)
    while True:
        try:
            await db.pool()
            break
        except Exception as exc:  # noqa: BLE001 -- e.g. the database does not exist yet: stay up, unhealthy
            logger.error("rt_audio_recordings database unavailable (%s); retrying in 60 s", exc)
            await asyncio.sleep(60)
    service = RecordingService(SeaweedFS(settings.seaweedfs_master, http, settings.recording_collection), redis, db,
                               settings.recording_retention_days)
    consumer = AIOKafkaConsumer(settings.recording_topic, bootstrap_servers=settings.kafka_bootstrap, group_id=GROUP,
                                enable_auto_commit=False, auto_offset_reset="earliest",
                                value_deserializer=lambda b: json.loads(b.decode()))
    await consumer.start()
    every = max(1.0, float(os.environ.get("RT_RECORDING_SWEEP_S", "60")))
    idle = float(os.environ.get("RT_RECORDING_ORPHAN_IDLE_S", "300"))   # no audio for this long: orphaned
    retention_every = max(1, int(3600 / every))
    count = [0]
    logger.info("recording consumer ready: topic %s, group %s", settings.recording_topic, GROUP)
    next_sweep = 0.0
    try:
        while True:
            if time.monotonic() >= next_sweep:
                await sweep(service, redis, retention_every, count, idle, settings.heartbeat_s)
                next_sweep = time.monotonic() + every
            batch = await consumer.getmany(timeout_ms=1000, max_records=20)
            for messages in batch.values():
                for msg in messages:
                    event = msg.value if isinstance(msg.value, dict) else {}
                    if event.get("event") == "session_ended" and event.get("session_id"):
                        try:
                            outcome = await service.finalize(str(event["session_id"]))
                            logger.info("session %s: %s", event["session_id"], outcome)
                        except Exception:
                            # Still in rt:rec:pending: the sweep retries it.
                            logger.exception("session %s not finalized", event["session_id"])
            if batch:
                await consumer.commit()
    finally:
        await consumer.stop()
        await db.close()
        await http.aclose()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
