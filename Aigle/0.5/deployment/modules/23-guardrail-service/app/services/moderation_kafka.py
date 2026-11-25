"""Kafka producer for CM-5 (VIE01-408): publishes one event per flag onto
KAFKA_MODERATION_TOPIC (default "moderation.flagged").

Short-lived producer (start/send/stop per call) -- module 23 has no other Kafka
usage and no app-lifespan producer to share, and flag creation is low-rate
(only when a moderation verdict crosses the flagging threshold), so a
dependency-free producer per call is simpler than wiring one into the app
lifespan for this alone (same reasoning as module 25's
app/services/kafka_producer.py::publish_index_request).
"""
from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from aiokafka import AIOKafkaProducer

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _default(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


async def publish_flagged(flag: dict[str, Any]) -> None:
    """Publish a moderation_flags row (as returned by the repo layer) to Kafka.
    Fire-and-forget from the caller's point of view: a failure here is logged
    and swallowed, never surfaced to the HTTP caller -- the flag row and its
    audit event are already committed to Postgres by the time this runs, so a
    lost Kafka event doesn't lose the flag itself, only this one notification."""
    settings = get_settings()
    try:
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v, default=_default).encode(),
        )
        await producer.start()
        try:
            await producer.send_and_wait(settings.kafka_moderation_topic, flag)
        finally:
            await producer.stop()
    except Exception as exc:  # noqa: BLE001 -- never let a lost Kafka notification surface
        logger.warning("[moderation_kafka] failed to publish flag %s: %s", flag.get("id"), exc)
