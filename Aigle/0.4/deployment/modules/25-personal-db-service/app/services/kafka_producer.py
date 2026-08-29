"""Kafka producer for publishing index-requests onto `personal-index-requests`.

Production ingest is driven by workers 09-12, which publish the index payload
onto this topic after their global-DB step. This helper lets a client publish
the *same* envelope directly — for tests, demos, or backfill — so the PA-8
consumer path (auto-create DB -> embed -> index chunks + graph) can be driven
end-to-end without standing up the whole worker fleet.

The envelope shape mirrors what `kafka_consumer._handle_message` expects:

    {"payload": {"branch_id": <branch>,
                 "parameters": {"version_id": <v>,
                                "chunks": [...], "entities": [...],
                                "relationships": [...], "temporal_facts": [...]}}}

Entries may be flat dicts (the consumer falls back to the entry itself when no
nested ``payload`` key is present); chunk entries additionally carry an ``id``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from aiokafka import AIOKafkaProducer

from app.core.config import settings
from app.services.index_events import compute_event_id


def build_envelope(
    branch_id: str,
    version_id: Optional[str],
    chunks: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    temporal_facts: List[Dict[str, Any]],
    asset_path: Optional[str] = None,
    source_module: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap flat entry lists into the worker index-request envelope.

    Stamps `event_id` here rather than in the consumer so the id is fixed at the
    moment of publication: a worker retrying its own send produces the identical
    id, which is what makes the consumer's dedup work across retries.
    """
    return {
        "event_id": compute_event_id(asset_path, version_id, branch_id),
        "schema_version": "personal-index-v1",
        "source_module": source_module,
        "payload": {
            "branch_id": branch_id,
            "parameters": {
                "version_id": version_id or "",
                "asset_path": asset_path or "",
                "chunks": chunks or [],
                "entities": entities or [],
                "relationships": relationships or [],
                "temporal_facts": temporal_facts or [],
            },
        }
    }


async def publish_index_request(envelope: Dict[str, Any]) -> None:
    """Publish one envelope to the index-requests topic (start/stop per call).

    A short-lived producer keeps this dependency-free of app lifespan wiring —
    fine for the low call rate of a test/demo/backfill trigger.
    """
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await producer.start()
    try:
        await producer.send_and_wait(settings.kafka_topic, envelope)
    finally:
        await producer.stop()
