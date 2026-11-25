"""RTA-X6 / RTA-07: where finished voice turns go besides the brain.

KafkaPublisher   one aiokafka producer for the service, started on first use (RT_KAFKA_BOOTSTRAP);
                 a send never raises -- failures are logged and reported as False.
Module 25        every screened TRANSCRIPT_FINAL is published to ``personal-index-requests`` in the
                 envelope Modules 09-12 use (schema personal-index-v1, event_id =
                 sha256(asset_path \\x1f version_id \\x1f user_id)), as one ``audios`` chunk of the
                 asset ``realtime-audio/<session_id>``, version ``<session_id>:<utterance>``.  The
                 payload also carries the backlog's {user_id, session_id, transcript, timestamp}.
Module 26        POST /memory/sessions/{session_id}/turns (X-User-ID) with the user's transcript
                 and the answer -- only for brains that do not archive their own turns: Module 15
                 already appends every chat turn to Module 26, so for it this would be a duplicate.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "personal-index-v1"
SOURCE_MODULE = "34"


class KafkaPublisher:
    def __init__(self, bootstrap: str) -> None:
        self.bootstrap = bootstrap
        self._producer = None
        self._lock = asyncio.Lock()

    async def _ensure(self):
        async with self._lock:
            if self._producer is None:
                from aiokafka import AIOKafkaProducer

                producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap,
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode(),
                    key_serializer=lambda k: k.encode() if k else None,
                    acks="all", request_timeout_ms=10000,
                )
                await producer.start()
                self._producer = producer
            return self._producer

    async def send(self, topic: str, value: dict[str, Any], key: str | None = None) -> bool:
        try:
            producer = await self._ensure()
            await producer.send_and_wait(topic, value, key=key)
            return True
        except Exception as exc:  # noqa: BLE001 -- best effort: a Kafka problem must not stall a voice turn
            logger.warning("kafka publish to %s failed: %s", topic, exc)
            return False

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None


def compute_event_id(asset_path: str, version_id: str, user_id: str) -> str:
    """Byte-identical to Module 25's app/services/index_events.py and Modules 09-12's publishers."""
    raw = "\x1f".join([asset_path or "", version_id or "", user_id or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def transcript_event(user_id: str, session_id: str, utterance: int, text: str, *, language: str | None,
                     start_ms: int, end_ms: int, timestamp: datetime | None = None) -> dict[str, Any]:
    when = (timestamp or datetime.now(timezone.utc)).isoformat()
    asset_path = f"realtime-audio/{session_id}"
    version_id = f"{session_id}:{utterance}"
    chunk = {
        "id": hashlib.sha256(version_id.encode()).hexdigest(),
        "payload": {
            "type": "audios", "embedding_type": "text", "text": text,
            "asset_path": asset_path, "version_id": version_id, "source": "realtime-audio",
            "filename": f"voice-session-{session_id}", "upload_time": when,
            "start_time": start_ms / 1000, "end_time": end_ms / 1000, "speaker": "user",
            "chunk_index": utterance, "language": language,
            "user_id": user_id, "session_id": session_id, "transcript": text, "timestamp": when,
        },
    }
    return {
        "event_id": compute_event_id(asset_path, version_id, user_id),
        "schema_version": SCHEMA_VERSION,
        "source_module": SOURCE_MODULE,
        "payload": {
            "branch_id": user_id, "correlation_id": "",
            "user_id": user_id, "session_id": session_id, "transcript": text, "timestamp": when,
            "parameters": {"version_id": version_id, "asset_path": asset_path, "chunks": [chunk], "moments": [],
                           "entities": [], "relationships": [], "temporal_facts": []},
        },
    }


class TranscriptIndexer:
    def __init__(self, kafka: KafkaPublisher, topic: str) -> None:
        self.kafka = kafka
        self.topic = topic

    async def publish(self, user_id: str, session_id: str, utterance: int, text: str, *, language: str | None,
                      start_ms: int, end_ms: int) -> bool:
        if not user_id or not text.strip():
            return False
        event = transcript_event(user_id, session_id, utterance, text, language=language,
                                 start_ms=start_ms, end_ms=end_ms)
        return await self.kafka.send(self.topic, event, key=user_id)


class MemoryArchive:
    """Module 26 session memory."""

    def __init__(self, base_url: str, client: httpx.AsyncClient, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.timeout_s = timeout_s

    async def append_turn(self, user_id: str, session_id: str, user_message: str, assistant_response: str) -> bool:
        url = f"{self.base_url}/memory/sessions/{session_id}/turns"
        try:
            resp = await self.client.post(url, headers={"X-User-ID": user_id}, timeout=self.timeout_s, json={
                "user_message": user_message, "assistant_response": assistant_response,
                "timestamp": datetime.now(timezone.utc).timestamp()})
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Module 26 append_turn for session %s failed: %s", session_id, exc)
            return False
