"""KafkaService.send_processing_request() -- the Redis "queued" placeholder
write must happen *before* the Kafka send, not after.

send_and_wait() only guarantees the broker received the message, not that
the consumer hasn't already picked it up and started processing. Writing
the placeholder afterward raced against the orchestrator's own, richer
state write to the same Redis key -- for a fast job, the orchestrator could
write first, then this function's delayed placeholder write would clobber
it back down to just {"step": "queued", ...} (confirmed live: an
aiohttp... no, a KeyError('original_message') crash downstream once that
happened). Writing it first establishes a strict happens-before instead.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_kafka_service_send_processing_request.py -v
"""
import json
from typing import ClassVar

import pytest
from app.services.kafka_service import KafkaService


class _FakeProducer:
    def __init__(self, raises=False):
        self.raises = raises
        self.sent = []

    async def send_and_wait(self, topic, message):
        if self.raises:
            raise RuntimeError("kafka unreachable")
        self.sent.append((topic, message))


class _FakeRedis:
    def __init__(self, raises=False):
        self.raises = raises
        self.calls = []

    async def setex(self, key, ttl, value):
        self.calls.append((key, ttl, value))
        if self.raises:
            raise RuntimeError("redis unreachable")


class _FakeSettings:
    kafka_topics: ClassVar[dict] = {"video": "video-processing-requests"}


def _upload_result():
    return {"asset_path": "video/mp4/foo", "version_id": "v1", "primary_filename": "foo.mp4", "status": "uploaded"}


@pytest.mark.asyncio
async def test_redis_write_happens_before_kafka_send():
    calls = []

    class _OrderTrackingProducer(_FakeProducer):
        async def send_and_wait(self, topic, message):
            calls.append("kafka")
            return await super().send_and_wait(topic, message)

    class _OrderTrackingRedis(_FakeRedis):
        async def setex(self, key, ttl, value):
            calls.append("redis")
            return await super().setex(key, ttl, value)

    producer = _OrderTrackingProducer()
    redis = _OrderTrackingRedis()
    svc = KafkaService(producer, _FakeSettings(), redis)

    await svc.send_processing_request(_upload_result(), user_id="u1", branch_id="b1", processing_mode=None)

    assert calls == ["redis", "kafka"]


@pytest.mark.asyncio
async def test_redis_placeholder_uses_the_same_correlation_id_sent_to_kafka():
    producer = _FakeProducer()
    redis = _FakeRedis()
    svc = KafkaService(producer, _FakeSettings(), redis)

    result = await svc.send_processing_request(_upload_result(), user_id="u1", branch_id="b1", processing_mode=None)

    redis_key, ttl, value = redis.calls[0]
    assert redis_key == f"video_orchestrator:{result['correlation_id']}"
    assert ttl == 86400
    assert json.loads(value) == {"step": "queued", "branch_id": "b1"}
    _, sent_message = producer.sent[0]
    assert sent_message["correlation_id"] == result["correlation_id"]


@pytest.mark.asyncio
async def test_redis_failure_does_not_block_the_kafka_send():
    """Best-effort, same as before the reorder -- a Redis outage must not
    prevent the actual processing request from going out."""
    producer = _FakeProducer()
    redis = _FakeRedis(raises=True)
    svc = KafkaService(producer, _FakeSettings(), redis)

    result = await svc.send_processing_request(_upload_result(), user_id="u1", branch_id="b1", processing_mode=None)

    assert result["correlation_id"]
    assert len(producer.sent) == 1


@pytest.mark.asyncio
async def test_kafka_failure_still_raises_even_though_redis_already_wrote():
    """The one new edge case from reordering: Redis write succeeds, then
    the Kafka send fails -- leaves an orphaned "queued" placeholder, but
    harmlessly so (this correlation_id is never returned to any caller, so
    nothing will ever poll for it; it expires via its own TTL)."""
    from fastapi import HTTPException

    producer = _FakeProducer(raises=True)
    redis = _FakeRedis()
    svc = KafkaService(producer, _FakeSettings(), redis)

    with pytest.raises(HTTPException) as exc_info:
        await svc.send_processing_request(_upload_result(), user_id="u1", branch_id="b1", processing_mode=None)
    assert exc_info.value.status_code == 500
    assert len(redis.calls) == 1  # the orphaned placeholder -- expected, not a regression
