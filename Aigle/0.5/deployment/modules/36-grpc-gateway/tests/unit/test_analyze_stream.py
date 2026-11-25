"""Tests for AnalyzeStream: real per-chunk Kafka forwarding (no buffering),
session_id/media_type propagation, idle timeout, cancellation cleanup, the
shared per-user stream limit, and the honest end-of-stream UNIMPLEMENTED.

This RPC's "receive AnalysisEvent back" half is deliberately not built --
see media_stream_service.py's module docstring and
docs/_internal/OPEN_QUESTION_analyze_stream_results.md for why: no consumer anywhere in
Raptor reads the analysis-requests topic, so there is nothing to verify
there beyond "the RPC correctly reports this as UNIMPLEMENTED rather than
silently returning zero events." These tests use a fake Kafka producer
(same convention as test_job_lifecycle.py) -- there is no live Kafka broker
or GPU/ML stack in this environment to test against for real (see this
ticket's own scoping discussion), so what's verified here is the wiring,
not real analysis results.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from interceptors.errors import (
    DeadlineExceededError,
    UnavailableError,
    UnimplementedError,
    ValidationError,
)
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import media_stream_service


class _FakeRedis:
    def __init__(self):
        self._kv = {}

    async def get(self, key):
        return self._kv.get(key)

    async def setex(self, key, ttl, value):
        self._kv[key] = value

    async def delete(self, key):
        self._kv.pop(key, None)

    async def incr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) + 1
        return self._kv[key]

    async def decr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) - 1
        return self._kv[key]

    async def expire(self, key, ttl):
        pass


class _FakeKafkaProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, message):
        self.sent.append((topic, message))


def _start_msg(session_id="s1", media_type="audio"):
    return raptor_pb2.AnalyzeStreamRequest(
        start=raptor_pb2.AnalyzeStreamStart(session_id=session_id, media_type=media_type)
    )


def _chunk_msg(data=b"x", sequence=0):
    return raptor_pb2.AnalyzeStreamRequest(chunk=raptor_pb2.MediaChunk(data=data, sequence=sequence))


class _RequestFeeder:
    """Async iterator a test can push AnalyzeStreamRequest messages into on
    demand -- same convention as test_job_session.py's _RequestFeeder, plus
    an explicit close() to simulate the client ending its send side."""

    _CLOSE = object()

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    def send(self, msg) -> None:
        self._queue.put_nowait(msg)

    def close(self) -> None:
        self._queue.put_nowait(self._CLOSE)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._queue.get()
        if item is self._CLOSE:
            raise StopAsyncIteration
        return item


# ── validation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_first_message_must_be_start():
    feeder = _RequestFeeder()
    feeder.send(_chunk_msg())
    with pytest.raises(ValidationError):
        async for _ in media_stream_service.analyze_stream(
            _FakeKafkaProducer(), "media-analysis-requests", _FakeRedis(), feeder, {"user_id": "u1", "branch_id": "u1"}, 5.0
        ):
            pass


@pytest.mark.asyncio
async def test_analyze_stream_invalid_media_type_is_validation_error():
    feeder = _RequestFeeder()
    feeder.send(_start_msg(media_type="image"))
    with pytest.raises(ValidationError):
        async for _ in media_stream_service.analyze_stream(
            _FakeKafkaProducer(), "media-analysis-requests", _FakeRedis(), feeder, {"user_id": "u1", "branch_id": "u1"}, 5.0
        ):
            pass


@pytest.mark.asyncio
async def test_analyze_stream_missing_session_id_is_validation_error():
    feeder = _RequestFeeder()
    feeder.send(_start_msg(session_id=""))
    with pytest.raises(ValidationError):
        async for _ in media_stream_service.analyze_stream(
            _FakeKafkaProducer(), "media-analysis-requests", _FakeRedis(), feeder, {"user_id": "u1", "branch_id": "u1"}, 5.0
        ):
            pass


@pytest.mark.asyncio
async def test_analyze_stream_kafka_not_configured_is_unavailable():
    feeder = _RequestFeeder()
    feeder.send(_start_msg())
    with pytest.raises(UnavailableError):
        async for _ in media_stream_service.analyze_stream(
            None, "media-analysis-requests", _FakeRedis(), feeder, {"user_id": "u1", "branch_id": "u1"}, 5.0
        ):
            pass


# ── real per-chunk forwarding, no buffering ─────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_forwards_each_chunk_immediately_no_buffering():
    fake_kafka = _FakeKafkaProducer()
    fake_redis = _FakeRedis()
    feeder = _RequestFeeder()
    feeder.send(_start_msg(session_id="sess-1", media_type="video"))

    events = []

    async def driver():
        gen = media_stream_service.analyze_stream(
            fake_kafka, "media-analysis-requests", fake_redis, feeder, {"user_id": "u1", "branch_id": "b1"}, 5.0
        )
        try:
            async for event in gen:
                events.append(event)  # pragma: no cover -- never reached, see docstring
        except UnimplementedError:
            pass

    task = asyncio.ensure_future(driver())
    await asyncio.sleep(0.02)
    assert len(fake_kafka.sent) == 0  # nothing published until a chunk actually arrives

    feeder.send(_chunk_msg(data=b"aaa", sequence=0))
    await asyncio.sleep(0.02)
    assert len(fake_kafka.sent) == 1  # published immediately -- not held back waiting for more chunks

    feeder.send(_chunk_msg(data=b"bbb", sequence=1))
    await asyncio.sleep(0.02)
    assert len(fake_kafka.sent) == 2

    feeder.close()  # simulates the client ending its send side
    await asyncio.wait_for(task, timeout=2)
    assert events == []  # AnalysisEvent is never emitted -- see module docstring


@pytest.mark.asyncio
async def test_analyze_stream_kafka_message_carries_session_id_and_media_type():
    fake_kafka = _FakeKafkaProducer()
    fake_redis = _FakeRedis()
    feeder = _RequestFeeder()
    feeder.send(_start_msg(session_id="corr-me", media_type="audio"))
    feeder.send(_chunk_msg(data=b"hello", sequence=0))
    feeder.close()

    gen = media_stream_service.analyze_stream(
        fake_kafka, "media-analysis-requests", fake_redis, feeder, {"user_id": "u1", "branch_id": "b1"}, 5.0
    )
    with pytest.raises(UnimplementedError):
        async for _ in gen:
            pass  # pragma: no cover

    assert len(fake_kafka.sent) == 1
    topic, message = fake_kafka.sent[0]
    assert topic == "media-analysis-requests"
    assert message["session_id"] == "corr-me"
    assert message["media_type"] == "audio"
    assert message["user_id"] == "u1"
    assert message["sequence"] == 0
    import base64
    assert base64.b64decode(message["chunk_data_b64"]) == b"hello"


# ── idle timeout ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_idle_timeout_is_deadline_exceeded():
    feeder = _RequestFeeder()
    feeder.send(_start_msg())
    # No chunk ever sent -- idle timeout should fire.
    with pytest.raises(DeadlineExceededError):
        async for _ in media_stream_service.analyze_stream(
            _FakeKafkaProducer(), "media-analysis-requests", _FakeRedis(), feeder, {"user_id": "u1", "branch_id": "u1"}, 0.05
        ):
            pass


@pytest.mark.asyncio
async def test_analyze_stream_idle_timeout_cleans_up_session_state():
    fake_redis = _FakeRedis()
    feeder = _RequestFeeder()
    feeder.send(_start_msg(session_id="idle-sess"))
    with pytest.raises(DeadlineExceededError):
        async for _ in media_stream_service.analyze_stream(
            _FakeKafkaProducer(), "media-analysis-requests", fake_redis, feeder, {"user_id": "u1", "branch_id": "u1"}, 0.05
        ):
            pass
    assert media_stream_service._analyze_session_key("idle-sess") not in fake_redis._kv


# ── off by default ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_is_off_by_default_and_forwards_nothing():
    """No service reads the topic, so a stream that is accepted would swallow the client's
    media for nothing. Disabled, it is refused before a byte reaches Kafka or Redis."""
    fake_kafka = _FakeKafkaProducer()
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), kafka_producer=fake_kafka, redis_client=fake_redis)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)

        async def request_gen():
            yield _start_msg(session_id="off-sess", media_type="video")

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.AnalyzeStream(request_gen(), metadata=(("authorization", f"Bearer {make_token()}"),)):
                pass
        assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
        assert "disabled" in exc_info.value.details()
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert fake_kafka.sent == []
    assert media_stream_service._analyze_session_key("off-sess") not in fake_redis._kv


def test_the_setting_defaults_to_off_and_is_read_from_the_environment(monkeypatch):
    from config import Settings

    assert Settings().analyze_stream_enabled is False
    monkeypatch.setenv("GRPC_GATEWAY_ANALYZE_STREAM_ENABLED", "true")
    assert Settings().analyze_stream_enabled is True


# ── UNIMPLEMENTED after forwarding everything ────────────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_ends_unimplemented_after_forwarding_all_chunks_rpc():
    fake_kafka = _FakeKafkaProducer()
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), kafka_producer=fake_kafka, redis_client=fake_redis, analyze_stream_enabled=True)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)

        async def request_gen():
            yield _start_msg(session_id="rpc-sess", media_type="video")
            yield _chunk_msg(data=b"chunk-1", sequence=0)
            yield _chunk_msg(data=b"chunk-2", sequence=1)

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.AnalyzeStream(request_gen(), metadata=(("authorization", f"Bearer {make_token()}"),)):
                pass
        assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert len(fake_kafka.sent) == 2  # both chunks were genuinely forwarded before the UNIMPLEMENTED


# ── cancellation: stops publishing, cleans session state ────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_cancellation_stops_publishing_and_cleans_session_state():
    fake_kafka = _FakeKafkaProducer()
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), kafka_producer=fake_kafka, redis_client=fake_redis, analyze_stream_enabled=True)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        hold = asyncio.Event()

        async def request_gen():
            yield _start_msg(session_id="cancel-sess", media_type="audio")
            yield _chunk_msg(data=b"before-cancel", sequence=0)
            await hold.wait()
            yield _chunk_msg(data=b"never-sent", sequence=1)  # pragma: no cover

        call = stub.AnalyzeStream(request_gen(), metadata=(("authorization", f"Bearer {make_token()}"),))
        await asyncio.sleep(0.2)  # let the first chunk get published
        call.cancel()
        with pytest.raises((grpc.aio.AioRpcError, asyncio.CancelledError)):
            async for _ in call:
                pass
    finally:
        await channel.close()
        await server.stop(grace=None)

    # Give the server-side generator's `finally` a moment to run.
    await asyncio.sleep(0.2)
    assert len(fake_kafka.sent) == 1  # the second chunk, held behind hold.wait(), was never published
    assert media_stream_service._analyze_session_key("cancel-sess") not in fake_redis._kv


# ── shared per-user stream limit ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_stream_shares_stream_limit_with_other_streaming_rpcs():
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), kafka_producer=_FakeKafkaProducer(), redis_client=fake_redis, max_streams_per_user=1, analyze_stream_enabled=True)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        hold = asyncio.Event()

        async def holding_request_gen():
            yield _start_msg(session_id="limit-sess", media_type="audio")
            await hold.wait()
            yield _chunk_msg()  # pragma: no cover

        call = stub.AnalyzeStream(holding_request_gen(), metadata=(("authorization", f"Bearer {make_token()}"),))
        deadline = asyncio.get_running_loop().time() + 5.0
        while int(fake_redis._kv.get("grpc:streams:dev-user", 0)) != 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)  # until it holds the single slot (blocked on the idle-timeout wait)
        assert int(fake_redis._kv.get("grpc:streams:dev-user", 0)) == 1

        # A second, different streaming RPC should be rejected -- proves
        # AnalyzeStream shares the same counter as every other streaming RPC.
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.DownloadAsset(
                raptor_pb2.DownloadAssetRequest(asset_path="a.bin", version_id="v1"),
                metadata=(("authorization", f"Bearer {make_token()}"),),
            ):
                pass
        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

        hold.set()
        with pytest.raises(grpc.aio.AioRpcError):
            async for _ in call:
                pass
    finally:
        await channel.close()
        await server.stop(grace=None)
