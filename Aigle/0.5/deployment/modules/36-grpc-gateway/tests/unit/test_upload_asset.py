"""Tests for UploadAsset: metadata-first framing, idempotency, idle timeout,
per-user concurrency limit, cancellation cleanup, and a large upload. Same
self-contained fake-double pattern as the other test_*.py files -- some
tests go through a real grpc.aio server+client (build_server(...)), others
drive the servicer's coroutine directly (like DownloadAsset's cancellation
test) when they need to tune settings or control timing precisely.
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
import tempfile
from pathlib import Path

import google_crc32c
import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from interceptors.errors import DeadlineExceededError
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import RaptorServiceServicer, build_server


class _FakeStorageService:
    def __init__(self, result=None):
        self._result = result or {
            "asset_path": "assets/a.bin", "version_id": "v1", "primary_filename": "a.bin", "status": "success",
        }
        self.calls = []

    async def upload_file(self, primary_file, archive_date_or_ttl, destroy_date_or_ttl, user):
        size = os.fstat(primary_file.file.fileno()).st_size
        self.calls.append({"filename": primary_file.filename, "content_type": primary_file.content_type, "size": size})
        return self._result


class _FakeRedis:
    """Dict-backed stand-in supporting the ops UploadAsset needs: get/setex
    (idempotency) and incr/decr/expire (concurrency counter)."""

    def __init__(self):
        self._store = {}

    async def get(self, key):
        return self._store.get(key)

    async def setex(self, key, ttl, value):
        self._store[key] = value

    async def incr(self, key):
        self._store[key] = int(self._store.get(key, 0)) + 1
        return self._store[key]

    async def decr(self, key):
        self._store[key] = int(self._store.get(key, 0)) - 1
        return self._store[key]

    async def expire(self, key, ttl):
        pass


class _FakeKafkaProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, message):
        self.sent.append((topic, message))


def _make_servicer(fake_storage=None, fake_redis=None, fake_kafka=None) -> RaptorServiceServicer:
    return RaptorServiceServicer(
        video_search_service=None,
        storage_service=fake_storage or _FakeStorageService(),
        http_client=None,
        redis_client=fake_redis or _FakeRedis(),
        kafka_producer=fake_kafka,
    )


async def _run_upload(fake_storage, fake_redis, request_gen, fake_kafka=None):
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=fake_redis, kafka_producer=fake_kafka)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        return await stub.UploadAsset(request_gen, metadata=(("authorization", f"Bearer {make_token()}"),))
    finally:
        await channel.close()
        await server.stop(grace=None)


def _metadata_msg(**kwargs):
    return raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(**kwargs))


def _chunk_msg(data: bytes):
    return raptor_pb2.UploadAssetRequest(chunk_data=data)


# ── validation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_asset_first_message_not_metadata_is_invalid_argument():
    async def gen():
        yield _chunk_msg(b"oops")

    fake_storage = _FakeStorageService()
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_upload(fake_storage, _FakeRedis(), gen())
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert fake_storage.calls == []


@pytest.mark.asyncio
async def test_upload_asset_empty_filename_is_invalid_argument():
    async def gen():
        yield _metadata_msg(filename="")
        yield _chunk_msg(b"data")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_upload(_FakeStorageService(), _FakeRedis(), gen())
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_upload_asset_total_size_mismatch_is_invalid_argument():
    async def gen():
        yield _metadata_msg(filename="a.bin", total_size=100)
        yield _chunk_msg(b"too short")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_upload(_FakeStorageService(), _FakeRedis(), gen())
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ── checksum (VIE01-161) ─────────────────────────────────────────────────

_PAYLOAD = [b"hello ", b"checksum ", b"world"]
_PAYLOAD_CRC = google_crc32c.value(b"".join(_PAYLOAD))


def _final_crc_msg(value: int):
    return raptor_pb2.UploadAssetRequest(final_crc32c=value)


async def _upload_with(meta_kwargs, trailer=None, extra_after_trailer=False):
    fake_storage = _FakeStorageService()

    async def gen():
        yield _metadata_msg(filename="a.bin", **meta_kwargs)
        for part in _PAYLOAD:
            yield _chunk_msg(part)
        if trailer is not None:
            yield _final_crc_msg(trailer)
        if extra_after_trailer:
            yield _chunk_msg(b"late")

    try:
        resp = await _run_upload(fake_storage, _FakeRedis(), gen())
    except grpc.aio.AioRpcError as exc:
        return fake_storage, exc
    return fake_storage, resp


@pytest.mark.asyncio
async def test_upload_asset_metadata_crc32c_match_is_accepted():
    storage, resp = await _upload_with({"crc32c": _PAYLOAD_CRC})
    assert resp.bytes_received == len(b"".join(_PAYLOAD))
    assert len(storage.calls) == 1


@pytest.mark.asyncio
async def test_upload_asset_trailing_crc32c_match_is_accepted():
    storage, resp = await _upload_with({}, trailer=_PAYLOAD_CRC)
    assert resp.bytes_received == len(b"".join(_PAYLOAD))
    assert len(storage.calls) == 1


@pytest.mark.asyncio
async def test_upload_asset_without_a_checksum_is_not_checked():
    storage, _resp = await _upload_with({})
    assert len(storage.calls) == 1


@pytest.mark.asyncio
async def test_upload_asset_metadata_crc32c_mismatch_is_data_loss_and_stores_nothing():
    storage, exc = await _upload_with({"crc32c": _PAYLOAD_CRC ^ 1})
    assert exc.code() == grpc.StatusCode.DATA_LOSS
    assert "metadata.crc32c" in exc.details()
    assert storage.calls == []


@pytest.mark.asyncio
async def test_upload_asset_trailing_crc32c_mismatch_is_data_loss_and_stores_nothing():
    storage, exc = await _upload_with({}, trailer=_PAYLOAD_CRC ^ 1)
    assert exc.code() == grpc.StatusCode.DATA_LOSS
    assert "final_crc32c" in exc.details()
    assert storage.calls == []


@pytest.mark.asyncio
async def test_upload_asset_both_checksums_must_match():
    storage, exc = await _upload_with({"crc32c": _PAYLOAD_CRC}, trailer=_PAYLOAD_CRC ^ 1)
    assert exc.code() == grpc.StatusCode.DATA_LOSS
    assert storage.calls == []


@pytest.mark.asyncio
async def test_upload_asset_an_explicit_zero_checksum_is_checked_not_ignored():
    # proto3 `optional` keeps "sent 0" apart from "not sent"; CRC32C of this data is not 0.
    storage, exc = await _upload_with({"crc32c": 0})
    assert exc.code() == grpc.StatusCode.DATA_LOSS
    assert storage.calls == []


@pytest.mark.asyncio
async def test_upload_asset_nothing_may_follow_the_trailing_checksum():
    storage, exc = await _upload_with({}, trailer=_PAYLOAD_CRC, extra_after_trailer=True)
    assert exc.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert storage.calls == []


@pytest.mark.asyncio
async def test_upload_asset_size_mismatch_is_reported_before_checksum():
    async def gen():
        yield _metadata_msg(filename="a.bin", total_size=999, crc32c=_PAYLOAD_CRC ^ 1)
        yield _chunk_msg(b"x")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_upload(_FakeStorageService(), _FakeRedis(), gen())
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_upload_asset_checksum_mismatch_leaves_no_temp_file_and_no_idempotency_record():
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "raptor-upload-*")))
    redis = _FakeRedis()

    async def gen():
        yield _metadata_msg(filename="a.bin", idempotency_key="k1", crc32c=_PAYLOAD_CRC ^ 1)
        for part in _PAYLOAD:
            yield _chunk_msg(part)

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_upload(_FakeStorageService(), redis, gen())
    assert exc_info.value.code() == grpc.StatusCode.DATA_LOSS
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "raptor-upload-*"))) == before
    assert not any(k.startswith("idempotency:") for k in redis._store)


# ── idempotency ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_asset_idempotent_duplicate_returns_already_exists_without_rewrite():
    fake_storage = _FakeStorageService()
    fake_redis = _FakeRedis()

    def gen():
        async def _g():
            yield _metadata_msg(filename="a.bin", idempotency_key="key-1")
            yield _chunk_msg(b"hello")
        return _g()

    first = await _run_upload(fake_storage, fake_redis, gen())
    assert len(fake_storage.calls) == 1

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_upload(fake_storage, fake_redis, gen())
    assert exc_info.value.code() == grpc.StatusCode.ALREADY_EXISTS
    trailing = exc_info.value.trailing_metadata()
    assert trailing["asset-id"] == first.asset_id
    assert trailing["version-id"] == first.version_id
    assert len(fake_storage.calls) == 1  # not re-written


# ── idle timeout / concurrency limit (direct servicer invocation, tuned settings) ──

@pytest.mark.asyncio
async def test_upload_asset_idle_timeout_is_deadline_exceeded():
    servicer = _make_servicer()
    servicer._upload_idle_timeout_s = 0.05

    async def hanging_gen():
        yield _metadata_msg(filename="a.bin")
        await asyncio.sleep(10)
        yield _chunk_msg(b"never reached")  # pragma: no cover

    with pytest.raises(DeadlineExceededError):
        await servicer.UploadAsset(hanging_gen(), context=None)


# Per-user concurrent-stream limiting moved to StreamLimitInterceptor --
# shared across every streaming RPC, not just UploadAsset -- see
# tests/test_stream_limit.py for that coverage.


# ── cancellation cleanup, x100 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_asset_cancellation_leaves_no_orphaned_temp_files_x100():
    fake_storage = _FakeStorageService()
    servicer = _make_servicer(fake_storage=fake_storage)

    def _leftover_temp_files():
        return set(glob.glob(os.path.join(tempfile.gettempdir(), "raptor-upload-*")))

    before = _leftover_temp_files()

    def gen():
        # A never-set Event guarantees the task is genuinely suspended
        # (not just "hasn't been scheduled yet") the moment we cancel it --
        # avoids a race where the whole upload finishes before cancel() runs.
        async def _g():
            yield _metadata_msg(filename="a.bin")
            yield _chunk_msg(b"x" * 1024)
            await asyncio.Event().wait()
            yield _chunk_msg(b"never reached")  # pragma: no cover
        return _g()

    for _ in range(100):
        task = asyncio.ensure_future(servicer.UploadAsset(gen(), context=None))
        # asyncio.wait_for wraps each __anext__() in its own Task, so a bare
        # sleep(0) isn't always enough scheduling rounds to reach the Event
        # -- a short real sleep reliably lets it get there before we cancel.
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert _leftover_temp_files() == before
    assert fake_storage.calls == []  # never reached the final upload -- nothing sent to module 04


# ── cancelled while the asset service is committing (KNOWN_ISSUES E15) ───

class _GatedStorage(_FakeStorageService):
    """Holds upload_file until the test lets it finish, so a cancel can land mid-commit."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = 0

    async def upload_file(self, primary_file, archive_date_or_ttl, destroy_date_or_ttl, user):
        self.started.set()
        await self.release.wait()
        result = await super().upload_file(primary_file, archive_date_or_ttl, destroy_date_or_ttl, user)
        self.finished += 1
        return result


class _Context:
    def __init__(self):
        self.trailing = None

    def set_trailing_metadata(self, md):
        self.trailing = dict(md)


def _complete_upload_gen(key="k-1"):
    async def _g():
        yield _metadata_msg(filename="a.bin", idempotency_key=key)
        yield _chunk_msg(b"x" * 1024)
    return _g()


@pytest.mark.asyncio
async def test_a_cancel_during_the_commit_still_finishes_the_asset_and_records_the_key():
    storage, redis, kafka = _GatedStorage(), _FakeRedis(), _FakeKafkaProducer()
    servicer = _make_servicer(fake_storage=storage, fake_redis=redis, fake_kafka=kafka)

    task = asyncio.ensure_future(servicer.UploadAsset(_complete_upload_gen(), context=_Context()))
    await asyncio.wait_for(storage.started.wait(), 5)  # all the data is in, the commit has begun
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()  # the cancel is held back until the commit is done
    assert storage.finished == 0

    storage.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert storage.finished == 1  # the asset was created, once
    assert "idempotency:dev-user:k-1" in redis._store  # and the key remembers it
    assert len(kafka.sent) == 1  # the finalized event went out


@pytest.mark.asyncio
async def test_a_retry_after_that_cancel_gets_already_exists_with_the_original_asset_id():
    storage, redis = _GatedStorage(), _FakeRedis()
    servicer = _make_servicer(fake_storage=storage, fake_redis=redis)

    task = asyncio.ensure_future(servicer.UploadAsset(_complete_upload_gen(), context=_Context()))
    await asyncio.wait_for(storage.started.wait(), 5)
    task.cancel()
    storage.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    from interceptors.errors import AlreadyExistsError

    ctx = _Context()
    with pytest.raises(AlreadyExistsError):
        await servicer.UploadAsset(_complete_upload_gen(), context=ctx)
    assert ctx.trailing == {"asset-id": "assets/a.bin", "version-id": "v1"}
    assert storage.finished == 1  # no second asset


@pytest.mark.asyncio
async def test_the_temp_file_is_removed_after_a_cancelled_commit_finishes():
    storage = _GatedStorage()
    servicer = _make_servicer(fake_storage=storage)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "raptor-upload-*")))

    task = asyncio.ensure_future(servicer.UploadAsset(_complete_upload_gen(), context=_Context()))
    await asyncio.wait_for(storage.started.wait(), 5)
    task.cancel()
    storage.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "raptor-upload-*"))) == before


@pytest.mark.asyncio
async def test_an_error_from_the_commit_still_reaches_the_caller_when_cancelled_meanwhile():
    class _Failing(_GatedStorage):
        async def upload_file(self, *a, **k):
            self.started.set()
            await self.release.wait()
            raise RuntimeError("storage down")

    storage = _Failing()
    servicer = _make_servicer(fake_storage=storage)
    task = asyncio.ensure_future(servicer.UploadAsset(_complete_upload_gen(), context=_Context()))
    await asyncio.wait_for(storage.started.wait(), 5)
    task.cancel()
    storage.release.set()
    with pytest.raises(RuntimeError, match="storage down"):
        await task


@pytest.mark.asyncio
async def test_a_cancel_before_the_data_is_complete_still_stores_nothing():
    storage = _GatedStorage()
    servicer = _make_servicer(fake_storage=storage)

    async def partial():
        yield _metadata_msg(filename="a.bin", idempotency_key="k-2")
        yield _chunk_msg(b"x" * 10)
        await asyncio.Event().wait()

    task = asyncio.ensure_future(servicer.UploadAsset(partial(), context=_Context()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not storage.started.is_set()


# ── large upload ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_asset_large_upload_reports_correct_bytes_and_publishes_event():
    chunk = b"x" * (1024 * 1024)
    total_chunks = 105  # > 100 MB
    total_size = len(chunk) * total_chunks

    async def gen():
        yield _metadata_msg(filename="big.bin", content_type="application/octet-stream", total_size=total_size)
        for _ in range(total_chunks):
            yield _chunk_msg(chunk)

    fake_storage = _FakeStorageService()
    fake_kafka = _FakeKafkaProducer()

    resp = await _run_upload(fake_storage, _FakeRedis(), gen(), fake_kafka=fake_kafka)

    assert resp.bytes_received == total_size
    assert fake_storage.calls[0]["size"] == total_size
    assert len(fake_kafka.sent) == 1
    topic, message = fake_kafka.sent[0]
    assert topic == "asset-upload-finalized"
    assert message["payload"]["bytes_received"] == total_size
