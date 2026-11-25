"""Tests for the metrics MetricsInterceptor adds beyond the base request/
latency counters: active-stream gauge, sent/received byte counters, and the
cancellation counter. Reads deltas around each action (Prometheus counters/
gauges are global per-process, shared across the whole test session).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import grpc
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from interceptors.metrics import (
    ACTIVE_STREAMS_GAUGE,
    STREAM_BYTES_COUNTER,
    STREAM_CANCELLATIONS_COUNTER,
)
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server

AUTH = (("authorization", f"Bearer {make_token()}"),)


class _FakeRedis:
    def __init__(self):
        self._kv = {}

    async def get(self, key):
        return self._kv.get(key)

    async def setex(self, key, ttl, value):
        self._kv[key] = value

    async def incr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) + 1
        return self._kv[key]

    async def decr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) - 1
        return self._kv[key]

    async def expire(self, key, ttl):
        pass


class _FakeStorageService:
    async def upload_file(self, primary_file, archive_date_or_ttl, destroy_date_or_ttl, user):
        os.fstat(primary_file.file.fileno())
        return {"asset_path": "assets/a.bin", "version_id": "v1", "primary_filename": "a.bin", "status": "success"}

    async def get_presign_url(self, asset_path, version_id, user):
        return "http://example.com/a.bin"

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        # Tests here pass the object key as asset_path: the platform's metadata lookup does not
        # know it, so the resolver falls back to using it as the key.
        raise HTTPException(status_code=404, detail="not a platform asset_path")


class _FakeStreamResp:
    def __init__(self, body: bytes):
        self._body = body
        self.status_code = 200
        self.headers = {"content-type": "application/octet-stream", "content-length": str(len(body))}

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size):
        yield self._body


class _FakeStreamCtx:
    def __init__(self, body: bytes):
        self._resp = _FakeStreamResp(body)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeHttpClient:
    def stream(self, method, url, headers=None):
        return _FakeStreamCtx(b"x" * 1000)


def _gauge_value(method_name: str) -> float:
    return ACTIVE_STREAMS_GAUGE.labels(grpc_method=method_name)._value.get()


def _counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


def _metadata_msg(**kwargs):
    return raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(**kwargs))


def _chunk_msg(data: bytes):
    return raptor_pb2.UploadAssetRequest(chunk_data=data)


@pytest.mark.asyncio
async def test_active_streams_gauge_tracks_concurrent_upload():
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=_FakeStorageService(), redis_client=fake_redis, max_streams_per_user=10)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    method = "/raptor.v1.RaptorService/UploadAsset"
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        before = _gauge_value(method)
        hold = asyncio.Event()

        async def gen():
            yield _metadata_msg(filename="a.bin")
            await hold.wait()
            yield _chunk_msg(b"x")

        task = asyncio.ensure_future(stub.UploadAsset(gen(), metadata=AUTH))
        deadline = asyncio.get_running_loop().time() + 5.0
        while _gauge_value(method) != before + 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)  # until the server has counted the stream, not a fixed guess
        assert _gauge_value(method) == before + 1

        hold.set()
        await task
        assert _gauge_value(method) == before
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_stream_bytes_counter_counts_sent_bytes_on_download():
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), 
        storage_service=_FakeStorageService(), http_client=_FakeHttpClient(),
        redis_client=fake_redis, max_streams_per_user=10,
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    method = "/raptor.v1.RaptorService/DownloadAsset"
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        before = _counter_value(STREAM_BYTES_COUNTER, grpc_method=method, direction="sent")

        async for _ in stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path="a.bin", version_id="v1"), metadata=AUTH
        ):
            pass

        after = _counter_value(STREAM_BYTES_COUNTER, grpc_method=method, direction="sent")
        assert after > before  # header + chunk message(s), each counted by serialized size
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_stream_bytes_counter_counts_received_bytes_on_upload():
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=_FakeStorageService(), redis_client=fake_redis, max_streams_per_user=10)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    method = "/raptor.v1.RaptorService/UploadAsset"
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        before = _counter_value(STREAM_BYTES_COUNTER, grpc_method=method, direction="received")

        async def gen():
            yield _metadata_msg(filename="a.bin")
            yield _chunk_msg(b"y" * 500)

        await stub.UploadAsset(gen(), metadata=AUTH)

        after = _counter_value(STREAM_BYTES_COUNTER, grpc_method=method, direction="received")
        assert after >= before + 500
    finally:
        await channel.close()
        await server.stop(grace=None)


class _SlowFakeStreamResp:
    """Never finishes on its own -- aiter_bytes yields forever, so the only
    way this call ends is a client cancel. Used to test cancellation on a
    server-streaming RPC (DownloadAsset), where -- unlike UploadAsset's
    client-streaming shape -- a real client-side cancel *does* reliably
    surface as asyncio.CancelledError on the server (verified empirically;
    UploadAsset's request-iterator read sees a clean StopAsyncIteration
    instead on cancel, indistinguishable from the client legitimately
    finishing -- a known grpc.aio behavior, not something this interceptor
    can compensate for. See the security review notes.).
    """
    status_code = 200
    headers = {"content-length": "999999999"}  # noqa: RUF012 -- fake response, never mutated in place

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size):
        while True:
            yield b"x" * chunk_size
            await asyncio.sleep(0.01)


class _SlowFakeStreamCtx:
    async def __aenter__(self):
        return _SlowFakeStreamResp()

    async def __aexit__(self, *exc):
        return False


class _SlowFakeHttpClient:
    def stream(self, method, url, headers=None):
        return _SlowFakeStreamCtx()


@pytest.mark.asyncio
async def test_stream_cancellations_counter_increments_on_client_cancel():
    fake_redis = _FakeRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), 
        storage_service=_FakeStorageService(), http_client=_SlowFakeHttpClient(),
        redis_client=fake_redis, max_streams_per_user=10,
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    method = "/raptor.v1.RaptorService/DownloadAsset"
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        before = _counter_value(STREAM_CANCELLATIONS_COUNTER, grpc_method=method)

        call = stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path="a.bin", version_id="v1"), metadata=AUTH
        )
        received = 0
        async for _ in call:
            received += 1
            if received == 3:
                call.cancel()
                break
        with pytest.raises((grpc.aio.AioRpcError, asyncio.CancelledError)):
            async for _ in call:
                pass
        # The server notices the cancel asynchronously; wait for the counter
        # (bounded) instead of a fixed sleep, which a loaded CI runner can outlast.
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            after = _counter_value(STREAM_CANCELLATIONS_COUNTER, grpc_method=method)
            if after == before + 1:
                break
            await asyncio.sleep(0.02)
        after = _counter_value(STREAM_CANCELLATIONS_COUNTER, grpc_method=method)
        assert after == before + 1
    finally:
        await channel.close()
        await server.stop(grace=None)
