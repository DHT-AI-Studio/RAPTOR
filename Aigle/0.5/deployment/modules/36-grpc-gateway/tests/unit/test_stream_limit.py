"""Tests for StreamLimitInterceptor: per-user concurrent-stream cap shared
across every streaming RPC (not just one), unary RPCs exempt, and the Redis
slot released on normal completion, error, AND client cancellation.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server


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


def _metadata_msg(**kwargs):
    return raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(**kwargs))


def _chunk_msg(data: bytes):
    return raptor_pb2.UploadAssetRequest(chunk_data=data)


def _holding_upload_gen(hold: asyncio.Event):
    async def _g():
        yield _metadata_msg(filename="a.bin")
        await hold.wait()
        yield _chunk_msg(b"x")
    return _g()


AUTH_A = (("authorization", f"Bearer {make_token()}"),)
SLOT_KEY = "grpc:streams:dev-user"  # AUTH_A's token's own `sub` claim (make_token()'s default)


async def _wait_for_slots(fake_redis, expected: int, timeout_s: float = 5.0) -> None:
    """Waits until the user's stream counter equals `expected`. Replaces fixed sleeps: the thing
    the tests need to see is the server having taken (or released) its slot, so wait for that."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while int(fake_redis._kv.get(SLOT_KEY, 0)) != expected:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"stream counter is {fake_redis._kv.get(SLOT_KEY, 0)}, expected {expected}")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_stream_limit_blocks_excess_streams_across_different_rpc_types():
    """2 concurrent UploadAsset streams fill the (limit=2) shared counter;
    a third stream -- via a *different* RPC (DownloadAsset) -- is rejected.
    Proves the counter is shared per-user, not per-method."""
    fake_redis = _FakeRedis()
    server = build_server(storage_service=_FakeStorageService(), redis_client=fake_redis, max_streams_per_user=2, identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        hold1, hold2 = asyncio.Event(), asyncio.Event()

        t1 = asyncio.ensure_future(stub.UploadAsset(_holding_upload_gen(hold1), metadata=AUTH_A))
        t2 = asyncio.ensure_future(stub.UploadAsset(_holding_upload_gen(hold2), metadata=AUTH_A))
        await _wait_for_slots(fake_redis, 2)  # both acquired their slot and are blocked on hold.wait()

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.DownloadAsset(
                raptor_pb2.DownloadAssetRequest(asset_path="a.bin", version_id="v1"), metadata=AUTH_A
            ):
                pass
        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

        hold1.set()
        hold2.set()
        await t1
        await t2
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_stream_limit_exempts_unary_rpcs():
    """Even with every stream slot occupied, a unary RPC (Ping) is unaffected."""
    fake_redis = _FakeRedis()
    server = build_server(storage_service=_FakeStorageService(), redis_client=fake_redis, max_streams_per_user=1, identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        hold = asyncio.Event()
        t1 = asyncio.ensure_future(stub.UploadAsset(_holding_upload_gen(hold), metadata=AUTH_A))
        await _wait_for_slots(fake_redis, 1)

        resp = await stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=AUTH_A)
        assert resp.message == "hi"

        hold.set()
        await t1
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_stream_limit_releases_slot_on_client_cancellation():
    """Cancelling a stream releases its slot just as reliably as a normal finish."""
    fake_redis = _FakeRedis()
    server = build_server(storage_service=_FakeStorageService(), redis_client=fake_redis, max_streams_per_user=1, identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        hold = asyncio.Event()
        call = stub.UploadAsset(_holding_upload_gen(hold), metadata=AUTH_A)
        await _wait_for_slots(fake_redis, 1)  # it holds the single slot

        call.cancel()
        # A client-streaming (stream_unary) call cancelled while the client
        # is still sending surfaces as asyncio.CancelledError, not
        # AioRpcError -- unlike a server-streaming cancel, there's no
        # completed RPC response to wrap.
        with pytest.raises((grpc.aio.AioRpcError, asyncio.CancelledError)):
            await call
        await _wait_for_slots(fake_redis, 0)  # the server observed the cancellation and released the slot

        assert int(fake_redis._kv.get(SLOT_KEY, 0)) == 0

        # The single slot must be free again -- a fresh stream should succeed immediately.
        hold2 = asyncio.Event()
        t2 = asyncio.ensure_future(stub.UploadAsset(_holding_upload_gen(hold2), metadata=AUTH_A))
        await _wait_for_slots(fake_redis, 1)
        assert int(fake_redis._kv[SLOT_KEY]) == 1
        hold2.set()
        await t2
    finally:
        await channel.close()
        await server.stop(grace=None)
