"""A cancelled UploadAsset must not be committed, and must be counted as cancelled.

Found on a deployed gateway: an upload that declared no `total_size` and was cancelled midway was
stored as a complete asset (6 cancelled uploads -> 6 assets, all counted as OK). grpc.aio ends the
handler's request iterator for a client-streaming cancel exactly as it does for a client that
finished sending, so the handler saw a clean end of stream and committed the partial bytes. The
cancellation itself only arrives at the handler's next suspension point. With `total_size`
declared the size check happened to reject the partial upload, which hid the problem.
"""
from __future__ import annotations

import asyncio

import grpc
import httpx
import pytest
from app.services.storage_service import StorageService
from config import Settings
from conftest import authz_http_client, identity_http_client, make_token
from interceptors.metrics import REQUEST_COUNTER, STREAM_CANCELLATIONS_COUNTER
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from test_upload_real_storage import _asset_service, _FakeRedis

METHOD = "/raptor.v1.RaptorService/UploadAsset"


def _count(counter, **labels):
    return counter.labels(grpc_method=METHOD, **labels)._value.get()


def _metadata(**kw):
    return raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(
        filename="clip.mp4", content_type="video/mp4", **kw))


async def _serve(captured):
    http_client = _asset_service(captured)
    storage = StorageService(http_client, Settings(_env_file=None))
    server = build_server(
        storage_service=storage, http_client=http_client, redis_client=_FakeRedis(),
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(),
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    return server, grpc.aio.insecure_channel(f"localhost:{port}")


def _uploads(captured):
    return [r for r in captured if r.url.path.endswith("/fileupload")]


async def _settle():
    # the handler's cleanup and the metrics wrapper's `finally` run after the client call has returned
    for _ in range(50):
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("declare_total_size", [False, True])
async def test_cancelled_upload_is_not_committed_and_counted_as_cancelled(declare_total_size):
    captured: list[httpx.Request] = []
    server, channel = await _serve(captured)
    sent_some = asyncio.Event()

    async def requests():
        kw = {"total_size": 1_000_000} if declare_total_size else {}
        yield _metadata(**kw)
        yield raptor_pb2.UploadAssetRequest(chunk_data=b"x" * 4096)
        sent_some.set()
        await asyncio.sleep(60)  # the client is still "sending" when it cancels
        yield raptor_pb2.UploadAssetRequest(chunk_data=b"never")

    before = {
        "cancelled": _count(REQUEST_COUNTER, grpc_code="CANCELLED"),
        "ok": _count(REQUEST_COUNTER, grpc_code="OK"),
        "invalid": _count(REQUEST_COUNTER, grpc_code="INVALID_ARGUMENT"),
        "cancellations": _count(STREAM_CANCELLATIONS_COUNTER),
    }
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        call = stub.UploadAsset(requests(), metadata=(("authorization", f"Bearer {make_token()}"),))
        await asyncio.wait_for(sent_some.wait(), 5)
        await asyncio.sleep(0.1)  # let the chunk reach the server
        call.cancel()
        await _settle()
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert _uploads(captured) == [], "a cancelled upload was sent to the asset service (committed)"
    assert _count(REQUEST_COUNTER, grpc_code="CANCELLED") == before["cancelled"] + 1
    assert _count(REQUEST_COUNTER, grpc_code="OK") == before["ok"]
    assert _count(REQUEST_COUNTER, grpc_code="INVALID_ARGUMENT") == before["invalid"]  # not mistaken for a size mismatch
    assert _count(STREAM_CANCELLATIONS_COUNTER) == before["cancellations"] + 1


@pytest.mark.asyncio
async def test_an_upload_that_finishes_normally_without_total_size_is_still_committed():
    captured: list[httpx.Request] = []
    server, channel = await _serve(captured)

    async def requests():
        yield _metadata()
        yield raptor_pb2.UploadAssetRequest(chunk_data=b"y" * 2048)

    ok_before = _count(REQUEST_COUNTER, grpc_code="OK")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        resp = await stub.UploadAsset(requests(), metadata=(("authorization", f"Bearer {make_token()}"),))
        await _settle()
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert resp.bytes_received == 2048 and resp.asset_id == "video/mp4/clip"
    assert len(_uploads(captured)) == 1
    assert _count(REQUEST_COUNTER, grpc_code="OK") == ok_before + 1
