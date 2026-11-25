"""Tests for DownloadAsset: request validation, error mapping, offset/Range
resume, constant-memory streaming (RSS), slow-client backpressure, and
cancellation handle-release. Same self-contained fake-double pattern as the
other test_*.py files here (real grpc.aio server + client where useful, no
mocking of grpc internals).
"""
from __future__ import annotations

import asyncio
import gc
import sys
from pathlib import Path

import grpc
import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services.asset_service import _parse_size_and_seekability

# ── fakes ──────────────────────────────────────────────────────────────────

class _FakeStorageService:
    def __init__(self, url: str | None):
        self._url = url

    async def get_presign_url(self, asset_path, version_id, user):
        return self._url

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        # These tests pass the full object key as asset_path, which the platform's metadata
        # lookup does not know -- the resolver then falls back to using it as the key.
        raise HTTPException(status_code=404, detail="not a platform asset_path")


class _FakeStreamResponse:
    def __init__(self, status_code: int, headers: dict[str, str], chunks):
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    async def aiter_bytes(self, chunk_size):
        async for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse):
        self.response = response
        self.exited = False

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc_info):
        self.exited = True
        return False


class _FakeHttpClient:
    """response_factory(headers_sent) -> _FakeStreamResponse, so tests can
    assert on the Range header the handler actually sent."""

    def __init__(self, response_factory):
        self._response_factory = response_factory
        self.last_context: _FakeStreamContext | None = None
        self.last_headers_sent: dict[str, str] | None = None

    def stream(self, method, url, headers=None):
        self.last_headers_sent = headers or {}
        response = self._response_factory(self.last_headers_sent)
        self.last_context = _FakeStreamContext(response)
        return self.last_context


async def _drain(fake_storage, fake_http, request):
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, http_client=fake_http)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        responses = [
            r async for r in stub.DownloadAsset(request, metadata=(("authorization", f"Bearer {make_token()}"),))
        ]
    finally:
        await channel.close()
        await server.stop(grace=None)
    return responses


def _read_rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found in /proc/self/status")


# ── _parse_size_and_seekability: pure unit tests ──────────────────────────

def test_parse_size_and_seekability_206_partial_content():
    resp = _FakeStreamResponse(206, {"content-range": "bytes 500-1499/1500000"}, chunks=None)
    assert _parse_size_and_seekability(resp) == (1500000, True)


def test_parse_size_and_seekability_200_with_accept_ranges():
    resp = _FakeStreamResponse(200, {"content-length": "42", "accept-ranges": "bytes"}, chunks=None)
    assert _parse_size_and_seekability(resp) == (42, True)


def test_parse_size_and_seekability_200_no_size_info():
    resp = _FakeStreamResponse(200, {}, chunks=None)
    assert _parse_size_and_seekability(resp) == (-1, False)


# ── validation / error mapping ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_download_asset_empty_asset_path_is_invalid_argument():
    fake_storage = _FakeStorageService("http://example.com/a")
    fake_http = _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {}, chunks=None))
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, http_client=fake_http)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.DownloadAsset(
                raptor_pb2.DownloadAssetRequest(asset_path="", version_id="v1"),
                metadata=(("authorization", f"Bearer {make_token()}"),),
            ):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_download_asset_negative_offset_is_invalid_argument():
    fake_storage = _FakeStorageService("http://example.com/a")
    fake_http = _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {}, chunks=None))
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, http_client=fake_http)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.DownloadAsset(
                raptor_pb2.DownloadAssetRequest(asset_path="a.mp4", version_id="v1", offset=-1),
                metadata=(("authorization", f"Bearer {make_token()}"),),
            ):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_download_asset_missing_presign_url_is_not_found():
    fake_storage = _FakeStorageService(None)
    fake_http = _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {}, chunks=None))
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _drain(
            fake_storage, fake_http, raptor_pb2.DownloadAssetRequest(asset_path="missing.mp4", version_id="v1")
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_download_asset_storage_404_is_not_found():
    fake_storage = _FakeStorageService("http://example.com/a.mp4")
    fake_http = _FakeHttpClient(lambda headers: _FakeStreamResponse(404, {}, chunks=None))
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _drain(
            fake_storage, fake_http, raptor_pb2.DownloadAssetRequest(asset_path="a.mp4", version_id="v1")
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


# ── offset/Range resume: byte-accurate reconstruction ──────────────────────

@pytest.mark.asyncio
async def test_download_asset_offset_resume_reconstructs_correct_bytes():
    payload = bytes(i % 256 for i in range(2_000_000))
    requested_offset = 500_000

    async def chunks_from(offset):
        remaining = payload[offset:]
        for i in range(0, len(remaining), 65536):
            yield remaining[i : i + 65536]

    def factory(headers_sent):
        assert headers_sent.get("Range") == f"bytes={requested_offset}-"
        return _FakeStreamResponse(
            206,
            {"content-range": f"bytes {requested_offset}-{len(payload) - 1}/{len(payload)}"},
            chunks_from(requested_offset),
        )

    fake_storage = _FakeStorageService("http://example.com/a.bin")
    fake_http = _FakeHttpClient(factory)
    responses = await _drain(
        fake_storage,
        fake_http,
        raptor_pb2.DownloadAssetRequest(asset_path="a.bin", version_id="v1", offset=requested_offset),
    )

    header = responses[0].header
    assert header.content_length == len(payload)
    assert header.seekable is True

    chunk_msgs = [r.chunk for r in responses[1:]]
    assert chunk_msgs[0].offset == requested_offset
    reconstructed = b"".join(c.data for c in chunk_msgs)
    assert reconstructed == payload[requested_offset:]
    assert chunk_msgs[-1].end_of_stream is True
    assert all(not c.end_of_stream for c in chunk_msgs[:-1])


# ── constant memory for a large download ────────────────────────────────────

@pytest.mark.asyncio
async def test_download_asset_constant_rss_for_large_file():
    chunk = b"x" * (1024 * 1024)
    total_chunks = 1200  # ~1.17 GB, comfortably over the ticket's >1GB bar

    async def large_chunks():
        for _ in range(total_chunks):
            yield chunk

    def factory(headers_sent):
        return _FakeStreamResponse(
            200, {"content-length": str(len(chunk) * total_chunks)}, large_chunks()
        )

    fake_storage = _FakeStorageService("http://example.com/big.bin")
    fake_http = _FakeHttpClient(factory)

    gc.collect()
    rss_before = _read_rss_kb()

    total_bytes = 0
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, http_client=fake_http)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        async for resp in stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path="big.bin", version_id="v1"),
            metadata=(("authorization", f"Bearer {make_token()}"),),
        ):
            if resp.HasField("chunk"):
                total_bytes += len(resp.chunk.data)
    finally:
        await channel.close()
        await server.stop(grace=None)

    gc.collect()
    rss_after = _read_rss_kb()

    assert total_bytes == len(chunk) * total_chunks
    growth_kb = rss_after - rss_before
    assert growth_kb < 300 * 1024, f"RSS grew by {growth_kb} KB while streaming {total_bytes} bytes -- looks buffered"


# ── slow-client backpressure ─────────────────────────────────────────────
#
# NOTE: this drives the servicer's async generator directly (like the
# cancellation test below), not through a live grpc round trip. A real
# end-to-end test through grpc.aio buffers several small writes ahead of
# what the peer has actually read (HTTP/2 flow control lets it absorb
# several of these 64KB messages before it would ever apply real
# backpressure), which is normal transport behavior and outside this
# handler's control. What the handler *is* responsible for -- and what
# "queue-bounded read loop" actually means here -- is that its own storage
# read loop doesn't run ahead of what it has handed off for delivery,
# regardless of how fast/slow the eventual consumer drains it. That's a
# claim about the handler's internal structure, so it's tested at that
# level, isolated from grpc's own buffering.

@pytest.mark.asyncio
async def test_download_asset_backpressure_bounded_lookahead():
    from interceptors.context import user_var
    from server import RaptorServiceServicer

    chunk = b"y" * 65536
    total_chunks = 20
    produced = [0]

    async def counted_chunks():
        for _ in range(total_chunks):
            produced[0] += 1
            yield chunk

    fake_storage = _FakeStorageService("http://example.com/slow.bin")
    fake_http = _FakeHttpClient(
        lambda headers: _FakeStreamResponse(200, {"content-length": str(len(chunk) * total_chunks)}, counted_chunks())
    )
    servicer = RaptorServiceServicer(
        video_search_service=None, storage_service=fake_storage, http_client=fake_http, redis_client=None,
    )

    token = user_var.set({"user_id": "dev-user", "branch_id": "dev-user"})
    max_lead = 0
    consumed = 0
    try:
        gen = servicer.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path="slow.bin", version_id="v1"), context=None
        )
        async for resp in gen:
            if resp.HasField("chunk"):
                consumed += 1
                max_lead = max(max_lead, produced[0] - consumed)
                await asyncio.sleep(0.05)  # simulates a slow consumer
    finally:
        user_var.reset(token)

    assert consumed == total_chunks
    # 1 from the queue's own maxsize=1, +1 from _iter_with_final_flag's
    # one-chunk lookahead -- verified in isolation (no grpc) to hold exactly
    # at 1 for the queue alone; allow a little slack for scheduling order.
    assert max_lead <= 2, f"read loop produced up to {max_lead} chunks ahead of what it had yielded"


# ── cancellation releases the storage handle ────────────────────────────────

@pytest.mark.asyncio
async def test_download_asset_cancellation_releases_handle():
    from interceptors.context import user_var
    from server import RaptorServiceServicer

    async def many_chunks():
        for _ in range(100):
            yield b"z" * 1024

    fake_storage = _FakeStorageService("http://example.com/cancel-me.bin")
    fake_http = _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {}, many_chunks()))
    servicer = RaptorServiceServicer(
        video_search_service=None, storage_service=fake_storage, http_client=fake_http, redis_client=None,
    )

    token = user_var.set({"user_id": "dev-user", "branch_id": "dev-user"})
    try:
        gen = servicer.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path="cancel-me.bin", version_id="v1"), context=None
        )
        await gen.__anext__()  # header
        await gen.__anext__()  # first chunk
        await gen.aclose()     # simulates gRPC tearing down a cancelled call
    finally:
        user_var.reset(token)

    assert fake_http.last_context.exited is True
