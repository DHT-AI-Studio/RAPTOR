"""Tests for StreamVideoSegment: request validation, the video_id branch's
UNIMPLEMENTED status, RAW_FILE_RANGE's no-FFmpeg byte-range path, and (with
a real FFmpeg subprocess against a real tiny local test video, served by a
real local HTTP server) transcode correctness, early-playability, and
cancellation cleanup.

No browser automation exists anywhere in this repo (no Selenium/Playwright),
so "playable within 3s" is verified via the closest automatable proxy: the
first MediaChunk must arrive within 3s, and the accumulated prefix bytes up
to that point must themselves be a structurally valid, ffprobe-parseable
fragmented-MP4 stream -- not a live <video> element render.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import grpc
import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import media_stream_service

# ── fakes ──────────────────────────────────────────────────────────────────

class _FakeRedis:
    """Avoids real DNS/connection overhead to a nonexistent redis-standalone
    host -- StreamLimitInterceptor fails open on a real connection error
    anyway, but the DNS resolution failure itself is slow enough to skew the
    timing-sensitive tests below (early-playback within 3s)."""

    def __init__(self):
        self._kv = {}

    async def incr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) + 1
        return self._kv[key]

    async def decr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) - 1
        return self._kv[key]

    async def expire(self, key, ttl):
        pass


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
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _FakeHttpClient:
    def __init__(self, response_factory):
        self._response_factory = response_factory
        self.last_headers_sent: dict[str, str] | None = None

    def stream(self, method, url, headers=None):
        self.last_headers_sent = headers or {}
        return _FakeStreamContext(self._response_factory(self.last_headers_sent))


def _asset_ref(asset_path="test.mp4", version_id="v1"):
    return raptor_pb2.AssetRef(asset_path=asset_path, version_id=version_id)


# ── real tiny test video, served over real HTTP ─────────────────────────────

@pytest.fixture(scope="module")
def test_video(tmp_path_factory):
    # These tests drive a real ffmpeg (the image installs it; see Dockerfile).
    # CI's unit-test runner doesn't have one, so skip there rather than fail.
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not available on this host")
    tmp_dir = tmp_path_factory.mktemp("media_stream_test")
    video_path = tmp_dir / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-shortest", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac",
            str(video_path),
        ],
        check=True, capture_output=True,
    )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}/test.mp4"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def track_spawned_processes(monkeypatch):
    """Wraps asyncio.create_subprocess_exec to record every Process it
    creates, so tests can assert on real subprocess lifecycle (spawned,
    terminated) without guessing at internal state."""
    spawned = []
    original = asyncio.create_subprocess_exec

    async def _tracking(*args, **kwargs):
        proc = await original(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(media_stream_service.asyncio, "create_subprocess_exec", _tracking)
    return spawned


# ── validation / video_id branch ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_video_segment_video_id_branch_is_unimplemented():
    fake_storage = _FakeStorageService("http://example.com/x.mp4")
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(
            video_id="some-video-id", format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),)):
                pass
        assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_stream_video_segment_missing_source_is_invalid_argument():
    fake_storage = _FakeStorageService("http://example.com/x.mp4")
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264)
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),)):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_stream_video_segment_invalid_time_range_is_invalid_argument_and_spawns_no_ffmpeg(
    track_spawned_processes,
):
    fake_storage = _FakeStorageService("http://example.com/x.mp4")
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(
            asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264, start_time=10.0, end_time=5.0,
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),)):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(grace=None)
    assert track_spawned_processes == []


# ── RAW_FILE_RANGE: no FFmpeg ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_video_segment_raw_file_range_does_not_spawn_ffmpeg_and_proxies_bytes(
    track_spawned_processes,
):
    payload = bytes(range(256)) * 100  # 25,600 bytes

    async def chunks():
        for i in range(0, len(payload), 4096):
            yield payload[i : i + 4096]

    fake_storage = _FakeStorageService("http://example.com/raw.bin")
    fake_http = _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {}, chunks()))
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, http_client=fake_http, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(
            asset=_asset_ref(asset_path="raw.bin"), format=raptor_pb2.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE,
        )
        chunks_received = [
            r async for r in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),))
        ]
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert b"".join(c.data for c in chunks_received) == payload
    assert chunks_received[-1].end_of_stream is True
    assert track_spawned_processes == []  # the whole point of RAW_FILE_RANGE


@pytest.mark.asyncio
async def test_stream_video_segment_raw_file_range_honors_byte_offset_length():
    payload = bytes(range(256)) * 10

    async def chunks_from(offset, length):
        window = payload[offset : offset + length] if length else payload[offset:]
        yield window

    def factory(headers_sent):
        assert headers_sent.get("Range") == "bytes=100-199"
        return _FakeStreamResponse(206, {"content-range": f"bytes 100-199/{len(payload)}"}, chunks_from(100, 100))

    fake_storage = _FakeStorageService("http://example.com/raw.bin")
    fake_http = _FakeHttpClient(factory)
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, http_client=fake_http, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(
            asset=_asset_ref(asset_path="raw.bin"),
            format=raptor_pb2.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE,
            byte_offset=100, byte_length=100,
        )
        chunks_received = [
            r async for r in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),))
        ]
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert b"".join(c.data for c in chunks_received) == payload[100:200]


# ── real FFmpeg: transcode formats ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_video_segment_mp4_first_chunk_within_3s_and_structurally_valid(test_video, tmp_path):
    fake_storage = _FakeStorageService(test_video)
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    accumulated = bytearray()
    first_chunk_elapsed = None
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(
            asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264,
        )
        t0 = time.monotonic()
        call = stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),))
        async for chunk in call:
            if first_chunk_elapsed is None:
                first_chunk_elapsed = time.monotonic() - t0
            accumulated.extend(chunk.data)
            if chunk.end_of_stream or len(accumulated) > 65536:
                break
        call.cancel()
        with contextlib.suppress(grpc.aio.AioRpcError, asyncio.CancelledError):
            async for _ in call:
                pass
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert first_chunk_elapsed is not None, "no chunk was ever received"
    assert first_chunk_elapsed < 3.0, f"first MediaChunk took {first_chunk_elapsed:.2f}s (budget: 3s)"

    prefix_path = tmp_path / "prefix.mp4"
    prefix_path.write_bytes(bytes(accumulated))
    result = subprocess.run(  # noqa: ASYNC221 -- test-only, no other task competing for this event loop
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "default=nw=1:nk=1", str(prefix_path)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"ffprobe could not parse the streamed prefix: {result.stderr}"
    assert "mp4" in result.stdout.lower()


@pytest.mark.asyncio
async def test_stream_video_segment_mpeg_ts_format_produces_valid_sync_byte(test_video):
    fake_storage = _FakeStorageService(test_video)
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        request = raptor_pb2.StreamVideoSegmentRequest(
            asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MPEG_TS,
        )
        first_chunk = None
        async for chunk in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),)):
            first_chunk = chunk
            break
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert first_chunk is not None
    assert first_chunk.data[0] == 0x47  # MPEG-TS sync byte, first byte of every 188-byte packet


@pytest.mark.asyncio
async def test_stream_video_segment_time_range_produces_shorter_output_than_full_clip(test_video):
    fake_storage = _FakeStorageService(test_video)
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)

        async def total_bytes(start_time, end_time):
            request = raptor_pb2.StreamVideoSegmentRequest(
                asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MPEG_TS,
                start_time=start_time, end_time=end_time,
            )
            total = 0
            async for chunk in stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),)):
                total += len(chunk.data)
            return total

        full = await total_bytes(0, 0)
        clipped = await total_bytes(0, 1)  # first 1s of a 3s clip
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert 0 < clipped < full


# ── cancellation: FFmpeg is actually terminated ──────────────────────────────

@pytest.mark.asyncio
async def test_stream_video_segment_cancellation_terminates_ffmpeg_within_5s(
    test_video, track_spawned_processes
):
    from interceptors.context import user_var
    from server import RaptorServiceServicer

    http_client = httpx.AsyncClient()
    fake_storage = _FakeStorageService(test_video)
    servicer = RaptorServiceServicer(
        video_search_service=None, storage_service=fake_storage, http_client=http_client, redis_client=None,
    )
    token = user_var.set({"user_id": "dev-user", "branch_id": "dev-user"})
    try:
        request = raptor_pb2.StreamVideoSegmentRequest(
            asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264,
        )
        gen = servicer.StreamVideoSegment(request, context=None)
        await gen.__anext__()  # first chunk -- proves ffmpeg is actually running
        await gen.aclose()     # simulates gRPC tearing down a cancelled call
    finally:
        user_var.reset(token)
        await http_client.aclose()

    assert len(track_spawned_processes) == 1
    proc = track_spawned_processes[0]
    await asyncio.wait_for(proc.wait(), timeout=5)
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_stream_video_segment_50_cancelled_streams_leave_no_orphaned_ffmpeg(
    test_video, track_spawned_processes
):
    fake_storage = _FakeStorageService(test_video)
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        for _ in range(50):
            request = raptor_pb2.StreamVideoSegmentRequest(
                asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264,
            )
            call = stub.StreamVideoSegment(request, metadata=(("authorization", f"Bearer {make_token()}"),))
            async for _ in call:
                break  # wait for the first chunk -- proves ffmpeg is running for this call
            call.cancel()
            with contextlib.suppress(grpc.aio.AioRpcError, asyncio.CancelledError):
                async for _ in call:
                    pass
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert len(track_spawned_processes) == 50
    results = await asyncio.gather(*(asyncio.wait_for(p.wait(), timeout=5) for p in track_spawned_processes))
    assert all(rc is not None for rc in results)
    assert all(p.returncode is not None for p in track_spawned_processes), "orphaned FFmpeg process(es) found"
