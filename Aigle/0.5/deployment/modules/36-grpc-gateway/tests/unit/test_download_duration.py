"""DownloadAsset fills MediaHeader.duration_sec from the probe, and the probe can never hurt the download (VIE01-160)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import grpc
import pytest
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import asset_service
from services.media_probe import make_duration_probe
from test_download_asset import (
    _FakeHttpClient,
    _FakeStorageService,
    _FakeStreamResponse,
)

_PAYLOAD = b"z" * 300_000
_KEY = "video/mp4/clip/clip.mp4"


def _http_client():
    async def body():
        for start in range(0, len(_PAYLOAD), 100_000):
            yield _PAYLOAD[start:start + 100_000]

    return _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {"content-length": str(len(_PAYLOAD))}, body()))


async def _download(duration_probe):
    server = build_server(
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(),
        storage_service=_FakeStorageService("http://storage.example/clip.mp4"), http_client=_http_client(),
        duration_probe=duration_probe,
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        header, received = None, 0
        async for response in stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path=_KEY, version_id="v1"),
            metadata=(("authorization", f"Bearer {make_token()}"),),
        ):
            if response.HasField("header"):
                header = response.header
            else:
                received += len(response.chunk.data)
        return header, received
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_the_header_carries_the_duration_the_probe_found():
    seen = []

    async def probe(url, filename):
        seen.append((url, filename))
        return 12.5

    header, received = await _download(probe)

    assert header.duration_sec == pytest.approx(12.5)
    assert received == len(_PAYLOAD)
    assert seen == [("http://storage.example/clip.mp4", "clip.mp4")]  # the presigned URL and the file's name


@pytest.mark.asyncio
async def test_without_a_probe_the_duration_is_unknown_and_the_download_is_unchanged():
    header, received = await _download(None)
    assert header.duration_sec == 0.0
    assert received == len(_PAYLOAD)


@pytest.mark.asyncio
async def test_a_probe_that_raises_does_not_fail_the_download():
    async def probe(url, filename):
        raise RuntimeError("ffprobe exploded")

    header, received = await _download(probe)
    assert header.duration_sec == 0.0
    assert received == len(_PAYLOAD)


@pytest.mark.asyncio
async def test_a_hung_ffprobe_delays_the_header_by_at_most_its_timeout(tmp_path):
    script = tmp_path / "hang"
    script.write_text("#!/bin/sh\nexec sleep 30\n")
    script.chmod(0o755)
    probe = make_duration_probe(SimpleNamespace(probe_duration=True, ffprobe_binary=str(script), probe_duration_timeout_s=0.4))

    started = time.monotonic()
    header, received = await _download(probe)

    assert header.duration_sec == 0.0
    assert received == len(_PAYLOAD)
    assert time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_closing_the_download_while_the_probe_runs_cancels_the_probe():
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def probe(url, filename):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 1.0

    storage = _FakeStorageService("http://storage.example/clip.mp4")
    gate = asyncio.Event()

    async def stuck_body():
        await gate.wait()  # the storage stream never produces anything, so the generator waits at its header
        yield b""

    http = _FakeHttpClient(lambda headers: _FakeStreamResponse(200, {"content-length": "1"}, stuck_body()))
    generator = asset_service.download_asset(
        storage, http, raptor_pb2.DownloadAssetRequest(asset_path=_KEY, version_id="v1"),
        {"user_id": "u", "branch_id": "b"}, probe,
    )
    first = asyncio.ensure_future(generator.__anext__())
    await asyncio.wait_for(started.wait(), 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await generator.aclose()

    assert cancelled.is_set(), "the probe outlived its download"


@pytest.mark.asyncio
async def test_a_download_that_fails_before_its_header_still_cancels_the_probe():
    """404 from storage: the generator raises before it ever awaits the probe's result."""
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def probe(url, filename):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 1.0

    async def empty():
        return
        yield b""  # pragma: no cover

    class _SlowToAnswer:
        """Storage takes a moment to say 404, so the probe has really started by then."""

        def stream(self, method, url, headers=None):
            class _Ctx:
                async def __aenter__(self_inner):
                    await asyncio.sleep(0.05)
                    return _FakeStreamResponse(404, {}, empty())

                async def __aexit__(self_inner, *exc_info):
                    return False

            return _Ctx()

    http = _SlowToAnswer()
    generator = asset_service.download_asset(
        _FakeStorageService("http://storage.example/clip.mp4"), http,
        raptor_pb2.DownloadAssetRequest(asset_path=_KEY, version_id="v1"), {"user_id": "u", "branch_id": "b"}, probe,
    )
    from interceptors.errors import NotFoundError

    with pytest.raises(NotFoundError):
        await generator.__anext__()

    assert started.is_set()
    assert cancelled.is_set(), "the probe kept running after the download failed"
