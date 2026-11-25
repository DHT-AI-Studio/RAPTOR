"""Closing a DownloadAsset / StreamVideoSegment stream while the producer is still ahead of the
consumer must always finish, and release everything.

Found on a deployed gateway: roughly 1 in 100 cancelled downloads never released its per-user
stream slot or its active-streams count, for good (checked after minutes and 30,000 other calls).
A task dump of the live process showed readers parked on `await queue.put(_DONE)` and the
generators' `aclose()` waiting for them: the reader's `finally` put `_DONE` into the bounded
(maxsize=1) queue, which was full because the consumer that would have drained it is the very
thing being closed; the cancel that interrupted the reader is delivered once, so nothing ever
woke it again. The reader and the cleanup waited on each other, and because the generator never
finished, the outer interceptors' `finally` blocks (slot release, gauge) never ran either.

The producers below never stop and never await real I/O, so the queue is full whenever the
stream is closed -- the exact state that used to deadlock, every time instead of 1 time in 100.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import stat
import sys
import textwrap

import httpx
import pytest
from fastapi import HTTPException
from raptor.v1 import raptor_pb2
from services import asset_service, media_stream_service

USER = {"user_id": "u", "branch_id": "u"}
CLOSE_TIMEOUT_S = 2


class _Storage:
    async def get_presign_url(self, key, version_id, user):
        return "http://storage.test/object"

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        raise HTTPException(status_code=404, detail="the path is already the object key")


class _Response:
    status_code = 206
    def __init__(self, source):
        self._source = source
        self.headers = {"content-range": "bytes 0-999999999/1000000000", "etag": '"e"', "content-type": "video/mp4"}

    def raise_for_status(self):
        pass

    def aiter_bytes(self, chunk_size):
        return self._source(chunk_size)


class _StreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc_info):
        return False


class _Client:
    def __init__(self, source):
        self._source = source

    def stream(self, method, url, headers=None):
        return _StreamContext(_Response(self._source))


async def _endless(chunk_size):
    while True:
        yield b"x" * chunk_size  # no await: the producer is always ahead, so the queue is always full


def _download(source):
    request = raptor_pb2.DownloadAssetRequest(asset_path="a/b.mp4", version_id="v1", preferred_chunk_size=64 * 1024)
    return asset_service.download_asset(_Storage(), _Client(source), request, USER)


async def _closes_promptly(gen) -> None:
    """Fail if `gen.aclose()` does not finish by itself within CLOSE_TIMEOUT_S.

    asyncio.wait_for is NOT used on purpose: on timeout it cancels the close, and the cleanup
    under test swallows CancelledError, so a deadlocked close would look like a normal one
    (that is how an earlier version of this test passed while hanging for the full timeout).
    """
    closing = asyncio.ensure_future(gen.aclose())
    _, pending = await asyncio.wait({closing}, timeout=CLOSE_TIMEOUT_S)
    if pending:
        closing.cancel()
        pytest.fail(f"aclose() was still running after {CLOSE_TIMEOUT_S}s: the stream cleanup is deadlocked")
    closing.result()


def _reader_tasks():
    return [t for t in asyncio.all_tasks() if "_read_into_queue" in repr(t.get_coro())]


@pytest.mark.asyncio
async def test_closing_a_download_while_the_producer_is_ahead_finishes_and_leaves_no_reader():
    gen = _download(_endless)
    await gen.__anext__()                      # header
    await gen.__anext__()                      # first chunk; the reader now fills the queue and parks on put()
    await asyncio.sleep(0.05)
    await _closes_promptly(gen)                # used to hang forever
    assert _reader_tasks() == []


@pytest.mark.asyncio
async def test_closing_a_download_repeatedly_never_accumulates_stuck_readers():
    for _ in range(8):
        gen = _download(_endless)
        await gen.__anext__()
        await gen.__anext__()
        await asyncio.sleep(0)
        await _closes_promptly(gen)
    assert _reader_tasks() == []


@pytest.mark.asyncio
async def test_a_storage_failure_mid_download_is_an_error_not_a_clean_end():
    async def fails_after_two_chunks(chunk_size):
        yield b"a" * chunk_size
        yield b"b" * chunk_size
        raise httpx.ReadError("connection reset by peer")

    gen = _download(fails_after_two_chunks)
    received = 0
    with pytest.raises(httpx.ReadError):
        async for response in gen:
            received += response.HasField("chunk")
    assert received < 3  # it stopped; it did not pretend the file ended there


@pytest.mark.asyncio
async def test_a_complete_download_still_ends_normally_with_end_of_stream():
    async def three_chunks(chunk_size):
        for part in (b"a", b"b", b"c"):
            yield part * 10

    chunks = [r.chunk async for r in _download(three_chunks) if r.HasField("chunk")]
    assert [c.data for c in chunks] == [b"a" * 10, b"b" * 10, b"c" * 10]
    assert [c.end_of_stream for c in chunks] == [False, False, True]


# ── the FFmpeg transcode path has the same reader/queue shape ────────────────


def _fake_ffmpeg(tmp_path) -> tuple[str, str]:
    """An executable that ignores its arguments and writes to stdout forever (and records its pid)."""
    pid_file = tmp_path / "fake-ffmpeg.pid"
    script = tmp_path / "fake-ffmpeg"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import os, sys
        __import__("pathlib").Path({str(pid_file)!r}).write_text(str(os.getpid()))
        block = b"x" * 65536
        while True:
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), str(pid_file)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.asyncio
async def test_closing_a_transcode_while_ffmpeg_is_ahead_finishes_and_stops_the_process(tmp_path):
    binary, pid_file = _fake_ffmpeg(tmp_path)
    request = raptor_pb2.StreamVideoSegmentRequest(
        asset=raptor_pb2.AssetRef(asset_path="a/b.mp4", version_id="v1"),
        format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264,
    )
    gen = media_stream_service._stream_transcode(binary, 2.0, "http://storage.test/object", request)
    await gen.__anext__()
    await asyncio.sleep(0.2)                   # ffmpeg keeps writing; the reader parks on a full queue
    pid = int(pathlib.Path(pid_file).read_text())
    assert _alive(pid)
    await _closes_promptly(gen)                # used to hang forever
    assert not _alive(pid), "the ffmpeg process must not outlive its stream"
    assert _reader_tasks() == []
