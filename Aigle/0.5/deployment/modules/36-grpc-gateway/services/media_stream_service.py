"""StreamVideoSegment and AnalyzeStream RPC handlers.

StreamVideoSegment — time-ranged video streaming.

Two families of format, handled completely differently:

- MP4_H264 / MPEG_TS ("transcode formats"): an FFmpeg subprocess reads the
  asset's presigned URL directly (module 04's StorageService already
  produces one for DownloadAsset -- reused as-is here), seeks to start_time,
  and remuxes (``-c copy``) into a fragmented-MP4 or MPEG-TS pipe on stdout.
  ``-c copy`` is a deliberate choice, not the ticket's literal flag list:
  a full re-encode would blow the "first frame within 3s" budget on
  anything but a tiny clip, and Raptor's video pipeline already stores
  H.264/AAC uploads almost universally. It assumes the source codec is
  container-compatible with the requested format -- if it isn't, FFmpeg
  exits non-zero and that's surfaced as an INTERNAL error (not silently
  swallowed). Same bounded-queue backpressure pattern as DownloadAsset in
  asset_service.py: the next `proc.stdout.read()` cannot start until the
  previous MediaChunk has been taken off the queue for the gRPC write.

- RAW_FILE_RANGE: no FFmpeg at all -- a direct ranged-GET proxy against the
  same presigned URL, structurally identical to DownloadAsset's chunk loop
  (minus the MediaHeader message, since StreamVideoSegment has no header in
  its response stream). byte_offset/byte_length are used instead of
  start_time/end_time for this format.

Cancellation: StreamVideoSegment is unary_stream (server-streaming). Per
this project's own empirical findings (see docs/_internal/SECURITY_REVIEW.md §4), that
shape surfaces a real client cancel inside the async generator -- as
asyncio.CancelledError, or as GeneratorExit when grpc.aio was blocked writing
the previous chunk; unlike UploadAsset's client-streaming shape, which doesn't.
Either way the `finally` block below runs on every real cancellation,
not just on normal completion/error, and is where the FFmpeg subprocess is
actually terminated (SIGTERM, then SIGKILL after
settings.ffmpeg_terminate_timeout_s if it hasn't exited).

video_id resolution is NOT implemented -- see StreamVideoSegmentRequest's
own proto comment for why (no reverse video_id -> asset_path lookup exists
anywhere in Raptor today). Requests using that oneof branch get
UNIMPLEMENTED, not a guess.

AnalyzeStream — live chunk forwarding, with an honestly-scoped limit.

This RPC's chunk-forwarding half is real: every MediaChunk the client sends
is immediately published to Kafka (settings.kafka_analysis_requests_topic,
default "media-analysis-requests") as it arrives -- no buffering, no
waiting for the stream to end. Session state (a Redis marker keyed by
session_id) is cleaned up in a `finally` block covering normal completion,
error, AND client cancellation alike.

What this RPC does NOT do: return any AnalysisEvent. No consumer anywhere
in Raptor reads media-analysis-requests and produces entity/transcript/
embedding/scene results -- modules 09-12 are batch, whole-file pipelines
(WhisperX/InternVL run against a complete uploaded asset, not incremental
partial-chunk inference), and there is no streaming-capable ML inference
anywhere in this codebase to consume from. Building that is a multi-module
ML/worker-architecture project, well outside grpc-gateway's boundary --
see docs/_internal/OPEN_QUESTION_analyze_stream_results.md. Rather than silently return an
empty event stream (which could read as "ran fine, found nothing" instead
of "this capability doesn't exist yet"), analyze_stream() forwards every
chunk for real, then raises UnimplementedError once the client's send side
closes -- so the RPC's final status code (UNIMPLEMENTED) makes the gap
impossible to miss, while everything before that point (forwarding, idle
timeout, cancellation cleanup, the shared per-user stream limit) is fully
real and covered by tests/test_analyze_stream.py.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import google_crc32c
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "generated"))
from app.services.storage_service import StorageService
from app.services.storage_service import User as StorageUser
from interceptors.errors import (
    InternalError,
    NotFoundError,
    UnavailableError,
    UnimplementedError,
    ValidationError,
)
from raptor.v1 import raptor_pb2

from services.asset_resolver import resolve_asset
from services.asset_service import _next_with_idle_timeout

_logger = logging.getLogger("grpc_gateway.media_stream_service")

_MIN_CHUNK_SIZE = 64 * 1024
# Chunk + framing must stay under gRPC's default 4 MiB receive limit (see asset_service.py).
_MAX_CHUNK_SIZE = 4 * 1024 * 1024 - 4096
_DEFAULT_CHUNK_SIZE = 1024 * 1024

_TRANSCODE_FORMATS = {
    raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264: ("mp4", "+frag_keyframe+empty_moov+default_base_moof"),
    raptor_pb2.VIDEO_STREAM_FORMAT_MPEG_TS: ("mpegts", None),
}


def _chunk_size(preferred: int) -> int:
    return min(max(preferred or _DEFAULT_CHUNK_SIZE, _MIN_CHUNK_SIZE), _MAX_CHUNK_SIZE)


def _validate_source(request: raptor_pb2.StreamVideoSegmentRequest) -> None:
    which = request.WhichOneof("source")
    if which == "video_id":
        raise UnimplementedError(
            "StreamVideoSegment's video_id branch is not implemented -- no endpoint anywhere in "
            "Raptor resolves a version_id back to its asset_path. Pass `asset` (asset_path + "
            "version_id) instead, e.g. from a prior SearchByPerson or UploadAsset response."
        )
    if which != "asset" or not request.asset.asset_path or not request.asset.version_id:
        raise ValidationError("source.asset (asset_path + version_id) is required.")


def _validate_time_range(start_time: float, end_time: float) -> None:
    if end_time > 0 and end_time - start_time <= 0:
        raise ValidationError(f"Invalid time range: start_time={start_time} >= end_time={end_time}.")


def _validate_request(request: raptor_pb2.StreamVideoSegmentRequest) -> None:
    """Everything that can be rejected without asking storage anything -- done first, so a bad
    request is INVALID_ARGUMENT whether or not the asset exists and costs no downstream call."""
    _validate_source(request)
    if request.format == raptor_pb2.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE:
        if request.byte_offset < 0 or request.byte_length < 0:
            raise ValidationError("byte_offset/byte_length must be non-negative.")
    elif request.format in _TRANSCODE_FORMATS:
        _validate_time_range(request.start_time, request.end_time)
    else:
        raise ValidationError(
            f"Unsupported format {raptor_pb2.VideoStreamFormat.Name(request.format)} for StreamVideoSegment "
            "(HLS/DASH are StreamVideo-only, not implemented here)."
        )


async def _resolve_url(
    storage_service: StorageService, request: raptor_pb2.StreamVideoSegmentRequest, user: dict[str, str]
) -> str:
    _validate_source(request)
    storage_user = StorageUser(user_id=user["user_id"], branch_id=user["branch_id"])
    resolved = await resolve_asset(storage_service, request.asset.asset_path, request.asset.version_id, storage_user)
    return resolved.url


async def _iter_with_final_flag(byte_iterator) -> AsyncIterator[tuple[bytes, bool]]:
    pending: bytes | None = None
    async for chunk in byte_iterator:
        if pending is not None:
            yield pending, False
        pending = chunk
    if pending is not None:
        yield pending, True


def _build_ffmpeg_args(
    ffmpeg_binary: str, url: str, start_time: float, end_time: float, container: str, movflags: str | None
) -> list:
    args = [ffmpeg_binary, "-loglevel", "error", "-hide_banner"]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    args += ["-i", url]
    if end_time > 0:
        _validate_time_range(start_time, end_time)
        args += ["-t", str(end_time - start_time)]
    args += ["-c", "copy"]
    if movflags:
        args += ["-movflags", movflags]
    args += ["-f", container, "pipe:1"]
    return args


async def _stream_raw_file_range(
    http_client: httpx.AsyncClient, url: str, request: raptor_pb2.StreamVideoSegmentRequest
) -> AsyncIterator[raptor_pb2.MediaChunk]:
    if request.byte_offset < 0 or request.byte_length < 0:
        raise ValidationError("byte_offset/byte_length must be non-negative.")
    chunk_size = _chunk_size(request.preferred_chunk_size)
    range_end = request.byte_offset + request.byte_length - 1 if request.byte_length else ""
    range_header = {"Range": f"bytes={request.byte_offset}-{range_end}"}

    async with http_client.stream("GET", url, headers=range_header) as resp:
        if resp.status_code == 404:
            raise NotFoundError("Asset not found in storage.")
        if resp.status_code == 416:
            raise ValidationError(f"Requested range not satisfiable: byte_offset={request.byte_offset}")
        resp.raise_for_status()

        sequence = 0
        offset = request.byte_offset
        async for chunk, is_final in _iter_with_final_flag(resp.aiter_bytes(chunk_size)):
            yield raptor_pb2.MediaChunk(
                data=chunk,
                offset=offset,
                sequence=sequence,
                crc32c=google_crc32c.value(chunk),
                end_of_stream=is_final,
            )
            offset += len(chunk)
            sequence += 1


async def _terminate_ffmpeg(proc: asyncio.subprocess.Process, timeout_s: float) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()  # SIGTERM
    try:
        await asyncio.wait_for(proc.wait(), timeout_s)
    except asyncio.TimeoutError:
        _logger.warning("FFmpeg pid=%d did not exit within %.1fs of SIGTERM -- sending SIGKILL.", proc.pid, timeout_s)
        proc.kill()
        await proc.wait()


async def _drain_stderr(proc: asyncio.subprocess.Process) -> bytearray:
    """Reads stderr to completion in the background -- required so a full
    stderr pipe can never block FFmpeg's stdout writes (classic subprocess
    pipe deadlock), and keeps a bounded tail for error reporting."""
    tail = bytearray()
    assert proc.stderr is not None
    async for line in proc.stderr:
        tail.extend(line)
        del tail[:-4096]  # keep only the last ~4 KiB
    return tail


async def _stream_transcode(
    ffmpeg_binary: str,
    ffmpeg_terminate_timeout_s: float,
    url: str,
    request: raptor_pb2.StreamVideoSegmentRequest,
) -> AsyncIterator[raptor_pb2.MediaChunk]:
    container, movflags = _TRANSCODE_FORMATS[request.format]
    args = _build_ffmpeg_args(ffmpeg_binary, url, request.start_time, request.end_time, container, movflags)
    chunk_size = _chunk_size(request.preferred_chunk_size)

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stderr_task = asyncio.create_task(_drain_stderr(proc))

    # Same bounded-queue backpressure pattern as DownloadAsset
    # (asset_service.py): maxsize=1 means the next `proc.stdout.read()`
    # can't start until the previous chunk has been taken off the queue
    # for the gRPC write.
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    _DONE = object()

    async def _read_into_queue():
        # See DownloadAsset's reader: `_DONE` is only put where a consumer can still take it. A
        # `finally: await queue.put(_DONE)` blocked forever on a full queue when the consumer
        # cancelled this task, deadlocking the stream's cleanup.
        assert proc.stdout is not None
        try:
            while True:
                data = await proc.stdout.read(chunk_size)
                if not data:
                    break
                await queue.put(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            await queue.put(_DONE)  # the consumer is still reading; `await reader_task` below re-raises
            raise
        await queue.put(_DONE)

    reader_task = asyncio.create_task(_read_into_queue())
    try:
        sequence = 0
        offset = 0
        pending: bytes | None = None
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if pending is not None:
                yield raptor_pb2.MediaChunk(
                    data=pending, offset=offset, sequence=sequence,
                    crc32c=google_crc32c.value(pending), end_of_stream=False,
                )
                offset += len(pending)
                sequence += 1
            pending = item

        await reader_task
        returncode = await proc.wait()
        stderr_tail = await stderr_task

        if returncode != 0:
            raise InternalError(
                f"FFmpeg exited with code {returncode}: {bytes(stderr_tail).decode('utf-8', 'replace').strip()[-500:]}"
            )

        if pending is not None:
            yield raptor_pb2.MediaChunk(
                data=pending, offset=offset, sequence=sequence,
                crc32c=google_crc32c.value(pending), end_of_stream=True,
            )
    finally:
        await _terminate_ffmpeg(proc, ffmpeg_terminate_timeout_s)
        for task in (reader_task, stderr_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception) as e:  # noqa: BLE001 -- cleanup after cancel, discard whatever the cancelled task raised
                _logger.debug(f"Task teardown raised after cancel: {e}")


async def stream_video_segment(
    storage_service: StorageService,
    http_client: httpx.AsyncClient,
    ffmpeg_binary: str,
    ffmpeg_terminate_timeout_s: float,
    request: raptor_pb2.StreamVideoSegmentRequest,
    user: dict[str, str],
) -> AsyncIterator[raptor_pb2.MediaChunk]:
    _validate_request(request)
    url = await _resolve_url(storage_service, request, user)

    if request.format == raptor_pb2.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE:
        agen = _stream_raw_file_range(http_client, url, request)
    elif request.format in _TRANSCODE_FORMATS:
        # Time-range validation happens inside _stream_transcode (via
        # _build_ffmpeg_args), before any subprocess is spawned -- not
        # duplicated here.
        agen = _stream_transcode(ffmpeg_binary, ffmpeg_terminate_timeout_s, url, request)
    else:
        raise ValidationError(
            f"Unsupported format {raptor_pb2.VideoStreamFormat.Name(request.format)} for StreamVideoSegment "
            "(HLS/DASH are StreamVideo-only, not implemented here)."
        )

    async for chunk in agen:
        yield chunk


# ── AnalyzeStream ────────────────────────────────────────────────────────

_VALID_ANALYZE_MEDIA_TYPES = {"video", "audio"}


def _analyze_session_key(session_id: str) -> str:
    return f"grpc:analyze_session:{session_id}"


async def _publish_analysis_chunk(
    kafka_producer,
    kafka_topic: str,
    session_id: str,
    media_type: str,
    user: dict[str, str],
    sequence: int,
    chunk: raptor_pb2.MediaChunk,
) -> None:
    message = {
        "message_id": str(uuid.uuid4()),
        "session_id": session_id,
        "media_type": media_type,
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user["user_id"],
        "branch_id": user["branch_id"],
        "chunk_data_b64": base64.b64encode(chunk.data).decode("ascii"),
        "offset": chunk.offset,
        "crc32c": chunk.crc32c,
        "end_of_stream": chunk.end_of_stream,
    }
    await kafka_producer.send_and_wait(kafka_topic, message)


async def analyze_stream(
    kafka_producer,
    kafka_topic: str,
    redis_client,
    request_iterator,
    user: dict[str, str],
    idle_timeout_s: float,
) -> AsyncIterator[raptor_pb2.AnalysisEvent]:
    if kafka_producer is None:
        raise UnavailableError("Kafka is not configured on this gateway; AnalyzeStream is unavailable.")

    first = await _next_with_idle_timeout(request_iterator, idle_timeout_s)
    if first is None or not first.HasField("start"):
        raise ValidationError("The first AnalyzeStream message must be `start`.")
    start = first.start
    if not start.session_id:
        raise ValidationError("start.session_id is required.")
    if start.media_type not in _VALID_ANALYZE_MEDIA_TYPES:
        raise ValidationError(
            f"start.media_type must be one of {sorted(_VALID_ANALYZE_MEDIA_TYPES)}, got {start.media_type!r}."
        )

    session_key = _analyze_session_key(start.session_id)
    await redis_client.setex(session_key, 86400, "1")
    try:
        sequence = 0
        while True:
            msg = await _next_with_idle_timeout(request_iterator, idle_timeout_s)
            if msg is None:
                break
            if not msg.HasField("chunk"):
                raise ValidationError("Expected `chunk` after the start message.")
            await _publish_analysis_chunk(
                kafka_producer, kafka_topic, start.session_id, start.media_type, user, sequence, msg.chunk
            )
            sequence += 1
    finally:
        # Covers normal completion, a validation error, AND client
        # cancellation alike -- this is the "cleans session state" half of
        # this RPC's cancellation contract. The other half ("stops Kafka
        # publishing") falls out naturally: once cancelled, the `while`
        # loop above simply never reaches another _publish_analysis_chunk
        # call, nothing extra to do.
        await redis_client.delete(session_key)

    # Every chunk the client sent was forwarded above, for real. See this
    # module's docstring for why the RPC ends here rather than yielding a
    # (permanently empty) AnalysisEvent stream.
    raise UnimplementedError(
        "AnalyzeStream forwarded every chunk to Kafka, but no analysis-results pipeline exists yet "
        "to produce AnalysisEvent -- see docs/_internal/OPEN_QUESTION_analyze_stream_results.md."
    )
    yield  # pragma: no cover - keeps this an async generator matching the RPC's streaming-response shape
