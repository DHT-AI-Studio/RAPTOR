"""DownloadAsset / UploadAsset RPC handlers — streaming asset transfer.

DownloadAsset: one ranged GET against a presigned URL (resolved via
StorageService, module 04's proxy) supplies both the MediaHeader fields
(from response headers) and the chunk stream (from the response body) — no
separate metadata lookup exists anywhere upstream of this. A bounded queue
sits between the storage read and the gRPC write so the read loop never runs
ahead of what it has handed off for delivery, keeping server memory constant
regardless of file size.

UploadAsset: chunks are spooled to a local temp file (bounded memory, not
buffering the full file in RAM) and sent to module 04 in ONE upload when the
stream completes — module 04 has no incremental-write primitive today (its
dedup needs a full-file MD5, and the LakeFS write itself is one atomic
call). A useful consequence: nothing is ever sent to storage until the whole
stream succeeds, so a mid-stream cancellation never touches LakeFS —
cleanup only has to delete our own local temp file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google_crc32c
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "generated"))
from app.services.storage_service import StorageService
from app.services.storage_service import User as StorageUser
from interceptors.errors import (
    AlreadyExistsError,
    DataLossError,
    DeadlineExceededError,
    NotFoundError,
    ValidationError,
)
from raptor.v1 import raptor_pb2

from services.asset_resolver import resolve_asset
from services.media_probe import DurationProbe, background_probe

_logger = logging.getLogger("grpc_gateway.asset_service")

_MIN_CHUNK_SIZE = 64 * 1024
# A chunk plus its framing (offset, sequence, crc32c, ...) must stay under gRPC's default 4 MiB
# receive limit, or a default client gets RESOURCE_EXHAUSTED for the biggest chunk it can ask for.
_MAX_CHUNK_SIZE = 4 * 1024 * 1024 - 4096
_DEFAULT_CHUNK_SIZE = 1024 * 1024


async def _iter_with_final_flag(byte_iterator) -> AsyncIterator[tuple[bytes, bool]]:
    """Yields (chunk, is_final) from an async byte iterator via one-chunk
    lookahead — the source iterator doesn't tell us when the last chunk has
    arrived until it's exhausted, so we hold one chunk back."""
    pending: bytes | None = None
    async for chunk in byte_iterator:
        if pending is not None:
            yield pending, False
        pending = chunk
    if pending is not None:
        yield pending, True


def _parse_size_and_seekability(resp: httpx.Response) -> tuple[int, bool]:
    """Derives (content_length, seekable) from a ranged-GET response's headers.

    On 206 the total size lives in Content-Range's "/total" suffix. On 200
    (server ignored our Range request) fall back to Content-Length, and
    seekable is only true if the server advertises Accept-Ranges.
    """
    if resp.status_code == 206:
        content_range = resp.headers.get("content-range", "")
        total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
        return (int(total), True) if total.isdigit() else (-1, True)
    content_length = resp.headers.get("content-length")
    seekable = resp.headers.get("accept-ranges") == "bytes"
    return (int(content_length), seekable) if content_length is not None else (-1, seekable)


async def download_asset(
    storage_service: StorageService,
    http_client: httpx.AsyncClient,
    request: raptor_pb2.DownloadAssetRequest,
    user: dict[str, str],
    duration_probe: DurationProbe | None = None,
) -> AsyncIterator[raptor_pb2.DownloadAssetResponse]:
    if not request.asset_path or request.offset < 0 or request.length < 0:
        raise ValidationError("asset_path is required and offset/length must be non-negative.")
    chunk_size = min(
        max(request.preferred_chunk_size or _DEFAULT_CHUNK_SIZE, _MIN_CHUNK_SIZE),
        _MAX_CHUNK_SIZE,
    )

    storage_user = StorageUser(user_id=user["user_id"], branch_id=user["branch_id"])
    resolved = await resolve_asset(storage_service, request.asset_path, request.version_id, storage_user)
    url = resolved.url

    range_end = request.offset + request.length - 1 if request.length else ""
    range_header = {"Range": f"bytes={request.offset}-{range_end}"}

    # The probe (ffprobe, for the header's duration_sec) runs while the storage stream is being opened.
    async with background_probe(duration_probe, url, resolved.filename) as probe, \
            http_client.stream("GET", url, headers=range_header) as resp:
        if resp.status_code == 404:
            raise NotFoundError(f"Asset not found in storage: {request.asset_path}")
        if resp.status_code == 416:
            raise ValidationError(f"Requested range not satisfiable: offset={request.offset}")
        resp.raise_for_status()

        content_length, seekable = _parse_size_and_seekability(resp)
        yield raptor_pb2.DownloadAssetResponse(
            header=raptor_pb2.MediaHeader(
                content_type=resp.headers.get("content-type", "application/octet-stream"),
                content_length=content_length,
                filename=resolved.filename,
                seekable=seekable,
                duration_sec=await probe.duration_s(),
                etag=resp.headers.get("etag", "").strip('"'),
            )
        )

        # Explicit bounded queue between the storage read and the gRPC
        # write: relying on "yield just won't be scheduled again until
        # the write completes" isn't actually true -- HTTP/2 flow
        # control lets grpc.aio buffer several small writes ahead of
        # what the peer has consumed. maxsize=1 means the next storage
        # read can't start until the previous chunk has been taken off
        # the queue for writing.
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        _DONE = object()

        async def _read_into_queue():
            # `_DONE` may only be put where a consumer is still there to take it. The old
            # `finally: await queue.put(_DONE)` also ran when the consumer cancelled this task
            # because it was going away; with the maxsize=1 queue full, that put waited forever,
            # the consumer's cleanup waited for this task, and the stream never finished -- so the
            # stream slot and the active-streams count were never released (about 1 in 100 cancels).
            try:
                async for item in _iter_with_final_flag(resp.aiter_bytes(chunk_size)):
                    await queue.put(item)
            except asyncio.CancelledError:
                raise  # the consumer is gone: nobody will take `_DONE`, so do not wait for room for it
            except Exception:
                await queue.put(_DONE)  # the consumer is still reading: let it stop, then re-raise below
                raise
            await queue.put(_DONE)

        reader_task = asyncio.create_task(_read_into_queue())
        try:
            sequence = 0
            offset = request.offset
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                chunk, is_final = item
                yield raptor_pb2.DownloadAssetResponse(
                    chunk=raptor_pb2.MediaChunk(
                        data=chunk,
                        offset=offset,
                        sequence=sequence,
                        crc32c=google_crc32c.value(chunk),
                        end_of_stream=is_final,
                    )
                )
                offset += len(chunk)
                sequence += 1
            # A storage failure mid-download used to end here as if the file had simply ended (the
            # reader's error was only ever swallowed by the cleanup below): surface it instead.
            await reader_task
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception) as e:  # noqa: BLE001 -- cleanup after cancel, discard whatever the cancelled task raised
                _logger.debug(f"Reader task teardown raised after cancel: {e}")


# ── UploadAsset ──────────────────────────────────────────────────────────

class _FileHandleAdapter:
    """Duck-typed stand-in for FastAPI's UploadFile -- StorageService.upload_file
    only touches .file/.filename/.content_type, so wrapping a plain open file
    handle lets UploadAsset reuse it without a real UploadFile instance."""

    def __init__(self, file_obj, filename: str, content_type: str):
        self.file = file_obj
        self.filename = filename
        self.content_type = content_type


async def _next_with_idle_timeout(request_iterator, idle_timeout_s: float):
    """Returns the next request message, or None at a clean end of stream;
    raises DeadlineExceededError if none arrives in time.

    Not `asyncio.wait_for`: on Python 3.10 it swallows a cancellation of the awaiting task when
    the inner awaitable finishes in the same moment, and grpc.aio finishes the request iterator
    (a plain end of stream) exactly when the client cancels. The handler then saw a clean end of
    stream, never the cancellation, and committed a half-sent upload. `asyncio.wait` does not
    swallow it."""
    next_message = asyncio.ensure_future(request_iterator.__anext__())
    try:
        done, _ = await asyncio.wait({next_message}, timeout=idle_timeout_s)
    except asyncio.CancelledError:
        next_message.cancel()
        raise
    if not done:
        next_message.cancel()
        raise DeadlineExceededError(f"No chunk received within {idle_timeout_s}s of the previous one.")
    try:
        return next_message.result()
    except StopAsyncIteration:
        return None


async def _publish_asset_finalized(
    kafka_producer,
    kafka_topic: str,
    correlation_id: str,
    asset_id: str,
    version_id: str,
    bytes_received: int,
    meta: Any,
) -> None:
    if kafka_producer is None:
        _logger.info("Kafka producer not configured -- skipping asset-upload-finalized event.")
        return
    message = {
        "message_id": str(uuid.uuid4()),
        "correlation_id": correlation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_service": "grpc-gateway",
        "message_type": "EVENT",
        "payload": {
            "asset_id": asset_id,
            "version_id": version_id,
            "bytes_received": bytes_received,
            "filename": meta.filename,
            "attributes": dict(meta.attributes),
        },
    }
    try:
        await kafka_producer.send_and_wait(kafka_topic, message)
    except Exception:
        _logger.warning("Failed to publish asset-upload-finalized event", exc_info=True)


async def _commit_upload(
    storage_service, redis_client, kafka_producer, kafka_topic, tmp_path, storage_user, user_id, meta,
    correlation_id, bytes_received,
) -> tuple[str, str]:
    """Hands the verified temp file to the asset service, then records the idempotency key and
    publishes the finalized event. Returns (asset_id, version_id)."""
    fh = await asyncio.to_thread(open, tmp_path, "rb")
    try:
        upload_result = await storage_service.upload_file(
            primary_file=_FileHandleAdapter(fh, meta.filename, meta.content_type),
            archive_date_or_ttl=None,
            destroy_date_or_ttl=None,
            user=storage_user,
        )
    finally:
        fh.close()

    asset_id = upload_result.get("asset_path", "")
    version_id = upload_result.get("version_id", "")

    if meta.idempotency_key:
        await redis_client.setex(
            f"idempotency:{user_id}:{meta.idempotency_key}",
            86400,
            json.dumps({"asset_id": asset_id, "version_id": version_id}),
        )
    await _publish_asset_finalized(kafka_producer, kafka_topic, correlation_id, asset_id, version_id, bytes_received, meta)
    return asset_id, version_id


async def _run_to_completion(coro):
    """Runs `coro` so that cancelling the caller does not cancel it: the caller waits for it to
    finish, then the cancellation is delivered. The result (or the coroutine's own error) wins
    when it finishes normally. Used for the commit step of an upload."""
    task = asyncio.ensure_future(coro)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                raise  # the commit itself was cancelled from outside; nothing to wait for
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


async def upload_asset(
    storage_service: StorageService,
    redis_client,
    kafka_producer,
    kafka_topic: str,
    request_iterator,
    user: dict[str, str],
    idle_timeout_s: float,
    context,
) -> raptor_pb2.UploadAssetResponse:
    # Per-user concurrent-stream limiting is StreamLimitInterceptor's job now
    # (shared across every streaming RPC, not just this one) -- see
    # interceptors/stream_limit.py.
    storage_user = StorageUser(user_id=user["user_id"], branch_id=user["branch_id"])

    first = await _next_with_idle_timeout(request_iterator, idle_timeout_s)
    if first is None or not first.HasField("metadata"):
        raise ValidationError("The first UploadAsset message must be metadata.")
    meta = first.metadata
    if not meta.filename:
        raise ValidationError("metadata.filename is required.")

    if meta.idempotency_key:
        existing = await redis_client.get(f"idempotency:{user['user_id']}:{meta.idempotency_key}")
        if existing:
            record = json.loads(existing)
            context.set_trailing_metadata(
                (("asset-id", record["asset_id"]), ("version-id", record["version_id"]))
            )
            raise AlreadyExistsError(f"Duplicate upload for idempotency_key={meta.idempotency_key}")

    correlation_id = str(uuid.uuid4())
    fd, tmp_path = tempfile.mkstemp(prefix="raptor-upload-")
    bytes_received = 0
    checksum = google_crc32c.Checksum()
    trailer_crc32c: int | None = None
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                msg = await _next_with_idle_timeout(request_iterator, idle_timeout_s)
                if msg is None:
                    break
                if trailer_crc32c is not None:
                    raise ValidationError("final_crc32c must be the last message.")
                kind = msg.WhichOneof("payload")
                if kind == "final_crc32c":
                    trailer_crc32c = msg.final_crc32c
                    continue
                if kind != "chunk_data":
                    raise ValidationError("Expected chunk_data after the metadata message.")
                f.write(msg.chunk_data)
                checksum.update(msg.chunk_data)
                bytes_received += len(msg.chunk_data)

        if meta.total_size and bytes_received != meta.total_size:
            raise ValidationError(
                f"Declared total_size={meta.total_size} but received {bytes_received} bytes."
            )

        # After total_size so a truncated upload reports the size problem, not a checksum one.
        # Still before the storage hand-off, so a corrupted upload never becomes an asset.
        computed = int.from_bytes(checksum.digest(), "big")
        for where, declared in (
            ("metadata.crc32c", meta.crc32c if meta.HasField("crc32c") else None),
            ("final_crc32c", trailer_crc32c),
        ):
            if declared is not None and declared != computed:
                raise DataLossError(
                    f"{where}={declared:#010x} does not match the received data "
                    f"(CRC32C {computed:#010x}, {bytes_received} bytes). Nothing was stored."
                )

        # From here the upload is complete and verified. Commit it even if the client goes away
        # (see _commit_upload): abandoning it half-way would leave an asset the client never
        # learns the id of, and a retry with the same idempotency key would create it again.
        asset_id, version_id = await _run_to_completion(
            _commit_upload(
                storage_service, redis_client, kafka_producer, kafka_topic,
                tmp_path, storage_user, user["user_id"], meta, correlation_id, bytes_received,
            )
        )
    finally:
        os.remove(tmp_path)

    return raptor_pb2.UploadAssetResponse(
        asset_id=asset_id,
        version_id=version_id,
        correlation_id=correlation_id,
        bytes_received=bytes_received,
    )
