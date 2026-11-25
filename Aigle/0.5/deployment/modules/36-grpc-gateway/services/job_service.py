"""SubmitJob / GetJob / WatchJob RPC handlers — async job lifecycle.

SubmitJob routes to the same Kafka-driven pipeline as REST's
`/processing/process-file` (`KafkaService.send_processing_request`) and
records its own Job entity in Redis (job_id == correlation_id) so GetJob/
WatchJob don't need the caller to repeat submission-time details like
media_type.

WatchJob's "stream" is built on top of the single mutable `step` field the
processing workers already write to `{media_type}_orchestrator:{job_id}` —
there is no append-only event log anywhere upstream. grpc-gateway polls that
key, and the first time it observes each new step value, appends a JobEvent
to its own Redis list (`grpc:job_events:{job_id}`, TTL 7 days) with the next
sequence number, then yields it. A (re)connecting client replays everything
already in that list past its last_sequence before the live poll resumes,
so reconnect never re-observes — or misses — a state transition, even
across a full process restart of grpc-gateway itself (the log lives in
Redis, not memory).

JobSession layers a bidirectional control channel on top of watch_job(): the
subscribe direction (first message, then live relay) is a thin wrapper
around watch_job() itself — real, not a stub. The cancel/control directions
are intent-only: no processing worker (modules 09-12) exposes any cancel or
control API today (see JOB_STATE_CANCELLED's own proto comment — it's a
reserved, currently-unreachable value for exactly this reason), so a
cancel=true or control message is recorded (Redis key
`grpc:job_cancel_intent:{job_id}` for cancel; nothing persisted for
control, since there's nowhere for it to take effect) and acknowledged with
an ephemeral JobEvent (sequence=-1, never written to the durable event log)
rather than silently no-op'd or, worse, reported as if the job actually
stopped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "generated"))
from app.services.kafka_service import KafkaService
from app.services.storage_service import StorageService
from app.services.storage_service import User as StorageUser
from interceptors.errors import NotFoundError, ValidationError
from raptor.v1 import raptor_pb2

_logger = logging.getLogger("grpc_gateway.job_service")

_JOB_TTL_S = 86400          # request-id idempotency key and cancel intent; matches the orchestrator key's TTL
_EVENT_LOG_TTL_S = 7 * 86400  # 7 days, per this ticket's acceptance criteria
# The job record is the way in to GetJob *and* to WatchJob's replay (both raise
# NOT_FOUND without it), so it has to live as long as the event log. At 24 h it cut
# the 7-day replay window down to a day.
_JOB_RECORD_TTL_S = _EVENT_LOG_TTL_S
# The final state, kept so a finished job is still reported once the orchestrator's
# own key (24 h after its last update) is gone.
_RESULT_TTL_S = _EVENT_LOG_TTL_S
_WATCH_POLL_INTERVAL_S = 1.0

_TERMINAL_STATES = {
    raptor_pb2.JOB_STATE_COMPLETE,
    raptor_pb2.JOB_STATE_FAILED,
    raptor_pb2.JOB_STATE_CANCELLED,
}

_STATE_BY_STEP = {
    "queued": raptor_pb2.JOB_STATE_QUEUED,
    # Intermediate steps the media orchestrators (modules 09-12) write between
    # "queued" and "indexing". Media-specific, so they all read as RUNNING; the
    # raw step stays in JobEvent.detail. docs/_internal/grpc-job-domain.md has the table.
    "parallel_processing": raptor_pb2.JOB_STATE_RUNNING,  # audio, image
    "parallel_analysis": raptor_pb2.JOB_STATE_RUNNING,    # video
    "analysis": raptor_pb2.JOB_STATE_RUNNING,             # document
    "chunking": raptor_pb2.JOB_STATE_RUNNING,             # video
    "contextualize": raptor_pb2.JOB_STATE_RUNNING,        # video
    "merge": raptor_pb2.JOB_STATE_RUNNING,                # image
    "summary": raptor_pb2.JOB_STATE_RUNNING,              # audio, video, document
    "save": raptor_pb2.JOB_STATE_RUNNING,                 # all four
    "indexing": raptor_pb2.JOB_STATE_INDEXING,
    "complete": raptor_pb2.JOB_STATE_COMPLETE,
    # Every orchestrator writes "error" (plus error_code / error_message) on
    # failure. Without it a failed job read as UNSPECIFIED and WatchJob never ended.
    "error": raptor_pb2.JOB_STATE_FAILED,
    # Not written by any worker today; kept for forward compatibility.
    "transcribing": raptor_pb2.JOB_STATE_TRANSCRIBING,
    "extracting": raptor_pb2.JOB_STATE_EXTRACTING,
    "failed": raptor_pb2.JOB_STATE_FAILED,
    "cancelled": raptor_pb2.JOB_STATE_CANCELLED,
}


def _job_key(job_id: str) -> str:
    return f"grpc:job:{job_id}"


def _request_key(user_id: str, request_id: str) -> str:
    return f"grpc:job_by_request:{user_id}:{request_id}"


def _events_key(job_id: str) -> str:
    return f"grpc:job_events:{job_id}"


def _result_key(job_id: str) -> str:
    return f"grpc:job_result:{job_id}"


def _cancel_intent_key(job_id: str) -> str:
    return f"grpc:job_cancel_intent:{job_id}"


def _orchestrator_key(media_type: str, job_id: str) -> str:
    return f"{media_type}_orchestrator:{job_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_to_proto(record: dict[str, Any]) -> raptor_pb2.Job:
    return raptor_pb2.Job(
        job_id=record["job_id"],
        state=record.get("state", raptor_pb2.JOB_STATE_UNSPECIFIED),
        job_type=record.get("job_type", ""),
        media_type=record.get("media_type", ""),
        input_asset_ids=[
            raptor_pb2.AssetRef(asset_path=a["asset_path"], version_id=a["version_id"])
            for a in record.get("input_asset_ids", [])
        ],
        request_id=record.get("request_id", ""),
        created_at=record.get("created_at", ""),
        detail=record.get("detail", ""),
    )


async def _stored_result(redis_client, job_id: str) -> tuple[int, str]:
    raw = await redis_client.get(_result_key(job_id))
    if raw is None:
        return raptor_pb2.JOB_STATE_UNSPECIFIED, ""
    try:
        stored = json.loads(raw)
        return int(stored["state"]), str(stored.get("detail", ""))
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return raptor_pb2.JOB_STATE_UNSPECIFIED, ""


async def _current_state(redis_client, media_type: str, job_id: str) -> tuple[int, str]:
    """Reads the orchestrator's live step field.

    The orchestrator key lives 24 h after its worker's last update, so a finished
    job outlives it. Whenever a terminal state is seen it is also stored under
    `grpc:job_result:{job_id}` (7 days); a missing orchestrator key then falls back to
    that. Only a terminal state is trusted this way: if the key is gone and no final
    state was ever seen, the answer stays UNSPECIFIED -- not QUEUED, and not the last
    non-terminal state, either of which would claim progress nobody has confirmed
    ("we lost track of it", mirroring GetJobStatus's own fallback).

    Limit: a job nobody asked about between finishing and 24 h later has no stored
    result, so it reads UNSPECIFIED. Closing that needs the workers to keep their key
    longer (docs/_internal/grpc-job-domain.md, story 2b)."""
    raw = await redis_client.get(_orchestrator_key(media_type, job_id))
    if raw is None:
        return await _stored_result(redis_client, job_id)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raptor_pb2.JOB_STATE_UNSPECIFIED, str(raw)
    step = parsed.get("step", "")
    state = _STATE_BY_STEP.get(step, raptor_pb2.JOB_STATE_UNSPECIFIED)
    detail = json.dumps(parsed) if parsed else ""
    if state in _TERMINAL_STATES:
        await redis_client.setex(_result_key(job_id), _RESULT_TTL_S, json.dumps({"state": state, "detail": detail}))
    return state, detail


async def _load_job_record(redis_client, job_id: str) -> dict[str, Any] | None:
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def submit_job(
    kafka_service: KafkaService,
    storage_service: StorageService,
    redis_client,
    request: raptor_pb2.SubmitJobRequest,
    user: dict[str, str],
) -> raptor_pb2.Job:
    if len(request.input_asset_ids) != 1:
        raise ValidationError(
            f"SubmitJob supports exactly one input asset per job today, got {len(request.input_asset_ids)}."
        )
    asset_ref = request.input_asset_ids[0]
    if not asset_ref.asset_path or not asset_ref.version_id:
        raise ValidationError("input_asset_ids[0] must have both asset_path and version_id set.")

    user_id = user["user_id"]

    if request.request_id:
        existing_job_id = await redis_client.get(_request_key(user_id, request.request_id))
        if existing_job_id:
            existing = await _load_job_record(redis_client, existing_job_id)
            if existing is not None:
                return _job_to_proto(existing)
            # The idempotency pointer outlived the job record itself (TTL
            # drift) -- fall through and submit fresh rather than erroring.

    storage_user = StorageUser(user_id=user_id, branch_id=user["branch_id"])
    metadata_resp = await storage_service.download_asset(
        asset_ref.asset_path, asset_ref.version_id, storage_user, return_file_content=False
    )
    meta = metadata_resp.get("metadata", {}) if isinstance(metadata_resp, dict) else {}
    if not meta.get("primary_filename"):
        raise NotFoundError(f"Asset not found: {asset_ref.asset_path}/{asset_ref.version_id}")

    upload_result = {
        "asset_path": meta.get("asset_path", asset_ref.asset_path),
        "version_id": meta.get("version_id", asset_ref.version_id),
        "primary_filename": meta.get("primary_filename"),
        "status": meta.get("status", "success"),
    }

    send_result = await kafka_service.send_processing_request(
        upload_result=upload_result,
        user_id=user_id,
        branch_id=user["branch_id"],
        processing_mode=request.job_type or None,
    )
    job_id = send_result["correlation_id"]
    media_type = kafka_service.get_file_type(meta.get("primary_filename", ""))

    record: dict[str, Any] = {
        "job_id": job_id,
        "state": raptor_pb2.JOB_STATE_QUEUED,
        "job_type": request.job_type,
        "media_type": media_type,
        "input_asset_ids": [
            {"asset_path": upload_result["asset_path"], "version_id": upload_result["version_id"]}
        ],
        "request_id": request.request_id,
        "user_id": user_id,
        "created_at": _now_iso(),
        "detail": "",
    }
    await redis_client.setex(_job_key(job_id), _JOB_RECORD_TTL_S, json.dumps(record))
    if request.request_id:
        await redis_client.setex(_request_key(user_id, request.request_id), _JOB_TTL_S, job_id)

    return _job_to_proto(record)


async def _load_owned_record(redis_client, job_id: str, user: dict[str, str]) -> dict[str, Any] | None:
    """The job's record, or None when there is none *or it belongs to someone else* -- the
    caller answers NOT_FOUND either way, so a job id does not reveal that another account's
    job exists. Records written before the owner was stored have no `user_id` and stay
    readable (they expire within 7 days)."""
    record = await _load_job_record(redis_client, job_id)
    if record is None or record.get("user_id", user["user_id"]) != user["user_id"]:
        return None
    return record


async def get_job(redis_client, request: raptor_pb2.GetJobRequest, *, user: dict[str, str]) -> raptor_pb2.Job:
    record = await _load_owned_record(redis_client, request.job_id, user)
    if record is None:
        raise NotFoundError(f"Job not found: {request.job_id}")
    state, detail = await _current_state(redis_client, record["media_type"], record["job_id"])
    return _job_to_proto({**record, "state": state, "detail": detail})


async def watch_job(
    redis_client,
    request: raptor_pb2.WatchJobRequest,
    poll_interval_s: float = _WATCH_POLL_INTERVAL_S,
    *,
    user: dict[str, str],
) -> AsyncIterator[raptor_pb2.JobEvent]:
    job_id = request.job_id
    record = await _load_owned_record(redis_client, job_id, user)
    if record is None:
        raise NotFoundError(f"Job not found: {job_id}")
    media_type = record["media_type"]
    events_key = _events_key(job_id)

    last_sequence = request.last_sequence
    last_state: int | None = None

    stored: list[bytes] = await redis_client.lrange(events_key, 0, -1)
    for raw in stored:
        event = json.loads(raw)
        last_state = event["state"]
        if event["sequence"] > last_sequence:
            last_sequence = event["sequence"]
            yield raptor_pb2.JobEvent(
                job_id=job_id,
                sequence=event["sequence"],
                state=event["state"],
                detail=event.get("detail", ""),
                timestamp=event.get("timestamp", ""),
            )
    if last_state in _TERMINAL_STATES:
        return

    while True:
        state, detail = await _current_state(redis_client, media_type, job_id)
        if state != last_state:
            last_state = state
            last_sequence += 1
            event = {
                "job_id": job_id,
                "sequence": last_sequence,
                "state": state,
                "detail": detail,
                "timestamp": _now_iso(),
            }
            await redis_client.rpush(events_key, json.dumps(event))
            await redis_client.expire(events_key, _EVENT_LOG_TTL_S)
            yield raptor_pb2.JobEvent(**event)
            if state in _TERMINAL_STATES:
                return
        await asyncio.sleep(poll_interval_s)


def _intent_event(job_id: str, detail: str) -> raptor_pb2.JobEvent:
    """An ephemeral (sequence=-1, unpersisted) acknowledgment of a cancel/
    control message -- see job_session()'s module docstring for why this
    isn't a real state transition."""
    return raptor_pb2.JobEvent(
        job_id=job_id, sequence=-1, state=raptor_pb2.JOB_STATE_UNSPECIFIED, detail=detail, timestamp=_now_iso()
    )


async def job_session(
    redis_client,
    request_iterator,
    poll_interval_s: float = _WATCH_POLL_INTERVAL_S,
    *,
    user: dict[str, str],
) -> AsyncIterator[raptor_pb2.JobSessionResponse]:
    try:
        first = await request_iterator.__anext__()
    except StopAsyncIteration:
        raise ValidationError("JobSession requires at least one message (subscribe=true).")
    if first.WhichOneof("action") != "subscribe" or not first.subscribe:
        raise ValidationError("The first JobSession message must set subscribe=true.")
    job_id = first.job_id
    if not job_id:
        raise ValidationError("job_id is required.")
    if await _load_owned_record(redis_client, job_id, user) is None:
        raise NotFoundError(f"Job not found: {job_id}")

    control_queue: asyncio.Queue[raptor_pb2.JobEvent] = asyncio.Queue()

    async def _consume_control() -> None:
        async for msg in request_iterator:
            action = msg.WhichOneof("action")
            if action == "cancel" and msg.cancel:
                await redis_client.setex(_cancel_intent_key(job_id), _JOB_TTL_S, "1")
                await control_queue.put(_intent_event(
                    job_id, "Cancellation requested -- recorded as intent only; no processing "
                    "worker actually stops a running job today."
                ))
            elif action == "control":
                await control_queue.put(_intent_event(
                    job_id, f"Control command '{msg.control.command}' received -- forwarded as "
                    "intent only; no processing worker exposes a control API today."
                ))

    consumer_task = asyncio.create_task(_consume_control())
    watch_gen = watch_job(redis_client, raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=first.last_sequence), poll_interval_s, user=user)
    watch_next: asyncio.Task | None = asyncio.create_task(watch_gen.__anext__())
    control_next: asyncio.Task | None = asyncio.create_task(control_queue.get())
    try:
        while watch_next is not None:
            done, _ = await asyncio.wait({t for t in (watch_next, control_next) if t}, return_when=asyncio.FIRST_COMPLETED)

            if control_next is not None and control_next in done:
                yield raptor_pb2.JobSessionResponse(job_event=control_next.result())
                control_next = asyncio.create_task(control_queue.get())

            if watch_next in done:
                try:
                    event = watch_next.result()
                except StopAsyncIteration:
                    watch_next = None
                    continue
                yield raptor_pb2.JobSessionResponse(job_event=event)
                watch_next = asyncio.create_task(watch_gen.__anext__())
    finally:
        consumer_task.cancel()
        if control_next is not None:
            control_next.cancel()
        if watch_next is not None:
            # Must await the cancelled __anext__() task before aclose() --
            # otherwise watch_gen's frame is still considered "running" (the
            # cancellation hasn't been delivered into it yet) and aclose()
            # raises RuntimeError("aclose(): asynchronous generator is
            # already running").
            watch_next.cancel()
            try:
                await watch_next
            except (asyncio.CancelledError, StopAsyncIteration, Exception) as e:  # noqa: BLE001 -- cleanup after cancel, discard whatever the cancelled task raised
                _logger.debug(f"watch_next teardown raised after cancel: {e}")
        await watch_gen.aclose()
        for task in (consumer_task, control_next):
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception) as e:  # noqa: BLE001 -- cleanup after cancel, discard whatever the cancelled task raised
                _logger.debug(f"Task teardown raised after cancel: {e}")
