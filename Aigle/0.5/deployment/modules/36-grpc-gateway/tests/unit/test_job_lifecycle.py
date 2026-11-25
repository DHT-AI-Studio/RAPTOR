"""Tests for SubmitJob / GetJob / WatchJob: submission -> Kafka + Redis job
record, idempotency, GetJob's current-state read, and WatchJob's
poll-to-event-log bridge (sequence numbers, terminal-state auto-close,
reconnect replay). Same self-contained fake-double pattern as the other
test_*.py files here.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from interceptors.errors import NotFoundError
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import job_service

# ── fakes ──────────────────────────────────────────────────────────────────

_USER = {"user_id": "u-test", "branch_id": "b-test"}


class _FakeRedis:
    """Dict/list-backed stand-in supporting the ops job_service needs:
    get/setex (job record + idempotency pointer) and rpush/lrange/expire
    (event log)."""

    def __init__(self):
        self._kv = {}
        self._lists = {}

    async def get(self, key):
        return self._kv.get(key)

    async def setex(self, key, ttl, value):
        self._kv[key] = value

    async def rpush(self, key, value):
        self._lists.setdefault(key, []).append(value)

    async def lrange(self, key, start, end):
        items = self._lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    async def expire(self, key, ttl):
        pass


class _FakeStorageService:
    def __init__(self, metadata=None):
        self._metadata = (
            metadata
            if metadata is not None
            else {"asset_path": "assets/a.mp4", "version_id": "v1", "primary_filename": "a.mp4", "status": "success"}
        )

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        return {"metadata": self._metadata}


class _FakeKafkaProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, message):
        self.sent.append((topic, message))


def _asset_ref(asset_path="assets/a.mp4", version_id="v1"):
    return raptor_pb2.AssetRef(asset_path=asset_path, version_id=version_id)


async def _run_submit(fake_storage, fake_redis, fake_kafka, request):
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=fake_storage, redis_client=fake_redis, kafka_producer=fake_kafka)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        return await stub.SubmitJob(request, metadata=(("authorization", f"Bearer {make_token()}"),))
    finally:
        await channel.close()
        await server.stop(grace=None)


async def _run_get_job(fake_redis, job_id):
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), redis_client=fake_redis)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        return await stub.GetJob(raptor_pb2.GetJobRequest(job_id=job_id), metadata=(("authorization", f"Bearer {make_token()}"),))
    finally:
        await channel.close()
        await server.stop(grace=None)


def _write_job_record(fake_redis, job_id, media_type="video", **overrides):
    record = {
        "job_id": job_id, "state": raptor_pb2.JOB_STATE_QUEUED, "job_type": "", "media_type": media_type,
        "input_asset_ids": [], "request_id": "", "created_at": "2024-01-01T00:00:00+00:00", "detail": "",
    }
    record.update(overrides)
    fake_redis._kv[job_service._job_key(job_id)] = json.dumps(record)


def _set_orchestrator_step(fake_redis, media_type, job_id, step):
    fake_redis._kv[job_service._orchestrator_key(media_type, job_id)] = json.dumps({"step": step})


# ── SubmitJob ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_job_creates_job_queues_kafka_and_writes_record():
    fake_storage = _FakeStorageService()
    fake_redis = _FakeRedis()
    fake_kafka = _FakeKafkaProducer()

    request = raptor_pb2.SubmitJobRequest(job_type="fast", input_asset_ids=[_asset_ref()], request_id="req-1")
    job = await _run_submit(fake_storage, fake_redis, fake_kafka, request)

    assert job.state == raptor_pb2.JOB_STATE_QUEUED
    assert job.job_id  # == the Kafka message's correlation_id
    assert job.media_type == "video"
    assert job.request_id == "req-1"
    assert len(fake_kafka.sent) == 1
    topic, message = fake_kafka.sent[0]
    assert topic == "video-processing-requests"
    assert message["correlation_id"] == job.job_id

    stored = json.loads(fake_redis._kv[job_service._job_key(job.job_id)])
    assert stored["media_type"] == "video"
    assert stored["input_asset_ids"] == [{"asset_path": "assets/a.mp4", "version_id": "v1"}]


@pytest.mark.asyncio
async def test_submit_job_requires_exactly_one_input_asset():
    fake_storage = _FakeStorageService()
    request = raptor_pb2.SubmitJobRequest(input_asset_ids=[])
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_submit(fake_storage, _FakeRedis(), _FakeKafkaProducer(), request)
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_submit_job_unknown_asset_is_not_found():
    fake_storage = _FakeStorageService(metadata={})  # no primary_filename -> asset doesn't exist
    request = raptor_pb2.SubmitJobRequest(input_asset_ids=[_asset_ref()])
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_submit(fake_storage, _FakeRedis(), _FakeKafkaProducer(), request)
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_submit_job_idempotent_duplicate_request_id_returns_same_job_no_resend():
    fake_storage = _FakeStorageService()
    fake_redis = _FakeRedis()
    fake_kafka = _FakeKafkaProducer()
    request = raptor_pb2.SubmitJobRequest(input_asset_ids=[_asset_ref()], request_id="dup-key")

    first = await _run_submit(fake_storage, fake_redis, fake_kafka, request)
    second = await _run_submit(fake_storage, fake_redis, fake_kafka, request)

    assert second.job_id == first.job_id
    assert len(fake_kafka.sent) == 1  # not resent


# ── GetJob ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_returns_current_orchestrator_state():
    fake_redis = _FakeRedis()
    _write_job_record(fake_redis, "job-1", media_type="audio")
    _set_orchestrator_step(fake_redis, "audio", "job-1", "extracting")

    job = await _run_get_job(fake_redis, "job-1")

    assert job.job_id == "job-1"
    assert job.state == raptor_pb2.JOB_STATE_EXTRACTING


@pytest.mark.asyncio
async def test_get_job_unknown_job_is_not_found():
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_get_job(_FakeRedis(), "does-not-exist")
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


# ── WatchJob ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_watch_job_unknown_job_is_not_found():
    fake_redis = _FakeRedis()
    with pytest.raises(NotFoundError):
        async for _ in job_service.watch_job(
            fake_redis, raptor_pb2.WatchJobRequest(job_id="nope"), poll_interval_s=0.01, user=_USER):
            pass


@pytest.mark.asyncio
async def test_watch_job_streams_transitions_and_closes_on_complete():
    fake_redis = _FakeRedis()
    job_id = "job-complete"
    _write_job_record(fake_redis, job_id, media_type="video")
    _set_orchestrator_step(fake_redis, "video", job_id, "queued")

    async def driver():
        events = []
        request = raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=0)
        async for event in job_service.watch_job(fake_redis, request, poll_interval_s=0.01, user=_USER):
            events.append(event)
            if len(events) == 1:
                _set_orchestrator_step(fake_redis, "video", job_id, "transcribing")
            elif len(events) == 2:
                _set_orchestrator_step(fake_redis, "video", job_id, "complete")
        return events

    events = await asyncio.wait_for(driver(), timeout=5)
    assert [e.sequence for e in events] == [1, 2, 3]
    assert [e.state for e in events] == [
        raptor_pb2.JOB_STATE_QUEUED, raptor_pb2.JOB_STATE_TRANSCRIBING, raptor_pb2.JOB_STATE_COMPLETE,
    ]


@pytest.mark.asyncio
async def test_watch_job_closes_after_failed():
    fake_redis = _FakeRedis()
    job_id = "job-failed"
    _write_job_record(fake_redis, job_id, media_type="document")
    _set_orchestrator_step(fake_redis, "document", job_id, "queued")

    async def driver():
        events = []
        request = raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=0)
        async for event in job_service.watch_job(fake_redis, request, poll_interval_s=0.01, user=_USER):
            events.append(event)
            if len(events) == 1:
                _set_orchestrator_step(fake_redis, "document", job_id, "failed")
        return events

    events = await asyncio.wait_for(driver(), timeout=5)
    assert events[-1].state == raptor_pb2.JOB_STATE_FAILED


# What modules 09-12 really write to {media}_orchestrator:{job_id} (see
# docs/_internal/grpc-job-domain.md): failure is step "error" with error_code /
# error_message, success is "complete".
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media_type,final_value,expected_state",
    [
        ("audio", {"step": "error", "error_code": "ASR_FAILED", "error_message": "boom"}, raptor_pb2.JOB_STATE_FAILED),
        ("image", {"step": "error", "error_code": "MESSAGE_EXPIRED", "error_message": "x"}, raptor_pb2.JOB_STATE_FAILED),
        ("video", {"step": "error", "error_code": "E", "error_message": "x"}, raptor_pb2.JOB_STATE_FAILED),
        ("document", {"step": "error", "error_code": "E", "error_message": "x"}, raptor_pb2.JOB_STATE_FAILED),
        ("document", {"step": "complete"}, raptor_pb2.JOB_STATE_COMPLETE),
    ],
)
async def test_watch_job_ends_on_the_terminal_values_workers_actually_write(media_type, final_value, expected_state):
    fake_redis = _FakeRedis()
    job_id = f"job-real-{media_type}"
    _write_job_record(fake_redis, job_id, media_type=media_type)
    _set_orchestrator_step(fake_redis, media_type, job_id, "queued")

    async def driver():
        events = []
        request = raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=0)
        async for event in job_service.watch_job(fake_redis, request, poll_interval_s=0.01, user=_USER):
            events.append(event)
            if len(events) == 1:
                fake_redis._kv[job_service._orchestrator_key(media_type, job_id)] = json.dumps(final_value)
        return events

    events = await asyncio.wait_for(driver(), timeout=5)  # a stream that never ends fails here
    assert events[-1].state == expected_state
    if expected_state == raptor_pb2.JOB_STATE_FAILED:
        assert json.loads(events[-1].detail)["error_code"] == final_value["error_code"]  # why it failed is not lost


# Intermediate steps written by the orchestrators; all read as RUNNING, raw step in detail.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media_type,step",
    [
        ("audio", "parallel_processing"), ("audio", "summary"), ("audio", "save"),
        ("image", "parallel_processing"), ("image", "merge"), ("image", "save"),
        ("video", "parallel_analysis"), ("video", "chunking"), ("video", "contextualize"),
        ("video", "summary"), ("video", "save"),
        ("document", "analysis"), ("document", "summary"), ("document", "save"),
    ],
)
async def test_get_job_reports_running_for_intermediate_worker_steps(media_type, step):
    fake_redis = _FakeRedis()
    _write_job_record(fake_redis, "job-run", media_type=media_type)
    _set_orchestrator_step(fake_redis, media_type, "job-run", step)

    job = await job_service.get_job(fake_redis, raptor_pb2.GetJobRequest(job_id="job-run"), user=_USER)

    assert job.state == raptor_pb2.JOB_STATE_RUNNING
    assert json.loads(job.detail)["step"] == step  # the media-specific step is not lost


@pytest.mark.asyncio
async def test_watch_job_reports_queued_running_indexing_complete_for_a_real_step_sequence():
    """A document job's real step order. Consecutive intermediate steps are ONE state
    (RUNNING), so a client sees four events, not one per step, and the state never goes
    back to UNSPECIFIED between them."""
    fake_redis = _FakeRedis()
    _write_job_record(fake_redis, "job-seq", media_type="document")
    sequence = ["queued", "analysis", "summary", "save", "indexing", "complete"]
    _set_orchestrator_step(fake_redis, "document", "job-seq", sequence[0])

    async def advance_steps():
        # A worker moving through its steps while WatchJob polls (every 10 ms here).
        for step in sequence[1:]:
            await asyncio.sleep(0.05)
            _set_orchestrator_step(fake_redis, "document", "job-seq", step)

    async def driver():
        advancing = asyncio.create_task(advance_steps())
        events = []
        request = raptor_pb2.WatchJobRequest(job_id="job-seq", last_sequence=0)
        async for event in job_service.watch_job(fake_redis, request, poll_interval_s=0.01, user=_USER):
            events.append(event)
        await advancing
        return events

    events = await asyncio.wait_for(driver(), timeout=10)

    assert [e.state for e in events] == [
        raptor_pb2.JOB_STATE_QUEUED,
        raptor_pb2.JOB_STATE_RUNNING,
        raptor_pb2.JOB_STATE_INDEXING,
        raptor_pb2.JOB_STATE_COMPLETE,
    ]
    assert [e.sequence for e in events] == [1, 2, 3, 4]
    assert raptor_pb2.JOB_STATE_UNSPECIFIED not in {e.state for e in events}


@pytest.mark.asyncio
async def test_get_job_reports_failed_for_worker_error_step():
    fake_redis = _FakeRedis()
    _write_job_record(fake_redis, "job-err", media_type="audio")
    fake_redis._kv[job_service._orchestrator_key("audio", "job-err")] = json.dumps(
        {"step": "error", "error_code": "ASR_FAILED", "error_message": "boom"}
    )
    job = await job_service.get_job(fake_redis, raptor_pb2.GetJobRequest(job_id="job-err"), user=_USER)
    assert job.state == raptor_pb2.JOB_STATE_FAILED


@pytest.mark.asyncio
async def test_watch_job_reconnect_replays_events_after_last_sequence():
    fake_redis = _FakeRedis()
    job_id = "job-reconnect"
    _write_job_record(fake_redis, job_id, media_type="video")
    _set_orchestrator_step(fake_redis, "video", job_id, "queued")

    # First watcher observes 2 events then disconnects (never reaches a terminal state).
    first_events = []
    gen1 = job_service.watch_job(fake_redis, raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=0), poll_interval_s=0.01, user=_USER)
    async for event in gen1:
        first_events.append(event)
        if len(first_events) == 1:
            _set_orchestrator_step(fake_redis, "video", job_id, "transcribing")
        if len(first_events) == 2:
            await gen1.aclose()
            break
    assert [e.sequence for e in first_events] == [1, 2]

    # Reconnect from sequence 1 -> event #2 (already observed by the first
    # watcher) replays from the Redis-backed log without needing a new poll.
    gen2 = job_service.watch_job(fake_redis, raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=1), poll_interval_s=0.01, user=_USER)
    replayed = await gen2.__anext__()
    assert replayed.sequence == 2
    assert replayed.state == raptor_pb2.JOB_STATE_TRANSCRIBING

    _set_orchestrator_step(fake_redis, "video", job_id, "complete")
    remaining = [e async for e in gen2]
    assert [e.sequence for e in remaining] == [3]
    assert remaining[0].state == raptor_pb2.JOB_STATE_COMPLETE
