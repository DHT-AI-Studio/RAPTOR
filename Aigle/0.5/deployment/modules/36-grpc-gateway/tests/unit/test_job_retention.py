"""How long a job stays reachable, and what GetJob/WatchJob say once the orchestrator's
own Redis key (24 h after its worker's last update) is gone.

Before: the job record was written with a 24 h TTL while the event log lives 7 days. Both
GetJob and WatchJob raise NOT_FOUND without the record, so the 7-day replay window was
really 24 hours. And a finished job whose orchestrator key had expired read UNSPECIFIED.
"""
from __future__ import annotations

import asyncio
import json

import grpc
import pytest
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import job_service

DAY = 86400


_USER = {"user_id": "u-test", "branch_id": "b-test"}


class _TtlRedis:
    """Dict-backed Redis that remembers the TTL each write asked for and can drop a key
    the way an expiry would."""

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        return self.kv.get(key)

    async def setex(self, key, ttl, value):
        self.kv[key] = value
        self.ttls[key] = ttl

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    async def expire(self, key, ttl):
        self.ttls[key] = ttl

    def expire_now(self, key):
        self.kv.pop(key, None)


def _record(redis, job_id, media_type="document"):
    redis.kv[job_service._job_key(job_id)] = json.dumps({
        "job_id": job_id, "state": raptor_pb2.JOB_STATE_QUEUED, "job_type": "", "media_type": media_type,
        "input_asset_ids": [], "request_id": "", "created_at": "2024-01-01T00:00:00+00:00", "detail": "",
    })


def _worker_writes(redis, media_type, job_id, **state):
    redis.kv[job_service._orchestrator_key(media_type, job_id)] = json.dumps(state)


def _orchestrator_key_expires(redis, media_type, job_id):
    redis.expire_now(job_service._orchestrator_key(media_type, job_id))


async def _get(redis, job_id):
    return await job_service.get_job(redis, raptor_pb2.GetJobRequest(job_id=job_id), user=_USER)


async def _watch_all(redis, job_id, last_sequence=0, timeout=5):
    async def run():
        request = raptor_pb2.WatchJobRequest(job_id=job_id, last_sequence=last_sequence)
        return [e async for e in job_service.watch_job(redis, request, poll_interval_s=0.01, user=_USER)]

    return await asyncio.wait_for(run(), timeout=timeout)


# ── the record must outlive the replay window it is the door to ──────────────


class _Storage:
    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        return {"metadata": {"asset_path": asset_path, "version_id": version_id,
                             "primary_filename": "a.mp4", "status": "success"}}


class _Producer:
    async def send_and_wait(self, topic, message):
        return None


@pytest.mark.asyncio
async def test_submit_job_keeps_the_record_as_long_as_the_event_log():
    redis = _TtlRedis()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=_Storage(), redis_client=redis, kafka_producer=_Producer())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        job = await stub.SubmitJob(
            raptor_pb2.SubmitJobRequest(
                input_asset_ids=[raptor_pb2.AssetRef(asset_path="assets/a.mp4", version_id="v1")], request_id="r-1"
            ),
            metadata=(("authorization", f"Bearer {make_token()}"),),
        )
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert redis.ttls[job_service._job_key(job.job_id)] == 7 * DAY == job_service._EVENT_LOG_TTL_S
    # The request-id idempotency pointer is deliberately unchanged: 24 h.
    pointers = [t for k, t in redis.ttls.items() if k.startswith("grpc:job_by_request:")]
    assert pointers == [DAY]


# ── a finished job is still reported after the orchestrator key is gone ──────


@pytest.mark.asyncio
async def test_get_job_still_reports_complete_after_the_orchestrator_key_expires():
    redis = _TtlRedis()
    _record(redis, "j1")
    _worker_writes(redis, "document", "j1", step="complete")

    live = await _get(redis, "j1")
    assert live.state == raptor_pb2.JOB_STATE_COMPLETE
    assert redis.ttls[job_service._result_key("j1")] == 7 * DAY

    _orchestrator_key_expires(redis, "document", "j1")
    later = await _get(redis, "j1")
    assert later.state == raptor_pb2.JOB_STATE_COMPLETE
    assert later.detail == live.detail


@pytest.mark.asyncio
async def test_get_job_still_reports_failed_with_its_error_after_the_orchestrator_key_expires():
    redis = _TtlRedis()
    _record(redis, "j2", media_type="audio")
    _worker_writes(redis, "audio", "j2", step="error", error_code="ASR_FAILED", error_message="boom")

    await _get(redis, "j2")
    _orchestrator_key_expires(redis, "audio", "j2")
    later = await _get(redis, "j2")

    assert later.state == raptor_pb2.JOB_STATE_FAILED
    assert json.loads(later.detail)["error_code"] == "ASR_FAILED"  # why it failed survives the expiry


@pytest.mark.asyncio
async def test_a_non_terminal_state_is_not_trusted_once_the_key_is_gone():
    """The last thing seen was INDEXING, then the key expired: the job may have died, so
    claiming it is still indexing would be a guess. It stays UNSPECIFIED ("lost track")."""
    redis = _TtlRedis()
    _record(redis, "j3")
    _worker_writes(redis, "document", "j3", step="indexing")
    assert (await _get(redis, "j3")).state == raptor_pb2.JOB_STATE_INDEXING

    _orchestrator_key_expires(redis, "document", "j3")
    assert (await _get(redis, "j3")).state == raptor_pb2.JOB_STATE_UNSPECIFIED
    assert job_service._result_key("j3") not in redis.kv


@pytest.mark.asyncio
async def test_a_job_nobody_looked_at_before_the_key_expired_reads_unspecified():
    # The documented limit: nothing saw the terminal state, so nothing could keep it.
    redis = _TtlRedis()
    _record(redis, "j4")
    _worker_writes(redis, "document", "j4", step="complete")
    _orchestrator_key_expires(redis, "document", "j4")

    assert (await _get(redis, "j4")).state == raptor_pb2.JOB_STATE_UNSPECIFIED


@pytest.mark.asyncio
async def test_a_corrupt_stored_result_is_ignored_not_fatal():
    redis = _TtlRedis()
    _record(redis, "j5")
    redis.kv[job_service._result_key("j5")] = "not json"
    assert (await _get(redis, "j5")).state == raptor_pb2.JOB_STATE_UNSPECIFIED


# ── WatchJob ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_watch_job_stores_the_final_state_and_can_replay_after_the_orchestrator_key_expires():
    redis = _TtlRedis()
    _record(redis, "j6")
    _worker_writes(redis, "document", "j6", step="complete")

    first = await _watch_all(redis, "j6")
    assert [e.state for e in first] == [raptor_pb2.JOB_STATE_COMPLETE]
    assert job_service._result_key("j6") in redis.kv

    _orchestrator_key_expires(redis, "document", "j6")
    again = await _watch_all(redis, "j6")  # a reconnecting client, a day or more later
    assert [(e.sequence, e.state) for e in again] == [(1, raptor_pb2.JOB_STATE_COMPLETE)]
    assert (await _get(redis, "j6")).state == raptor_pb2.JOB_STATE_COMPLETE


# ── what happens when the 7 days are up ──────────────────────────────────────


def _seven_days_pass(redis, job_id):
    """Every key the gateway keeps for a job carries the 7-day TTL, so after 7 days Redis
    has dropped all of them. Fails if a key was written with a shorter or no TTL."""
    keys = [job_service._job_key(job_id), job_service._result_key(job_id), job_service._events_key(job_id)]
    for key in keys:
        assert redis.ttls[key] == 7 * DAY, f"{key} would not live the full retention period"
        redis.kv.pop(key, None)
        redis.lists.pop(key, None)


@pytest.mark.asyncio
async def test_every_key_a_finished_job_leaves_behind_lives_exactly_seven_days():
    redis = _TtlRedis()
    _record(redis, "j7")
    redis.ttls[job_service._job_key("j7")] = job_service._JOB_RECORD_TTL_S
    _worker_writes(redis, "document", "j7", step="complete")
    await _watch_all(redis, "j7")

    for key in (job_service._job_key("j7"), job_service._result_key("j7"), job_service._events_key("j7")):
        assert redis.ttls[key] == 7 * DAY


@pytest.mark.asyncio
async def test_after_the_retention_period_get_job_and_watch_job_answer_not_found():
    from interceptors.errors import NotFoundError

    redis = _TtlRedis()
    _record(redis, "j8")
    redis.ttls[job_service._job_key("j8")] = job_service._JOB_RECORD_TTL_S
    _worker_writes(redis, "document", "j8", step="complete")
    await _watch_all(redis, "j8")
    assert (await _get(redis, "j8")).state == raptor_pb2.JOB_STATE_COMPLETE

    _seven_days_pass(redis, "j8")
    _orchestrator_key_expires(redis, "document", "j8")

    with pytest.raises(NotFoundError):
        await _get(redis, "j8")
    with pytest.raises(NotFoundError):
        await _watch_all(redis, "j8")  # a replay after the window is refused, not an empty stream

