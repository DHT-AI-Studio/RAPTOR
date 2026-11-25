"""Tests for JobSession: the subscribe direction is a real wrapper around
watch_job() (proven by reusing the same event-sequence assertions as
test_job_lifecycle.py's WatchJob tests); the cancel/control directions are
proven to be intent-only -- acknowledged with an ephemeral (sequence=-1)
event, recorded to Redis, but never changing the job's own real state.
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
from interceptors.errors import NotFoundError, ValidationError
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import job_service

_USER = {"user_id": "u-test", "branch_id": "b-test"}


class _FakeRedis:
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


class _RequestFeeder:
    """An async iterator a test can push JobSessionRequest messages into on
    demand -- mirrors how grpc.aio's real request_iterator behaves (usable
    both via `.__anext__()` directly and `async for`), so job_service.job_session
    can be driven exactly like it would be by a real client."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    def send(self, msg) -> None:
        self._queue.put_nowait(msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()


async def _empty_iterator():
    return
    yield  # pragma: no cover - never reached, keeps this an async generator


def _write_job_record(fake_redis, job_id, media_type="video", **overrides):
    record = {
        "job_id": job_id, "state": raptor_pb2.JOB_STATE_QUEUED, "job_type": "", "media_type": media_type,
        "input_asset_ids": [], "request_id": "", "created_at": "2024-01-01T00:00:00+00:00", "detail": "",
    }
    record.update(overrides)
    fake_redis._kv[job_service._job_key(job_id)] = json.dumps(record)


def _set_orchestrator_step(fake_redis, media_type, job_id, step):
    fake_redis._kv[job_service._orchestrator_key(media_type, job_id)] = json.dumps({"step": step})


# ── service-level: fine-grained event interleaving ──────────────────────────

@pytest.mark.asyncio
async def test_job_session_requires_at_least_one_message():
    with pytest.raises(ValidationError):
        async for _ in job_service.job_session(_FakeRedis(), _empty_iterator(), poll_interval_s=0.01, user=_USER):
            pass


@pytest.mark.asyncio
async def test_job_session_first_message_must_subscribe():
    feeder = _RequestFeeder()
    feeder.send(raptor_pb2.JobSessionRequest(job_id="x", cancel=True))
    with pytest.raises(ValidationError):
        async for _ in job_service.job_session(_FakeRedis(), feeder, poll_interval_s=0.01, user=_USER):
            pass


@pytest.mark.asyncio
async def test_job_session_unknown_job_not_found():
    feeder = _RequestFeeder()
    feeder.send(raptor_pb2.JobSessionRequest(job_id="nope", subscribe=True))
    with pytest.raises(NotFoundError):
        async for _ in job_service.job_session(_FakeRedis(), feeder, poll_interval_s=0.01, user=_USER):
            pass


@pytest.mark.asyncio
async def test_job_session_relays_watch_events_and_closes_on_terminal_state():
    fake_redis = _FakeRedis()
    job_id = "job-session-relay"
    _write_job_record(fake_redis, job_id, media_type="video")
    _set_orchestrator_step(fake_redis, "video", job_id, "queued")

    feeder = _RequestFeeder()
    feeder.send(raptor_pb2.JobSessionRequest(job_id=job_id, subscribe=True, last_sequence=0))

    events = []

    async def driver():
        async for resp in job_service.job_session(fake_redis, feeder, poll_interval_s=0.01, user=_USER):
            events.append(resp.job_event)
            if len(events) == 1:
                _set_orchestrator_step(fake_redis, "video", job_id, "complete")

    await asyncio.wait_for(driver(), timeout=5)

    assert [e.state for e in events] == [raptor_pb2.JOB_STATE_QUEUED, raptor_pb2.JOB_STATE_COMPLETE]
    assert [e.sequence for e in events] == [1, 2]  # both real, persisted events -- no -1 ephemeral ones here


@pytest.mark.asyncio
async def test_job_session_cancel_is_intent_only_does_not_change_job_state():
    fake_redis = _FakeRedis()
    job_id = "job-session-cancel"
    _write_job_record(fake_redis, job_id, media_type="video")
    _set_orchestrator_step(fake_redis, "video", job_id, "queued")

    feeder = _RequestFeeder()
    feeder.send(raptor_pb2.JobSessionRequest(job_id=job_id, subscribe=True, last_sequence=0))

    events = []

    async def driver():
        gen = job_service.job_session(fake_redis, feeder, poll_interval_s=0.05, user=_USER)
        async for resp in gen:
            events.append(resp.job_event)
            if len(events) == 1:
                feeder.send(raptor_pb2.JobSessionRequest(job_id=job_id, cancel=True))
            if len(events) == 2:
                await gen.aclose()
                break

    await asyncio.wait_for(driver(), timeout=5)

    assert events[0].sequence == 1  # the real QUEUED event
    assert events[1].sequence == -1  # the ephemeral cancel-intent ack
    assert "intent" in events[1].detail.lower()

    # Recorded as intent...
    assert await fake_redis.get(job_service._cancel_intent_key(job_id)) == "1"
    # ...but the job's actual state is completely untouched -- still queued,
    # never transitions to JOB_STATE_CANCELLED, because no worker honors this.
    state, _ = await job_service._current_state(fake_redis, "video", job_id)
    assert state == raptor_pb2.JOB_STATE_QUEUED


@pytest.mark.asyncio
async def test_job_session_control_message_acknowledged_but_not_forwarded_anywhere():
    fake_redis = _FakeRedis()
    job_id = "job-session-control"
    _write_job_record(fake_redis, job_id, media_type="video")
    _set_orchestrator_step(fake_redis, "video", job_id, "queued")

    feeder = _RequestFeeder()
    feeder.send(raptor_pb2.JobSessionRequest(job_id=job_id, subscribe=True, last_sequence=0))

    events = []

    async def driver():
        gen = job_service.job_session(fake_redis, feeder, poll_interval_s=0.05, user=_USER)
        async for resp in gen:
            events.append(resp.job_event)
            if len(events) == 1:
                feeder.send(raptor_pb2.JobSessionRequest(
                    job_id=job_id, control=raptor_pb2.JobControl(command="pause")
                ))
            if len(events) == 2:
                await gen.aclose()
                break

    await asyncio.wait_for(driver(), timeout=5)

    assert events[1].sequence == -1
    assert "pause" in events[1].detail
    assert "intent" in events[1].detail.lower() or "forwarded" in events[1].detail.lower()


# ── full RPC wiring smoke test ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_session_rpc_end_to_end():
    fake_redis = _FakeRedis()
    job_id = "job-session-e2e"
    _write_job_record(fake_redis, job_id, media_type="video")
    _set_orchestrator_step(fake_redis, "video", job_id, "queued")

    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), redis_client=fake_redis)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)

        async def request_gen():
            yield raptor_pb2.JobSessionRequest(job_id=job_id, subscribe=True, last_sequence=0)

        call = stub.JobSession(request_gen(), metadata=(("authorization", f"Bearer {make_token()}"),))
        first = await call.read()
        assert first.job_event.state == raptor_pb2.JOB_STATE_QUEUED
        call.cancel()
    finally:
        await channel.close()
        await server.stop(grace=None)
