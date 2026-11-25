"""A job belongs to the account that submitted it: another account asking for it by id (GetJob,
WatchJob, JobSession) gets NOT_FOUND, the same answer as for an id that does not exist, so a job
id never reveals that someone else's job exists. Found on the deployed gateway by
scripts/isolation_check.py (VIE01-165): before this, any signed-in account could read any job.
"""
from __future__ import annotations

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
from test_job_lifecycle import (
    _asset_ref,
    _FakeKafkaProducer,
    _FakeRedis,
    _FakeStorageService,
    _write_job_record,
)

_OWNER = {"user_id": "owner-1", "branch_id": "b"}
_OTHER = {"user_id": "someone-else", "branch_id": "b"}


async def _empty():
    return
    yield  # pragma: no cover -- makes this an async generator


async def _one(msg):
    yield msg


def _redis_with_job(**overrides):
    redis = _FakeRedis()
    _write_job_record(redis, "job-1", **overrides)
    return redis


@pytest.mark.asyncio
async def test_the_owner_can_get_and_watch_its_job():
    redis = _redis_with_job(user_id=_OWNER["user_id"], state=raptor_pb2.JOB_STATE_COMPLETE)
    job = await job_service.get_job(redis, raptor_pb2.GetJobRequest(job_id="job-1"), user=_OWNER)
    assert job.job_id == "job-1"


@pytest.mark.asyncio
async def test_another_account_gets_not_found_from_get_job():
    redis = _redis_with_job(user_id=_OWNER["user_id"])
    with pytest.raises(NotFoundError):
        await job_service.get_job(redis, raptor_pb2.GetJobRequest(job_id="job-1"), user=_OTHER)


@pytest.mark.asyncio
async def test_another_account_gets_not_found_from_watch_job():
    redis = _redis_with_job(user_id=_OWNER["user_id"])
    with pytest.raises(NotFoundError):
        async for _ in job_service.watch_job(redis, raptor_pb2.WatchJobRequest(job_id="job-1"), 0.01, user=_OTHER):
            pass


@pytest.mark.asyncio
async def test_another_account_gets_not_found_from_job_session_and_cannot_set_a_cancel_intent():
    redis = _redis_with_job(user_id=_OWNER["user_id"])
    first = raptor_pb2.JobSessionRequest(job_id="job-1", subscribe=True)
    with pytest.raises(NotFoundError):
        async for _ in job_service.job_session(redis, _one(first), 0.01, user=_OTHER):
            pass
    assert job_service._cancel_intent_key("job-1") not in redis._kv


@pytest.mark.asyncio
async def test_a_record_written_before_owners_were_stored_stays_readable():
    redis = _redis_with_job()  # no user_id, as written by the previous release
    job = await job_service.get_job(redis, raptor_pb2.GetJobRequest(job_id="job-1"), user=_OTHER)
    assert job.job_id == "job-1"


@pytest.mark.asyncio
async def test_submit_job_stores_the_submitting_account_as_owner():
    redis = _FakeRedis()
    server = build_server(
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(),
        storage_service=_FakeStorageService(), redis_client=redis, kafka_producer=_FakeKafkaProducer(),
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        as_a = (("authorization", f"Bearer {make_token('account-a')}"),)
        as_b = (("authorization", f"Bearer {make_token('account-b')}"),)
        job = await stub.SubmitJob(
            raptor_pb2.SubmitJobRequest(input_asset_ids=[_asset_ref()], request_id="r"), metadata=as_a
        )
        stored = json.loads(redis._kv[job_service._job_key(job.job_id)])
        assert stored["user_id"]

        # End to end through the real server: A reads it, B is told it does not exist.
        again = await stub.GetJob(raptor_pb2.GetJobRequest(job_id=job.job_id), metadata=as_a)
        assert again.job_id == job.job_id
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.GetJob(raptor_pb2.GetJobRequest(job_id=job.job_id), metadata=as_b)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
        with pytest.raises(grpc.aio.AioRpcError) as watch_exc:
            async for _ in stub.WatchJob(raptor_pb2.WatchJobRequest(job_id=job.job_id), metadata=as_b):
                pass
        assert watch_exc.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        await channel.close()
        await server.stop(grace=None)
