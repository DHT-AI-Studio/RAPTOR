"""StreamVideoSegment and JobSession draw from the *same* per-user stream cap, and the cap
holds when many of them arrive at once.

test_stream_limit.py proves the counter is shared using UploadAsset and DownloadAsset. This
file covers the two RPCs VIE01-414 names, and does it under a burst: far more streams than
the cap are opened in the same instant, and exactly `cap` of them must get through, however
the two RPC types are mixed. This is the item the security sign-off asks for.
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
from fastapi import HTTPException
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from services import job_service

CAP = 4
AUTH = (("authorization", f"Bearer {make_token()}"),)
SLOT_KEY = "grpc:streams:dev-user"  # the token's own `sub`


class _Redis:
    def __init__(self):
        self.kv: dict = {}
        self.lists: dict = {}

    async def get(self, key):
        return self.kv.get(key)

    async def setex(self, key, ttl, value):
        self.kv[key] = value

    async def incr(self, key):
        self.kv[key] = int(self.kv.get(key, 0)) + 1
        return self.kv[key]

    async def decr(self, key):
        self.kv[key] = int(self.kv.get(key, 0)) - 1
        return self.kv[key]

    async def expire(self, key, ttl):
        pass

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]


class _Storage:
    async def get_presign_url(self, asset_path, version_id, user):
        return "http://example.invalid/raw.bin"

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        raise HTTPException(status_code=404, detail="not a platform asset_path")


class _HeldBody:
    """A response body that sends two chunks and then waits for the test to let go. Two,
    because the handler holds each chunk back until it sees the next one (to mark the last)."""

    def __init__(self, hold: asyncio.Event):
        self._hold = hold

    async def chunks(self):
        yield b"x" * 16
        yield b"y" * 16
        await self._hold.wait()
        yield b"z" * 16


class _Response:
    status_code = 200

    def __init__(self, hold):
        self.headers: dict = {}
        self._body = _HeldBody(hold)

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size):
        async for c in self._body.chunks():
            yield c


class _Stream:
    def __init__(self, hold):
        self._response = _Response(hold)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _Http:
    def __init__(self, hold):
        self._hold = hold

    def stream(self, method, url, headers=None):
        return _Stream(self._hold)


def _job_record(redis, job_id):
    redis.kv[job_service._job_key(job_id)] = json.dumps({
        "job_id": job_id, "state": raptor_pb2.JOB_STATE_QUEUED, "job_type": "", "media_type": "video",
        "input_asset_ids": [], "request_id": "", "created_at": "2024-01-01T00:00:00+00:00", "detail": "",
    })
    redis.kv[job_service._orchestrator_key("video", job_id)] = json.dumps({"step": "indexing"})


async def _server(redis, hold):
    server = build_server(
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(),
        storage_service=_Storage(), http_client=_Http(hold), redis_client=redis, max_streams_per_user=CAP,
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    return server, grpc.aio.insecure_channel(f"localhost:{port}")


def _segment(stub):
    request = raptor_pb2.StreamVideoSegmentRequest(
        asset=raptor_pb2.AssetRef(asset_path="raw.bin", version_id="v1"),
        format=raptor_pb2.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE, preferred_chunk_size=16,
    )
    return stub.StreamVideoSegment(request, metadata=AUTH)


def _session(stub, hold):
    async def requests():
        yield raptor_pb2.JobSessionRequest(job_id="j1", subscribe=True)
        await hold.wait()

    return stub.JobSession(requests(), metadata=AUTH)


async def _open_and_hold(call, hold, admitted: list):
    """Reads the first message (proof the stream got a slot), then stays open until `hold`
    is set. Returns "admitted", or the status code if the cap turned it away."""
    try:
        async for _ in call:
            admitted.append(1)
            await hold.wait()
            call.cancel()
            return "admitted"
        return "ended-without-a-message"
    except grpc.aio.AioRpcError as e:
        return e.code()


async def _wait_until(predicate, timeout_s=5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("sessions,segments", [(0, 12), (12, 0), (6, 6), (1, 11), (11, 1)])
async def test_a_burst_of_mixed_streams_admits_exactly_the_cap(sessions, segments):
    redis = _Redis()
    _job_record(redis, "j1")
    hold = asyncio.Event()
    server, channel = await _server(redis, hold)
    admitted: list = []
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        calls = [_session(stub, hold) for _ in range(sessions)] + [_segment(stub) for _ in range(segments)]
        tasks = [asyncio.ensure_future(_open_and_hold(c, hold, admitted)) for c in calls]

        # Everyone has had their say once CAP are in and the rest have been refused.
        await _wait_until(lambda: len(admitted) == CAP and sum(t.done() for t in tasks) == len(tasks) - CAP)
        assert int(redis.kv.get(SLOT_KEY, 0)) == CAP  # the counter settled at the cap, not above it

        hold.set()
        results = await asyncio.gather(*tasks)
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert results.count("admitted") == CAP
    assert results.count(grpc.StatusCode.RESOURCE_EXHAUSTED) == sessions + segments - CAP
    assert len(admitted) == CAP


async def _one_round(stub, redis):
    """Six sessions and six segments at once; returns how many were admitted."""
    admitted: list = []
    hold = asyncio.Event()
    calls = [_session(stub, hold) for _ in range(6)] + [_segment(stub) for _ in range(6)]
    tasks = [asyncio.ensure_future(_open_and_hold(c, hold, admitted)) for c in calls]
    await _wait_until(lambda: len(admitted) == CAP and sum(t.done() for t in tasks) == len(tasks) - CAP)
    hold.set()
    results = await asyncio.gather(*tasks)
    await _wait_until(lambda: int(redis.kv.get(SLOT_KEY, 0)) == 0)
    return results.count("admitted")


@pytest.mark.asyncio
async def test_every_slot_comes_back_so_the_next_burst_gets_the_full_cap_again():
    redis = _Redis()
    _job_record(redis, "j1")
    server, channel = await _server(redis, asyncio.Event())
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        assert [await _one_round(stub, redis) for _ in range(3)] == [CAP, CAP, CAP]
    finally:
        await channel.close()
        await server.stop(grace=None)
