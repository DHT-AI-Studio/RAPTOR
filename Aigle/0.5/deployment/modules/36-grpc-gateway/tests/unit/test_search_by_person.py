"""Unit tests for SearchByPerson / StreamSearchByPerson, backed by a fake
VideoSearchService — mirrors tests/test_dummy_call.py's self-contained
pattern (build_server(...) directly, not conftest.py's shared fixture).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from app.services.video_search_service import SegmentHit, VideoHit, VideoSearchResult
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server


class _FakeVideoSearchService:
    """Duck-typed stand-in for VideoSearchService.search_by_person — no real HTTP calls."""

    def __init__(self, result: VideoSearchResult):
        self.result = result
        self.last_call = None

    async def search_by_person(self, person_name, user, top_k=10, candidate_multiplier=5, score_threshold=0.52):
        self.last_call = {
            "person_name": person_name, "user": user, "top_k": top_k,
            "candidate_multiplier": candidate_multiplier, "score_threshold": score_threshold,
        }
        return self.result


_TWO_HIT_RESULT = VideoSearchResult(
    query="alice",
    videos=[
        VideoHit(
            video_id="v1", filename="a.mp4", score=0.9, asset_url="http://x/a.mp4", upload_time="2024-01-01",
            segments=[
                SegmentHit(start_time=1.0, end_time=2.0, score=0.9, text="alice speaking", sources=["bm25"]),
                SegmentHit(start_time=10.0, end_time=12.0, score=0.7, text="alice again", sources=["vector"]),
            ],
        ),
    ],
)


async def _build_and_call(fake, request, method_name="SearchByPerson"):
    server = build_server(video_search_service=fake, identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        method = getattr(stub, method_name)
        if method_name == "StreamSearchByPerson":
            return [hit async for hit in method(request, metadata=(("authorization", f"Bearer {make_token()}"),))]
        return await method(request, metadata=(("authorization", f"Bearer {make_token()}"),))
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_search_by_person_flattens_segments_to_clip_hits():
    fake = _FakeVideoSearchService(_TWO_HIT_RESULT)
    reply = await _build_and_call(fake, raptor_pb2.SearchByPersonRequest(person_name="alice", top_k=5))

    assert reply.total == 2
    assert [h.clip_id for h in reply.hits] == ["v1|1", "v1|10"]
    assert reply.hits[0].video_id == "v1"
    assert reply.hits[0].asset_url == "http://x/a.mp4"
    assert reply.hits[0].start_time == 1.0
    assert reply.hits[0].text == "alice speaking"
    assert fake.last_call["person_name"] == "alice"


@pytest.mark.asyncio
async def test_search_by_person_zero_hits_returns_empty_list():
    empty = VideoSearchResult(query="nobody", videos=[])
    fake = _FakeVideoSearchService(empty)
    reply = await _build_and_call(fake, raptor_pb2.SearchByPersonRequest(person_name="nobody"))
    assert list(reply.hits) == []
    assert reply.total == 0


@pytest.mark.asyncio
async def test_search_by_person_defaults_top_k_when_unset():
    fake = _FakeVideoSearchService(VideoSearchResult(query="x", videos=[]))
    await _build_and_call(fake, raptor_pb2.SearchByPersonRequest(person_name="x"))
    assert fake.last_call["top_k"] == 10
    assert fake.last_call["candidate_multiplier"] == 5
    assert fake.last_call["score_threshold"] == 0.52


@pytest.mark.asyncio
async def test_search_by_person_uses_identity_from_bearer_token():
    fake = _FakeVideoSearchService(VideoSearchResult(query="x", videos=[]))
    server = build_server(video_search_service=fake, identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        await stub.SearchByPerson(
            raptor_pb2.SearchByPersonRequest(person_name="x"),
            metadata=(("authorization", f"Bearer {make_token(sub='user-42')}"),),
        )
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert fake.last_call["user"] == {"user_id": "user-42", "branch_id": "user-42"}


@pytest.mark.asyncio
async def test_stream_search_by_person_yields_each_hit_incrementally():
    fake = _FakeVideoSearchService(_TWO_HIT_RESULT)
    hits = await _build_and_call(
        fake, raptor_pb2.SearchByPersonRequest(person_name="alice"), method_name="StreamSearchByPerson"
    )
    assert [h.clip_id for h in hits] == ["v1|1", "v1|10"]


class _SlowFakeVideoSearchService:
    """Simulates Module 25's real latency with a measurable delay -- used to
    prove StreamSearchByPerson's documented semantics (see
    docs/_internal/grpc-streaming-decision.md, Path A): the full result must be
    computed before the first item is streamed, not interleaved with
    retrieval."""

    def __init__(self, result: VideoSearchResult, delay_s: float):
        self.result = result
        self._delay_s = delay_s

    async def search_by_person(self, person_name, user, top_k=10, candidate_multiplier=5, score_threshold=0.52):
        await asyncio.sleep(self._delay_s)
        return self.result


@pytest.mark.asyncio
async def test_stream_search_by_person_is_buffered_not_incremental():
    """Proves StreamSearchByPerson's buffered-delivery contract: the first
    streamed ClipHit cannot arrive before Module 25's (simulated) full
    response does -- i.e. retrieval completes before any emission starts.
    If a future refactor ever made this RPC start emitting mid-computation,
    this test would catch it by failing on timing."""
    delay_s = 0.1
    fake = _SlowFakeVideoSearchService(_TWO_HIT_RESULT, delay_s)
    server = build_server(video_search_service=fake, identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        t0 = time.monotonic()
        first_item_elapsed = None
        async for _ in stub.StreamSearchByPerson(
            raptor_pb2.SearchByPersonRequest(person_name="alice"), metadata=(("authorization", f"Bearer {make_token()}"),)
        ):
            if first_item_elapsed is None:
                first_item_elapsed = time.monotonic() - t0
            break
    finally:
        await channel.close()
        await server.stop(grace=None)

    assert first_item_elapsed is not None
    assert first_item_elapsed >= delay_s, (
        f"first ClipHit arrived after {first_item_elapsed:.3f}s, before the simulated "
        f"{delay_s}s full-response delay elapsed -- streaming started before retrieval completed"
    )


class _RaisingVideoSearchService:
    def __init__(self, exc):
        self._exc = exc

    async def search_by_person(self, **kwargs):
        raise self._exc


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["SearchByPerson", "StreamSearchByPerson"])
@pytest.mark.parametrize("status_code", [502, 503, 504])
async def test_search_backend_down_maps_to_unavailable(method_name, status_code):
    from fastapi import HTTPException

    fake = _RaisingVideoSearchService(HTTPException(status_code=status_code, detail="Personal DB service unreachable"))
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _build_and_call(fake, raptor_pb2.SearchByPersonRequest(person_name="alice"), method_name)
    assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
    assert "module 25" in exc_info.value.details()


@pytest.mark.asyncio
async def test_search_other_backend_errors_are_not_reported_as_unavailable():
    from fastapi import HTTPException

    fake = _RaisingVideoSearchService(HTTPException(status_code=422, detail="bad request"))
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _build_and_call(fake, raptor_pb2.SearchByPersonRequest(person_name="alice"))
    assert exc_info.value.code() != grpc.StatusCode.UNAVAILABLE
