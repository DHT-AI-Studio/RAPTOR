"""Unit tests for Search / SearchBm25 / SearchVector, backed by a fake PersonalDBService.

Same self-contained pattern as test_search_by_person.py: build_server(...) directly with a
duck-typed backend, so no module 25 is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import grpc
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server

_RPC_TO_BACKEND = {"Search": "hybrid_search", "SearchBm25": "bm25_search", "SearchVector": "vector_search"}
_RPCS = list(_RPC_TO_BACKEND)

# What module 25 returns for a video chunk (the fields seen on a real account).
_VIDEO_HIT = {
    "id": "chunk-1",
    "score": 4.38,
    "payload": {
        "version_id": "ver-1",
        "asset_path": "video/mp4/clip",
        "filename": "clip.mp4",
        "type": "videos",
        "chunk_id": "chunk-1",
        "chunk_index": 3,
        "embedding_type": "text",
        "text": "asr:{hello there}",
        "summary": "A short clip.",
        "start_time": 10.0,
        "end_time": 20.0,
        "speaker": "alice",
        "source": "video",
        "status": "complete",
        "asset_url": "http://storage/clip.mp4?X-Amz-Expires=86400",
    },
}


class _FakePersonalDB:
    """Duck-typed stand-in for PersonalDBService: records every call, answers from `response`."""

    def __init__(self, response=None, error: Exception | None = None):
        self.response = {"results": [_VIDEO_HIT]} if response is None else response
        self.error = error
        self.calls: list[tuple[str, str, dict]] = []
        self.ensured: list[str] = []

    async def ensure_database(self, user_id):
        self.ensured.append(user_id)

    async def _answer(self, name, user_id, body):
        self.calls.append((name, user_id, body))
        if self.error is not None:
            raise self.error
        return self.response

    async def hybrid_search(self, user_id, body):
        return await self._answer("hybrid_search", user_id, body)

    async def bm25_search(self, user_id, body):
        return await self._answer("bm25_search", user_id, body)

    async def vector_search(self, user_id, body):
        return await self._answer("vector_search", user_id, body)


async def _call(fake, rpc, request, token=None):
    server = build_server(
        personal_db=fake, identity_http_client=identity_http_client(), authz_http_client=authz_http_client()
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        metadata = (("authorization", f"Bearer {token or make_token()}"),)
        return await getattr(stub, rpc)(request, metadata=metadata)
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _RPCS)
async def test_each_rpc_calls_only_its_own_retriever(rpc):
    fake = _FakePersonalDB()
    await _call(fake, rpc, raptor_pb2.SearchRequest(query="alice"))
    assert [c[0] for c in fake.calls] == [_RPC_TO_BACKEND[rpc]]


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _RPCS)
async def test_a_hit_is_mapped_field_by_field(rpc):
    reply = await _call(_FakePersonalDB(), rpc, raptor_pb2.SearchRequest(query="alice"))
    assert len(reply.results) == 1
    r = reply.results[0]
    assert (r.asset_path, r.version_id, r.filename, r.media_type) == ("video/mp4/clip", "ver-1", "clip.mp4", "videos")
    assert r.score == pytest.approx(4.38)
    assert r.snippet == "asr:{hello there}"
    assert (r.chunk_id, r.speaker) == ("chunk-1", "alice")
    assert r.HasField("start_time") and r.start_time == 10.0
    assert r.HasField("end_time") and r.end_time == 20.0


@pytest.mark.asyncio
async def test_the_presigned_asset_url_is_not_exposed():
    """It expires after a day; callers download with asset_path + version_id instead."""
    assert "asset_url" not in raptor_pb2.SearchResult.DESCRIPTOR.fields_by_name


@pytest.mark.asyncio
async def test_a_hit_without_a_time_range_leaves_the_times_unset():
    hit = {"id": "d1", "score": 0.5, "payload": {"version_id": "v", "asset_path": "document/pdf/a", "type": "documents"}}
    reply = await _call(_FakePersonalDB({"results": [hit]}), "Search", raptor_pb2.SearchRequest(query="x"))
    r = reply.results[0]
    assert not r.HasField("start_time") and not r.HasField("end_time")
    assert (r.snippet, r.speaker, r.filename) == ("", "", "")


@pytest.mark.asyncio
async def test_a_list_can_mix_whole_file_records_and_chunks():
    """Module 25 returns a video's whole-file summary (no chunk_id, no times) next to its 10 s chunks."""
    whole = {"id": "v", "score": 0.97, "payload": {"version_id": "ver-1", "asset_path": "video/mp4/clip", "summary": "About a clip."}}
    reply = await _call(_FakePersonalDB({"results": [whole, _VIDEO_HIT]}), "Search", raptor_pb2.SearchRequest(query="x"))
    first, second = reply.results
    assert first.chunk_id == "" and not first.HasField("start_time") and first.snippet == "About a clip."
    assert second.chunk_id == "chunk-1" and second.HasField("start_time")
    assert first.version_id == second.version_id == "ver-1"


@pytest.mark.asyncio
async def test_the_snippet_falls_back_to_the_summary():
    hit = {"id": "d1", "score": 0.5, "payload": {"summary": "only a summary", "text": None}}
    reply = await _call(_FakePersonalDB({"results": [hit]}), "Search", raptor_pb2.SearchRequest(query="x"))
    assert reply.results[0].snippet == "only a summary"


@pytest.mark.asyncio
async def test_a_hit_with_no_payload_does_not_crash():
    reply = await _call(_FakePersonalDB({"results": [{"id": "x", "score": 1}]}), "Search", raptor_pb2.SearchRequest(query="x"))
    assert len(reply.results) == 1 and reply.results[0].score == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{"results": []}, {}, {"results": None}])
async def test_no_matches_is_an_empty_list_not_an_error(response):
    reply = await _call(_FakePersonalDB(response), "SearchBm25", raptor_pb2.SearchRequest(query="nothing"))
    assert list(reply.results) == []


@pytest.mark.asyncio
async def test_results_keep_the_backends_order():
    hits = [{"id": str(i), "score": 10 - i, "payload": {"chunk_id": str(i)}} for i in range(4)]
    reply = await _call(_FakePersonalDB({"results": hits}), "Search", raptor_pb2.SearchRequest(query="x"))
    assert [r.chunk_id for r in reply.results] == ["0", "1", "2", "3"]


@pytest.mark.asyncio
async def test_the_request_body_sent_to_module_25():
    fake = _FakePersonalDB()
    await _call(fake, "Search", raptor_pb2.SearchRequest(query="  alice  ", top_k=7, media_type="videos"))
    assert fake.calls[0][2] == {"query": "alice", "top_k": 7, "type": "videos"}


@pytest.mark.asyncio
async def test_top_k_defaults_to_10_and_media_type_is_left_out_when_empty():
    fake = _FakePersonalDB()
    await _call(fake, "Search", raptor_pb2.SearchRequest(query="alice"))
    assert fake.calls[0][2] == {"query": "alice", "top_k": 10}


@pytest.mark.asyncio
async def test_top_k_100_is_accepted():
    fake = _FakePersonalDB()
    await _call(fake, "Search", raptor_pb2.SearchRequest(query="a", top_k=100))
    assert fake.calls[0][2]["top_k"] == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _RPCS)
@pytest.mark.parametrize(
    "request_message",
    [
        raptor_pb2.SearchRequest(),
        raptor_pb2.SearchRequest(query="   "),
        raptor_pb2.SearchRequest(query="a", top_k=-1),
        raptor_pb2.SearchRequest(query="a", top_k=101),
    ],
    ids=["no query", "blank query", "negative top_k", "top_k over 100"],
)
async def test_a_bad_request_is_invalid_argument_and_never_reaches_module_25(rpc, request_message):
    fake = _FakePersonalDB()
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await _call(fake, rpc, request_message)
    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert fake.calls == []


@pytest.mark.asyncio
async def test_the_callers_database_is_provisioned_and_searched_as_that_user():
    fake = _FakePersonalDB()
    await _call(fake, "Search", raptor_pb2.SearchRequest(query="a"), token=make_token(sub="user-42"))
    assert fake.ensured == ["user-42"]
    assert fake.calls[0][1] == "user-42"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (503, grpc.StatusCode.UNAVAILABLE),
        (504, grpc.StatusCode.DEADLINE_EXCEEDED),
        (422, grpc.StatusCode.INVALID_ARGUMENT),
        (500, grpc.StatusCode.INTERNAL),
    ],
)
async def test_a_module_25_failure_maps_to_the_matching_status(status, expected):
    fake = _FakePersonalDB(error=HTTPException(status_code=status, detail="boom"))
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await _call(fake, "SearchVector", raptor_pb2.SearchRequest(query="a"))
    assert caught.value.code() == expected


@pytest.mark.asyncio
async def test_module_25s_internal_error_text_is_not_sent_to_the_caller():
    fake = _FakePersonalDB(error=HTTPException(status_code=500, detail="secret internal detail"))
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        await _call(fake, "Search", raptor_pb2.SearchRequest(query="a"))
    assert "secret" not in caught.value.details()


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _RPCS)
async def test_a_call_without_a_token_is_refused_before_module_25(rpc):
    fake = _FakePersonalDB()
    server = build_server(
        personal_db=fake, identity_http_client=identity_http_client(), authz_http_client=authz_http_client()
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await getattr(stub, rpc)(raptor_pb2.SearchRequest(query="a"))
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert fake.calls == []
    finally:
        await channel.close()
        await server.stop(grace=None)


def test_the_three_rpcs_share_one_request_and_response_type():
    service = raptor_pb2.DESCRIPTOR.services_by_name["RaptorService"]
    for rpc in _RPCS:
        method = service.methods_by_name[rpc]
        assert method.input_type.name == "SearchRequest"
        assert method.output_type.name == "SearchResponse"
        assert not method.client_streaming and not method.server_streaming
