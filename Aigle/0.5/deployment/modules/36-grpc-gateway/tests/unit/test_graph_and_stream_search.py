"""StreamSearch*, GraphRagSearch and TKGSearch, backed by a fake PersonalDBService, plus the
asset_types / speaker_filter fields the three unary searches gained.

Same self-contained pattern as test_search_rpcs.py: build_server(...) with a duck-typed backend,
so no module 25 is needed. The module 25 response shapes below follow its own models
(app/models/graph_search.py) and searcher.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import grpc
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server

_STREAMS = {"StreamSearch": "hybrid_search", "StreamSearchBm25": "bm25_search", "StreamSearchVector": "vector_search"}

_CHUNK = {"id": "c1", "score": 0.9, "payload": {
    "version_id": "ver-1", "asset_path": "video/mp4/clip", "filename": "clip.mp4", "type": "videos",
    "chunk_id": "c1", "text": "hello", "start_time": 1.0, "end_time": 11.0, "speaker": "alice"}}

# What module 25's /personal/search/graphrag returns.
_GRAPHRAG = {
    "query": "q", "strategy": "hybrid",
    "matched_entities": [{"id": "e1", "name": "Alice"}],
    "matched_moments": [
        {"chunk_id": "m1", "type": "videos", "version_id": "ver-1", "asset_path": "video/mp4/clip",
         "filename": "clip.mp4", "start_sec": 5.0, "end_sec": 15.0, "score": 3.2,
         "asr_text": "full asr", "matched_via": ["asr_text"], "snippets": {"asr_text": "alice says hi"}},
        {"chunk_id": "m2", "type": "audios", "version_id": "ver-2", "asset_path": "audio/mp3/talk",
         "filename": "talk.mp3", "start_sec": 0.0, "end_sec": 8.0, "score": 1.1, "lvlm_desc": "a talk"},
    ],
    "nodes": [{"id": "e1"}], "relationships": [{"id": "r1", "from": "e1", "to": "m1"}],
    "moment_ids": [], "citations": [{"label": "[1]", "moment_id": "m1"}],
}

# What /personal/search/tkg returns.
_TKG = {
    "query": "q", "matched_entities": [{"id": "e1", "name": "Alice"}],
    "subgraph_nodes": [{"id": "e1"}], "subgraph_edges": [{"id": "r1"}],
    "temporal_facts": [
        {"fact_id": "f1", "entity": "Alice", "relation": "joined", "value": "Acme", "when": "2024-07",
         "confidence": 0.8, "source_version_id": "ver-1"},
        {"fact_id": "f2", "entity": "Alice", "relation": "left", "value": "Acme", "when": "2025",
         "confidence": 0.5, "source_version_id": "ver-2"},
    ],
    "moment_ids": [{"moment_id": "m1"}],
}


class _Fake:
    def __init__(self, chunks=None, graphrag=None, tkg=None, error: Exception | None = None):
        self.chunks = {"results": [_CHUNK]} if chunks is None else chunks
        self.graphrag = _GRAPHRAG if graphrag is None else graphrag
        self.tkg = _TKG if tkg is None else tkg
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def ensure_database(self, user_id):
        pass

    def _answer(self, name, body, response):
        self.calls.append((name, body))
        if self.error is not None:
            raise self.error
        return response

    async def hybrid_search(self, user_id, body):
        return self._answer("hybrid_search", body, self.chunks)

    async def bm25_search(self, user_id, body):
        return self._answer("bm25_search", body, self.chunks)

    async def vector_search(self, user_id, body):
        return self._answer("vector_search", body, self.chunks)

    async def graphrag_search(self, user_id, body):
        return self._answer("graphrag_search", body, self.graphrag)

    async def temporal_search(self, user_id, body):
        return self._answer("temporal_search", body, self.tkg)


class _Redis:
    """The stream-cap counter only; without it every stream waits for a Redis DNS timeout."""

    def __init__(self):
        self.n = 0

    async def incr(self, key):
        self.n += 1
        return self.n

    async def decr(self, key):
        self.n -= 1
        return self.n

    async def expire(self, key, ttl):
        pass


async def _run(fake, rpc, request):
    server = build_server(personal_db=fake, redis_client=_Redis(), identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        call = getattr(stub, rpc)(request, metadata=(("authorization", f"Bearer {make_token()}"),))
        if rpc in _STREAMS:
            return [r async for r in call]
        return await call
    finally:
        await channel.close()
        await server.stop(grace=None)


async def _code(fake, rpc, request):
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await _run(fake, rpc, request)
    return e.value.code()


# ── StreamSearch* ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _STREAMS)
async def test_each_stream_rpc_calls_only_its_own_retriever_and_sends_one_item_per_hit(rpc):
    hits = [{"id": str(i), "score": 9 - i, "payload": {"chunk_id": str(i)}} for i in range(3)]
    fake = _Fake(chunks={"results": hits})
    items = await _run(fake, rpc, raptor_pb2.SearchRequest(query="alice"))
    assert [c[0] for c in fake.calls] == [_STREAMS[rpc]]
    assert [i.chunk_id for i in items] == ["0", "1", "2"]  # backend order kept


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _STREAMS)
async def test_a_streamed_hit_is_the_same_as_the_unary_hit(rpc):
    unary = await _run(_Fake(), rpc.replace("Stream", ""), raptor_pb2.SearchRequest(query="alice"))
    streamed = await _run(_Fake(), rpc, raptor_pb2.SearchRequest(query="alice"))
    assert streamed == list(unary.results)


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _STREAMS)
async def test_a_stream_with_no_matches_is_empty_not_an_error(rpc):
    assert await _run(_Fake(chunks={"results": []}), rpc, raptor_pb2.SearchRequest(query="x")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc", _STREAMS)
async def test_a_stream_validates_like_the_unary_call(rpc):
    fake = _Fake()
    assert await _code(fake, rpc, raptor_pb2.SearchRequest(query=" ")) == grpc.StatusCode.INVALID_ARGUMENT
    assert await _code(fake, rpc, raptor_pb2.SearchRequest(query="a", top_k=101)) == grpc.StatusCode.INVALID_ARGUMENT
    assert fake.calls == []


@pytest.mark.asyncio
async def test_a_stream_reports_module_25_being_down_as_unavailable():
    fake = _Fake(error=HTTPException(status_code=503, detail="down"))
    assert await _code(fake, "StreamSearch", raptor_pb2.SearchRequest(query="a")) == grpc.StatusCode.UNAVAILABLE


# ── asset_types / speaker_filter on the unary searches ───────────────────────


@pytest.mark.asyncio
async def test_asset_types_and_speaker_filter_reach_module_25():
    fake = _Fake()
    await _run(fake, "Search", raptor_pb2.SearchRequest(
        query="a", asset_types=["videos", "audios"], speaker_filter=["alice", "bob"]))
    assert fake.calls[0][1] == {"query": "a", "top_k": 10, "type": ["videos", "audios"], "speaker": ["alice", "bob"]}


@pytest.mark.asyncio
async def test_media_type_and_asset_types_combine_without_duplicates():
    fake = _Fake()
    await _run(fake, "Search", raptor_pb2.SearchRequest(query="a", media_type="videos", asset_types=["videos", "images"]))
    assert fake.calls[0][1]["type"] == ["videos", "images"]


@pytest.mark.asyncio
async def test_blank_entries_in_the_filters_are_ignored():
    fake = _Fake()
    await _run(fake, "Search", raptor_pb2.SearchRequest(query="a", asset_types=[" "], speaker_filter=[""]))
    assert fake.calls[0][1] == {"query": "a", "top_k": 10}


# ── GraphRagSearch ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graphrag_returns_the_matching_moments_as_hits():
    reply = await _run(_Fake(), "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="alice"))
    first, second = reply.results
    assert (first.chunk_id, first.asset_path, first.version_id, first.filename, first.media_type) == (
        "m1", "video/mp4/clip", "ver-1", "clip.mp4", "videos")
    assert first.score == pytest.approx(3.2)
    assert (first.start_time, first.end_time) == (5.0, 15.0)
    assert first.snippet == "alice says hi"  # the matched snippet, not the whole transcript
    assert second.snippet == "a talk"  # falls back to the description when nothing was highlighted


@pytest.mark.asyncio
async def test_graphrag_hits_carry_the_speaker_like_the_other_searches_do():
    graphrag = {**_GRAPHRAG, "matched_moments": [
        {**_GRAPHRAG["matched_moments"][0], "speaker": "alice"}, _GRAPHRAG["matched_moments"][1]]}
    reply = await _run(_Fake(graphrag=graphrag), "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="a"))
    assert [r.speaker for r in reply.results] == ["alice", ""]  # absent -> empty, not an error


@pytest.mark.asyncio
async def test_graphrag_puts_the_graph_in_subgraph_json():
    reply = await _run(_Fake(), "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="alice"))
    assert json.loads(reply.subgraph_json) == {
        "matched_entities": [{"id": "e1", "name": "Alice"}],
        "nodes": [{"id": "e1"}],
        "edges": [{"id": "r1", "from": "e1", "to": "m1"}],
        "citations": [{"label": "[1]", "moment_id": "m1"}],
    }


@pytest.mark.asyncio
async def test_graphrag_sends_only_what_the_caller_set():
    fake = _Fake()
    await _run(fake, "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="  alice "))
    assert fake.calls[0] == ("graphrag_search", {"query": "alice"})  # module 25's own defaults apply

    fake = _Fake()
    await _run(fake, "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(
        query="a", max_depth=3, strategy="literal", score_threshold=0.0))
    assert fake.calls[0][1] == {"query": "a", "max_depth": 3, "strategy": "literal", "score_threshold": 0.0}


@pytest.mark.asyncio
async def test_graphrag_top_k_caps_the_hits_but_not_the_graph():
    reply = await _run(_Fake(), "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="a", top_k=1))
    assert len(reply.results) == 1
    assert json.loads(reply.subgraph_json)["nodes"] == [{"id": "e1"}]


@pytest.mark.asyncio
async def test_graphrag_with_nothing_indexed_is_an_empty_result():
    empty = {"query": "q", "strategy": "hybrid", "matched_entities": [], "matched_moments": [], "nodes": [],
             "relationships": [], "moment_ids": [], "citations": []}
    reply = await _run(_Fake(graphrag=empty), "GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="a"))
    assert list(reply.results) == [] and reply.subgraph_json == "{}"


@pytest.mark.asyncio
@pytest.mark.parametrize("request_message", [
    raptor_pb2.GraphRagSearchRequest(),
    raptor_pb2.GraphRagSearchRequest(query="a", top_k=101),
    raptor_pb2.GraphRagSearchRequest(query="a", top_k=-1),
    raptor_pb2.GraphRagSearchRequest(query="a", max_depth=5),
    raptor_pb2.GraphRagSearchRequest(query="a", max_depth=-1),
    raptor_pb2.GraphRagSearchRequest(query="a", strategy="fuzzy"),
], ids=["no query", "top_k over 100", "negative top_k", "depth over 4", "negative depth", "unknown strategy"])
async def test_graphrag_bad_requests_are_invalid_argument_and_never_reach_module_25(request_message):
    fake = _Fake()
    assert await _code(fake, "GraphRagSearch", request_message) == grpc.StatusCode.INVALID_ARGUMENT
    assert fake.calls == []


# ── TKGSearch ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tkg_returns_the_facts_as_hits_in_module_25s_order():
    reply = await _run(_Fake(), "TKGSearch", raptor_pb2.TKGSearchRequest(query="alice"))
    first, second = reply.results
    assert first.snippet == "Alice joined Acme (2024-07)"
    assert first.score == pytest.approx(0.8) and first.version_id == "ver-1"
    assert first.asset_path == "" and first.chunk_id == "" and not first.HasField("start_time")
    assert second.snippet == "Alice left Acme (2025)" and second.version_id == "ver-2"


@pytest.mark.asyncio
async def test_tkg_puts_the_temporal_subgraph_in_subgraph_json():
    reply = await _run(_Fake(), "TKGSearch", raptor_pb2.TKGSearchRequest(query="alice"))
    assert json.loads(reply.subgraph_json) == {
        "matched_entities": [{"id": "e1", "name": "Alice"}],
        "nodes": [{"id": "e1"}],
        "edges": [{"id": "r1"}],
        "moments": [{"moment_id": "m1"}],
    }


@pytest.mark.asyncio
async def test_tkg_forwards_the_window_the_entity_and_the_limit():
    fake = _Fake()
    await _run(fake, "TKGSearch", raptor_pb2.TKGSearchRequest(
        query="a", top_k=5, max_depth=2, score_threshold=0.2, time_start="2024-01-01", time_end="2024-12-31", entity="Alice"))
    assert fake.calls[0] == ("temporal_search", {
        "query": "a", "limit": 5, "max_depth": 2, "score_threshold": 0.2,
        "time_start": "2024-01-01", "time_end": "2024-12-31", "entity": "Alice"})

    fake = _Fake()
    await _run(fake, "TKGSearch", raptor_pb2.TKGSearchRequest(query="a"))
    assert fake.calls[0][1] == {"query": "a"}


@pytest.mark.asyncio
async def test_tkg_with_nothing_indexed_is_an_empty_result():
    empty = {"query": "q", "matched_entities": [], "subgraph_nodes": [], "subgraph_edges": [],
             "temporal_facts": [], "moment_ids": []}
    reply = await _run(_Fake(tkg=empty), "TKGSearch", raptor_pb2.TKGSearchRequest(query="a"))
    assert list(reply.results) == [] and reply.subgraph_json == "{}"


@pytest.mark.asyncio
@pytest.mark.parametrize("request_message", [
    raptor_pb2.TKGSearchRequest(),
    raptor_pb2.TKGSearchRequest(query="a", top_k=101),
    raptor_pb2.TKGSearchRequest(query="a", max_depth=5),
], ids=["no query", "top_k over 100", "depth over 4"])
async def test_tkg_bad_requests_are_invalid_argument_and_never_reach_module_25(request_message):
    fake = _Fake()
    assert await _code(fake, "TKGSearch", request_message) == grpc.StatusCode.INVALID_ARGUMENT
    assert fake.calls == []


# ── errors from module 25, both graph RPCs ───────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc,request_message", [
    ("GraphRagSearch", raptor_pb2.GraphRagSearchRequest(query="a")),
    ("TKGSearch", raptor_pb2.TKGSearchRequest(query="a")),
])
@pytest.mark.parametrize("status,expected", [
    (503, grpc.StatusCode.UNAVAILABLE),
    (404, grpc.StatusCode.NOT_FOUND),
    (400, grpc.StatusCode.INVALID_ARGUMENT),
])
async def test_module_25_errors_keep_their_meaning(rpc, request_message, status, expected):
    fake = _Fake(error=HTTPException(status_code=status, detail="x"))
    assert await _code(fake, rpc, request_message) == expected
