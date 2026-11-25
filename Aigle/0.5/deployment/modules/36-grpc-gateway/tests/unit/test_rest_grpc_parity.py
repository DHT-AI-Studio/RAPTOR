"""REST / gRPC parity: for the same module 25 data, `POST .../search/video_search`
and `SearchByPerson` must return the same clips.

Module 13's REST route and this gateway each run their own copy of the 4-way
RRF + rerank pipeline, so the only honest way to compare them is at the one
place both talk to the same thing: module 25. A fake PersonalDBService answers
the four retrievers and the reranker; both pipelines run end to end on it.
"""
from __future__ import annotations

import os

import pytest
from fastapi import Response

# app.routers.video_search imports app.core.config at module scope (via
# app.api.dependencies), which needs these REST-app-only settings to exist.
# Only needed for the import to succeed; their values are never exercised here.
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", "{}")

from app.routers.video_search import VideoSearchRequest, video_search
from app.services.video_search_service import VideoSearchService
from raptor.v1 import raptor_pb2
from services.search_service import search_by_person

USER = {"user_id": "user-1", "branch_id": "user-1"}


def _payload(version_id, start, end, text, filename, asset_path, upload_time):
    return {
        "version_id": version_id, "start_time": start, "end_time": end, "asr_text": text,
        "asset_path": asset_path, "filename": filename, "upload_time": upload_time,
    }


# Shapes follow module 25's SearchResponse / GraphRAGResponse / TKGResponse.
_BM25 = {"results": [
    {"id": "c1", "score": 9.1, "payload": _payload("v1", 1.0, 2.0, "alice speaking", "a.mp4", "u/a.mp4", "2024-01-01")},
    {"id": "c2", "score": 7.2, "payload": _payload("v1", 10.0, 12.0, "alice again", "a.mp4", "u/a.mp4", "2024-01-01")},
]}
_VECTOR = {"results": [
    {"id": "c1", "score": 0.91, "payload": _payload("v1", 1.0, 2.0, "alice speaking", "a.mp4", "u/a.mp4", "2024-01-01")},
    {"id": "c3", "score": 0.80, "payload": _payload("v2", 30.0, 32.0, "alice mentioned", "b.mp4", "u/b.mp4", "2024-02-01")},
]}
_GRAPHRAG = {
    "matched_moments": [
        {"id": "m1", "version_id": "v3", "start_sec": 50.0, "end_sec": 55.0, "score": 0.8,
         "asr_text": "alice in the graph", "asset_path": "u/c.mp4", "filename": "c.mp4", "upload_time": "2024-03-01"},
    ],
    "moment_ids": [
        {"moment_id": "m2", "version_id": "v1", "start_sec": 10.0, "end_sec": 12.0,
         "contextual_text": "alice again", "asset_path": "u/a.mp4", "filename": "a.mp4", "upload_time": "2024-01-01"},
    ],
}
_TKG = {"moment_ids": [
    {"moment_id": "m3", "version_id": "v2", "start_sec": 30.0, "end_sec": 32.0,
     "contextual_text": "alice event", "asset_path": "u/b.mp4", "filename": "b.mp4", "upload_time": "2024-02-01"},
]}
# Cross-encoder scores by moment key "{video}|{round(start)}"; v3 falls below the 0.52 default threshold.
_RERANK = {"v1|1": 0.9, "v1|10": 0.7, "v2|30": 0.6, "v3|50": 0.45}


class _FakePersonalDB:
    """Module 25 at the HTTP boundary: canned answers, and a record of what it was asked."""

    def __init__(self, bm25=None, vector=None, graphrag=None, tkg=None, rerank=None):
        self.answers = {
            "bm25": bm25 if bm25 is not None else _BM25,
            "vector": vector if vector is not None else _VECTOR,
            "graphrag": graphrag if graphrag is not None else _GRAPHRAG,
            "tkg": tkg if tkg is not None else _TKG,
        }
        self.rerank_scores = rerank if rerank is not None else _RERANK
        self.bodies: dict[str, dict] = {}

    async def ensure_database(self, user_id):
        return None

    async def bm25_search(self, user_id, body):
        self.bodies["bm25"] = body
        return self.answers["bm25"]

    async def vector_search(self, user_id, body):
        self.bodies["vector"] = body
        return self.answers["vector"]

    async def graphrag_search(self, user_id, body):
        self.bodies["graphrag"] = body
        return self.answers["graphrag"]

    async def temporal_search(self, user_id, body):
        self.bodies["tkg"] = body
        return self.answers["tkg"]

    async def rerank(self, user_id, body):
        return {"results": [{"id": d["id"], "score": self.rerank_scores.get(d["id"], 0.5)} for d in body["documents"]]}


class _FakeUrlResolver:
    async def resolve_version_urls(self, version_map, user):
        return {vid: f"http://files/{vid}" for vid in version_map}


async def _run_rest(db, query="alice", **overrides):
    req = VideoSearchRequest(query=query, top_k=10, candidate_multiplier=5, score_threshold=0.52, **overrides)
    return await video_search(req, Response(), svc=db, search_svc=_FakeUrlResolver(), current_user={"sub": "user-1"})


async def _run_grpc(db, query="alice"):
    service = VideoSearchService(personal_db=db, url_resolver=_FakeUrlResolver())
    return await search_by_person(service, raptor_pb2.SearchByPersonRequest(person_name=query, top_k=10), USER)


def _rest_clips(resp):
    return sorted(
        (
            f"{v.video_id}|{round(s.start_time) if s.start_time is not None else 0}",
            v.video_id, round(s.score, 6), s.text, v.asset_url, v.filename, v.upload_time, tuple(s.sources),
        )
        for v in resp.results for s in v.segments
    )


def _grpc_clips(resp):
    return sorted(
        (h.clip_id, h.video_id, round(h.score, 6), h.text, h.asset_url, h.filename, h.upload_time, tuple(h.sources))
        for h in resp.hits
    )


@pytest.mark.asyncio
async def test_rest_and_grpc_return_identical_clips_for_the_same_module25_data():
    rest = _rest_clips(await _run_rest(_FakePersonalDB()))
    grpc = _grpc_clips(await _run_grpc(_FakePersonalDB()))

    assert rest == grpc
    # Guard against a vacuous pass: the fixture must really exercise fusion and the threshold.
    keys = {c[0] for c in rest}
    assert {"v1|1", "v1|10", "v2|30"} <= keys, "expected the fused moments"
    assert "v3|50" not in keys, "0.45 is below the 0.52 threshold and must be filtered by both"
    merged = next(c for c in rest if c[0] == "v1|10")
    assert set(merged[7]) >= {"bm25", "graphrag"}, "a moment found by several retrievers keeps all its sources"


@pytest.mark.asyncio
async def test_rest_and_grpc_agree_on_zero_hits():
    empty = {"results": []}
    kwargs = {"bm25": empty, "vector": empty, "graphrag": {}, "tkg": {}}
    rest = await _run_rest(_FakePersonalDB(**kwargs), query="nobody")
    grpc = await _run_grpc(_FakePersonalDB(**kwargs), query="nobody")
    assert rest.total == 0
    assert grpc.total == 0


@pytest.mark.asyncio
async def test_both_pipelines_send_module25_the_same_requests_except_score_threshold():
    rest_db, grpc_db = _FakePersonalDB(), _FakePersonalDB()
    await _run_rest(rest_db)
    await _run_grpc(grpc_db)

    def without_threshold(body):
        return {k: v for k, v in body.items() if k != "score_threshold"}

    for retriever in ("bm25", "vector", "graphrag", "tkg"):
        assert without_threshold(rest_db.bodies[retriever]) == without_threshold(grpc_db.bodies[retriever]), retriever


@pytest.mark.asyncio
async def test_known_difference_graph_score_threshold_sent_to_module25():
    """REST hard-codes 0.3 for GraphRAG/TKG; the gateway forwards the caller's score_threshold
    (default 0.52). With module 25 filtering server-side, the two can return different graph
    moments for the same query. This test pins the difference so changing either side is a
    deliberate decision (see docs/_internal/grpc-module25-contract.md), not an accident."""
    rest_db, grpc_db = _FakePersonalDB(), _FakePersonalDB()
    await _run_rest(rest_db)
    await _run_grpc(grpc_db)

    for retriever in ("graphrag", "tkg"):
        assert rest_db.bodies[retriever]["score_threshold"] == 0.3
        assert grpc_db.bodies[retriever]["score_threshold"] == 0.52
