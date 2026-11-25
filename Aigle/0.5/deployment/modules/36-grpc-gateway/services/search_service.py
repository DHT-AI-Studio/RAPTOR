"""SearchByPerson / StreamSearchByPerson, Search / SearchBm25 / SearchVector (and their Stream versions),
and GraphRagSearch / TKGSearch, RPC handlers.

Both delegate to the shared `VideoSearchService` (copied into this image
from module 13's REST app — see the Dockerfile) so the gRPC and REST
transports run the exact same retrieval/ranking pipeline. No person-entity
index exists in Raptor today, so "by person" means passing person_name
straight through as the free-text query (see VideoSearchService.search_by_person).
"""
from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "generated"))
from app.services.personal_db_service import PersonalDBService
from app.services.video_search_service import ClipHitData, VideoSearchService
from fastapi import HTTPException
from interceptors.errors import UnavailableError, ValidationError
from raptor.v1 import raptor_pb2


def _to_clip_hit(hit: ClipHitData) -> raptor_pb2.ClipHit:
    kwargs = {
        "clip_id": hit.clip_id,
        "video_id": hit.video_id,
        "filename": hit.filename or "",
        "score": hit.score,
        "asset_url": hit.asset_url or "",
        "upload_time": hit.upload_time or "",
        "text": hit.text or "",
        "sources": hit.sources,
    }
    # proto3 `optional double` fields can't be passed as None — omit if unset.
    if hit.start_time is not None:
        kwargs["start_time"] = hit.start_time
    if hit.end_time is not None:
        kwargs["end_time"] = hit.end_time
    return raptor_pb2.ClipHit(**kwargs)


# Statuses PersonalDBService raises (as HTTPException) when module 25 is down
# (503 "unreachable") or a gateway in front of it fails.
_BACKEND_DOWN_STATUSES = {502, 503, 504}


async def _search(video_search_service, request: raptor_pb2.SearchByPersonRequest, user: dict[str, str]):
    try:
        return await video_search_service.search_by_person(
            person_name=request.person_name,
            user=user,
            top_k=request.top_k or 10,
            candidate_multiplier=request.candidate_multiplier or 5,
            score_threshold=request.score_threshold or 0.52,
        )
    except HTTPException as exc:
        if exc.status_code in _BACKEND_DOWN_STATUSES:
            raise UnavailableError("Search backend (module 25) is unavailable.") from exc
        raise


async def search_by_person(
    video_search_service: VideoSearchService,
    request: raptor_pb2.SearchByPersonRequest,
    user: dict[str, str],
) -> raptor_pb2.SearchByPersonResponse:
    result = await _search(video_search_service, request, user)
    hits = VideoSearchService.flatten_to_hits(result)
    return raptor_pb2.SearchByPersonResponse(hits=[_to_clip_hit(h) for h in hits], total=len(hits))


async def stream_search_by_person(
    video_search_service: VideoSearchService,
    request: raptor_pb2.SearchByPersonRequest,
    user: dict[str, str],
) -> AsyncIterator[raptor_pb2.ClipHit]:
    # Progressive per-item delivery *after* the RRF computation completes —
    # not genuine mid-computation streaming (that would need the 4-way
    # fan-out restructured around as_completed()); a documented scope choice.
    result = await _search(video_search_service, request, user)
    for hit in VideoSearchService.flatten_to_hits(result):
        yield _to_clip_hit(hit)


# ── Search / SearchBm25 / SearchVector ───────────────────────────────────

DEFAULT_TOP_K = 10
MAX_TOP_K = 100


def _chunk_to_result(item: dict) -> raptor_pb2.SearchResult:
    """One module 25 hit ({id, score, payload{...}}) -> SearchResult. Only the fields every
    media type is known to carry are read; anything else in the payload is ignored."""
    payload = item.get("payload") or {}
    kwargs = {
        "asset_path": payload.get("asset_path") or "",
        "score": float(item.get("score") or 0.0),
        "snippet": payload.get("text") or payload.get("summary") or "",
        "version_id": payload.get("version_id") or "",
        "filename": payload.get("filename") or "",
        "media_type": payload.get("type") or "",
        "chunk_id": payload.get("chunk_id") or "",
        "speaker": payload.get("speaker") or "",
    }
    # proto3 `optional double` fields can't be passed as None -- omit if unset.
    if payload.get("start_time") is not None:
        kwargs["start_time"] = float(payload["start_time"])
    if payload.get("end_time") is not None:
        kwargs["end_time"] = float(payload["end_time"])
    return raptor_pb2.SearchResult(**kwargs)


async def _search_chunks(
    search_call,
    personal_db: PersonalDBService,
    request: raptor_pb2.SearchRequest,
    user: dict[str, str],
) -> raptor_pb2.SearchResponse:
    query = request.query.strip()
    if not query:
        raise ValidationError("query is required.")
    if request.top_k < 0 or request.top_k > MAX_TOP_K:
        raise ValidationError(f"top_k must be between 0 and {MAX_TOP_K}.")
    body: dict = {"query": query, "top_k": request.top_k or DEFAULT_TOP_K}
    asset_types = [t.strip() for t in request.asset_types if t.strip()]
    if asset_types:
        # media_type and asset_types together mean "any of these"; module 25 takes a list.
        wanted = ([request.media_type] if request.media_type else []) + asset_types
        body["type"] = list(dict.fromkeys(wanted))
    elif request.media_type:
        body["type"] = request.media_type
    speakers = [sp.strip() for sp in request.speaker_filter if sp.strip()]
    if speakers:
        body["speaker"] = speakers
    user_id = user["user_id"]
    await personal_db.ensure_database(user_id)
    # A downstream failure (module 25 down, slow, or rejecting the request) is an HTTPException
    # here; ExceptionMappingInterceptor turns it into the matching gRPC status.
    data = await search_call(user_id, body)
    return raptor_pb2.SearchResponse(results=[_chunk_to_result(r) for r in data.get("results") or []])


async def search(personal_db, request, user):
    return await _search_chunks(personal_db.hybrid_search, personal_db, request, user)


async def search_bm25(personal_db, request, user):
    return await _search_chunks(personal_db.bm25_search, personal_db, request, user)


async def search_vector(personal_db, request, user):
    return await _search_chunks(personal_db.vector_search, personal_db, request, user)


# ── StreamSearch / StreamSearchBm25 / StreamSearchVector ─────────────────
#
# Buffered, like StreamSearchByPerson (docs/_internal/grpc-streaming-decision.md): module 25 answers with
# the complete list, and only then is it handed out one item at a time.


async def _stream_chunks(search_call, personal_db, request, user):
    response = await _search_chunks(search_call, personal_db, request, user)
    for result in response.results:
        yield result


def stream_search(personal_db, request, user):
    return _stream_chunks(personal_db.hybrid_search, personal_db, request, user)


def stream_search_bm25(personal_db, request, user):
    return _stream_chunks(personal_db.bm25_search, personal_db, request, user)


def stream_search_vector(personal_db, request, user):
    return _stream_chunks(personal_db.vector_search, personal_db, request, user)


# ── GraphRagSearch / TKGSearch ───────────────────────────────────────────

_STRATEGIES = ("literal", "semantic", "hybrid")
MAX_GRAPH_DEPTH = 4


def _validate_graph_request(request) -> str:
    query = request.query.strip()
    if not query:
        raise ValidationError("query is required.")
    if request.top_k < 0 or request.top_k > MAX_TOP_K:
        raise ValidationError(f"top_k must be between 0 and {MAX_TOP_K}.")
    if request.max_depth < 0 or request.max_depth > MAX_GRAPH_DEPTH:
        raise ValidationError(f"max_depth must be between 0 (default) and {MAX_GRAPH_DEPTH}.")
    return query


def _first_text(*candidates) -> str:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return ""


def _moment_to_result(moment: dict) -> raptor_pb2.SearchResult:
    """A module 25 graph moment (a Chunk vertex that matched the query) -> SearchResult."""
    snippets = moment.get("snippets") or {}
    kwargs = {
        "asset_path": moment.get("asset_path") or "",
        "score": float(moment.get("score") or 0.0),
        "snippet": _first_text(
            *(snippets.values() if isinstance(snippets, dict) else ()),
            moment.get("asr_text"), moment.get("lvlm_desc"), moment.get("contextual_text"),
        ),
        "version_id": moment.get("version_id") or "",
        "filename": moment.get("filename") or "",
        "media_type": moment.get("type") or "",
        "chunk_id": moment.get("chunk_id") or "",
        "speaker": moment.get("speaker") or "",
    }
    if moment.get("start_sec") is not None:
        kwargs["start_time"] = float(moment["start_sec"])
    if moment.get("end_sec") is not None:
        kwargs["end_time"] = float(moment["end_sec"])
    return raptor_pb2.SearchResult(**kwargs)


def _fact_to_result(fact: dict) -> raptor_pb2.SearchResult:
    """A dated fact -> SearchResult. A fact is not a chunk of a file, so asset_path and the
    times stay empty: the snippet is the fact in words, version_id says which source it came
    from, and score is module 25's confidence in it."""
    when = fact.get("when") or ""
    said = " ".join(str(fact[k]) for k in ("entity", "relation", "value") if fact.get(k) not in (None, ""))
    return raptor_pb2.SearchResult(
        score=float(fact.get("confidence") or 0.0),
        snippet=f"{said} ({when})" if said and when else said or when,
        version_id=fact.get("source_version_id") or "",
    )


async def graphrag_search(personal_db, request, user) -> raptor_pb2.GraphSearchResponse:
    query = _validate_graph_request(request)
    if request.strategy and request.strategy not in _STRATEGIES:
        raise ValidationError(f"strategy must be one of {', '.join(_STRATEGIES)}.")
    body: dict = {"query": query}
    if request.max_depth:
        body["max_depth"] = request.max_depth
    if request.strategy:
        body["strategy"] = request.strategy
    if request.HasField("score_threshold"):
        body["score_threshold"] = request.score_threshold
    user_id = user["user_id"]
    await personal_db.ensure_database(user_id)
    data = await personal_db.graphrag_search(user_id, body)

    moments = data.get("matched_moments") or []
    if request.top_k:
        moments = moments[: request.top_k]
    subgraph = {
        "matched_entities": data.get("matched_entities") or [],
        "nodes": data.get("nodes") or [],
        "edges": data.get("relationships") or [],
        "citations": data.get("citations") or [],
    }
    return raptor_pb2.GraphSearchResponse(
        results=[_moment_to_result(m) for m in moments], subgraph_json=_subgraph_json(subgraph)
    )


async def tkg_search(personal_db, request, user) -> raptor_pb2.GraphSearchResponse:
    query = _validate_graph_request(request)
    body: dict = {"query": query}
    if request.max_depth:
        body["max_depth"] = request.max_depth
    if request.top_k:
        body["limit"] = request.top_k
    if request.HasField("score_threshold"):
        body["score_threshold"] = request.score_threshold
    for name in ("time_start", "time_end", "entity"):
        value = getattr(request, name).strip()
        if value:
            body[name] = value
    user_id = user["user_id"]
    await personal_db.ensure_database(user_id)
    data = await personal_db.temporal_search(user_id, body)

    subgraph = {
        "matched_entities": data.get("matched_entities") or [],
        "nodes": data.get("subgraph_nodes") or [],
        "edges": data.get("subgraph_edges") or [],
        "moments": data.get("moment_ids") or [],
    }
    return raptor_pb2.GraphSearchResponse(
        results=[_fact_to_result(f) for f in data.get("temporal_facts") or []],
        subgraph_json=_subgraph_json(subgraph),
    )


def _subgraph_json(subgraph: dict) -> str:
    if not any(subgraph.values()):
        return "{}"
    return json.dumps(subgraph, ensure_ascii=False, default=str)
