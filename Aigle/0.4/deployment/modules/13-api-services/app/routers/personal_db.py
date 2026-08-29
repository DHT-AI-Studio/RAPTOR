"""Personal DB gateway routes — `/api/0.4/personal-db/*` (VIE01-191).

The public face of module 25 for internal/admin use. The six search routes
this module originally added (hybrid/bm25/vector/temporal/graphrag/
video_search) are commented out below, not deleted -- once `search.py` and
`video_search.py` were cut over to call module 25 directly under the old
0.3-style `/api/{version}/search/*` paths, these became exact duplicates.
Old naming is the one surviving public search surface; this file's job now
is just: (1) module 25's proxy service (`PersonalDBService`) that both this
file and `search.py`/`video_search.py` depend on, and (2) `/search/graph`,
`/status`, `DELETE /`, kept commented out below for the reasons already
noted there -- not something an end user needs, module 25's own endpoints
stay available directly for internal/admin use.

Every route still active here is **authenticated** (`get_current_user` runs
as a router-level dependency, so an unauthenticated request is rejected with
401 before any handler executes) and only ever acts on the caller's own
database — the JWT `sub` is the sole source of identity throughout, there is
no `user_id` field anywhere on these routes for a caller to (mis)supply.

The database is provisioned on the caller's first request to any of these routes
— the client never sees an init step.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_settings, get_storage_service
from app.core.config import Settings
from app.routers.video_search import (
    _RERANK_CANDIDATES, _format_graph_text, _rrf_add, _seg_key,
    Segment, VideoResult, VideoSearchResponse,
)
from app.services.personal_db_service import PersonalDBService
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["Personal DB"])


def get_personal_db_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> PersonalDBService:
    return PersonalDBService(client=client, settings=settings)


def get_hybrid_search_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage_svc: StorageService = Depends(get_storage_service),
) -> HybridSearchService:
    return HybridSearchService(
        hybrid_search_url=settings.hybrid_search_url,
        http_client=client,
        storage_service=storage_svc,
    )


def _caller_id(current_user: Dict[str, Any], claimed_user_id: Optional[str]) -> str:
    """The authenticated `sub`, after refusing any attempt to name someone else.

    An explicit 403 rather than silently substituting the caller's own id: a
    client that asks for another user's data has a bug or worse, and quietly
    returning its own data would hide both.
    """
    sub = current_user.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has no subject")
    if claimed_user_id and claimed_user_id != sub:
        _logger.warning("cross-user personal-db request: %s tried to access %s", sub, claimed_user_id)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Cannot access another user's personal database")
    return sub


class PersonalSearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language query.")
    top_k: int = Field(10, ge=1, le=100)
    # No user_id field: the database is always the caller's own JWT sub —
    # this field used to exist as a claim the caller could optionally repeat
    # (rejected with 403 if it didn't match), but it could never actually grant
    # access to anything, only cause a caller's own request to fail on a typo.
    # Explicit, typed filters — matches app.routers.search.HybridSearchRequest's
    # shape so the two search surfaces are structurally interchangeable.
    type: Optional[Union[str, List[str]]] = Field(None, description="Filter by media type.")
    embedding_type: Optional[str] = Field(None, description="`text` or `summary`.")
    filename: Optional[List[str]] = Field(None, description="Filter by filename; accepts multiple values.")
    speaker: Optional[List[str]] = Field(None, description="Filter by speaker name.")
    source: Optional[str] = Field(None, description="Filter by original file extension, e.g. `pdf`, `mp4`.")


class PersonalGraphSearchRequest(BaseModel):
    """Distinct from PersonalSearchRequest -- module 25's graph_search takes a
    seed entity name and a hop count, not a free-text query + media filters
    (this used to reuse PersonalSearchRequest wholesale, which has no
    entity_name field at all; every call here 422'd on module 25's side)."""
    entity_name: str = Field(..., description="Seed entity to traverse RELATION edges from.")
    max_depth: int = Field(2, ge=1, le=5, description="RELATION hops from the seed entity.")


class PersonalTemporalSearchRequest(BaseModel):
    """Batch 5 of the graph/TKG parity plan -- module 25's TKGRequest moved
    from an entity_name exact-match filter to a natural-language query (entity
    fulltext search -> subgraph expansion -> time-windowed TemporalFacts),
    matching Module 20's TkgQueryRequest shape. Updated here in the same
    change as module 25's, not deferred to a later gateway-only batch --
    leaving this on the old entity_name shape after module 25's request model
    changed would have made /search/temporal 422 on every call, same failure
    mode Batch 1 fixed for /search/graph."""
    query: str = Field(..., description="Natural-language query.")
    time_start: Optional[str] = Field(None, description="ISO 8601 inclusive lower bound.")
    time_end: Optional[str] = Field(None, description="ISO 8601 inclusive upper bound.")
    max_depth: int = Field(2, ge=1, le=4, description="Subgraph hops from each matched entity.")
    limit: int = Field(50, ge=1, le=200)
    score_threshold: float = Field(
        0.5, ge=0.0, le=10.0,
        description=(
            "Minimum BM25 relevance score to keep (ArcadeDB's own $score, same idea as Neo4j's "
            "Lucene score on /api/0.4/search/tkg). Default 0.5 matches /api/0.4/search/tkg's "
            "own default and module 20's -- see graph_query.py."
        ),
    )


class PersonalGraphRAGSearchRequest(BaseModel):
    """Batch 6 of the graph/TKG/GraphRAG parity plan -- module 25 has had
    /personal/search/graphrag since Batch 4 (entity/moment fulltext search ->
    subgraph expansion -> citations); this gateway route never existed at
    all (404), unlike /search/graph and /search/temporal which existed but
    were wired wrong. Matches module 25's GraphRAGRequest shape."""
    query: str = Field(..., description="Natural-language query.")
    max_depth: int = Field(2, ge=1, le=4, description="Subgraph hops from each matched entity.")
    limit: int = Field(50, ge=1, le=200, description="Node cap per matched entity's subgraph.")
    strategy: Literal["literal", "semantic", "hybrid"] = Field(
        "hybrid",
        description=("'literal' = keep moment hits in ASR/visual-description text only; "
                     "'semantic' = keep hits in contextual text only; "
                     "'hybrid' = no filter (default)."),
    )
    score_threshold: float = Field(
        0.5, ge=0.0, le=10.0,
        description=(
            "Minimum BM25 relevance score to keep (ArcadeDB's own $score, same idea as Neo4j's "
            "Lucene score on /api/0.4/search/graphrag). Default 0.5 matches both /search/graphrag "
            "and module 20's own internal default -- see graph_query.py."
        ),
    )


def _user_dict(user_id: str) -> Dict[str, str]:
    return {"user_id": user_id, "branch_id": user_id}


# ── search ────────────────────────────────────────────────────────────────────
# Commented out, not deleted -- superseded by app/routers/search.py's
# /api/{version}/search/{hybrid,bm25,vector,tkg,graphrag} and
# app/routers/video_search.py's /search/video_search, both of which now call
# PersonalDBService (module 25) directly under the old 0.3-style path names.
# Two live surfaces doing the same thing under different names was the thing
# being retired here. PersonalSearchRequest/PersonalTemporalSearchRequest/
# PersonalGraphRAGSearchRequest are left uncommented even though nothing
# references them now (harmless, kept for anyone diffing against module 25's
# own request shapes); PersonalVideoSearchRequest is commented out along
# with its route below since it isn't used anywhere else.

# @router.post("/search/hybrid", summary="Hybrid search (dense + BM25, fused with RRF, reranked)")
# async def personal_hybrid_search(
#     req: PersonalSearchRequest,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     search_svc: HybridSearchService = Depends(get_hybrid_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Search the caller's own indexed content. Never sees another user's data —
#     each user has a physically separate ArcadeDB database. Reranked by module
#     25 itself (see module docstring) before this ever sees the results."""
#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     data = await svc.hybrid_search(user_id, req.model_dump(exclude_none=True))
#     payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
#     await search_svc.enrich_with_urls(payloads, _user_dict(user_id))
#     return data


# @router.post("/search/bm25", summary="BM25 keyword search")
# async def personal_bm25_search(
#     req: PersonalSearchRequest,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     search_svc: HybridSearchService = Depends(get_hybrid_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """BM25-only search over the caller's own indexed content. No rerank —
#     matches module 17's /bm25, which is also un-reranked."""
#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     data = await svc.bm25_search(user_id, req.model_dump(exclude_none=True))
#     payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
#     await search_svc.enrich_with_urls(payloads, _user_dict(user_id))
#     return data


# @router.post("/search/vector", summary="Vector semantic search")
# async def personal_vector_search(
#     req: PersonalSearchRequest,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     search_svc: HybridSearchService = Depends(get_hybrid_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Vector-only search over the caller's own indexed content. No rerank —
#     matches module 17's /vector, which is also un-reranked."""
#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     data = await svc.vector_search(user_id, req.model_dump(exclude_none=True))
#     payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
#     await search_svc.enrich_with_urls(payloads, _user_dict(user_id))
#     return data


# /search/graph is commented out, not deleted -- module 25's own endpoint and
# PersonalDBService.graph_search() are untouched. Not something an end user
# needs directly: /search/graphrag (below) already does everything this did
# (entity lookup + subgraph expansion) plus moment fulltext search and
# citations, so this narrower route is redundant on the gateway surface.
# @router.post("/search/graph", summary="Graph traversal over the caller's entities")
# async def personal_graph_search(
#     req: PersonalGraphSearchRequest,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     return await svc.graph_search(user_id, req.model_dump(exclude_none=True))


# @router.post("/search/temporal", summary="Temporal knowledge-graph query (timeline of facts)")
# async def personal_temporal_search(
#     req: PersonalTemporalSearchRequest,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     search_svc: HybridSearchService = Depends(get_hybrid_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Enriches moment_ids with asset_url, matching /api/0.4/search/tkg -- this
#     was missing entirely until now (silent gap, not a 404/422/500, so nothing
#     surfaced it until output was compared field-by-field against the old
#     endpoint with real data)."""
#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     data = await svc.temporal_search(user_id, req.model_dump(exclude_none=True))
#     await search_svc.enrich_with_urls(data.get("moment_ids", []), _user_dict(user_id))
#     return data


# @router.post("/search/graphrag", summary="GraphRAG (entity/moment search -> subgraph -> citations)")
# async def personal_graphrag_search(
#     req: PersonalGraphRAGSearchRequest,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     search_svc: HybridSearchService = Depends(get_hybrid_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Enriches matched_moments + moment_ids with asset_url, matching
#     /api/0.4/search/graphrag -- same missing-enrichment gap as
#     personal_temporal_search above."""
#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     data = await svc.graphrag_search(user_id, req.model_dump(exclude_none=True))
#     await search_svc.enrich_with_urls(
#         data.get("matched_moments", []) + data.get("moment_ids", []), _user_dict(user_id))
#     return data


# class PersonalVideoSearchRequest(BaseModel):
#     """Personal-db equivalent of video_search.py's VideoSearchRequest --
#     same fields/defaults/bounds, see that file for the score_threshold
#     rationale (0.52 assumes the sigmoid-normalised rerank score)."""
#     query: str = Field(..., description="Natural language search query")
#     top_k: int = Field(10, ge=1, le=50, description="Number of videos to return")
#     candidate_multiplier: int = Field(5, ge=1, le=20,
#         description="Each retriever fetches top_k × candidate_multiplier segments as candidates before RRF")
#     score_threshold: float = Field(0.52, ge=0.0, le=1.0,
#         description="Minimum rerank score (sigmoid-normalised) for a segment to appear in results")


# @router.post("/search/video_search", response_model=VideoSearchResponse,
#             summary="Video search over the caller's own database (4-way RRF + rerank)")
# async def personal_video_search(
#     req: PersonalVideoSearchRequest,
#     response: Response,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     search_svc: HybridSearchService = Depends(get_hybrid_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> VideoSearchResponse:
#     """Personal-db equivalent of video_search.py -- same 4-way RRF (BM25 +
#     Vector + GraphRAG + TKG) -> cross-encoder rerank -> score_threshold
#     filter -> group-by-video pipeline, verbatim (imported, not copied) for
#     the RRF pool/grouping logic, since that part is pure Python with no
#     dependency on which backend the payloads came from. The two things that
#     actually change:

#     - the 4-way fan-out calls PersonalDBService (module 25, the caller's own
#       isolated database) instead of HybridSearchService + a direct call to
#       module 20's graph-service
#     - the rerank step calls module 25's own POST /personal/search/rerank
#       (new -- see reranker.py's rerank_documents()) instead of module 17's
#       /api/v1/search/rerank

#     Every field _rrf_add()/_format_graph_text() reads (version_id, start_sec/
#     start_time, asr_text, contextual_text, lvlm_description, score, id) was
#     confirmed present on module 25's bm25/vector/graphrag/temporal responses
#     by the shape-parity work this surface already went through -- this
#     endpoint is what that parity work was actually for.
#     """
#     import time as _time
#     _t0 = _time.perf_counter()

#     user_id = _caller_id(current_user, None)
#     await svc.ensure_database(user_id)
#     user = _user_dict(user_id)
#     candidate_k = req.top_k * req.candidate_multiplier
#     graph_limit = min(candidate_k, 200)  # module 25's TKGRequest/GraphRAGRequest limit cap

#     # ── Parallel fan-out to all 4 retrievers, all scoped to the caller's own database ──
#     bm25_coro = svc.bm25_search(
#         user_id, {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"})
#     vector_coro = svc.vector_search(
#         user_id, {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"})
#     graphrag_coro = svc.graphrag_search(
#         user_id, {"query": req.query, "max_depth": 2, "limit": graph_limit, "score_threshold": 0.3})
#     tkg_coro = svc.temporal_search(
#         user_id, {"query": req.query, "max_depth": 2, "limit": graph_limit, "score_threshold": 0.3})

#     bm25_res, vector_res, graphrag_res, tkg_res = await asyncio.gather(
#         bm25_coro, vector_coro, graphrag_coro, tkg_coro, return_exceptions=True)
#     _t_fanout = _time.perf_counter()
#     pool: Dict[str, dict] = {}
#     video_meta: Dict[str, dict] = {}

#     # ── 1. BM25 ──────────────────────────────────────────────────────────────
#     if not isinstance(bm25_res, Exception):
#         for rank, hit in enumerate(bm25_res.get("results", [])):
#             p = hit.get("payload", {})
#             vid_id = p.get("version_id") or hit.get("id", "")
#             if not vid_id:
#                 continue
#             _rrf_add(
#                 pool, video_meta, video_id=vid_id,
#                 start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
#                 end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
#                 rank=rank, text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
#                 source_tag="bm25", asset_path=p.get("asset_path"),
#                 filename=p.get("filename"), upload_time=p.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"personal-db BM25 search failed: {bm25_res}")

#     # ── 2. Vector ────────────────────────────────────────────────────────────
#     if not isinstance(vector_res, Exception):
#         for rank, hit in enumerate(vector_res.get("results", [])):
#             p = hit.get("payload", {})
#             vid_id = p.get("version_id") or hit.get("id", "")
#             if not vid_id:
#                 continue
#             _rrf_add(
#                 pool, video_meta, video_id=vid_id,
#                 start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
#                 end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
#                 rank=rank, text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
#                 source_tag="vector", asset_path=p.get("asset_path"),
#                 filename=p.get("filename"), upload_time=p.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"personal-db vector search failed: {vector_res}")

#     # ── 3. GraphRAG ──────────────────────────────────────────────────────────
#     if not isinstance(graphrag_res, Exception):
#         gr = graphrag_res
#         matched = sorted(gr.get("matched_moments", []), key=lambda m: m.get("score", 0.0), reverse=True)
#         for rank, m in enumerate(matched):
#             src = m.get("version_id", "")
#             if not src:
#                 continue
#             _rrf_add(
#                 pool, video_meta, video_id=src, start_sec=m.get("start_sec"), end_sec=m.get("end_sec"),
#                 rank=rank, text=m.get("asr_text") or m.get("contextual_text") or "",
#                 source_tag="graphrag", asset_path=m.get("asset_path"),
#                 filename=m.get("filename"), upload_time=m.get("upload_time"),
#             )

#         subgraph_rank = len(matched)
#         matched_ids = {m.get("id") or m.get("moment_id") for m in matched}
#         for m in gr.get("moment_ids", []):
#             src = m.get("version_id", "")
#             mid = m.get("moment_id", "")
#             if not src or mid in matched_ids:
#                 continue
#             _rrf_add(
#                 pool, video_meta, video_id=src, start_sec=m.get("start_sec"), end_sec=m.get("end_sec"),
#                 rank=subgraph_rank, text=_format_graph_text(m), source_tag="graphrag",
#                 asset_path=m.get("asset_path"), filename=m.get("filename"), upload_time=m.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"personal-db GraphRAG search failed: {graphrag_res}")

#     # ── 4. TKG ───────────────────────────────────────────────────────────────
#     if not isinstance(tkg_res, Exception):
#         for m in tkg_res.get("moment_ids", []):
#             src = m.get("version_id", "")
#             if not src:
#                 continue
#             _rrf_add(
#                 pool, video_meta, video_id=src, start_sec=m.get("start_sec"), end_sec=m.get("end_sec"),
#                 rank=0, text=_format_graph_text(m), source_tag="tkg",
#                 asset_path=m.get("asset_path"), filename=m.get("filename"), upload_time=m.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"personal-db TKG search failed: {tkg_res}")

#     # ── Rerank top candidates with module 25's own cross-encoder ─────────────
#     candidates = sorted(pool.values(), key=lambda s: s["score"], reverse=True)[:_RERANK_CANDIDATES]
#     if candidates:
#         try:
#             rerank_docs = [
#                 {"id": _seg_key(s["video_id"], s["start_time"]), "text": s["text"]}
#                 for s in candidates if s.get("text")
#             ]
#             for s in candidates:
#                 if not s.get("text"):
#                     pool[_seg_key(s["video_id"], s["start_time"])]["score"] = 0.0

#             if rerank_docs:
#                 rerank_data = await svc.rerank(user_id, {"query": req.query, "documents": rerank_docs})
#                 for item in rerank_data.get("results", []):
#                     if item["id"] in pool:
#                         pool[item["id"]]["score"] = item["score"]
#         except Exception as exc:
#             _logger.warning(f"personal-db rerank failed, keeping RRF scores: {exc}")
#     _t_rerank = _time.perf_counter()

#     # ── URL resolution ─────────────────────────────────────────────────────
#     version_map = {
#         vid_id: (meta["asset_path"], meta.get("filename"))
#         for vid_id, meta in video_meta.items() if meta.get("asset_path")
#     }
#     url_map = await search_svc.resolve_version_urls(version_map, user)
#     _t_url = _time.perf_counter()

#     # ── Group segments by video ───────────────────────────────────────────────
#     videos: Dict[str, dict] = {}
#     for seg in candidates:
#         if seg["score"] < req.score_threshold:
#             continue
#         vid_id = seg["video_id"]
#         if vid_id not in videos:
#             meta = video_meta.get(vid_id, {})
#             videos[vid_id] = {
#                 "video_id": vid_id, "filename": meta.get("filename"),
#                 "asset_url": url_map.get(vid_id), "upload_time": meta.get("upload_time"),
#                 "score": 0.0, "segments": [],
#             }
#         videos[vid_id]["segments"].append(seg)
#         if seg["score"] > videos[vid_id]["score"]:
#             videos[vid_id]["score"] = seg["score"]

#     for v in videos.values():
#         v["segments"].sort(key=lambda s: s["score"], reverse=True)

#     ranked = sorted(videos.values(), key=lambda v: v["score"], reverse=True)[: req.top_k]

#     results = [
#         VideoResult(
#             video_id=v["video_id"], filename=v["filename"], score=v["score"],
#             asset_url=v["asset_url"], upload_time=v["upload_time"],
#             segments=[
#                 Segment(start_time=s["start_time"], end_time=s["end_time"], score=s["score"],
#                        text=s["text"] or None, sources=s["sources"])
#                 for s in v["segments"]
#             ],
#         )
#         for v in ranked
#     ]

#     _t_end = _time.perf_counter()
#     ms = lambda a, b: str(round((b - a) * 1000, 1))
#     response.headers["X-Process-Time-Total"] = ms(_t0, _t_end)
#     response.headers["X-Process-Time-Fanout"] = ms(_t0, _t_fanout)
#     response.headers["X-Process-Time-Rerank"] = ms(_t_fanout, _t_rerank)
#     response.headers["X-Process-Time-Url"] = ms(_t_rerank, _t_url)

#     return VideoSearchResponse(query=req.query, total=len(results), results=results)


# ── lifecycle ─────────────────────────────────────────────────────────────────

# /status and DELETE / are commented out, not deleted -- module 25's own
# endpoints and PersonalDBService.status()/delete_database() are untouched.
# Not something an end user needs exposed on the gateway; kept available at
# module 25 directly for internal/admin use.
# @router.get("/status", summary="Whether the caller's database exists, and what it holds")
# async def personal_db_status(
#     user_id: Optional[str] = None,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Answers 200 with `db_exists: false` for a user who has never uploaded —
#     not having a database yet is a normal state, not an error."""
#     return await svc.status(_caller_id(current_user, user_id))
#
#
# @router.delete("/", summary="Delete the caller's personal database and all its contents")
# async def delete_personal_db(
#     user_id: Optional[str] = None,
#     svc: PersonalDBService = Depends(get_personal_db_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Irreversible. Module 25 records the deletion in PostgreSQL before dropping
#     anything, and refuses with 503 if that audit cannot be written — which this
#     route passes through unchanged."""
#     return await svc.delete_database(_caller_id(current_user, user_id))
