"""Per-user search endpoints (PA-6) — hybrid / vector / bm25, and people (dev plan v1.4 §4.6).

Scoped to the caller's ArcadeDB database via the X-Branch-ID header.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException

from app.models.identity import (
    PeopleInVideosRequest,
    PeopleInVideosResponse,
    PeopleSearchRequest,
    PeopleSearchResponse,
)
from app.models.search import (
    RerankRequest,
    RerankResponse,
    SearchRequest,
    SearchResponse,
)
from app.services import identity_store, reranker, searcher
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import DatabaseNotInitializedError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/personal/search", tags=["search"])
client = ArcadeDBClient()


def _require_branch(x_branch_id: str | None) -> str:
    if not x_branch_id:
        raise HTTPException(status_code=400, detail="Missing X-Branch-ID header")
    return x_branch_id


async def _run(fn, x_branch_id, req):
    branch_id = _require_branch(x_branch_id)
    try:
        return await fn(client, branch_id, req)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- wraps searcher/reranker/ArcadeDB calls; converts any failure to 500, logged
        logger.error(f"search endpoint failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/hybrid", response_model=SearchResponse, summary="Dense + BM25 fused with RRF")
async def hybrid(req: SearchRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _run(searcher.hybrid_search, x_branch_id, req)


@router.post("/vector", response_model=SearchResponse, summary="Dense vector search only")
async def vector(req: SearchRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _run(searcher.vector_search, x_branch_id, req)


@router.post("/bm25", response_model=SearchResponse, summary="BM25 full-text only")
async def bm25(req: SearchRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _run(searcher.bm25_search, x_branch_id, req)


@router.post("/rerank", response_model=RerankResponse,
            summary="Rerank arbitrary (id, text) pairs -- not scoped to a user's database")
async def rerank(req: RerankRequest) -> RerankResponse:
    """No X-Branch-ID: this is a stateless cross-encoder call over whatever
    the caller supplies, same contract as Module 17's /api/v1/search/rerank."""
    results = await reranker.rerank_documents(req.query, req.documents, req.top_k)
    return RerankResponse(results=results)


@router.post("/people", response_model=PeopleSearchResponse,
             summary="Find people by name or alias, with every segment they appear in")
async def people(req: PeopleSearchRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    """Only videos whose Source is active are returned, the same rule as content search.
    Fail-open: a user whose database isn't initialized has no people -- an empty result, not 404."""
    async def search(client, branch_id, req):
        try:
            return await identity_store.search_people(client, branch_id, req)
        except DatabaseNotInitializedError:
            return PeopleSearchResponse(query=req.query, identities=[], total=0, results=[])
    return await _run(search, x_branch_id, req)


@router.post("/people/in-videos", response_model=PeopleInVideosResponse,
             summary="Every trace for these identities, restricted to these videos")
async def people_in_videos(req: PeopleInVideosRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    """Scoped counterpart to /people: no limit, since the result is already bounded by
    (identities) x (videos) -- see identity_store.people_in_videos(). Fail-open same as
    /people: a user whose database isn't initialized has no people, not a 404."""
    async def search(client, branch_id, req):
        try:
            return await identity_store.people_in_videos(client, branch_id, req.identity_ids, req.version_ids)
        except DatabaseNotInitializedError:
            return PeopleInVideosResponse(results=[])
    return await _run(search, x_branch_id, req)
