"""Per-user search endpoints (PA-6) — hybrid / vector / bm25.

Scoped to the caller's ArcadeDB database via the X-Branch-ID header.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from app.models.search import RerankRequest, RerankResponse, SearchRequest, SearchResponse
from app.services import reranker, searcher
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import DatabaseNotInitializedError

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
    except Exception as exc:  # noqa: BLE001
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
