"""Search router — `/api/{version}/search/*`.

Backed by module 25 (personal-db, per-user isolated ArcadeDB) as of the 17/20
retirement. Endpoint paths and request/response shapes are kept exactly as
they were under module 17/20 -- only the implementation underneath changed --
so no caller (module 27, or anything else hitting this gateway) needs to
change what it calls. Verified field-for-field against
`app/routers/personal_db.py`'s already-shipped `/personal-db/search/*`
surface before this cutover (that surface's routes are now commented out
below/in personal_db.py, having become exact duplicates of this file).

The one real behavioral gap found during verification: module 17's
`payload_schema` (`contextual` vs `temporal`) actually switched filter/BM25/
rerank-text-extraction logic; module 25's equivalent field is accepted but a
no-op. Confirmed with the user that nothing depends on this -- the field is
dropped here, not forwarded.

Old module 17/20-backed implementation kept commented out below each
function, not deleted.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.personal_db_service import PersonalDBService
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Dependencies
# ============================================================================

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
    """Kept for `enrich_with_urls`/`resolve_version_urls` -- pure URL-resolution
    helpers with no dependency on module 17 being alive."""
    return HybridSearchService(
        hybrid_search_url=settings.hybrid_search_url,
        http_client=client,
        storage_service=storage_svc,
    )


# def get_search_service(
#     client: httpx.AsyncClient = Depends(get_http_client),
#     settings: Settings = Depends(get_settings),
#     storage_svc: StorageService = Depends(get_storage_service),
# ) -> HybridSearchService:
#     return HybridSearchService(
#         hybrid_search_url=settings.hybrid_search_url,
#         http_client=client,
#         storage_service=storage_svc,
#     )


def _user_dict(user_id: str) -> Dict[str, str]:
    return {"user_id": user_id, "branch_id": user_id}


# ============================================================================
# Request model (module 25's personal-db equivalent -- no `payload_schema`,
# see module docstring)
# ============================================================================

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text or natural language question")
    top_k: Optional[int] = Field(10, description="Maximum number of results to return")
    embedding_type: Optional[str] = Field(None, description="Embedding type: `text` (raw content) or `summary`; omit to search both")
    type: Optional[Union[str, List[str]]] = Field(None, description="Filter by media type: `document` / `image` / `video` / `audio`; accepts a list for multiple types")
    filename: Optional[List[str]] = Field(None, description="Filter by filename; accepts multiple values")
    speaker: Optional[List[str]] = Field(None, description="Filter by speaker name (applicable to audio/video segments)")
    source: Optional[str] = Field(None, description="Filter by original file extension, e.g. `pdf`, `csv`, `mp4`")

# class HybridSearchRequest(BaseModel):
#     query: str = Field(..., description="Search query text or natural language question")
#     top_k: Optional[int] = Field(10, description="Maximum number of results to return")
#     payload_schema: str = Field("contextual", description="Result schema: `contextual` (general documents) or `temporal` (audio/video segments with timestamps)")
#     embedding_type: Optional[str] = Field(None, description="Embedding type: `text` (raw content) or `summary`; omit to search both")
#     type: Optional[Union[str, List[str]]] = Field(None, description="Filter by media type: `document` / `image` / `video` / `audio`; accepts a list for multiple types")
#     filename: Optional[List[str]] = Field(None, description="Filter by filename; accepts multiple values")
#     speaker: Optional[List[str]] = Field(None, description="Filter by speaker name (applicable to audio/video segments)")
#     source: Optional[str] = Field(None, description="Filter by original file extension, e.g. `pdf`, `csv`, `mp4`")


# ============================================================================
# Search endpoints
# ============================================================================

@router.post(
    "/hybrid",
    tags=["Search"],
    status_code=status.HTTP_200_OK,
    summary="Hybrid Search (BM25 + Vector + Rerank)",
    description="""
Runs BM25 keyword search and vector semantic search in parallel, merges the ranked lists with Reciprocal Rank Fusion (RRF), then re-ranks the top candidates with a cross-encoder reranker.

Covers all uploaded and processed **document / image / video / audio** assets.

**Response fields:**
- `results[]` — each result includes a relevance score, payload (content, filename, timestamps), and a pre-signed access URL
- `collection_stats` — hit counts per collection
""",
)
async def hybrid_search(
    req: HybridSearchRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    search_svc: HybridSearchService = Depends(get_hybrid_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Unified hybrid search (BM25 + Vector + Rerank), scoped to the caller's
    own personal database (module 25). Reranked inside module 25 itself."""
    user_id = current_user["sub"]
    try:
        await svc.ensure_database(user_id)
        data = await svc.hybrid_search(user_id, req.model_dump(exclude_none=True))
        payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
        await search_svc.enrich_with_urls(payloads, _user_dict(user_id))
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"Hybrid search failed: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Hybrid search failed: {e.response.text}")
    except Exception as e:
        _logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# async def hybrid_search(
#     req: HybridSearchRequest,
#     svc: HybridSearchService = Depends(get_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Unified hybrid search (BM25 + Vector + Rerank) via module 17."""
#     user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
#     try:
#         return await svc.hybrid_search(req.model_dump(exclude_none=True), user=user)
#     except httpx.HTTPStatusError as e:
#         _logger.error(f"Hybrid search failed: {e.response.status_code} {e.response.text}")
#         raise HTTPException(status_code=e.response.status_code, detail=f"Hybrid search failed: {e.response.text}")
#     except Exception as e:
#         _logger.error(f"Hybrid search error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/bm25",
    tags=["Search"],
    status_code=status.HTTP_200_OK,
    summary="BM25 Keyword Search",
    description="Full-text keyword search using the BM25 algorithm. Best for exact term matching; no semantic inference.",
)
async def bm25_search(
    req: HybridSearchRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    search_svc: HybridSearchService = Depends(get_hybrid_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """BM25-only search, scoped to the caller's own personal database (module
    25). No rerank -- matches module 17's old /bm25, also un-reranked."""
    user_id = current_user["sub"]
    try:
        await svc.ensure_database(user_id)
        data = await svc.bm25_search(user_id, req.model_dump(exclude_none=True))
        payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
        await search_svc.enrich_with_urls(payloads, _user_dict(user_id))
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"BM25 search failed: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=f"BM25 search failed: {e.response.text}")
    except Exception as e:
        _logger.error(f"BM25 search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# async def bm25_search(
#     req: HybridSearchRequest,
#     svc: HybridSearchService = Depends(get_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """BM25-only search via module 17 (OpenSearch)."""
#     user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
#     try:
#         return await svc.bm25_search(req.model_dump(exclude_none=True), user=user)
#     except httpx.HTTPStatusError as e:
#         _logger.error(f"BM25 search failed: {e.response.status_code}")
#         raise HTTPException(status_code=e.response.status_code, detail=f"BM25 search failed: {e.response.text}")
#     except Exception as e:
#         _logger.error(f"BM25 search error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector",
    tags=["Search"],
    status_code=status.HTTP_200_OK,
    summary="Vector Semantic Search",
    description="Semantic similarity search using embedding vectors. Finds relevant content even when the exact keywords are not present.",
)
async def vector_search(
    req: HybridSearchRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    search_svc: HybridSearchService = Depends(get_hybrid_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Vector-only search, scoped to the caller's own personal database
    (module 25). No rerank -- matches module 17's old /vector, also
    un-reranked."""
    user_id = current_user["sub"]
    try:
        await svc.ensure_database(user_id)
        data = await svc.vector_search(user_id, req.model_dump(exclude_none=True))
        payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
        await search_svc.enrich_with_urls(payloads, _user_dict(user_id))
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"Vector search failed: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Vector search failed: {e.response.text}")
    except Exception as e:
        _logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# async def vector_search(
#     req: HybridSearchRequest,
#     svc: HybridSearchService = Depends(get_search_service),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """Vector-only search via module 17 (Qdrant)."""
#     user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
#     try:
#         return await svc.vector_search(req.model_dump(exclude_none=True), user=user)
#     except httpx.HTTPStatusError as e:
#         _logger.error(f"Vector search failed: {e.response.status_code}")
#         raise HTTPException(status_code=e.response.status_code, detail=f"Vector search failed: {e.response.text}")
#     except Exception as e:
#         _logger.error(f"Vector search error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TKG / GraphRAG search (module 20)
# ============================================================================

class TkgSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query text")
    time_start: Optional[str] = Field(None, description="Start of time range (ISO 8601, e.g. `2024-01-01T00:00:00`)")
    time_end:   Optional[str] = Field(None, description="End of time range (ISO 8601)")
    max_depth:  int = Field(2, ge=1, le=4, description="Graph traversal depth (1–4)")
    limit:      int = Field(50, ge=1, le=200, description="Maximum number of nodes/events to return")
    score_threshold: float = Field(
        0.5, ge=0.0, le=10.0,
        description="Minimum relevance score",
    )
    # Was 0.3 -- copied from module 20's own TkgQueryRequest json_schema_extra
    # *example*, not its actual Field default, which is 0.5. Fixed to agree
    # with what module 20 actually defaults to when this field is omitted.


@router.post(
    "/tkg",
    tags=["Search"],
    status_code=status.HTTP_200_OK,
    summary="Temporal Knowledge Graph Search (TKG)",
    description="""
Graph traversal query over a Temporal Knowledge Graph (TKG).

**Best for:**
- Querying what happened at a specific time in a video or audio recording
- Event queries that require a time range filter

**Response fields:**
- `subgraph_nodes` / `subgraph_edges` — relevant graph nodes and edges
- `temporal_facts` — timestamped event records
- `moment_ids` — corresponding audio/video segment IDs (with pre-signed access URLs)
""",
)
async def tkg_search(
    req: TkgSearchRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    search_svc: HybridSearchService = Depends(get_hybrid_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Temporal knowledge-graph query, scoped to the caller's own personal
    database (module 25 calls this route `/tkg` internally too, just not on
    this gateway's public path)."""
    user_id = current_user["sub"]
    try:
        await svc.ensure_database(user_id)
        data = await svc.temporal_search(user_id, req.model_dump(exclude_none=True))
        await search_svc.enrich_with_urls(data.get("moment_ids", []), _user_dict(user_id))
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"TKG search failed: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"TKG search failed: {e.response.text}")
    except Exception as e:
        _logger.error(f"TKG search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# async def tkg_search(
#     req: TkgSearchRequest,
#     svc: HybridSearchService = Depends(get_search_service),
#     client: httpx.AsyncClient = Depends(get_http_client),
#     settings: Settings = Depends(get_settings),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """TKG + GraphRAG combined search via module 20 (graph-service)."""
#     try:
#         user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
#         body = req.model_dump(exclude_none=True)
#         body["branch_id"] = current_user["sub"]
#         resp = await client.post(
#             f"{settings.graph_service_url}/tkg/query",
#             json=body,
#             timeout=30.0,
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         await svc.enrich_with_urls(data.get("moment_ids", []), user)
#         return data
#     except httpx.HTTPStatusError as e:
#         _logger.error(f"TKG search failed: {e.response.status_code} {e.response.text}")
#         raise HTTPException(status_code=e.response.status_code, detail=f"TKG search failed: {e.response.text}")
#     except Exception as e:
#         _logger.error(f"TKG search error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GraphRAG search (module 20)
# ============================================================================

class GraphRagSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query text")
    max_depth: int = Field(2, ge=1, le=4, description="Graph traversal depth (1–4)")
    limit: int = Field(50, ge=1, le=200, description="Maximum number of entities/nodes to return")
    score_threshold: float = Field(0.5, ge=0.0, le=10.0, description="Minimum relevance score")
    strategy: Literal["hybrid", "literal", "semantic"] = Field(
        "hybrid",
        description="`hybrid`: keyword + semantic; `literal`: keyword only; `semantic`: semantic only",
    )


@router.post(
    "/graphrag",
    tags=["Search"],
    status_code=status.HTTP_200_OK,
    summary="Knowledge Graph Entity Search (GraphRAG)",
    description="""
Entity and subgraph search over a structured knowledge graph.

**Best for:**
- Finding people, events, or locations mentioned in documents or media
- Complex queries that require understanding relationships between entities

**Response fields:**
- `matched_entities` — matched knowledge graph entities
- `matched_moments` — associated audio/video segments (with pre-signed URLs)
- `nodes` / `relationships` — subgraph structure
- `citations` — source references
""",
)
async def graphrag_search(
    req: GraphRagSearchRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    search_svc: HybridSearchService = Depends(get_hybrid_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """GraphRAG entity/subgraph search, scoped to the caller's own personal
    database (module 25)."""
    user_id = current_user["sub"]
    try:
        await svc.ensure_database(user_id)
        data = await svc.graphrag_search(user_id, req.model_dump(exclude_none=True))
        await search_svc.enrich_with_urls(
            data.get("matched_moments", []) + data.get("moment_ids", []),
            _user_dict(user_id),
        )
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"GraphRAG search failed: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"GraphRAG search failed: {e.response.text}")
    except Exception as e:
        _logger.error(f"GraphRAG search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# async def graphrag_search(
#     req: GraphRagSearchRequest,
#     svc: HybridSearchService = Depends(get_search_service),
#     client: httpx.AsyncClient = Depends(get_http_client),
#     settings: Settings = Depends(get_settings),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Dict[str, Any]:
#     """GraphRAG entity/subgraph search via module 20 (graph-service)."""
#     try:
#         user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
#         body = req.model_dump(exclude_none=True)
#         body["branch_id"] = current_user["sub"]
#         resp = await client.post(
#             f"{settings.graph_service_url}/query/graph_rag",
#             json=body,
#             timeout=30.0,
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         await svc.enrich_with_urls(
#             data.get("matched_moments", []) + data.get("moment_ids", []),
#             user,
#         )
#         return data
#     except httpx.HTTPStatusError as e:
#         _logger.error(f"GraphRAG search failed: {e.response.status_code} {e.response.text}")
#         raise HTTPException(status_code=e.response.status_code, detail=f"GraphRAG search failed: {e.response.text}")
#     except Exception as e:
#         _logger.error(f"GraphRAG search error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
