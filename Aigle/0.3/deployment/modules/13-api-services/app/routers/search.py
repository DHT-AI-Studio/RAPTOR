"""Search router — delegates to HybridSearchService (module 17)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, Literal

import httpx
from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Dependencies
# ============================================================================

def get_search_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage_svc: StorageService = Depends(get_storage_service),
) -> HybridSearchService:
    return HybridSearchService(
        hybrid_search_url=settings.hybrid_search_url,
        http_client=client,
        storage_service=storage_svc,
    )


# ============================================================================
# Request model (mirrors module 17's SearchRequest)
# ============================================================================

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: Optional[int] = Field(10, description="Maximum number of results")
    payload_schema: str = Field("contextual", description="Schema: 'contextual' or 'temporal'")
    embedding_type: Optional[str] = Field(None, description="'text' or 'summary'")
    type: Optional[Union[str, List[str]]] = Field(None, description="Collection type filter")
    filename: Optional[List[str]] = Field(None, description="Filename filter")
    speaker: Optional[List[str]] = Field(None, description="Speaker filter (audio/video)")
    source: Optional[str] = Field(None, description="File type filter (e.g. pdf, csv)")


# ============================================================================
# Search endpoints
# ============================================================================

@router.post("/hybrid", tags=["Search"], status_code=status.HTTP_200_OK)
async def hybrid_search(
    req: HybridSearchRequest,
    svc: HybridSearchService = Depends(get_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Unified hybrid search (BM25 + Vector + Rerank) via module 17."""
    user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
    try:
        return await svc.hybrid_search(req.model_dump(exclude_none=True), user=user)
    except httpx.HTTPStatusError as e:
        _logger.error(f"Hybrid search failed: {e.response.status_code} {e.response.text}")
        return {"error": "Hybrid search failed", "detail": str(e), "status_code": e.response.status_code}
    except Exception as e:
        _logger.error(f"Hybrid search error: {e}")
        return {"error": "Internal server error", "detail": str(e)}


@router.post("/bm25", tags=["Search"], status_code=status.HTTP_200_OK)
async def bm25_search(
    req: HybridSearchRequest,
    svc: HybridSearchService = Depends(get_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """BM25-only search via module 17 (OpenSearch)."""
    user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
    try:
        return await svc.bm25_search(req.model_dump(exclude_none=True), user=user)
    except httpx.HTTPStatusError as e:
        _logger.error(f"BM25 search failed: {e.response.status_code}")
        return {"error": "BM25 search failed", "detail": str(e)}
    except Exception as e:
        _logger.error(f"BM25 search error: {e}")
        return {"error": "Internal server error", "detail": str(e)}


@router.post("/vector", tags=["Search"], status_code=status.HTTP_200_OK)
async def vector_search(
    req: HybridSearchRequest,
    svc: HybridSearchService = Depends(get_search_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Vector-only search via module 17 (Qdrant)."""
    user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
    try:
        return await svc.vector_search(req.model_dump(exclude_none=True), user=user)
    except httpx.HTTPStatusError as e:
        _logger.error(f"Vector search failed: {e.response.status_code}")
        return {"error": "Vector search failed", "detail": str(e)}
    except Exception as e:
        _logger.error(f"Vector search error: {e}")
        return {"error": "Internal server error", "detail": str(e)}


# ============================================================================
# TKG / GraphRAG search (module 20)
# ============================================================================

class TkgSearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language query text")
    time_start: Optional[str] = Field(None, description="ISO 8601 時間區間起點（可選）")
    time_end:   Optional[str] = Field(None, description="ISO 8601 時間區間終點（可選）")
    max_depth:  int = Field(2, ge=1, le=4)
    limit:      int = Field(50, ge=1, le=200)
    score_threshold: float = Field(0.3, ge=0.0, le=10.0)


@router.post("/tkg", tags=["Search"], status_code=status.HTTP_200_OK)
async def tkg_search(
    req: TkgSearchRequest,
    svc: HybridSearchService = Depends(get_search_service),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """TKG + GraphRAG combined search via module 20 (graph-service).

    回傳 subgraph_nodes / subgraph_edges / temporal_facts / moment_ids。
    """
    try:
        user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
        body = req.model_dump(exclude_none=True)
        body["branch_id"] = current_user["sub"]
        resp = await client.post(
            f"{settings.graph_service_url}/tkg/query",
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        await svc.enrich_with_urls(data.get("moment_ids", []), user)
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"TKG search failed: {e.response.status_code} {e.response.text}")
        return {"error": "TKG search failed", "detail": str(e), "status_code": e.response.status_code}
    except Exception as e:
        _logger.error(f"TKG search error: {e}")
        return {"error": "Internal server error", "detail": str(e)}


# ============================================================================
# GraphRAG search (module 20)
# ============================================================================

class GraphRagSearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language query text")
    max_depth: int = Field(2, ge=1, le=4)
    limit: int = Field(50, ge=1, le=200)
    score_threshold: float = Field(0.5, ge=0.0, le=10.0)
    strategy: Literal["hybrid", "literal", "semantic"] = Field("hybrid")


@router.post("/graphrag", tags=["Search"], status_code=status.HTTP_200_OK)
async def graphrag_search(
    req: GraphRagSearchRequest,
    svc: HybridSearchService = Depends(get_search_service),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """GraphRAG entity/subgraph search via module 20 (graph-service).

    回傳 matched_entities / matched_moments / nodes / relationships / citations。
    """
    try:
        user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
        body = req.model_dump(exclude_none=True)
        body["branch_id"] = current_user["sub"]
        resp = await client.post(
            f"{settings.graph_service_url}/query/graph_rag",
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        await svc.enrich_with_urls(
            data.get("matched_moments", []) + data.get("moment_ids", []),
            user,
        )
        return data
    except httpx.HTTPStatusError as e:
        _logger.error(f"GraphRAG search failed: {e.response.status_code} {e.response.text}")
        return {"error": "GraphRAG search failed", "detail": str(e), "status_code": e.response.status_code}
    except Exception as e:
        _logger.error(f"GraphRAG search error: {e}")
        return {"error": "Internal server error", "detail": str(e)}
