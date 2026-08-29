"""Search request/response models (PA-6) — mirrors Module 17's SearchRequest so
the per-user ArcadeDB search is contract-compatible with the global search."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    # optional filters (applied as WHERE clauses on Chunk)
    type: Optional[Union[str, List[str]]] = None       # documents|videos|images|audios
    embedding_type: Optional[str] = None               # text | summary (granularity)
    status: Optional[str] = None
    version_id: Optional[str] = None
    filename: Optional[List[str]] = None
    source: Optional[str] = None
    speaker: Optional[List[str]] = None                  # video/audio segments
    payload_schema: Optional[str] = None   # accepted for Module-17 parity; unused for now


class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    timing: Dict[str, float]


# ---------------------------------------------------------------------------
# Rerank -- generic "rerank arbitrary (id, text) pairs against a query"
# utility, not tied to any one user's database (no X-Branch-ID needed).
# Mirrors Module 17's /api/v1/search/rerank contract exactly (RerankDocument/
# RerankRequest/RerankResult/RerankResponse in that module's schemas.py) so
# a caller (video_search.py's personal-db equivalent) can swap backends
# without reshaping the request/response.
# ---------------------------------------------------------------------------
class RerankDocument(BaseModel):
    id: str
    text: str
    payload: Optional[Dict[str, Any]] = None


class RerankRequest(BaseModel):
    query: str
    documents: List[RerankDocument]
    top_k: Optional[int] = None   # None = return all


class RerankResult(BaseModel):
    id: str
    score: float
    payload: Optional[Dict[str, Any]] = None


class RerankResponse(BaseModel):
    results: List[RerankResult]
