"""Search request/response models (PA-6) — mirrors Module 17's SearchRequest so
the per-user ArcadeDB search is contract-compatible with the global search."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    # optional filters (applied as WHERE clauses on Chunk)
    type: str | list[str] | None = None       # documents|videos|images|audios
    embedding_type: str | None = None               # text | summary (granularity)
    status: str | None = None
    version_id: str | list[str] | None = None        # P21: one source, or a scope of several
    filename: list[str] | None = None
    source: str | None = None
    speaker: list[str] | None = None                  # video/audio segments
    payload_schema: str | None = None   # accepted for Module-17 parity; unused for now
    # CM-5 / VIE01-408: content that is 'flagged' (pending review) or
    # 'removed' (confirmed violation) is excluded by default; 'cleared'
    # content is not. An admin-only caller sets this to see it anyway.
    include_flagged: bool = False


class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]
    timing: dict[str, float]


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
    payload: dict[str, Any] | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[RerankDocument]
    top_k: int | None = None   # None = return all


class RerankResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] | None = None


class RerankResponse(BaseModel):
    results: list[RerankResult]
