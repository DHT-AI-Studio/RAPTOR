"""
app/models.py
Shared request/response models for the RAG pipeline.
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class SortOrder(str, Enum):
    RELEVANCE = "relevance"
    TIME_ASC  = "time_asc"
    TIME_DESC = "time_desc"


class QueryPlan(BaseModel):
    modes:          set[str]     = Field(default_factory=lambda: {"vector", "keyword"})
    sort_by:        SortOrder    = SortOrder.RELEVANCE
    group_by_doc:   bool         = False
    top_k_override: int | None   = None
    rel_position:   str | None   = None   # "start" | "middle" | "end"
    tail_seconds:   float | None = None
    entities:       list[str]    = Field(default_factory=list)
    time_range:     dict | None  = None   # {"start": ISO8601|None, "end": ISO8601|None}


# ── Search agent request models ───────────────────────────────────────────────

class VectorSearchRequest(BaseModel):
    query:           str
    top_k:           int                        = 10
    type:            str | list[str] | None     = None   # "videos"|"audios"|"documents"|"images"
    score_threshold: float                      = 0.3
    branch_id:       str | None                 = None


class BM25SearchRequest(BaseModel):
    query:     str
    top_k:     int                    = 10
    type:      str | list[str] | None = None     # "videos"|"audios"|"documents"|"images"
    branch_id: str | None             = None


class GraphRAGRequest(BaseModel):
    query:           str
    entity:          str | None = None       # seed entity for graph traversal
    max_depth:       int   = 2
    score_threshold: float = 0.3
    branch_id:       str | None = None


class TKGRequest(BaseModel):
    query:           str
    max_depth:       int   = 2
    score_threshold: float = 0.3
    time_start:      str | None = None       # ISO8601, from QueryPlan.time_range
    time_end:        str | None = None
    branch_id:       str | None = None


class RerankerRequest(BaseModel):
    query:      str
    candidates: list[dict]
    top_k:      int = 5


# ── Result models ─────────────────────────────────────────────────────────────

class SearchHit(BaseModel):
    id:          str
    doc_id:      str
    score:       float
    content:     str
    metadata:    dict         = Field(default_factory=dict)
    start_time:  float | None = None
    end_time:    float | None = None
    # asset info for URL resolution (populated from search payload)
    asset_path:  str | None   = None
    version_id:  str | None   = None
    storage_uri: str | None   = None


class SearchResult(BaseModel):
    hits:   list[SearchHit]
    total:  int
    source: str   # "vector" | "bm25" | "reranker"


class GraphResult(BaseModel):
    triples:      list[dict]   # [{subject, predicate, object, valid_from?, valid_to?}]
    summary:      str
    entity_count: int
    source:       str = ""     # "graphrag" | "tkg"


class VideoClipResult(BaseModel):
    """Video clip with resolved playback URL — built from search result metadata."""
    moment_id:   str
    clip_url:    str           # presigned URL with #t=start,end fragment
    start_time:  float
    end_time:    float
    storage_uri: str   = ""
    doc_id:      str   = ""
    score:       float = 0.0
    content:     str   = ""


class OrchestratorResult(BaseModel):
    answer:        str
    sources:       list[dict]
    graph_context: str  = ""
    chunks_used:   int  = 0
