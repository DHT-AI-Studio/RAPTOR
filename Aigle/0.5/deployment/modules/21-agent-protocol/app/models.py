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
    keywords:       list[str]    = Field(default_factory=list)   # TMP-4: Module 18's evidence terms
    time_range:     dict | None  = None   # {"start": ISO8601|None, "end": ISO8601|None}
    match:          str | None   = None   # P07: TKG Allen match mode from Module 18 ("overlaps": a closed window)
    # P16: the event the answer is relative to ("after Google acquired Android") and the Allen relations
    # a fact must have to it, from Module 18
    reference:           str | None       = None
    reference_relations: list[str] | None = None
    # MTR-25: "people" | "people+semantic" | "people+temporal" from Module 18 -- who appears in the videos
    people_intent:       str | None       = None
    # TMP-7: Module 18 could not read the time condition (`parse` failed or cannot read the date): the
    # question is not answered without it
    temporal_status:     str | None       = None   # "RESOLVED" | "NO_TEMPORAL" | "UNRESOLVED"
    temporal_reason:     str | None       = None


# ── Search agent request models ───────────────────────────────────────────────

class VectorSearchRequest(BaseModel):
    query:           str
    top_k:           int                        = 10
    type:            str | list[str] | None     = None   # "videos"|"audios"|"documents"|"images"
    score_threshold: float                      = 0.3
    branch_id:       str | None                 = None
    version_id:      list[str] | None           = None   # P21: only these sources


class BM25SearchRequest(BaseModel):
    query:     str
    top_k:     int                    = 10
    type:      str | list[str] | None = None     # "videos"|"audios"|"documents"|"images"
    branch_id: str | None             = None
    version_id: list[str] | None      = None     # P21: only these sources


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
    # TMP-4: Module 25 restricts facts to an entity only when one is named; keywords
    # filter the evidence; with neither, every fact in the window is a candidate
    entity:          str | None = None
    keywords:        list[str] | None = None
    # P07: how facts must relate to the window -- "overlaps" also keeps a fact that only shares time
    # with it ("what was X doing in 2010" finds a 2008-2013 role); None keeps Module 25's "inside"
    match:           str | None = None
    # P16: a reference event (its text) and the Allen relations to it; sent as TKG contract v2
    reference:           str | None       = None
    reference_relations: list[str] | None = None
    # P21: only these sources' facts (TKG v2 source_filter)
    source_filter:       list[str] | None = None


class PeopleSearchRequest(BaseModel):
    """MTR-25 PeopleAgent: the people named in `query` (as the user named them in Module 25) and the
    segments they appear in. `auth` is the end user's bearer token -- PeopleAgent calls Module 13's
    /search/people, which acts as the token's user, not as branch_id."""
    query:     str
    top_k:     int        = 10
    branch_id: str | None = None
    auth:      str | None = None


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
    triples:      list[dict]   # [{subject, predicate, object, valid_from?, valid_to?, when?}] -- when: D8 dates at their precision
    summary:      str
    entity_count: int
    source:       str = ""     # "graphrag" | "tkg"
    # P17 (tkg): each claim the answer stage proposed and whether Module 25 verified it
    claims:       list[dict]   = Field(default_factory=list)


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
    answer:        str | None   # None: not answered -- see temporal_status (TMP-7)
    sources:       list[dict]
    graph_context: str  = ""
    chunks_used:   int  = 0
    # P17: the TKG's temporal claims and their verification ({claim, verified, reason})
    temporal_claims: list[dict] = Field(default_factory=list)
    # MTR-25: PeopleAgent's segments -- {identity_name, version_id, trace_id, first_sec, last_sec, score}
    people:          list[dict] = Field(default_factory=list)
    # TMP-7: UNRESOLVED -- the question's time condition could not be read, so nothing was retrieved and
    # there is no answer (never an answer from unfiltered evidence)
    temporal_status:             str | None = None
    temporal_reason:             str | None = None
    temporal_constraint_applied: bool | None = None
