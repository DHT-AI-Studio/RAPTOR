"""Request/response models for graph, TKG and GraphRAG search (PA-7).

All scoped to the caller's ArcadeDB database (X-Branch-ID). Graph traversal runs
over the `RELATION` edges between `Entity` vertices; GraphRAG fuses dense
`vectorNeighbors` on `Chunk` with `out('MENTIONS')` entity context; TKG queries
`TemporalFact` vertices with an optional time window.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


# ------------------------------------------------------------------ graph
class GraphSearchRequest(BaseModel):
    entity_name: str
    max_depth: int = 2                     # RELATION hops from the seed entity (clamped 1..5)
    query: Optional[str] = None            # power-user override: a raw SELECT graph query


class GraphEdge(BaseModel):
    relation: str
    from_name: Optional[str] = None
    from_id: Optional[str] = None
    to_name: Optional[str] = None
    to_id: Optional[str] = None
    confidence: Optional[float] = None


class GraphSearchResponse(BaseModel):
    entities: List[Dict[str, Any]]
    edges: List[GraphEdge]
    paths: List[List[str]]                 # shortest path (entity names) seed -> each reachable entity


# ------------------------------------------------------------------ tkg
# Batch 5 of the graph/TKG/GraphRAG parity plan -- matches Module 20's
# TkgQueryRequest/TkgQueryResponse shape (natural-language query -> entity
# fulltext search -> subgraph expansion -> time-windowed TemporalFacts),
# replacing the old entity_name-exact-match-only version. No branch_id field,
# same reasoning as GraphRAGRequest above.
class TKGRequest(BaseModel):
    query: str
    time_start: Optional[str] = None       # inclusive lower bound on fact.time_start
    time_end: Optional[str] = None         # inclusive upper bound on fact.time_end
    max_depth: int = 2                     # subgraph hops from each matched entity (clamped 1..4,
                                            # matching Module 20's TkgQueryRequest le=4)
    limit: int = 50
    score_threshold: float = 0.5           # matches Module 20's TkgQueryRequest default -- was
                                            # 0.3 (copied from Module 13's gateway, which had
                                            # copied 20's own stale json_schema_extra example
                                            # instead of its real Field default; both now fixed
                                            # to 0.5, see search.py's TkgSearchRequest). Filters
                                            # ArcadeDB's real BM25 $score, same as Module 20
                                            # filters Neo4j's Lucene score -- see graph_query.py.


class TKGResponse(BaseModel):
    query: str
    matched_entities: List[Dict[str, Any]]
    subgraph_nodes: List[Dict[str, Any]]
    subgraph_edges: List[Dict[str, Any]]
    temporal_facts: List[Dict[str, Any]]   # sorted by time_start ASC
    moment_ids: List[Dict[str, Any]]


# ------------------------------------------------------------------ graphrag
# Batch 4 of the graph/TKG/GraphRAG parity plan -- matches Module 20's
# GraphRagRequest/GraphRagResponse shape (natural-language query -> entity/
# moment fulltext search -> subgraph expansion -> citations), not the
# original simple dense-vector-search version this replaces (never exposed
# through Module 13's gateway, so nothing depends on the old shape).
# No branch_id field: Module 20 needs one because it's a single shared
# Neo4j; Module 25 is already isolated per user via a separate ArcadeDB
# database per branch_id.
class GraphRAGRequest(BaseModel):
    query: str
    max_depth: int = 2                     # RELATION|MENTIONS|HAS_TEMPORAL_FACT hops (clamped 1..4,
                                            # matching Module 20's GraphRagRequest le=4)
    limit: int = 50                        # node cap per matched entity's subgraph
    strategy: Literal["literal", "semantic", "hybrid"] = "hybrid"
    score_threshold: float = 0.5           # matches Module 20's GraphRagRequest default (both
                                            # 20 and its gateway agree on 0.5, unlike TKG's --
                                            # see TKGRequest above). Filters ArcadeDB's real BM25
                                            # $score, same as Module 20 filters Neo4j's Lucene
                                            # score -- see graph_query.py.


class GraphRAGResponse(BaseModel):
    query: str
    strategy: str
    matched_entities: List[Dict[str, Any]]
    matched_moments: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    moment_ids: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]


# ------------------------------------------------------------------ raw query
class RawGraphQueryRequest(BaseModel):
    query: str                             # raw ArcadeDB SQL; must be a read-only SELECT


class RawGraphQueryResponse(BaseModel):
    result: List[Dict[str, Any]]
