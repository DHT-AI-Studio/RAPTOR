"""Request/response models for graph, TKG and GraphRAG search (PA-7).

All scoped to the caller's ArcadeDB database (X-Branch-ID). Graph traversal runs
over the `RELATION` edges between `Entity` vertices; GraphRAG fuses dense
`vectorNeighbors` on `Chunk` with `out('MENTIONS')` entity context; TKG queries
`TemporalFact` vertices with an optional time window.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------------ graph
class GraphSearchRequest(BaseModel):
    entity_name: str
    max_depth: int = 2                     # RELATION hops from the seed entity (clamped 1..5)
    query: str | None = None            # power-user override: a raw SELECT graph query


class GraphEdge(BaseModel):
    relation: str
    from_name: str | None = None
    from_id: str | None = None
    to_name: str | None = None
    to_id: str | None = None
    confidence: float | None = None


class GraphSearchResponse(BaseModel):
    entities: list[dict[str, Any]]
    edges: list[GraphEdge]
    paths: list[list[str]]                 # shortest path (entity names) seed -> each reachable entity


# ------------------------------------------------------------------ tkg
# Batch 5 of the graph/TKG/GraphRAG parity plan -- matches Module 20's
# TkgQueryRequest/TkgQueryResponse shape (natural-language query -> entity
# fulltext search -> subgraph expansion -> time-windowed TemporalFacts),
# replacing the old entity_name-exact-match-only version. No branch_id field,
# same reasoning as GraphRAGRequest above.
AllenRelation = Literal["before", "meets", "overlaps", "starts", "during", "finishes", "equals",
                        "finished-by", "contains", "started-by", "overlapped-by", "met-by", "after"]


class TKGWindow(BaseModel):
    """Contract v2's window (§5.9): either side may be omitted -- unbounded, never an invented sentinel."""
    start: str | None = None
    end: str | None = None


# Contract v2 fields the plan defines that this deployment does not serve yet (capabilities lists them as
# unsupported). A v2 request giving one a value is refused, never silently ignored.
V2_UNSUPPORTED = ("as_of",)


class TKGRequest(BaseModel):
    # Unknown fields are kept (not dropped) so a v2 request carrying one can be refused by name;
    # a v1 request (no `version`) keeps ignoring them, as before.
    model_config = ConfigDict(extra="allow")

    query: str
    time_start: str | None = None       # inclusive lower bound on fact.time_start
    time_end: str | None = None         # inclusive upper bound on fact.time_end
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
    # TMP-4 (D13): how temporal facts are retrieved. Facts are restricted to entities
    # only when one is named -- `entity`, or an entity the query text fulltext-matches;
    # otherwise keywords filter the evidence, and with neither every fact in the window
    # is a candidate. A question naming no entity is never answered with nothing.
    entity: str | None = None              # matched instead of the query text; no match: no facts
    keywords: list[str] | None = None      # every one must appear in a fact's evidence text; with a
                                           # matched entity, also adds other facts that mention them
    # P07 (§5.9): how a fact must relate to the window, decided by Allen classification of its
    # endpoint ranges. Unset keeps the containment semantics above. `relations` (any of the 13
    # Allen names) overrides `match`; `certainty: certain` keeps only facts whose every possible
    # relation is accepted. Without a window these change nothing.
    match: Literal["inside", "overlaps", "contains", "any"] | None = None
    relations: list[AllenRelation] | None = None
    certainty: Literal["possible", "certain"] = "possible"

    # P08 / P09 (§5.9): contract v2. With `version: 2` the whole result is ranked (`sort`) and paged
    # (`limit` + `cursor`) -- entity facts included -- and unknown or unsupported fields are refused.
    # Without it (v1) everything above behaves exactly as before.
    version: Literal[1, 2] | None = None
    window: TKGWindow | None = None        # v2 form of time_start / time_end
    timezone: str | None = None            # the zone calendar dates are read in (default: the deployment's)
    timeline_id: str | None = None         # only "world" is served
    sort: Literal["time", "relevance"] | None = None   # v2 default: time
    cursor: str | None = None              # from the previous page's response
    # P16 (v2): the event the answer is relative to -- {"fact_id": ...} or {"text": ...} -- and the Allen
    # relations a fact must have to it ("after Google acquired Android": ["after"]). Checked by
    # contract.check_v2 (structured 400s); see app.temporal.reference.
    reference: Any = None
    reference_relations: Any = None
    as_of: Any = None                      # V2_UNSUPPORTED -- accepted only to be refused by name
    # P21 (v2): only these sources' facts -- a list of source version ids (contract.check_source_filter)
    source_filter: Any = None


class TKGResponse(BaseModel):
    query: str
    matched_entities: list[dict[str, Any]]
    subgraph_nodes: list[dict[str, Any]]
    subgraph_edges: list[dict[str, Any]]
    temporal_facts: list[dict[str, Any]]   # sorted by time_start ASC; each has `when` -- its dates at
                                           # the precision it has ("2024-07", "2008 to 2013") -- and
                                           # `precision`; with a window also allen: {vs_window, certainty}
                                           # (None: not classifiable)
    moment_ids: list[dict[str, Any]]
    retrieval: str = "entity"              # TMP-4: entity | keywords | temporal
    complete: bool = True                  # False when more facts matched than `limit` returned
    # P08 (v2 only): `cursor` fetches the next page; `total` is how many facts matched in all
    version: int = 1
    sort: str | None = None
    cursor: str | None = None
    total: int | None = None
    # P16: how `reference` was resolved -- status resolved | ambiguous | not_found | undated, the event's fact,
    # and the candidates of an ambiguous one. Unless resolved, no facts are returned.
    reference: dict[str, Any] | None = None


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
    matched_entities: list[dict[str, Any]]
    matched_moments: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    moment_ids: list[dict[str, Any]]
    citations: list[dict[str, Any]]


# ------------------------------------------------------------------ raw query
class RawGraphQueryRequest(BaseModel):
    query: str                             # raw ArcadeDB SQL; must be a read-only SELECT


class RawGraphQueryResponse(BaseModel):
    result: list[dict[str, Any]]
