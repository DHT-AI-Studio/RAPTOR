"""Per-user hybrid / vector / BM25 search over ArcadeDB (PA-6, unified `Chunk`
+ Source). Mirrors Module 17's search contract but scoped to one user's
ArcadeDB database.

`Chunk` holds per-item/per-segment content across all four media types, so a
single query covers everything and `type` (+ `embedding_type`, filename,
source, speaker) filters narrow it. Whole-asset summaries (previously a fake
"summary" Chunk row with no start_sec/end_sec) live on `Source` instead now,
matching Module 20's Source.summary -- see indexer.index_source_summary().
Every query here runs against both and merges the results, so this remains a
drop-in match for Module 17's own results, which do include summary-level
hits in the same list.

Verified ArcadeDB templates:
  dense  : vectorNeighbors('Chunk[embedding]'|'Source[embedding]', :qvec, k) -> rows w/ distance
  bm25   : WHERE SEARCH_INDEX('Chunk[text]'|'Source[summary]',:q)=true, real $score --
         except for a lone-CJK-character query, which falls back to a plain
         `field LIKE :q` scan (see _bm25_condition): CJKAnalyzer's bigram
         index structurally cannot match a single character (confirmed live),
         and ArcadeDB's METADATA has no outputUnigrams-equivalent to fix that
         at the index level (confirmed against a live error listing every
         supported FULL_TEXT metadata key, and the official docs).
  hybrid : vector.fuse(<dense rows>, <bm25 rows>, {'fusion':'RRF','limit':k}),
         run once per vertex type, concatenated and handed to one cross-
         encoder rerank pass (see reranker.py) — self-contained, does not
         depend on module 17.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from app.core.config import settings
from app.services import graph_query
from app.models.graph_search import (GraphEdge, GraphRAGRequest, GraphRAGResponse,
                                      GraphSearchRequest, GraphSearchResponse,
                                      RawGraphQueryResponse, TKGRequest, TKGResponse)
from app.models.search import SearchRequest, SearchResponse, SearchResult
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.embedder import embed_texts
from app.services.indexer import DatabaseNotInitializedError
from app.services.reranker import rerank

logger = logging.getLogger("personal_db.searcher")

_PAYLOAD = ("chunk_id, type, embedding_type, text, summary, filename, source, "
            "asset_path, version_id, status, start_sec, end_sec, speaker, "
            "chunk_index")
_DROP = {"@rid", "@type", "@cat", "@props", "distance", "score", "embedding",
         "sparse_indices", "sparse_weights"}
# text (chunked media) covers the per-segment/per-item content; whole-asset
# summaries used to be faked as a same-table "summary" Chunk row (no
# start_sec/end_sec) so this one SEARCH_INDEX on Chunk covered both -- now
# summary lives on Source instead (indexer.index_source_summary()), queried
# separately below and merged into the same result list, so this only needs
# `text` any more.
# Source's own payload shaped to match _PAYLOAD's Chunk columns as closely as
# possible (embedding_type/type synthesized since Source has no such
# columns of its own) so a caller can't tell from the payload shape alone
# whether a hit came from a Chunk or a Source -- same reasoning as the
# GraphRAG/TKG field-parity work.
_SOURCE_PAYLOAD = ("version_id, summary, filename, asset_path, "
                   "media_type AS type, 'summary' AS embedding_type")


def _single_cjk_char(query: str) -> str | None:
    """Returns the query's lone CJK character when its CJK run is exactly one
    character, else None -- mirrors graph_query._cjk_phrases()'s own
    unigram-fallback condition exactly. A lone CJK character produces no
    bigrams for a CJKAnalyzer-backed FULL_TEXT index to match against:
    confirmed live (SEARCH_INDEX on a single character always returns zero
    rows) and confirmed ArcadeDB has no way to fix this at the index level
    (no outputUnigrams-equivalent METADATA key -- checked against both a
    live error message listing every supported FULL_TEXT metadata key and
    the official docs).

    Returns the character itself, not just a bool: this file's callers (via
    pipeline.py's BM25SearchRequest(query=question, ...)) often pass a whole
    natural-language question, not a bare keyword -- e.g. "who is 狗
    associated with?". A LIKE pattern built from that entire sentence would
    essentially never match anything indexed; only the single CJK character
    should go into the pattern. Confirmed live this was a real, not
    theoretical, bug: a bare "狗" query matched fine, the identical entity
    embedded in an English sentence did not, until this returned just the
    character instead of the caller re-using the full query string."""
    cjk_run = "".join(c for c in query if graph_query._is_cjk(c))
    return cjk_run if len(cjk_run) == 1 else None


def _bm25_condition(vertex_type: str, field: str, query: str, param: str, params: dict) -> str:
    """SEARCH_INDEX normally; a plain substring LIKE for a lone-CJK-character
    query instead, since SEARCH_INDEX can never match one (see
    _single_cjk_char). Sets params[param] to whichever value the chosen
    condition needs (the single CJK character for LIKE, raw query for
    SEARCH_INDEX)."""
    single_char = _single_cjk_char(query)
    if single_char is not None:
        params[param] = f"%{single_char}%"
        return f"{field} LIKE :{param}"
    params[param] = query
    return f"SEARCH_INDEX('{vertex_type}[{field}]', :{param}) = true"


def _include_source(req: SearchRequest) -> bool:
    """Whether the Source-side query should run at all. Source's payload
    always carries a synthesized, literal embedding_type='summary' (Source
    has no such column of its own -- see _SOURCE_PAYLOAD), so a caller who
    explicitly asked for embedding_type="text" (or anything else that isn't
    "summary") means "no summaries" and Source has nothing else to offer;
    running the query and then filtering its results out in Python would
    waste a round-trip for a result set guaranteed to always be discarded."""
    return req.embedding_type is None or req.embedding_type == "summary"


def _include_chunk(req: SearchRequest) -> bool:
    """Symmetric to _include_source(): a caller who explicitly asked for
    embedding_type="summary" wants Source's content, and Chunk has none of
    its own any more -- summary is no longer written there (the Source-
    summary migration), and old pre-migration summary Chunks are not kept
    around, so there is nothing on the Chunk side for a summary-only
    request to ever find. Unlike _include_source(), this used to stay
    unconditional deliberately for backward compatibility with legacy
    summary Chunks; dropped now that old data isn't being kept."""
    return req.embedding_type is None or req.embedding_type == "text"


def _source_filters(req: SearchRequest, params: dict) -> str:
    """Subset of _filters() that actually applies to Source -- Source has no
    speaker/source/start_sec columns, so those filters simply don't narrow
    the Source side (they still narrow the Chunk side as before), not an
    error and not silently wrong: a filter that names a Chunk-only concept
    has nothing to exclude on Source in the first place. status DOES apply
    (Source.status exists and is now actually written by
    set_status_by_version()) -- same active-by-default reasoning as
    _filters()."""
    clauses = []
    if req.type is not None:
        val = req.type
        clauses.append("media_type IN :sf_type" if isinstance(val, list) else "media_type = :sf_type")
        params["sf_type"] = val
    if req.filename is not None:
        val = req.filename
        clauses.append("filename IN :sf_filename" if isinstance(val, list) else "filename = :sf_filename")
        params["sf_filename"] = val
    if req.version_id is not None:
        clauses.append("version_id = :sf_version_id")
        params["sf_version_id"] = req.version_id
    clauses.append("status = :sf_status")
    params["sf_status"] = req.status if req.status is not None else "active"
    return " AND ".join(clauses)


async def _ensure_ready(client: ArcadeDBClient, branch_id: str) -> str:
    db = db_name_for(branch_id)
    if not await client.database_exists(db):
        raise DatabaseNotInitializedError(
            "Personal database not initialized. Call POST /internal/db/init first.")
    return db


async def _rid_filter_matches(client: ArcadeDBClient, db: str, vertex_type: str,
                               where: str, params: dict) -> bool:
    """True if at least one `vertex_type` row satisfies `where`.

    vectorNeighbors()'s `filter` option takes a RID subquery
    (`(SELECT @rid AS rid FROM ... WHERE ...).rid`) -- confirmed live that
    when that subquery is legitimately empty (the filter combination matches
    zero rows, e.g. type=["audios"] for a user with no audio content at all),
    ArcadeDB treats the empty RID list as "no filter" rather than "match
    nothing" and vectorNeighbors returns unfiltered neighbors instead. Plain
    SQL WHERE (bm25_search(), and the BM25 half of hybrid_search()'s fuse())
    doesn't have this problem -- only call sites that build a vectorNeighbors
    filter subquery need this pre-check, to skip that call entirely rather
    than let an empty filter silently leak every vertex regardless of it."""
    rows = await client.query(db, f"SELECT @rid FROM {vertex_type} WHERE {where} LIMIT 1", params=params)
    return bool(rows)


def _filters(req: SearchRequest, params: dict) -> str:
    """WHERE fragment (no leading keyword) from the optional filters."""
    clauses = []
    for field in ("type", "filename", "speaker"):
        val = getattr(req, field, None)
        if val is not None:
            clauses.append(f"{field} IN :f_{field}" if isinstance(val, list) else f"{field} = :f_{field}")
            params[f"f_{field}"] = val
    for field in ("version_id", "source", "embedding_type"):
        val = getattr(req, field, None)
        if val is not None:
            clauses.append(f"{field} = :f_{field}")
            params[f"f_{field}"] = val
    # Default to active-only, matching Module 17's own read-side filter
    # (qdrant_service.py/opensearch_service.py: status == "active" unless
    # explicitly overridden) -- now that Module 04's search_sync can archive
    # a version (set_status_by_version()), search needs to actually respect
    # that instead of still surfacing archived content. An explicit
    # req.status (including "archived", for a caller that wants it) always
    # wins over this default.
    clauses.append("status = :f_status")
    params["f_status"] = req.status if req.status is not None else "active"
    return " AND ".join(clauses)


def _results(rows, rank_score: bool) -> list[SearchResult]:
    out = []
    for i, r in enumerate(rows):
        # chunk_id first -- @rid is ArcadeDB's internal row position, not a
        # stable public identifier (unlike Module 17's Qdrant/OpenSearch UUID
        # chunk ids); every other Module 25 surface (matched_moments,
        # citations, get_moment_subgraph) already keys moments by chunk_id.
        # version_id next -- a Source-origin row (whole-asset summary) has no
        # chunk_id at all any more, so this is its only stable identifier.
        rid = r.get("chunk_id") or r.get("version_id") or r.get("@rid") or ""
        if "distance" in r and r["distance"] is not None:
            score = 1.0 - float(r["distance"])
        elif r.get("score") is not None:
            # Real BM25 score ($score AS score in the SQL) -- needed whenever
            # Chunk and Source rows get merged into one ranked list, since
            # two independently-computed 1/(i+1) position ranks from two
            # different queries aren't comparable to each other.
            score = float(r["score"])
        elif rank_score:
            score = 1.0 / (i + 1)
        else:
            score = 0.0
        payload = {k: v for k, v in r.items() if k not in _DROP}
        # Module 17 parity: start_time/end_time as strings, not start_sec/end_sec floats.
        start_sec = payload.pop("start_sec", None)
        end_sec = payload.pop("end_sec", None)
        if start_sec is not None:
            payload["start_time"] = str(start_sec)
        if end_sec is not None:
            payload["end_time"] = str(end_sec)
        out.append(SearchResult(id=str(rid), score=round(score, 6), payload=payload))
    return out


async def vector_search(client: ArcadeDBClient, branch_id: str, req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    db = await _ensure_ready(client, branch_id)
    te = time.perf_counter()
    qvec = (await embed_texts([req.query]))[0]
    embed_sec = time.perf_counter() - te
    # Filters (status='active' by default, plus type/filename/etc.) are pushed
    # into vectorNeighbors' own `filter` option -- a RID list from a subquery,
    # confirmed live this ArcadeDB version accepts (must be `.rid` off a
    # `SELECT @rid AS rid ...` projection; a bare `SELECT @rid FROM ...` or
    # `SELECT FROM ...` both error with "must contain RIDs, got: ResultInternal").
    # Previously the filter was a post-filter applied *after* vectorNeighbors
    # already picked its top_k -- an archived Chunk within true vector
    # distance of the query could occupy one of those top_k slots and get
    # discarded afterward, silently shrinking the result count below top_k
    # even when a further-but-still-active real match existed just outside
    # the raw (unfiltered) top_k cutoff. Pushing the filter into the ANN
    # search itself means top_k candidates are already all valid.
    #
    # A legitimately-empty filter (e.g. type=["audios"] for a user with no
    # audio content) needs its own guard -- see _rid_filter_matches()'s
    # docstring for why an empty RID subquery isn't safe to hand straight to
    # vectorNeighbors.
    filter_params: dict = {}
    where = _filters(req, filter_params)
    src_filter_params: dict = {}
    src_where = _source_filters(req, src_filter_params)

    tq = time.perf_counter()
    rows: list = []
    if _include_chunk(req) and await _rid_filter_matches(client, db, "Chunk", where, filter_params):
        params = {"qvec": qvec, **filter_params}
        sql = (f"SELECT {_PAYLOAD}, @rid, distance "
               f"FROM (SELECT expand(vectorNeighbors('Chunk[embedding]', :qvec, {req.top_k}, "
               f"{{'filter': (SELECT @rid AS rid FROM Chunk WHERE {where}).rid}})))")
        rows = await client.query(db, sql, params=params)
    src_rows: list = []
    if _include_source(req) and await _rid_filter_matches(client, db, "Source", src_where, src_filter_params):
        src_params = {"qvec": qvec, **src_filter_params}
        src_sql = (f"SELECT {_SOURCE_PAYLOAD}, @rid, distance "
                  f"FROM (SELECT expand(vectorNeighbors('Source[embedding]', :qvec, {req.top_k}, "
                  f"{{'filter': (SELECT @rid AS rid FROM Source WHERE {src_where}).rid}})))")
        src_rows = await client.query(db, src_sql, params=src_params)
    vector_sec = time.perf_counter() - tq
    merged = sorted(_results(rows + src_rows, rank_score=False), key=lambda r: r.score, reverse=True)
    return SearchResponse(results=merged[:req.top_k],
                          timing={"total_sec": round(time.perf_counter() - t0, 6),
                                  "embed_sec": round(embed_sec, 6), "vector_sec": round(vector_sec, 6)})


async def bm25_search(client: ArcadeDBClient, branch_id: str, req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    db = await _ensure_ready(client, branch_id)
    params = {}
    bm25_cond = _bm25_condition("Chunk", "text", req.query, "q", params)
    where = _filters(req, params)
    cond = bm25_cond + (f" AND {where}" if where else "")
    sql = f"SELECT {_PAYLOAD}, $score AS score, @rid FROM Chunk WHERE {cond} LIMIT {req.top_k}"
    src_params = {}
    src_bm25_cond = _bm25_condition("Source", "summary", req.query, "q", src_params)
    src_where = _source_filters(req, src_params)
    src_cond = src_bm25_cond + (f" AND {src_where}" if src_where else "")
    src_sql = f"SELECT {_SOURCE_PAYLOAD}, $score AS score, @rid FROM Source WHERE {src_cond} LIMIT {req.top_k}"
    tq = time.perf_counter()
    rows = await client.query(db, sql, params=params) if _include_chunk(req) else []
    src_rows = await client.query(db, src_sql, params=src_params) if _include_source(req) else []
    bm25_sec = time.perf_counter() - tq
    merged = sorted(_results(rows + src_rows, rank_score=True), key=lambda r: r.score, reverse=True)
    return SearchResponse(results=merged[:req.top_k],
                          timing={"total_sec": round(time.perf_counter() - t0, 6),
                                  "bm25_sec": round(bm25_sec, 6)})


# -------------------------------------------------------------------- PA-7 graph
_META = {"@rid", "@type", "@cat", "@props", "@in", "@out"}


def _clean(row: dict) -> dict:
    """Drop ArcadeDB record metadata keys from a projected row."""
    return {k: v for k, v in row.items() if k not in _META}


# A read-only SQL guard for the raw /graph/query power-user endpoint. We only
# allow a single SELECT and reject anything that can mutate schema or data.
_SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|create|drop|alter|truncate|import|"
    r"grant|revoke|rebuild|move|traverse\s+into)\b", re.IGNORECASE)


def is_read_only_select(sql: str) -> bool:
    """True iff sql is a single read-only SELECT (no DDL/DML, no statement chaining)."""
    s = (sql or "").strip().rstrip(";")
    if ";" in s:                       # no statement chaining
        return False
    return bool(_SELECT_ONLY.match(s)) and not _FORBIDDEN.search(s)


async def _resolve_entity_id(client: ArcadeDBClient, db: str, name: str) -> str | None:
    """Resolve an exact Entity.name to its entity_id -- NOT via `WHERE name =
    :n`. Confirmed live: since Batch 2 put a FULL_TEXT/CJKAnalyzer index on
    Entity(name) (schema_init.py) to power fulltext search, ArcadeDB routes a
    direct equality comparison against that same property through the same
    index instead of an exact lookup -- querying name="金門豬肉" also matched
    an unrelated entity named "金門" purely because they share a bigram, and
    graph_search()'s shortestPath() call on the resulting 2-vertex "single"
    seed then hard-errored ("Only one sourceVertex is allowed"). Entity.entity_id
    keeps its own plain (non-fulltext) unique index and isn't affected --
    confirmed the same bug does NOT reproduce for `entity_id = :n`, or for
    `outV().name = :n` / `inV().name = :n` on RELATION (a computed expression,
    not a direct property lookup on the indexed type). This narrows candidates
    via the same over-inclusive bigram SEARCH_INDEX match Batch 2 already
    uses, then does the real exact comparison in Python."""
    rows = await client.query(
        db, "SELECT entity_id, name FROM Entity WHERE SEARCH_INDEX('Entity[name]', :n) = true",
        params={"n": name})
    for r in rows:
        if r.get("name") == name:
            return r.get("entity_id")
    return None


async def graph_search(client: ArcadeDBClient, branch_id: str,
                       req: GraphSearchRequest) -> GraphSearchResponse:
    """Entity-graph neighbourhood: entities reachable within max_depth RELATION
    hops of the seed, the RELATION edges among them, and the shortest path from
    the seed to each. `req.query`, if given, is a raw SELECT override."""
    db = await _ensure_ready(client, branch_id)

    if req.query:                       # power-user override
        if not is_read_only_select(req.query):
            raise ValueError("query override must be a read-only SELECT")
        rows = [_clean(r) for r in await client.query(db, req.query)]
        return GraphSearchResponse(entities=rows, edges=[], paths=[])

    depth = max(1, min(int(req.max_depth), 5))   # MAXDEPTH needs an int literal
    seed_id = await _resolve_entity_id(client, db, req.entity_name)
    if seed_id is None:
        return GraphSearchResponse(entities=[], edges=[], paths=[])

    entity_sql = (
        f"SELECT name, entity_id, type, mention_count FROM ("
        f"TRAVERSE both('RELATION') FROM (SELECT FROM Entity WHERE entity_id = :id) "
        f"MAXDEPTH {depth}) WHERE @this INSTANCEOF 'Entity'")
    seen, entities = set(), []
    for r in await client.query(db, entity_sql, params={"id": seed_id}):
        eid = r.get("entity_id")
        if eid not in seen:
            seen.add(eid)
            entities.append(_clean(r))

    names = [e["name"] for e in entities]
    edges: list[GraphEdge] = []
    paths: list[list[str]] = []
    if names:
        edge_rows = await client.query(
            db, "SELECT relation, outV().name AS from_name, outV().entity_id AS from_id, "
                "inV().name AS to_name, inV().entity_id AS to_id, confidence "
                "FROM RELATION WHERE outV().name IN :ns AND inV().name IN :ns",
            params={"ns": names})
        # Dedup by (from_id, relation, to_id) -- index_relationship() no longer
        # collapses two independent sources asserting the same fact onto one
        # edge (each now gets its own, tagged with its own source_version_id,
        # so delete_by_version() on one doesn't remove a fact another source
        # still supports -- see that function's docstring). That means the
        # same logical relationship can now genuinely be N edges here, same
        # (from,relation,to) with possibly different confidence; last write
        # wins on which one's confidence survives into the response, matching
        # tkg_search()/graphrag_search()'s own all_edges dict-merge pattern
        # for get_subgraph()'s edges (unaffected by this change -- it already
        # merged on the same key before this fix existed to need it).
        deduped: dict[tuple, GraphEdge] = {}
        for e in edge_rows:
            edge = GraphEdge(**_clean(e))
            deduped[(edge.from_id, edge.relation, edge.to_id)] = edge
        edges = list(deduped.values())

        for target in entities:
            target_id = target["entity_id"]
            if target_id == seed_id:
                continue
            prows = await client.query(
                db, "SELECT shortestPath((SELECT FROM Entity WHERE entity_id = :a), "
                    "(SELECT FROM Entity WHERE entity_id = :b), 'BOTH', 'RELATION').name AS path "
                    "FROM (SELECT 1)",
                params={"a": seed_id, "b": target_id})
            path = (prows[0].get("path") if prows else None) or []
            if path:
                paths.append(path)
    return GraphSearchResponse(entities=entities, edges=edges, paths=paths)


def _entity_for_response(e: dict) -> dict:
    """Adds Module 20's matched_entities field names (`id`, `node_kind`) on
    top of Module 25's own (`entity_id`, `mention_count`) -- additive, not a
    rename, so every internal caller still reading e["entity_id"] (subgraph
    expansion, TemporalFact filtering) is unaffected. Found missing by
    comparing live responses from both endpoints against the same real data,
    not from the Pydantic schemas alone (matched_entities is List[Dict[str,
    Any]] on both sides, so the schemas never caught this)."""
    return {**e, "id": e.get("entity_id"), "node_kind": "entity"}


def _moment_for_response(m: dict) -> dict:
    """Same reasoning as _entity_for_response, for matched_moments/moment_ids
    (`id`, `node_kind`, `lvlm_description` on top of `chunk_id`/`lvlm_desc`).
    lvlm_desc is absent on whole-asset summary-embedding moments (no ASR/LVLM
    per-segment fields), so the alias is only added when the source key
    actually exists."""
    out = {**m, "id": m.get("chunk_id"), "node_kind": "moment"}
    if "lvlm_desc" in m:
        out["lvlm_description"] = m["lvlm_desc"]
    return out


async def tkg_search(client: ArcadeDBClient, branch_id: str, req: TKGRequest) -> TKGResponse:
    """TKG (Batch 5 of the graph/TKG parity plan): natural-language query ->
    entity fulltext search -> subgraph expansion -> time-windowed TemporalFacts
    for the matched entities. Ported from Module 20's tkg_query() handler
    (main.py:383), reusing graph_query.py's Batch 2/3 primitives -- replaces
    the old entity_name-exact-match-only version (no NL understanding at all).

    Same node/edge accumulation as graphrag_search(), plus Module 20's own
    fallback: if the matched entities' subgraphs found no moments at all, fall
    back to searching moments directly by the query text."""
    db = await _ensure_ready(client, branch_id)

    entities = await graph_query.fulltext_search_entities(
        client, branch_id, req.query, limit=req.limit, score_threshold=req.score_threshold)

    all_nodes: dict[str, dict] = {}
    all_edges: dict[tuple, dict] = {}
    all_moments: dict[str, dict] = {}

    if entities:
        sg_results = await asyncio.gather(
            *[graph_query.get_subgraph(client, branch_id, e["entity_id"], max_depth=req.max_depth, limit=req.limit)
              for e in entities],
            return_exceptions=True)
        for sg in sg_results:
            if isinstance(sg, Exception):
                continue
            for n in sg["nodes"]:
                all_nodes[n["id"]] = n
                if "Chunk" in n["labels"]:
                    all_moments.setdefault(n["id"], {"moment_id": n["id"], **n["properties"]})
            for edge in sg["edges"]:
                all_edges[(edge["from_id"], edge["type"], edge["to_id"])] = edge

    if not all_moments:
        try:
            moment_hits = await graph_query.fulltext_search_moments(
                client, branch_id, req.query, limit=req.limit,
                score_threshold=req.score_threshold * 0.6)  # same 0.6 discount Module 20 uses for moments
            for m in moment_hits:
                all_moments.setdefault(m["chunk_id"], {"moment_id": m["chunk_id"], **m})
        except Exception as exc:
            logger.warning("[tkg_search] moment fulltext fallback failed: %s", exc)

    entity_ids = [e["entity_id"] for e in entities]
    temporal_facts: list[dict] = []
    if entity_ids:
        # status IS NULL treated as active, not excluded -- TemporalFact.status
        # is a new property (see schema_init.py); existing per-user databases
        # created before it existed have no way to backfill it, so NULL means
        # "predates the migration", not "archived".
        clauses, params = ["entity_id IN :eids", "(status IS NULL OR status = 'active')"], {"eids": entity_ids}
        if req.time_start is not None:
            clauses.append("(time_start IS NULL OR time_start >= :ts)")
            params["ts"] = req.time_start
        if req.time_end is not None:
            clauses.append("(time_end IS NULL OR time_end <= :te)")
            params["te"] = req.time_end
        sql = (f"SELECT fact_id, entity, entity_id, relation, value, time_start, time_end, "
               f"confidence, source_version_id FROM TemporalFact WHERE {' AND '.join(clauses)} "
               f"ORDER BY time_start ASC")
        temporal_facts = [_clean(r) for r in await client.query(db, sql, params=params)]

    return TKGResponse(
        query=req.query, matched_entities=[_entity_for_response(e) for e in entities],
        subgraph_nodes=list(all_nodes.values()), subgraph_edges=list(all_edges.values()),
        temporal_facts=temporal_facts,
        moment_ids=[_moment_for_response(m) for m in all_moments.values()],
    )


async def graphrag_search(client: ArcadeDBClient, branch_id: str,
                          req: GraphRAGRequest) -> GraphRAGResponse:
    """GraphRAG (Batch 4 of the graph/TKG parity plan): natural-language query
    -> entity + moment fulltext search -> subgraph expansion -> citations.
    Ported from Module 20's query_graph_rag() (neo4j_reader.py), same
    assembly order, onto graph_query.py's ArcadeDB/Cypher primitives:

    1. entity fulltext search (graph_query.fulltext_search_entities)
    2. moment fulltext search (graph_query.fulltext_search_moments), independent
       of (1) -- a query can hit moments with no entity match at all
    3. subgraph expansion per matched entity, in parallel (graph_query.get_subgraph)
       -- collects nodes/edges/moment ids reachable from each matched entity
    4. per-moment subgraph per matched moment, in parallel (graph_query.get_moment_subgraph)
       -- collects the Source + Entities around each fulltext-matched moment
    5. dedupe nodes/edges by id (last write wins, same as Module 20); moments
       found by (2) overwrite any bare entry from (3) so citations carry
       matched_via/snippets wherever fulltext actually found the moment
    6. sort moments by (version_id, start_sec), build [n]-labeled citations

    Unlike Module 20, no separate "enrich moments with filename/upload_time
    from Source nodes" step is needed -- Module 25's Chunk rows already carry
    filename/upload_time directly (Module 20's Moment nodes don't)."""
    await _ensure_ready(client, branch_id)  # fail fast if the caller's DB doesn't exist yet

    entities = await graph_query.fulltext_search_entities(
        client, branch_id, req.query, limit=req.limit, score_threshold=req.score_threshold)
    moments = await graph_query.fulltext_search_moments(
        client, branch_id, req.query, limit=req.limit, strategy=req.strategy,
        score_threshold=req.score_threshold * 0.6)  # same 0.6 discount Module 20 uses for moments

    all_nodes: dict[str, dict] = {}
    all_edges: dict[tuple, dict] = {}
    all_moment_ids: dict[str, dict] = {}

    if entities:
        sg_results = await asyncio.gather(
            *[graph_query.get_subgraph(client, branch_id, e["entity_id"], max_depth=req.max_depth, limit=req.limit)
              for e in entities],
            return_exceptions=True)
        for sg in sg_results:
            if isinstance(sg, Exception):
                continue
            for n in sg["nodes"]:
                all_nodes[n["id"]] = n
                if "Chunk" in n["labels"]:
                    all_moment_ids.setdefault(n["id"], {"moment_id": n["id"], **n["properties"]})
            for edge in sg["edges"]:
                all_edges[(edge["from_id"], edge["type"], edge["to_id"])] = edge

    if moments:
        moment_sg_results = await asyncio.gather(
            *[graph_query.get_moment_subgraph(client, branch_id, m["chunk_id"]) for m in moments],
            return_exceptions=True)
        for moment, sg in zip(moments, moment_sg_results):
            if not isinstance(sg, Exception):
                for n in sg["nodes"]:
                    all_nodes[n["id"]] = n
            # fulltext-matched moment overwrites any bare entry from the
            # subgraph pass above, so matched_via/snippets survive into citations
            all_moment_ids[moment["chunk_id"]] = {"moment_id": moment["chunk_id"], **moment}

    sorted_moments = sorted(
        all_moment_ids.values(), key=lambda m: (m.get("version_id") or "", m.get("start_sec") or 0))
    citations = [
        {
            "label": f"[{i + 1}]",
            "moment_id": m["moment_id"],
            "version_id": m.get("version_id"),
            "start_sec": m.get("start_sec"),
            "end_sec": m.get("end_sec"),
            "matched_via": m.get("matched_via", []),
            "snippets": m.get("snippets", {}),
        }
        for i, m in enumerate(sorted_moments)
    ]

    return GraphRAGResponse(
        query=req.query, strategy=req.strategy,
        matched_entities=[_entity_for_response(e) for e in entities],
        matched_moments=[_moment_for_response(m) for m in moments],
        nodes=list(all_nodes.values()), relationships=list(all_edges.values()),
        moment_ids=[_moment_for_response(m) for m in sorted_moments], citations=citations,
    )


async def list_entities(client: ArcadeDBClient, branch_id: str, *,
                        type: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    """Paginated list of Entity vertices with type and mention_count."""
    db = await _ensure_ready(client, branch_id)
    params: dict = {}
    where = ""
    if type is not None:
        where = " WHERE type = :t"
        params["t"] = type
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    rows = await client.query(
        db, f"SELECT entity_id, name, type, description, mention_count FROM Entity{where} "
            f"ORDER BY mention_count DESC SKIP {offset} LIMIT {limit}", params=params)
    total = (await client.query(db, f"SELECT count(*) AS total FROM Entity{where}", params=params))
    return {"entities": [_clean(r) for r in rows],
            "total": total[0].get("total", 0) if total else 0,
            "limit": limit, "offset": offset}


async def get_entity(client: ArcadeDBClient, branch_id: str, name: str) -> dict | None:
    """One Entity vertex plus its outgoing and incoming RELATION edges."""
    db = await _ensure_ready(client, branch_id)
    seed_id = await _resolve_entity_id(client, db, name)  # not `WHERE name = :n` -- see _resolve_entity_id
    if seed_id is None:
        return None
    verts = await client.query(
        db, "SELECT entity_id, name, type, description, mention_count "
            "FROM Entity WHERE entity_id = :id", params={"id": seed_id})
    if not verts:
        return None
    out_e = await client.query(
        db, "SELECT relation, inV().name AS to_name, inV().entity_id AS to_id, confidence "
            "FROM RELATION WHERE outV().name = :n", params={"n": name})
    in_e = await client.query(
        db, "SELECT relation, outV().name AS from_name, outV().entity_id AS from_id, confidence "
            "FROM RELATION WHERE inV().name = :n", params={"n": name})
    # Dedup by (relation, to_id)/(relation, from_id) -- same reasoning as
    # graph_search()'s edge dedup: index_relationship() now gives independent
    # sources their own edge for the same fact (see its docstring), so this
    # raw per-edge query can return what looks like duplicate rows for one
    # relationship. Last write wins on confidence, matching graph_search().
    out_dedup: dict[tuple, dict] = {}
    for e in out_e:
        e = _clean(e)
        out_dedup[(e.get("relation"), e.get("to_id"))] = e
    in_dedup: dict[tuple, dict] = {}
    for e in in_e:
        e = _clean(e)
        in_dedup[(e.get("relation"), e.get("from_id"))] = e
    return {"entity": _clean(verts[0]),
            "outgoing": list(out_dedup.values()),
            "incoming": list(in_dedup.values())}


async def co_occurring_entities(client: ArcadeDBClient, branch_id: str, name: str, limit: int = 20) -> dict | None:
    """Entities that most often co-occur with the named entity (CO_OCCURS_WITH,
    ranked by weight -- see graph_query.co_occurring_entities's docstring for
    why not count(r)). Resolves name -> entity_id the same guarded way
    get_entity() does, not a direct `WHERE name = :n` -- see _resolve_entity_id."""
    db = await _ensure_ready(client, branch_id)
    seed_id = await _resolve_entity_id(client, db, name)
    if seed_id is None:
        return None
    others = await graph_query.co_occurring_entities(client, branch_id, seed_id, limit=limit)
    return {"entity_id": seed_id, "co_occurring": others}


async def raw_graph_query(client: ArcadeDBClient, branch_id: str, sql: str) -> list:
    """Run a raw, read-only SELECT against the user's DB (power-user escape hatch)."""
    db = await _ensure_ready(client, branch_id)
    if not is_read_only_select(sql):
        raise ValueError("only a single read-only SELECT statement is allowed")
    return [_clean(r) for r in await client.query(db, sql)]


async def hybrid_search(client: ArcadeDBClient, branch_id: str, req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    db = await _ensure_ready(client, branch_id)
    te = time.perf_counter()
    qvec = (await embed_texts([req.query]))[0]
    embed_sec = time.perf_counter() - te
    bm25_params: dict = {}
    bm25_cond = _bm25_condition("Chunk", "text", req.query, "q", bm25_params)
    filter_params: dict = {}
    where = _filters(req, filter_params)
    # Widen the fusion pool before reranking, then truncate to the caller's
    # top_k after — mirrors module 17's own hybrid_search() (search_depth =
    # max(top_k*3, RERANK_DEPTH)).
    fusion_limit = max(req.top_k * 3, settings.rerank_depth)
    # `where` (status='active' by default, plus type/filename/etc.) pushed
    # into BOTH fuse() inputs, not applied as an outer post-filter -- the old
    # `){where}` post-filter let an archived Chunk occupy one of the RRF
    # inputs' own limited candidate slots (vectorNeighbors' {fusion_limit}
    # count, or the bm25_cond subquery competing for the same RRF rank
    # positions) before being discarded afterward, which can skew which
    # *active* chunks survive the fusion_limit cutoff, not just leak archived
    # rows into the final output. vectorNeighbors' filter option (a RID list
    # from a subquery) confirmed live on this ArcadeDB version -- see
    # vector_search() above for the exact `.rid`-projection syntax it needs,
    # and _rid_filter_matches()'s docstring for why a legitimately-empty
    # filter needs its own pre-check before this query runs at all.
    sql = (f"SELECT {_PAYLOAD}, @rid FROM (SELECT expand(vector.fuse("
           f"vectorNeighbors('Chunk[embedding]', :qvec, {fusion_limit}, "
           f"{{'filter': (SELECT @rid AS rid FROM Chunk WHERE {where}).rid}}), "
           f"(SELECT FROM Chunk WHERE {bm25_cond} AND {where}), "
           f"{{'fusion':'RRF','limit':{fusion_limit}}})))")
    # Source gets its own, separate RRF fusion (its own vector index, its own
    # BM25 index) -- not merged into the SQL above (vectorNeighbors/
    # SEARCH_INDEX are both type-specific, no single query spans two vertex
    # types). The two already-fused pools are concatenated and handed to the
    # cross-encoder rerank below together, which naturally picks the true
    # top_k regardless of which pool a candidate came from -- no separate
    # RRF-of-RRFs merge step needed, the rerank step already does that job
    # for the Chunk-only pool today.
    src_bm25_params: dict = {}
    src_bm25_cond = _bm25_condition("Source", "summary", req.query, "q", src_bm25_params)
    src_filter_params: dict = {}
    src_where = _source_filters(req, src_filter_params)
    src_sql = (f"SELECT {_SOURCE_PAYLOAD}, @rid FROM (SELECT expand(vector.fuse("
              f"vectorNeighbors('Source[embedding]', :qvec, {fusion_limit}, "
              f"{{'filter': (SELECT @rid AS rid FROM Source WHERE {src_where}).rid}}), "
              f"(SELECT FROM Source WHERE {src_bm25_cond} AND {src_where}), "
              f"{{'fusion':'RRF','limit':{fusion_limit}}})))")
    tq = time.perf_counter()
    # Both fuse() branches above have a vectorNeighbors half, so a
    # legitimately-empty filter needs the same _rid_filter_matches() guard
    # as vector_search() -- otherwise the empty RID subquery makes
    # vectorNeighbors ignore the filter and leak unfiltered candidates into
    # the RRF fusion even though the BM25 half of the same fuse() call
    # correctly contributes nothing for that filter.
    rows: list = []
    if _include_chunk(req) and await _rid_filter_matches(client, db, "Chunk", where, filter_params):
        params = {"qvec": qvec, **bm25_params, **filter_params}
        rows = await client.query(db, sql, params=params)
    src_rows: list = []
    if _include_source(req) and await _rid_filter_matches(client, db, "Source", src_where, src_filter_params):
        src_params = {"qvec": qvec, **src_bm25_params, **src_filter_params}
        src_rows = await client.query(db, src_sql, params=src_params)
    # Named to match module 17's `fusion_sec` — the closest honest match.
    # ArcadeDB does vector retrieval + BM25 retrieval + RRF fusion inside one
    # query, unlike module 17 (separate OpenSearch/Qdrant calls it can time
    # individually as bm25_sec/vector_sec, then a distinct fusion_sec step),
    # so there is no real bm25_sec/vector_sec split here to report.
    fusion_sec = time.perf_counter() - tq

    tr = time.perf_counter()
    results = await rerank(req.query, _results(rows + src_rows, rank_score=True), req.top_k)
    rerank_sec = time.perf_counter() - tr

    return SearchResponse(results=results,
                          timing={"total_sec": round(time.perf_counter() - t0, 6),
                                  "embed_sec": round(embed_sec, 6), "fusion_sec": round(fusion_sec, 6),
                                  "rerank_sec": round(rerank_sec, 6)})
