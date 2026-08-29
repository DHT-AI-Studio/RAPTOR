"""Entity / moment full-text search primitives for TKG / GraphRAG (PA-7 parity
work, Batch 2 of the plan to bring Module 25's TKG/GraphRAG up to Module 20's
level: natural-language query -> entity search -> subgraph expansion ->
citations).

Design note -- why the Python-side matching below still exists even though
the underlying indexes (schema_init.py) ARE now configured with a proper CJK
analyzer (org.apache.lucene.analysis.cjk.CJKAnalyzer, bigram tokenization --
same concept as Module 20's `cjk` analyzer on its Neo4j fulltext indexes;
confirmed live this actually works on this ArcadeDB version and eliminates
the false-positive matches an earlier version of this file hit with the
default StandardAnalyzer, e.g. "以色列" matching unrelated text via the bare
character "色"):

- SEARCH_INDEX's bigram match is still an OR across a query's bigrams, not a
  phrase match (querying "美國前總統" matches text containing any of "美國"/
  "國前"/"前總"/"總統" individually) -- the coarse retrieval below is over-
  inclusive by design, same as Module 20's own Lucene score-threshold cutoff
  being a coarse filter, not a final answer.
- Citation snippets and the literal/semantic/hybrid strategy filter are pure
  Python logic in Module 20 too (`_detect_matches`, neo4j_reader.py:592) --
  ported here practically verbatim, generalized to take an arbitrary field
  dict instead of 20's hardcoded three Moment fields.
- Primary ranking/score_threshold filtering now uses ArcadeDB's own `$score`
  query variable (a real BM25 relevance score, confirmed live -- see
  fulltext_search_entities' docstring), the same underlying idea as Module
  20's Neo4j Lucene score. `_detect_matches`' rank is kept only as a cheap
  false-positive backstop (drop rank == -1, i.e. no literal phrase/bigram
  actually found in any field text) and for snippet extraction, not as the
  primary sort key -- an earlier version of this file had no access to a
  real score and used the rank itself as a fake substitute; that assumption
  ("ArcadeDB has no relevance score, only true/false") was never verified
  and turned out to be wrong.
- A lone CJK character produces no bigrams for CJKAnalyzer to match at all,
  so SEARCH_INDEX on one always returns zero rows -- confirmed live this was
  silently starving _detect_matches()'s own unigram-safe re-rank of any row
  to even look at, even though that re-rank was written with exactly this
  case in mind (_cjk_phrases()'s unigram tier). searcher.py's BM25 search
  already had the fix for this (_single_cjk_char/_bm25_condition: fall
  back to a plain LIKE scan instead of SEARCH_INDEX for a lone-character
  query) but it was never ported to this file's own SEARCH_INDEX calls until
  now -- see _single_cjk_char/_fulltext_or_conditions below.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.indexer import DatabaseNotInitializedError

_SNIPPET_RADIUS = 40  # characters around the match, for citation snippets

_META = {"@rid", "@type", "@cat", "@props", "@in", "@out"}


def _clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _META}


async def _ensure_ready(client: ArcadeDBClient, branch_id: str) -> str:
    db = db_name_for(branch_id)
    if not await client.database_exists(db):
        raise DatabaseNotInitializedError(
            "Personal database not initialized. Call POST /internal/db/init first.")
    return db


def _is_cjk(ch: str) -> bool:
    """True for Chinese / Japanese / Korean unified ideographs. Same ranges as
    kafka_consumer.py's _is_cjk (video ASR word-joining) -- kept independent
    rather than shared since the two live in different deployable services."""
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
    )


def _single_cjk_char(query_text: str) -> Optional[str]:
    """Returns the query's lone CJK character when its CJK run is exactly one
    character, else None. Mirrors searcher._single_cjk_char() -- that
    function's own docstring says it mirrors this file's _cjk_phrases()
    unigram-fallback condition; this is the missing SQL-layer half of that
    same fix, which was applied to searcher.bm25_search's SEARCH_INDEX calls
    but never ported over here, to fulltext_search_entities/
    fulltext_search_moments's SEARCH_INDEX calls -- confirmed live: a lone
    CJK character produces no bigrams for a CJKAnalyzer-backed FULL_TEXT
    index to match against, so SEARCH_INDEX on one always returns zero rows,
    silently starving _detect_matches()'s own already-written unigram-safe
    re-rank of any row to even look at.

    Returns the character itself, not just a bool, because callers building
    a LIKE pattern must match on *that one character*, not the caller's full
    query_text -- a caller here is typically a whole natural-language
    question (pipeline.py's GraphRAGRequest/TKGRequest pass `query=question`
    verbatim, e.g. "who is 狗 associated with?"), and `LIKE '%<question>%'`
    against that whole English+CJK sentence would essentially never match
    anything indexed. Confirmed live this was a real bug, not theoretical:
    a bare "狗" query matched fine, but the identical entity failed to match
    once embedded in an English sentence, until this returned just the
    character "狗", not the sentence, for the LIKE pattern below."""
    cjk_run = "".join(c for c in query_text if _is_cjk(c))
    return cjk_run if len(cjk_run) == 1 else None


def _fulltext_or_conditions(
    vertex_type: str, fields: List[str], query_text: str, param: str, params: dict,
) -> str:
    """`field1 <op> :param OR field2 <op> :param OR ...` -- SEARCH_INDEX
    normally, a plain substring LIKE for a lone-CJK-character query instead
    (see _single_cjk_char). Same fix as searcher._bm25_condition(),
    generalized across multiple OR'd fields since the two callers below
    query 2-3 fields per SQL statement, not one. Sets params[param] to
    whichever value the chosen condition needs (the single CJK character for
    LIKE, raw query_text for SEARCH_INDEX) -- shared across every field's OR
    clause since they all key off the same query_text and therefore the
    same mode."""
    single_char = _single_cjk_char(query_text)
    if single_char is not None:
        params[param] = f"%{single_char}%"
        return " OR ".join(f"{f} LIKE :{param}" for f in fields)
    params[param] = query_text
    return " OR ".join(f"SEARCH_INDEX('{vertex_type}[{f}]', :{param}) = true" for f in fields)


def _cjk_phrases(query_text: str) -> List[str]:
    """Priority-ordered candidate phrases to look for in a field's text:
    1. the full query as one phrase
    2. whitespace-split tokens (for English)
    3. CJK bigrams (overlapping), from the query's CJK run
    4. CJK unigrams -- only when the query's CJK run itself is a single
       character (nothing longer to fall back from). A live test against
       real data caught single-character unigram matching as real noise:
       querying "以色列" (strategy=literal) surfaced clips about a sleeping
       dog and a cat purely because "色" -- an extremely common character --
       appears in "淺藍色" ("light blue"). This was found (and re-verified
       fixed) even after the underlying index was reconfigured with a proper
       CJK bigram analyzer, because SEARCH_INDEX's bigram match is still an
       OR across bigrams, not a phrase match -- a bare unigram tier in this
       Python re-rank would still add matches the index-level bigram
       matching wouldn't have produced on its own, for no real benefit.

    Earlier entries are more specific / higher-confidence matches -- used as
    a ranking key by the callers below (lower index = better match)."""
    phrases: List[str] = [query_text]
    phrases.extend(t for t in query_text.split() if t and t != query_text)

    cjk_run = "".join(c for c in query_text if _is_cjk(c))
    if len(cjk_run) >= 2:
        for i in range(len(cjk_run) - 1):
            phrases.append(cjk_run[i:i + 2])
    elif cjk_run:
        phrases.append(cjk_run)

    seen, unique = set(), []
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _detect_matches(query_text: str, fields: Dict[str, str]) -> Tuple[List[str], Dict[str, str], int]:
    """Inspect each named field's text for the query. Returns:
      matched_via: field names where some phrase was found
      snippets:    {field_name: short context window around the first match}
      best_rank:   index into _cjk_phrases()'s priority list of the highest-
                    priority phrase that matched anywhere (lower = better
                    match); -1 if nothing matched at all.

    Ported from Module 20's _detect_matches (neo4j_reader.py:592), generalized
    to take an arbitrary field dict (20's version hardcodes the three Moment
    text fields) and to also return best_rank, since here this drives primary
    ranking rather than just tagging citation snippets."""
    if not query_text:
        return [], {}, -1

    phrases = _cjk_phrases(query_text)
    matched_via: List[str] = []
    snippets: Dict[str, str] = {}
    best_rank = -1

    for field_name, field_text in fields.items():
        if not field_text:
            continue
        lower_text = field_text.lower()
        for rank, phrase in enumerate(phrases):
            idx = lower_text.find(phrase.lower())
            if idx >= 0:
                matched_via.append(field_name)
                start = max(0, idx - _SNIPPET_RADIUS)
                end = min(len(field_text), idx + len(phrase) + _SNIPPET_RADIUS)
                snippet = field_text[start:end].strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(field_text):
                    snippet = snippet + "…"
                snippets[field_name] = snippet
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank
                break  # one match per field is enough

    return matched_via, snippets, best_rank


async def fulltext_search_entities(
    client: ArcadeDBClient, branch_id: str, query_text: str, limit: int = 10,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Coarse SEARCH_INDEX over Entity(name)/Entity(description), re-ranked by
    CJK-aware phrase matching in Python. Returns entity dicts (entity_id,
    name, type, description, mention_count, score), best matches first.

    score_threshold, if given, drops matches at/below it (strict `>`, same as
    Module 20's `WHERE score > $threshold`) against ArcadeDB's own `$score` --
    confirmed live (this ArcadeDB instance, 26.6.1) that FULL_TEXT indexes
    return a real BM25 relevance score via the `$score` query variable, not
    just a true/false match (an earlier version of this function assumed
    otherwise and built a fake Python-side rank->score approximation instead;
    that assumption was wrong and never verified -- ArcadeDB's docs and a
    live query both confirm `$score` works, including across an OR of
    multiple SEARCH_INDEX() calls, and BM25 is the default similarity model
    for a FULL_TEXT index unless METADATA explicitly sets `similarity:
    "CLASSIC"` -- ours don't, so schema_init.py's indexes are already BM25).
    Same BM25 family as Module 20's Neo4j Lucene fulltext score, though the
    exact numeric scale isn't guaranteed identical (different corpora /
    analyzer tuning) -- a best-effort comparable filter, not byte-identical."""
    db = await _ensure_ready(client, branch_id)
    if not query_text.strip():
        return []

    single_char = _single_cjk_char(query_text) is not None
    params: dict = {}
    cond = _fulltext_or_conditions("Entity", ["name", "description"], query_text, "q", params)
    # LIKE-fallback rows have no real BM25 $score (ArcadeDB's $score is a
    # FULL_TEXT-index-only feature -- a LIKE predicate leaves it null), so
    # score_threshold can't be applied to them the normal way; see below.
    rows = await client.query(
        db,
        "SELECT entity_id, name, type, description, mention_count, $score AS _score FROM Entity "
        f"WHERE {cond} "
        f"ORDER BY $score DESC LIMIT {max(limit * 5, 50)}",
        params=params,
    )

    scored = []
    for r in rows:
        r = _clean(r)
        score = float(r.pop("_score", 0.0) or 0.0)
        fields = {"name": r.get("name") or "", "description": r.get("description") or ""}
        _, _, rank = _detect_matches(query_text, fields)
        if rank == -1:
            continue  # coarse index false-positive (shared a character with a different phrase entirely)
        # single_char rows have no real $score to threshold against -- rank
        # (an actual literal/unigram match, just confirmed above) is already
        # the real quality gate for them, same as searcher.bm25_search
        # falling back to position rank instead of a null $score.
        if not single_char and score_threshold is not None and score <= score_threshold:
            continue
        r["score"] = round(score, 4)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


async def fulltext_search_moments(
    client: ArcadeDBClient, branch_id: str, query_text: str, limit: int = 10,
    strategy: str = "hybrid", score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Coarse SEARCH_INDEX over Chunk's asr_text/lvlm_desc/contextual_text,
    re-ranked the same way as fulltext_search_entities. Each result carries
    matched_via/snippets for citation building.

    strategy (same semantics as Module 20):
      'literal'  -- keep only hits in asr_text or lvlm_desc (fact-style query)
      'semantic' -- keep only hits in contextual_text (concept-style query)
      'hybrid'   -- keep everything (default)

    score_threshold, if given, drops matches at/below it against ArcadeDB's
    real `$score` -- see fulltext_search_entities for why this is a genuine
    BM25 score, not a Python approximation.
    """
    db = await _ensure_ready(client, branch_id)
    if not query_text.strip():
        return []

    single_char = _single_cjk_char(query_text) is not None
    params: dict = {}
    cond = _fulltext_or_conditions(
        "Chunk", ["asr_text", "lvlm_desc", "contextual_text"], query_text, "q", params)
    # status='active': matches searcher.py's _filters()/_source_filters() default --
    # without this, an archived Chunk (e.g. the original half of a clone_point()
    # dedup pair) surfaces in graphrag/tkg moment results indistinguishably from
    # active content. Found doing a full sweep of every Chunk/Source query in this
    # module after a status-filter gap was found in get_subgraph(); no caller
    # currently needs archived moments back from this function, unlike
    # searcher.SearchRequest's explicit req.status override, so this is a plain
    # literal, not a parameter -- add one if a real caller ever needs it.
    #
    # `cond` MUST be parenthesized here: it's a 3-way OR chain from
    # _fulltext_or_conditions() (asr_text OR lvlm_desc OR contextual_text), and
    # SQL's AND binds tighter than OR -- an unparenthesized `{cond} AND
    # status='active'` parses as `A OR B OR (C AND status='active')`, applying
    # the status check to only the last OR arm. Shipped once without the
    # parens (real live bug, not caught by py_compile/unit tests since it's a
    # runtime WHERE-clause semantics issue, not a syntax error); caught by
    # re-testing PR #107's own fix live post-deploy and finding archived
    # content still leaking through graphrag_search -- confirmed via a raw
    # ArcadeDB query showing 4 archived rows without the parens, 0 with them.
    rows = await client.query(
        db,
        "SELECT chunk_id, type, version_id, asset_path, filename, upload_time, "
        "chunk_index, start_sec, end_sec, asr_text, lvlm_desc, contextual_text, "
        "$score AS _score "
        f"FROM Chunk WHERE ({cond}) AND status = 'active' "
        f"ORDER BY $score DESC LIMIT {max(limit * 5, 50)}",
        params=params,
    )

    scored = []
    for r in rows:
        r = _clean(r)
        score = float(r.pop("_score", 0.0) or 0.0)
        fields = {
            "asr_text": r.get("asr_text") or "",
            "lvlm_desc": r.get("lvlm_desc") or "",
            "contextual_text": r.get("contextual_text") or "",
        }
        matched_via, snippets, rank = _detect_matches(query_text, fields)
        if rank == -1:
            continue
        # see fulltext_search_entities for why single_char rows skip this
        if not single_char and score_threshold is not None and score <= score_threshold:
            continue
        if strategy == "literal" and not ({"asr_text", "lvlm_desc"} & set(matched_via)):
            continue
        if strategy == "semantic" and "contextual_text" not in matched_via:
            continue
        r["matched_via"] = matched_via
        r["snippets"] = snippets
        r["score"] = round(score, 4)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


# ---------------------------------------------------------------------------
# Subgraph expansion (Batch 3) -- ArcadeDB's own Cypher dialect, not a hand-
# rolled SQL TRAVERSE. Verified live that ArcadeDB's /api/v1/query accepts
# language="cypher" and supports everything Module 20's APOC-based
# apoc.path.subgraphAll needs an equivalent for: multi-edge-type variable-
# length patterns (`[:A|B*1..n]`), named $param bindings, `path`/
# `relationships(path)`, and `OPTIONAL MATCH` + `collect(DISTINCT ...)` for
# in-database dedup. No APOC-equivalent redesign needed -- this is a close
# port of Module 20's Cypher patterns onto ArcadeDB's own Cypher engine.
#
# One real constraint found live: a variable-length range bound (`*1..N`)
# cannot be a bound $parameter -- ArcadeDB rejects it ("Parameters cannot be
# used as predicates in MATCH patterns"). max_depth is clamped to a small int
# range server-side first (same clamp searcher.py::graph_search already
# uses), then that already-validated int is f-string-interpolated into the
# query text -- safe because it's never raw caller input by that point, only
# actual entity/chunk ids go through real $-bound params.
# ---------------------------------------------------------------------------

_SUBGRAPH_EDGE_TYPES = "RELATION|MENTIONS|HAS_TEMPORAL_FACT|CO_OCCURS_WITH|HAS_CHUNK|MENTIONED_IN"
# Now matches Module 20's own relationshipFilter (neo4j_reader.py:220) in
# full: "RELATION|CO_OCCURS_WITH|MENTIONED_IN|HAS_TEMPORAL_FACT|HAS_MOMENT|
# APPEARS_IN" -- APPEARS_IN (Entity->Moment) is MENTIONS here, HAS_MOMENT
# (Source->Moment, structural) is HAS_CHUNK here. MENTIONED_IN (Entity->
# Source, document-level mention, from summary-level extraction) used to
# have no equivalent -- Module 25 had no document-level mention distinct
# from moment-level ones because the whole-asset summary was faked as a
# Chunk with a moment-level MENTIONS edge; now that summary lives on Source
# directly (indexer.index_source_summary()) and its entities get a real
# MENTIONED_IN edge (graph_indexer.index_mentioned_in()), this is a genuine
# 1:1 port, not an architecture difference anymore.


def _serialize_node(raw: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    """Clean an ArcadeDB Cypher node object into a dict shaped like Module 20's
    _serialize_node -- id + labels + properties, PLUS the same label-specific
    fields Module 20 flattens onto the node itself (neo4j_reader.py:487).
    Found missing by comparing live responses from both endpoints against the
    same real data: `nodes` is List[Dict[str, Any]] on both response models,
    so the schemas alone never caught that Module 20 duplicates these fields
    at the top level, not just inside `properties`.

    No branch_id on Chunk (Module 20's Moment carries one; Module 25 has no
    such property to report -- each user's ArcadeDB database already IS the
    branch, same reasoning as GraphRAGRequest/TKGRequest never taking a
    branch_id param). Source's "title" doesn't exist as its own property
    here, unlike Module 20's Source.title -- filename fills that role.

    One deliberate naming departure: this field is chunk_index, not Module
    20's moment_index -- Module 20 has a dedicated "Moment" node type, so a
    moment-specific name made sense there; Module 25 unifies every media type
    into one Chunk type (video/audio/image/document alike), so keeping a
    video-flavored name for what's really "this chunk's sequence index
    within its source asset" stopped making sense. Same property name change
    all the way down to the ArcadeDB schema and the Kafka ingest path
    (kafka_consumer.py, schema_init.py) -- not just this serializer.

    Chunk carries a 1024-dim embedding vector as a raw property; always
    dropped here (never useful to a subgraph caller, and large enough to
    bloat every response)."""
    props = _clean(raw)
    props.pop("embedding", None)
    props.pop("sparse_indices", None)
    props.pop("sparse_weights", None)

    if "Entity" in labels:
        node_id = props.get("entity_id", "")
    elif "Chunk" in labels:
        node_id = props.get("chunk_id", "")
    elif "TemporalFact" in labels:
        node_id = props.get("fact_id", "")
    elif "Source" in labels:
        node_id = props.get("version_id", "")
    else:
        node_id = raw.get("@rid", "")

    node = {"id": node_id, "labels": labels, "properties": props}

    if "Entity" in labels:
        node.update({
            "name": props.get("name", ""),
            "type": props.get("type", ""),
            "description": props.get("description", ""),
        })
    elif "Chunk" in labels:
        node.update({
            "version_id": props.get("version_id", ""),
            "asset_path": props.get("asset_path", ""),
            "chunk_index": props.get("chunk_index"),
            "start_sec": props.get("start_sec"),
            "end_sec": props.get("end_sec"),
            "asr_text": props.get("asr_text", ""),
            "lvlm_description": props.get("lvlm_desc", ""),
            "contextual_text": props.get("contextual_text", ""),
        })
    elif "TemporalFact" in labels:
        node.update({
            "entity": props.get("entity", ""),
            "relation": props.get("relation", ""),
            "value": props.get("value", ""),
            "time_start": props.get("time_start"),
            "time_end": props.get("time_end"),
            "confidence": props.get("confidence"),
            "entity_id": props.get("entity_id"),
            "moment_id": props.get("moment_id"),
            "source_document_id": props.get("source_version_id"),
        })
    elif "Source" in labels:
        node.update({
            "title": props.get("filename", ""),
            "media_type": props.get("media_type", ""),
            "asset_path": props.get("asset_path", ""),
        })

    return node


def _serialize_edge(raw: Dict[str, Any], rid_to_id: Dict[str, str]) -> Dict[str, Any]:
    """rid_to_id maps ArcadeDB's internal @rid (e.g. "#7:5") to the same
    semantic id _serialize_node put in a node's "id" field -- an edge's raw
    @in/@out are only meaningful as rids, and rids aren't exposed anywhere
    else in this API, so this resolves them to the ids a caller can actually
    correlate against `nodes`. Falls back to the raw rid if a node somehow
    wasn't in the node set (shouldn't happen -- both queries share the same
    MATCH pattern -- but better than silently dropping the edge).

    start_id/end_id duplicate from_id/to_id -- Module 20's _serialize_rel
    uses start_id/end_id (neo4j_reader.py:543); from_id/to_id are kept too
    since searcher.py's tkg_search/graphrag_search already dedupe edges by
    (edge["from_id"], edge["type"], edge["to_id"])."""
    out_rid, in_rid = raw.get("@out"), raw.get("@in")
    from_id, to_id = rid_to_id.get(out_rid, out_rid), rid_to_id.get(in_rid, in_rid)
    return {
        "type": raw.get("@type", ""),
        "from_id": from_id,
        "to_id": to_id,
        "start_id": from_id,
        "end_id": to_id,
        "properties": _clean(raw),
    }


async def get_subgraph(
    client: ArcadeDBClient, branch_id: str, entity_id: str, max_depth: int = 2, limit: int = 50,
) -> Dict[str, Any]:
    """Multi-hop subgraph from a seed Entity, across RELATION/MENTIONS/
    HAS_TEMPORAL_FACT edges -- Module 25's equivalent of Module 20's
    get_subgraph() (APOC apoc.path.subgraphAll), via ArcadeDB's own Cypher.
    Returns {"nodes": [...], "edges": [...]}, each node/edge cleaned via
    _serialize_node/_serialize_edge.

    `ALL(n IN nodes(path) WHERE n.status IS NULL OR n.status = 'active')`
    filters out archived Chunk/Source anywhere along the path -- endpoint AND
    intermediate hops -- while passing Entity/TemporalFact through untouched
    (neither has a `status` property, so `.status` is null for them).
    Confirmed live: ArcadeDB's Cypher supports `nodes(path)` + `ALL()`, and a
    real archive-then-query test (archiving a whole video's Source+Chunks via
    Module 13's actual filearchive endpoint, not a direct DB edit) showed
    every path touching the archived version disappearing entirely, at any
    depth -- not just when the archived node happened to be the endpoint."""
    db = await _ensure_ready(client, branch_id)
    depth = max(1, min(int(max_depth), 4))  # matches Module 20's TkgQueryRequest/GraphRagRequest le=4
    lim = max(1, min(int(limit), 200))
    _status_ok = "ALL(n IN nodes(path) WHERE n.status IS NULL OR n.status = 'active')"

    # Seed entity fetched separately -- it's `a` in the pattern below, never
    # `b`, so it wouldn't otherwise appear in the node set at all (edges that
    # point directly at/from the seed still need its rid resolved for
    # from_id/to_id below).
    seed_rows = await client.query(
        db, "MATCH (a:Entity {entity_id: $id}) RETURN a, labels(a) AS node_labels",
        params={"id": entity_id}, language="cypher",
    )

    node_rows = await client.query(
        db,
        f"MATCH path=(a:Entity {{entity_id: $id}})-[:{_SUBGRAPH_EDGE_TYPES}*1..{depth}]-(b) "
        f"WHERE {_status_ok} "
        f"RETURN DISTINCT b, labels(b) AS node_labels LIMIT {lim}",
        params={"id": entity_id}, language="cypher",
    )

    rid_to_id: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    seen_ids = set()
    for key, rows in (("a", seed_rows), ("b", node_rows)):
        for r in rows:
            raw = r[key]
            node = _serialize_node(raw, r["node_labels"])
            rid_to_id[raw.get("@rid")] = node["id"]
            if node["id"] not in seen_ids:
                seen_ids.add(node["id"])
                nodes.append(node)

    # Same ALL(nodes(path)) filter as node_rows above -- per-hop, not just the
    # endpoint, so an edge touching an archived intermediate node can't leak
    # through even when the path's final b is active (e.g. Entity -MENTIONS->
    # archived Chunk -HAS_CHUNK-> active Source: the old endpoint-only filter
    # let this path's edges through since b=Source passed, even though the
    # archived Chunk itself was correctly absent from `nodes` -- a dangling
    # edge reference. ALL(nodes(path)) rejects the whole path instead.
    edge_rows = await client.query(
        db,
        f"MATCH path=(a:Entity {{entity_id: $id}})-[:{_SUBGRAPH_EDGE_TYPES}*1..{depth}]-(b) "
        f"WHERE {_status_ok} "
        f"UNWIND relationships(path) AS r RETURN collect(DISTINCT r) AS edges",
        params={"id": entity_id}, language="cypher",
    )
    edges = [_serialize_edge(e, rid_to_id) for e in (edge_rows[0]["edges"] if edge_rows else [])]

    return {"nodes": nodes, "edges": edges}


async def get_moment_subgraph(client: ArcadeDBClient, branch_id: str, chunk_id: str) -> Dict[str, Any]:
    """Single hop from one Chunk (a "moment") to its Source and the Entities it
    MENTIONS -- Module 25's equivalent of Module 20's get_moment_subgraph()
    (neo4j_reader.py:288). Includes the Chunk itself in `nodes`, matching
    Module 20's `RETURN [m] + sources + entities` -- an earlier version of
    this function omitted it (only returned Source + Entities), a real gap
    found by directly comparing this function's output against Module 20's
    read source, not from the response schemas (nodes is untyped there too).
    Source and the MATCH anchor are both OPTIONAL now so the Chunk itself is
    still returned even if HAS_CHUNK somehow doesn't resolve, matching Module
    20's own MATCH (m) unconditional + OPTIONAL MATCH (s) structure.

    Narrower than Module 20's version in one respect, not fixed here: Module
    20 pulls every Entity MENTIONED_IN the whole Source (document-level);
    this only pulls Entities this specific Chunk MENTIONS (moment-level) --
    Module 25 has no document-level mention edge to walk (see
    _SUBGRAPH_EDGE_TYPES's comment on MENTIONED_IN), so there's nothing
    broader available to query even if this matched Module 20's scope.

    status: 'active' on the anchor match -- every caller today already only
    passes chunk_ids discovered through fulltext_search_moments()/
    get_subgraph() (both filtered to active as of the same sweep this
    comment was added in), so this is defense in depth, not currently load-
    bearing: don't rely on that staying true for a future direct caller.
    set_status_by_version() always updates Chunk and Source together for the
    same version_id, so an active Chunk's own Source is never independently
    archived -- no separate check needed on `s`."""
    db = await _ensure_ready(client, branch_id)
    rows = await client.query(
        db,
        "MATCH (c:Chunk {chunk_id: $cid, status: 'active'}) "
        "OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(s:Source) "
        "OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity) "
        "RETURN c, s, collect(DISTINCT e) AS entities",
        params={"cid": chunk_id}, language="cypher",
    )
    if not rows:
        return {"nodes": []}

    nodes = [_serialize_node(rows[0]["c"], ["Chunk"])]
    if rows[0]["s"]:
        nodes.append(_serialize_node(rows[0]["s"], ["Source"]))
    nodes.extend(_serialize_node(e, ["Entity"]) for e in rows[0]["entities"] if e)
    return {"nodes": nodes}


async def co_occurring_entities(
    client: ArcadeDBClient, branch_id: str, entity_id: str, limit: int = 20,
) -> List[Dict[str, Any]]:
    """Entities that most often co-occur with entity_id, i.e. Module 20's
    GET /graph/co-topics (main.py:502) -- but ranked by CO_OCCURS_WITH's own
    `weight` property, not `count(r)` like Module 20's query does. Module
    20's version returns 1 for every row: CO_OCCURS_WITH is MERGE-deduped to
    one edge per pair (create_co_occurs_with(), neo4j_writer.py:299), so a
    single-pair MATCH can only ever traverse that one edge once -- count(r)
    can't distinguish "co-occurred once" from "co-occurred in ten different
    videos" the way the edge's own accumulated weight can, so ORDER BY
    count(r) DESC ties every row at 1 and the ordering it produces is
    meaningless. Not copied here."""
    db = await _ensure_ready(client, branch_id)
    rows = await client.query(
        db,
        "MATCH (e:Entity {entity_id: $id})-[r:CO_OCCURS_WITH]-(other:Entity) "
        "RETURN other.entity_id AS entity_id, other.name AS name, other.type AS type, "
        "r.weight AS co_count "
        f"ORDER BY r.weight DESC LIMIT {max(1, min(int(limit), 200))}",
        params={"id": entity_id}, language="cypher",
    )
    return [_clean(r) for r in rows]
