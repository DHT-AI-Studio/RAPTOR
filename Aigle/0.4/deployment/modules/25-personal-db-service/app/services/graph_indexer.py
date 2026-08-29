"""Entity / Relationship / TemporalFact indexing into per-user ArcadeDB (PA-5,
unified `Chunk` model).

Vertices UPSERT (idempotent on entity_id / fact_id, backed by UNIQUE indexes).
Edges have no native UPSERT → check-then-create to avoid duplicates. Content↔
Entity linkage is `MENTIONS(Chunk→Entity)`, which makes the cross-media graph
work (one Entity linked from chunks of any media). DB creation stays PA-2's job.
"""
from __future__ import annotations

import logging

from app.models.graph_index import (EntityIndexRequest, RelationshipIndexRequest,
                                     TemporalFactIndexRequest)
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.indexer import DatabaseNotInitializedError

logger = logging.getLogger("personal_db.graph_indexer")


async def _ensure_ready(client: ArcadeDBClient, branch_id: str) -> str:
    db = db_name_for(branch_id)
    if not await client.database_exists(db):
        raise DatabaseNotInitializedError(
            "Personal database not initialized. Call POST /internal/db/init first.")
    return db


def _rid(result) -> str:
    return result[0].get("@rid", "") if result else ""


async def _edge_exists(client, db, etype, from_key, from_val, to_key, to_val) -> str:
    rows = await client.query(
        db, f"SELECT @rid FROM {etype} WHERE outV().{from_key}=:f AND inV().{to_key}=:t",
        params={"f": from_val, "t": to_val})
    return _rid(rows)


async def index_entity(client: ArcadeDBClient, branch_id: str, req: EntityIndexRequest) -> str:
    db = await _ensure_ready(client, branch_id)
    sets = ["entity_id=:eid", "name=:n", "type=:t"]
    params = {"eid": req.entity_id, "n": req.name, "t": req.type}
    if req.description is not None:
        sets.append("description=:d")
        params["d"] = req.description
    sql = f"UPDATE Entity SET {', '.join(sets)} UPSERT RETURN AFTER @rid WHERE entity_id=:eid"
    rid = _rid(await client.command(db, sql, params=params))

    # link the chunk that mentioned it. Only recompute mention_count when a NEW
    # MENTIONS edge is actually created — idempotent re-indexing (edge already
    # there) leaves the count unchanged, so we skip the count query entirely.
    if req.source_chunk_id and not await _edge_exists(
            client, db, "MENTIONS", "chunk_id", req.source_chunk_id, "entity_id", req.entity_id):
        edge_sql = ("CREATE EDGE MENTIONS FROM (SELECT FROM Chunk WHERE chunk_id=:c) "
                    "TO (SELECT FROM Entity WHERE entity_id=:e)")
        ep = {"c": req.source_chunk_id, "e": req.entity_id}
        if req.modality:
            edge_sql += " SET modality=:m"
            ep["m"] = req.modality
        await client.command(db, edge_sql, params=ep)
        cnt = (await client.query(
            db, "SELECT count(*) AS c FROM MENTIONS WHERE inV().entity_id=:e",
            params={"e": req.entity_id}))[0]["c"]
        await client.command(db, "UPDATE Entity SET mention_count=:mc WHERE entity_id=:e",
                             params={"mc": cnt, "e": req.entity_id})
    return rid


async def index_relationship(client: ArcadeDBClient, branch_id: str,
                             req: RelationshipIndexRequest) -> str:
    """Upsert-ish: idempotent per source, but NOT deduped across sources.

    The existence check is keyed on (relation, from, to, source_version_id)
    together when source_version_id is given -- not just (relation, from,
    to) -- so two independent sources that both extract the exact same fact
    each get their own edge. Previously (relation, from, to) alone decided
    "already exists", meaning the second source's index_relationship() call
    was a silent no-op: its source_version_id was never recorded anywhere,
    so delete_by_version() on the *first* source's version_id deleted the
    only edge for that fact -- even while the second source's content was
    still active and still asserting it. See delete_by_version()'s docstring
    for the fuller writeup; this is the actual fix, not a workaround --
    ArcadeDB doesn't forbid parallel edges of the same type between the same
    two vertices, that was purely this function's own choice.

    Still idempotent for repeat processing of the SAME source (matching
    source_version_id -> matching existing edge -> no duplicate). A caller
    that omits source_version_id falls back to the old (relation, from, to)
    -only check, since there's nothing else to key on."""
    db = await _ensure_ready(client, branch_id)
    if req.source_version_id is not None:
        existing = await client.query(
            db, "SELECT @rid FROM RELATION WHERE relation=:r AND outV().entity_id=:f "
                "AND inV().entity_id=:t AND source_version_id=:sv",
            params={"r": req.relation, "f": req.from_entity_id, "t": req.to_entity_id,
                    "sv": req.source_version_id})
    else:
        existing = await client.query(
            db, "SELECT @rid FROM RELATION WHERE relation=:r AND outV().entity_id=:f AND inV().entity_id=:t",
            params={"r": req.relation, "f": req.from_entity_id, "t": req.to_entity_id})
    if existing:
        return _rid(existing)
    sets = ["relation=:r"]
    params = {"r": req.relation, "f": req.from_entity_id, "t": req.to_entity_id}
    if req.confidence is not None:
        sets.append("confidence=:c")
        params["c"] = req.confidence
    if req.source_version_id is not None:
        sets.append("source_version_id=:sv")
        params["sv"] = req.source_version_id
    sql = ("CREATE EDGE RELATION FROM (SELECT FROM Entity WHERE entity_id=:f) "
           "TO (SELECT FROM Entity WHERE entity_id=:t) SET " + ", ".join(sets))
    return _rid(await client.command(db, sql, params=params))


async def index_mentioned_in(client: ArcadeDBClient, branch_id: str,
                             entity_id: str, version_id: str) -> str:
    """MENTIONED_IN(Entity -> Source) -- document-level mention, for entities
    extracted from a Source's whole-asset summary (summary is no longer a
    fake Chunk, see indexer.index_source_summary()). Matches Module 20's
    create_mentioned_in() (pipeline.py:127), which runs on exactly the same
    summary-level extraction step. Check-then-create like every other edge
    here, keyed on (entity_id, version_id) so re-processing the same summary
    (retry, redelivery) doesn't create a duplicate."""
    db = await _ensure_ready(client, branch_id)
    if await _edge_exists(client, db, "MENTIONED_IN", "entity_id", entity_id, "version_id", version_id):
        return ""
    sql = ("CREATE EDGE MENTIONED_IN FROM (SELECT FROM Entity WHERE entity_id=:e) "
           "TO (SELECT FROM Source WHERE version_id=:v)")
    return _rid(await client.command(db, sql, params={"e": entity_id, "v": version_id}))


async def index_co_occurrence(client: ArcadeDBClient, branch_id: str,
                              entity_id_a: str, entity_id_b: str) -> str:
    """Undirected CO_OCCURS_WITH edge between two entities that appeared in
    the same moment -- weight increments on repeat co-occurrence (same video
    or across videos), same semantics as Module 20's create_co_occurs_with()
    (neo4j_writer.py:284, `MERGE (a)-[r:CO_OCCURS_WITH]-(b) ON CREATE SET
    r.weight = 1 ON MATCH SET r.weight = r.weight + 1`). ArcadeDB confirmed
    live to support that exact Cypher MERGE/ON CREATE/ON MATCH idiom, but
    this file's own established convention is SQL check-then-create (every
    other edge type here does the same), so this follows that instead of
    introducing Cypher into a file that has none.

    Canonical (sorted) pair ordering so the same two entities always land on
    one edge regardless of which one this gets called as "a" vs "b" -- an
    edge type is inherently directed in ArcadeDB even when read back with an
    undirected Cypher pattern, so without this a second call with the
    arguments swapped would create a second, opposite-direction edge for the
    same pair instead of incrementing the first one's weight."""
    db = await _ensure_ready(client, branch_id)
    a, b = sorted([entity_id_a, entity_id_b])
    if a == b:
        return ""
    existing = await client.query(
        db, "SELECT @rid, weight FROM CO_OCCURS_WITH WHERE outV().entity_id=:a AND inV().entity_id=:b",
        params={"a": a, "b": b})
    if existing:
        rid = existing[0]["@rid"]
        new_weight = (existing[0].get("weight") or 1) + 1
        await client.command(db, "UPDATE CO_OCCURS_WITH SET weight=:w WHERE @rid=:rid",
                             params={"w": new_weight, "rid": rid})
        return rid
    sql = ("CREATE EDGE CO_OCCURS_WITH FROM (SELECT FROM Entity WHERE entity_id=:a) "
           "TO (SELECT FROM Entity WHERE entity_id=:b) SET weight=1")
    return _rid(await client.command(db, sql, params={"a": a, "b": b}))


async def index_temporal_fact(client: ArcadeDBClient, branch_id: str,
                              req: TemporalFactIndexRequest) -> str:
    db = await _ensure_ready(client, branch_id)
    # status: new facts are always active -- archiving happens later, in bulk,
    # via set_status_by_version() when the whole source_version_id is
    # archived, same as Chunk/Source. No caller-supplied status here (unlike
    # Chunk's req.status) since nothing creates a fact pre-archived.
    sets = ["fact_id=:fid", "entity=:en", "relation=:r", "value=:v", "status=:st"]
    params = {"fid": req.fact_id, "en": req.entity, "r": req.relation, "v": req.value, "st": "active"}
    for field in ("entity_id", "time_start", "time_end", "confidence", "source_version_id"):
        val = getattr(req, field)
        if val is not None:
            sets.append(f"{field}=:{field}")
            params[field] = val
    sql = f"UPDATE TemporalFact SET {', '.join(sets)} UPSERT RETURN AFTER @rid WHERE fact_id=:fid"
    rid = _rid(await client.command(db, sql, params=params))

    if req.entity_id and not await _edge_exists(client, db, "HAS_TEMPORAL_FACT",
                                                "entity_id", req.entity_id, "fact_id", req.fact_id):
        await client.command(db, "CREATE EDGE HAS_TEMPORAL_FACT FROM (SELECT FROM Entity WHERE entity_id=:e) "
                                 "TO (SELECT FROM TemporalFact WHERE fact_id=:f)",
                             params={"e": req.entity_id, "f": req.fact_id})
    if req.chunk_id and not await _edge_exists(client, db, "OBSERVED_IN",
                                               "fact_id", req.fact_id, "chunk_id", req.chunk_id):
        await client.command(db, "CREATE EDGE OBSERVED_IN FROM (SELECT FROM TemporalFact WHERE fact_id=:f) "
                                 "TO (SELECT FROM Chunk WHERE chunk_id=:c)",
                             params={"f": req.fact_id, "c": req.chunk_id})
    return rid
