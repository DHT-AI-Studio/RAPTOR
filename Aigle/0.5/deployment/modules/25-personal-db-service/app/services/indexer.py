"""Chunk indexing into per-user ArcadeDB (PA-4, unified `Chunk` model).

One `index_chunk` handles all four media types — the request only carries the
fields relevant to its media, and we upsert exactly those. Idempotent on
`chunk_id` (backed by the UNIQUE index). All values are param-bound (injection
safe, and array binding is verified against arcadedb:latest).

DB creation is PA-2's job (POST /internal/db/init) — a missing DB raises 404.
"""
from __future__ import annotations

import hashlib
import logging

from app.models.index import (
    ChunkIndexRequest,
    CloneVersionRequest,
    SourceSummaryIndexRequest,
)
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.embedder import embed_texts

logger = logging.getLogger("personal_db.indexer")


class DatabaseNotInitializedError(Exception):
    """Raised when the user's ArcadeDB database has not been created yet."""


async def _ensure_ready(client: ArcadeDBClient, branch_id: str) -> str:
    db = db_name_for(branch_id)
    if not await client.database_exists(db):
        raise DatabaseNotInitializedError(
            "Personal database not initialized. Call POST /internal/db/init first.")
    return db


def _count(result) -> int:
    """Row count from an ArcadeDB DELETE/UPDATE result ([{'count': n}])."""
    return int(result[0].get("count", 0)) if result else 0


def _rid(result) -> str:
    """@rid from a CREATE/UPDATE...RETURN result. Duplicated from
    graph_indexer.py's own _rid() rather than imported -- graph_indexer.py
    already imports DatabaseNotInitializedError from this module, so the
    reverse import would be circular."""
    return result[0].get("@rid", "") if result else ""


async def _edge_exists(client: ArcadeDBClient, db: str, etype: str,
                       from_key: str, from_val: str, to_key: str, to_val: str) -> str:
    """Same check-then-create guard as graph_indexer.py's own _edge_exists()
    (duplicated for the same circular-import reason as _rid() above) --
    ArcadeDB edges have no native UPSERT."""
    rows = await client.query(
        db, f"SELECT @rid FROM {etype} WHERE outV().{from_key}=:f AND inV().{to_key}=:t",  # nosec B608 -- etype/from_key/to_key are always hardcoded literals at every call site
        params={"f": from_val, "t": to_val})
    return _rid(rows)


async def version_exists(client: ArcadeDBClient, branch_id: str, version_id: str, status: str | None = None) -> bool:
    """Whether this version has been indexed, as Chunk or Source -- checking
    Chunk alone missed summary-only content (e.g. audio with no speech to
    transcribe), reporting "not indexed" and spuriously re-triggering
    analysis on every duplicate upload. Module 04's search_sync.check_indexed()
    equivalent.

    `status`, when given, narrows the match to that lifecycle state (e.g.
    "active") instead of any row regardless of status -- a caller using this
    as proof that content is genuinely *live* (not just present from some
    earlier clone/archive) should pass it; the default stays status-blind
    for callers that only care whether anything was ever indexed at all."""
    db = await _ensure_ready(client, branch_id)
    for entity in ("Chunk", "Source"):
        sql = f"SELECT count(*) AS c FROM {entity} WHERE version_id = :v"  # nosec B608 -- entity iterates the hardcoded tuple ("Chunk", "Source")
        params = {"v": version_id}
        if status is not None:
            sql += " AND status = :status"
            params["status"] = status
        rows = await client.query(db, sql + " LIMIT 1", params=params)
        if rows and _count([{"count": rows[0]["c"]}]) > 0:
            return True
    return False


def _clone_id(target_version_id: str, old_id: str) -> str:
    """Deterministic new id derived from (target_version_id, old_id) --
    makes clone_version() idempotent: retrying it for the same source/target
    pair re-upserts the same target vertices instead of creating duplicates,
    matching every other write in this module (index_chunk/index_temporal_fact
    are UPSERTs keyed on a stable id for the same reason)."""
    return hashlib.sha256(f"{target_version_id}:{old_id}".encode()).hexdigest()


async def clone_version(client: ArcadeDBClient, branch_id: str, req: CloneVersionRequest) -> dict:
    """Clone one asset version's Source + Chunks + TemporalFacts onto a new
    version_id/asset_path -- Module 04's content-dedup optimization
    (client.py: identical MD5 found at a different, already-archived
    asset_path -- clone its index instead of re-running analysis).

    What gets cloned vs shared, and why:
    - Source, Chunk: version-scoped, genuinely duplicated with new identity
      (chunk_id/version_id/asset_path), status forced to 'cloned' regardless
      of the source's current status -- Module 04's cloned_from column
      records this succeeded, and Register flips it to 'active' via
      set_status_by_version() -- never assumed live the moment this
      function returns.
    - HAS_CHUNK (Source->Chunk), MENTIONS (Chunk->Entity), MENTIONED_IN
      (Entity->Source): recreated pointing at the new Chunk/Source vertices.
      MENTIONS re-derives Entity.mention_count the same way index_entity()
      does when it creates a fresh MENTIONS edge, so the count reflects the
      clone actually being a new place this entity is mentioned.
    - Entity: NOT cloned. Entities are already a cross-content, deduplicated
      pool (see schema_init.py's own module docstring) -- the new Chunks
      point at the SAME Entity vertices the source Chunks did.
    - RELATION: cloned -- but only the edges the SOURCE version itself
      contributed (source_version_id = source_version_id), given its own
      new source_version_id = target_version_id, not every RELATION edge
      touching an entity these chunks happen to mention (which could
      include facts an unrelated source extracted). index_relationship()
      now dedupes per-source (relation+from+to+source_version_id, not just
      relation+from+to -- see its own docstring for the fix and why it was
      needed: two sources independently asserting the same fact used to
      collapse onto one edge, so deleting whichever source got there first
      silently dropped a relation the other source still supported). Giving
      the clone its own edge means deleting the source (often archived --
      see this function's own docstring) won't do that to the clone.
    - CO_OCCURS_WITH: NOT cloned/incremented here. Its weight is a derived
      count of currently-mentioning Chunks (see delete_by_version()'s
      _recompute_co_occurs_with()), not stamped with any one source's id, so
      there's no per-source edge to give the clone its own copy of the way
      RELATION now gets. The clone's new MENTIONS edges (above) do mean the
      *true* count for a pair is now higher than whatever weight is
      currently stored -- but nothing recomputes it proactively at clone
      time, only _recompute_co_occurs_with() does, and only when
      delete_by_version() next touches one of the entities in the pair.
      Until then, weight under-counts the clone's contribution. Not fixed
      here -- would mean running the same recompute proactively after every
      clone, which is a real gap but a separate piece of work from what this
      function set out to do.
    - TemporalFact: cloned (new fact_id, source_version_id=target), since
      unlike RELATION it has no cross-source dedup -- index_temporal_fact()
      is a plain UPSERT on fact_id, so each fact genuinely belongs to one
      extraction event. OBSERVED_IN(TemporalFact->Chunk) is recreated
      pointing at the corresponding *new* Chunk via the chunk_id remap built
      while cloning Chunks above.
    - FaceTrace/FaceGroup/their representative Faces: cloned the same way as
      Chunk (new ids, version_id=target); GROUP_OF_IDENTITY is recreated
      pointing at the SAME Identity a source group already carries (never
      cloned, same reasoning as Entity above) -- see
      identity_store.clone_face_data()'s own docstring. Previously this
      function didn't touch face data at all, which silently dropped every
      dedup-cloned asset's People/face data entirely.
    """
    db = await _ensure_ready(client, branch_id)
    sv, sa = req.source_version_id, req.source_asset_path  # noqa: F841 -- Source.version_id is UNIQUE-indexed, sa adds no scoping
    tv, ta = req.target_version_id, req.target_asset_path
    if sv == tv:
        # _clone_id(tv, old_id) derives an id different from old_id even when
        # tv == sv, so this wouldn't actually no-op -- it would duplicate
        # every Chunk/fact under a second, hash-derived id in the SAME
        # version, silently doubling the index. The real caller (module 04's
        # search_sync, dedup onto a different asset_path/version_id) can't
        # hit this, but nothing else about this endpoint enforces it.
        raise ValueError("source_version_id and target_version_id must differ")

    # ---- Source ----
    src_rows = await client.query(db, "SELECT FROM Source WHERE version_id = :v", params={"v": sv})
    if not src_rows:
        raise ValueError(f"source version {sv} has no Source vertex to clone")
    source_fields = {k: v for k, v in src_rows[0].items() if not str(k).startswith("@")}
    source_filename = source_fields.get("filename")  # pre-override: a Chunk's own filename is
    source_fields["version_id"] = tv                  # compared against this, not the new one, below
    source_fields["asset_path"] = ta
    if req.target_filename is not None:
        # Content-dedup means the same bytes, not the same name -- without
        # this, every presign-url lookup for the clone used the SOURCE's
        # filename, which doesn't exist at the new asset_path in LakeFS
        # (confirmed live: a video re-uploaded under a different filename
        # hit 404 "File ... or commit ... not found" for every search hit).
        source_fields["filename"] = req.target_filename
    # "cloned", not "active" -- Register flips it via set_status_by_version(),
    # matching VIE01-214's rule that nothing is "active" before Register says so.
    source_fields["status"] = "cloned"
    src_set = ", ".join(f"{k} = :{k}" for k in source_fields)
    await client.command(
        db, f"UPDATE Source SET {src_set} UPSERT WHERE version_id = :version_id", params=source_fields)  # nosec B608 -- source_fields' keys are an existing Source row's own column names (SELECTed above), not request-derived

    # ---- Chunks (+ chunk_id remap for TemporalFact.OBSERVED_IN below) ----
    chunk_rows = await client.query(db, "SELECT FROM Chunk WHERE version_id = :v", params={"v": sv})
    chunk_id_map: dict[str, str] = {}
    for row in chunk_rows:
        old_chunk_id = row["chunk_id"]
        new_chunk_id = _clone_id(tv, old_chunk_id)
        chunk_id_map[old_chunk_id] = new_chunk_id
        fields = {k: v for k, v in row.items() if not str(k).startswith("@")}
        fields["chunk_id"] = new_chunk_id
        fields["version_id"] = tv
        fields["asset_path"] = ta
        if req.target_filename is not None and fields.get("filename") == source_filename:
            # Only remap chunks that carried the asset's own (renamed) file --
            # an associated-file chunk with some other filename is left alone,
            # since nothing here knows what that file was renamed to, if
            # anything (not the confirmed bug; a known, documented gap).
            fields["filename"] = req.target_filename
        fields["status"] = "cloned"
        set_parts = ", ".join(f"{k} = :{k}" for k in fields)
        await client.command(
            db, f"UPDATE Chunk SET {set_parts} UPSERT WHERE chunk_id = :chunk_id", params=fields)  # nosec B608 -- fields' keys are an existing Chunk row's own column names (SELECTed above), not request-derived

        if not await _edge_exists(client, db, "HAS_CHUNK", "version_id", tv, "chunk_id", new_chunk_id):
            await client.command(
                db, "CREATE EDGE HAS_CHUNK FROM (SELECT FROM Source WHERE version_id=:v) "
                    "TO (SELECT FROM Chunk WHERE chunk_id=:c)",
                params={"v": tv, "c": new_chunk_id})

        mention_rows = await client.query(
            db, "SELECT out('MENTIONS').entity_id AS ids FROM Chunk WHERE chunk_id = :c",
            params={"c": old_chunk_id})
        for eid in ((mention_rows[0].get("ids") if mention_rows else None) or []):
            if not eid:
                continue
            if await _edge_exists(client, db, "MENTIONS", "chunk_id", new_chunk_id, "entity_id", eid):
                continue
            await client.command(
                db, "CREATE EDGE MENTIONS FROM (SELECT FROM Chunk WHERE chunk_id=:c) "
                    "TO (SELECT FROM Entity WHERE entity_id=:e)",
                params={"c": new_chunk_id, "e": eid})
            cnt = await client.query(
                db, "SELECT count(*) AS c FROM MENTIONS WHERE inV().entity_id=:e", params={"e": eid})
            await client.command(
                db, "UPDATE Entity SET mention_count=:mc WHERE entity_id=:e",
                params={"mc": cnt[0]["c"] if cnt else 0, "e": eid})

    # ---- MENTIONED_IN (Entity -> Source), document-level mentions ----
    src_mention_rows = await client.query(
        db, "SELECT in('MENTIONED_IN').entity_id AS ids FROM Source WHERE version_id = :v", params={"v": sv})
    for eid in ((src_mention_rows[0].get("ids") if src_mention_rows else None) or []):
        if not eid:
            continue
        if await _edge_exists(client, db, "MENTIONED_IN", "entity_id", eid, "version_id", tv):
            continue
        await client.command(
            db, "CREATE EDGE MENTIONED_IN FROM (SELECT FROM Entity WHERE entity_id=:e) "
                "TO (SELECT FROM Source WHERE version_id=:v)",
            params={"e": eid, "v": tv})

    # ---- RELATION (Entity<->Entity, tagged with the SOURCE's own source_version_id) ----
    # Only edges index_relationship() attributed to THIS source_version_id --
    # not every RELATION edge touching an entity these chunks happen to
    # mention, which could include facts a completely different, unrelated
    # source extracted. Now that index_relationship() dedupes per-source
    # (relation+from+to+source_version_id, not just relation+from+to -- see
    # its own docstring), giving the clone its own edge here means deleting
    # the (often-archived) source later won't silently drop a relation the
    # clone's own content still asserts.
    rel_rows = await client.query(
        db, "SELECT relation, confidence, outV().entity_id AS f, inV().entity_id AS t "
            "FROM RELATION WHERE source_version_id = :v", params={"v": sv})
    relations_cloned = 0
    for row in rel_rows:
        exists = await client.query(
            db, "SELECT @rid FROM RELATION WHERE relation=:r AND outV().entity_id=:f "
                "AND inV().entity_id=:t AND source_version_id=:sv",
            params={"r": row["relation"], "f": row["f"], "t": row["t"], "sv": tv})
        if exists:
            continue
        sets = ["relation=:r", "source_version_id=:sv"]
        rparams = {"r": row["relation"], "f": row["f"], "t": row["t"], "sv": tv}
        if row.get("confidence") is not None:
            sets.append("confidence=:c")
            rparams["c"] = row["confidence"]
        await client.command(
            db, "CREATE EDGE RELATION FROM (SELECT FROM Entity WHERE entity_id=:f) "  # nosec B608 -- sets built above from hardcoded "x=:x"-shaped literals only
                "TO (SELECT FROM Entity WHERE entity_id=:t) SET " + ", ".join(sets),
            params=rparams)
        relations_cloned += 1

    # ---- TemporalFact ----
    fact_rows = await client.query(
        db, "SELECT FROM TemporalFact WHERE source_version_id = :v", params={"v": sv})
    facts_cloned = 0
    for row in fact_rows:
        old_fact_id = row["fact_id"]
        new_fact_id = _clone_id(tv, old_fact_id)
        fields = {k: v for k, v in row.items() if not str(k).startswith("@")}
        fields["fact_id"] = new_fact_id
        fields["source_version_id"] = tv
        fields["status"] = "cloned"
        set_parts = ", ".join(f"{k} = :{k}" for k in fields)
        await client.command(
            db, f"UPDATE TemporalFact SET {set_parts} UPSERT WHERE fact_id = :fact_id", params=fields)  # nosec B608 -- fields' keys are an existing TemporalFact row's own column names, not request-derived
        facts_cloned += 1

        entity_id = fields.get("entity_id")
        if entity_id and not await _edge_exists(
                client, db, "HAS_TEMPORAL_FACT", "entity_id", entity_id, "fact_id", new_fact_id):
            await client.command(
                db, "CREATE EDGE HAS_TEMPORAL_FACT FROM (SELECT FROM Entity WHERE entity_id=:e) "
                    "TO (SELECT FROM TemporalFact WHERE fact_id=:f)",
                params={"e": entity_id, "f": new_fact_id})

        obs_rows = await client.query(
            db, "SELECT out('OBSERVED_IN').chunk_id AS ids FROM TemporalFact WHERE fact_id = :f",
            params={"f": old_fact_id})
        for old_cid in ((obs_rows[0].get("ids") if obs_rows else None) or []):
            new_cid = chunk_id_map.get(old_cid)
            if not new_cid:
                continue
            if await _edge_exists(client, db, "OBSERVED_IN", "fact_id", new_fact_id, "chunk_id", new_cid):
                continue
            await client.command(
                db, "CREATE EDGE OBSERVED_IN FROM (SELECT FROM TemporalFact WHERE fact_id=:f) "
                    "TO (SELECT FROM Chunk WHERE chunk_id=:c)",
                params={"f": new_fact_id, "c": new_cid})

    # P14: the temporal job (model replies included) and the undated candidates go with the clone
    from app.services.temporal_jobs import clone_temporal
    temporal = await clone_temporal(client, db, branch_id, sv, tv, chunk_id_map, _clone_id)

    from app.services.identity_store import clone_face_data
    faces = await clone_face_data(client, db, sv, tv, _clone_id)

    result = {"source_version_id": sv, "target_version_id": tv, "sources": 1,
              "chunks": len(chunk_rows), "relationships": relations_cloned, "temporal_facts": facts_cloned,
              **temporal, **faces}
    logger.info("[indexer] clone-version %s → %s: %s", sv, tv, result)
    return result


async def index_chunk(client: ArcadeDBClient, branch_id: str, req: ChunkIndexRequest) -> str:
    """Upsert one Chunk (any media type). Returns its ArcadeDB @rid.

    When the chunk belongs to an uploaded asset (``version_id`` set), also upsert
    the asset's ``Source`` vertex and link ``Source -HAS_CHUNK-> Chunk`` — this
    is the provenance backbone that makes delete-by-version clean.
    """
    db = await _ensure_ready(client, branch_id)

    # No pre-computed vector? Embed the content in-process with the service's BGE-M3.
    if req.embedding is None:
        content = (req.text if req.embedding_type == "text" else req.summary) or req.text or req.summary
        if not content:
            raise ValueError("chunk has no `embedding` and no `text`/`summary` to embed")
        req.embedding = (await embed_texts([content]))[0]

    data = req.model_dump(exclude_none=True)          # only send provided fields
    set_parts = [f"{k} = :{k}" for k in data]
    sql = ("UPDATE Chunk SET " + ", ".join(set_parts)  # nosec B608 -- data's keys come from req.model_dump(), a Pydantic model's own declared fields, never arbitrary extra keys
           + " UPSERT RETURN AFTER @rid WHERE chunk_id = :chunk_id")
    result = await client.command(db, sql, params=data)
    rid = result[0].get("@rid", "") if result else ""
    if req.version_id:
        await _link_source(client, db, req)
    logger.debug("[indexer] chunk %s (%s) → %s", req.chunk_id, req.type, rid)
    return rid


async def index_source_summary(client: ArcadeDBClient, branch_id: str,
                               req: SourceSummaryIndexRequest) -> str:
    """Upsert a whole-asset summary directly onto its Source vertex -- the
    replacement for indexing it as a fake summary Chunk (no start_sec/
    end_sec, embedding_type="summary"). Every media type calls this the same
    way; there is no media_type-specific branch because Source.summary/
    Source.embedding aren't media-specific fields."""
    db = await _ensure_ready(client, branch_id)

    if req.embedding is None:
        req.embedding = (await embed_texts([req.summary]))[0]

    sets = ["version_id = :version_id", "summary = :summary", "embedding = :embedding", "status = :status"]
    params = {"version_id": req.version_id, "summary": req.summary, "embedding": req.embedding,
              "status": req.status}
    if req.filename is not None:
        sets.append("filename = :filename"); params["filename"] = req.filename
    if req.asset_path is not None:
        sets.append("asset_path = :asset_path"); params["asset_path"] = req.asset_path
    if req.media_type is not None:
        sets.append("media_type = :media_type"); params["media_type"] = req.media_type
    if req.capture_time is not None:          # P13: as the worker read it
        sets.append("capture_time = :capture_time"); params["capture_time"] = req.capture_time
        sets.append("capture_time_source = :capture_time_source")
        params["capture_time_source"] = req.capture_time_source
    if req.moderation_flags is not None:      # CM-2
        sets.append("moderation_flags = :moderation_flags"); params["moderation_flags"] = req.moderation_flags

    sql = ("UPDATE Source SET " + ", ".join(sets)  # nosec B608 -- sets built from hardcoded "x = :x"-shaped literals only
           + " UPSERT RETURN AFTER @rid WHERE version_id = :version_id")
    result = await client.command(db, sql, params=params)
    rid = result[0].get("@rid", "") if result else ""
    await _link_waiting_face_traces(client, db, req.version_id)
    logger.debug("[indexer] source summary %s → %s", req.version_id, rid)
    return rid


async def _link_waiting_face_traces(client: ArcadeDBClient, db: str, version_id: str) -> None:
    """Face indexing never creates a Source (see identity_store.write_face_traces),
    so traces written before this video's content indexing are linked to its Source
    here, once it exists. Imported lazily: identity_store imports from this module."""
    from app.services.identity_store import link_face_traces_to_source
    await link_face_traces_to_source(client, db, version_id)


async def _link_source(client: ArcadeDBClient, db: str, req: ChunkIndexRequest) -> None:
    """Upsert the asset's Source vertex and create HAS_CHUNK(Source→Chunk) once."""
    sets, p = ["version_id = :v"], {"v": req.version_id, "c": req.chunk_id}
    if req.filename is not None:
        sets.append("filename = :f"); p["f"] = req.filename
    if req.asset_path is not None:
        sets.append("asset_path = :ap"); p["ap"] = req.asset_path
    if req.type is not None:
        sets.append("media_type = :mt"); p["mt"] = req.type
    if req.capture_time is not None:          # P13: as the worker read it; validated where it is used
        sets.append("capture_time = :ct"); p["ct"] = req.capture_time
        sets.append("capture_time_source = :cts"); p["cts"] = req.capture_time_source

    # Source.summary is an indexed NOT-NULL field, but index_chunk() (this
    # function's only caller) never sets it -- only index_source_summary()
    # does, as a separate call elsewhere in the real pipeline, typically
    # around the same time. If a chunk is indexed before that call has run
    # for this version_id (or that call never runs at all, e.g. this chunk's
    # asset has no whole-document summary), UPSERT would otherwise create a
    # brand-new Source row with summary left NULL and ArcadeDB rejects it.
    # Only supply a placeholder on first creation -- an existing row (summary
    # already set for real, or already placeholdered by an earlier chunk)
    # must never be touched here, or a later real index_source_summary()
    # write and this one could race to clobber each other.
    existing = await client.query(db, "SELECT @rid FROM Source WHERE version_id = :v LIMIT 1", params={"v": req.version_id})
    if not existing:
        sets.append("summary = :s")
        p["s"] = ""

    await client.command(
        db, f"UPDATE Source SET {', '.join(sets)} UPSERT WHERE version_id = :v", params=p)  # nosec B608 -- sets built above from hardcoded "x = :x"-shaped literals only
    if not existing:
        await _link_waiting_face_traces(client, db, req.version_id)
    exists = await client.query(
        db, "SELECT @rid FROM HAS_CHUNK WHERE outV().version_id = :v AND inV().chunk_id = :c",
        params={"v": req.version_id, "c": req.chunk_id})
    if not exists:
        await client.command(
            db, "CREATE EDGE HAS_CHUNK FROM (SELECT FROM Source WHERE version_id = :v) "
                "TO (SELECT FROM Chunk WHERE chunk_id = :c)",
            params={"v": req.version_id, "c": req.chunk_id})


async def set_status_by_version(client: ArcadeDBClient, branch_id: str,
                                version_id: str, status: str) -> dict:
    """Set status on every Chunk + Source + TemporalFact for one asset version
    (idempotent). Module 25's equivalent of Module 20's POST /source/set_status
    (main.py:235) -- Module 04's search_sync fan-out calls this on archive
    (status="archived") and reactivate (status="active"). Verified live that
    a plain UPDATE...SET...WHERE returns {"count": n} the same way DELETE
    does, so this reuses _count() the same way delete_by_version() below
    does.

    TemporalFact keyed by source_version_id, not version_id (see
    schema_init.py's comment on that property) -- added alongside Chunk/
    Source so tkg_search()'s temporal_facts results actually respect archive
    status instead of a fact outliving its source indefinitely."""
    db = await _ensure_ready(client, branch_id)
    # P21: an expired inline source is never active again ("no resurrection")
    from app.services.source_expiry import is_expired
    if status == "active" and await is_expired(client, db, version_id):
        logger.info("[indexer] %s is expired; reactivation refused", version_id)
        status = "archived"
    chunks = _count(await client.command(
        db, "UPDATE Chunk SET status = :s WHERE version_id = :v", params={"s": status, "v": version_id}))
    # P14: every archive / reactivation advances the source's generation -- temporal work started before it
    # sees it is stale and never writes, even when the source is active again by the time it finishes.
    sources = _count(await client.command(
        db, "UPDATE Source SET status = :s, generation = ifnull(generation, 0) + 1 WHERE version_id = :v",
        params={"s": status, "v": version_id}))
    facts = _count(await client.command(
        db, "UPDATE TemporalFact SET status = :s WHERE source_version_id = :v",
        params={"s": status, "v": version_id}))
    # Imported lazily: temporal_jobs imports from this module.
    from app.services.temporal_jobs import on_status_change
    temporal = await on_status_change(client, db, branch_id, version_id, status)
    result = {"version_id": version_id, "status": status, "chunks": chunks, "sources": sources,
              "temporal_facts": facts, **temporal}
    logger.info("[indexer] set-status-by-version %s → %s", version_id, result)
    return result


async def _recompute_co_occurs_with(client: ArcadeDBClient, db: str, entity_ids: list[str]) -> int:
    """Recompute (or delete) every CO_OCCURS_WITH edge touching any of these
    entities, from currently-surviving MENTIONS evidence -- see
    delete_by_version()'s own docstring for why this has to be a full
    recompute rather than a decrement. weight = size of the intersection of
    "Chunks that mention entity A" and "Chunks that mention entity B" (both
    fetched via Entity's incoming MENTIONS, then intersected in Python --
    ArcadeDB has no native set-intersection-of-two-traversals SQL/Cypher
    idiom simpler than this). Returns the number of edges deleted (weight
    recomputed to 0)."""
    if not entity_ids:
        return 0
    edges = await client.query(
        db, "SELECT @rid, outV().entity_id AS a, inV().entity_id AS b FROM CO_OCCURS_WITH "
            "WHERE outV().entity_id IN :ids OR inV().entity_id IN :ids",
        params={"ids": entity_ids})
    if not edges:
        return 0
    chunk_sets: dict[str, set] = {}

    async def _chunks_mentioning(eid: str) -> set:
        if eid not in chunk_sets:
            rows = await client.query(
                db, "SELECT in('MENTIONS').chunk_id AS ids FROM Entity WHERE entity_id = :e",
                params={"e": eid})
            chunk_sets[eid] = set((rows[0].get("ids") if rows else None) or [])
        return chunk_sets[eid]

    deleted = 0
    for edge in edges:
        a, b = edge["a"], edge["b"]
        new_weight = len(await _chunks_mentioning(a) & await _chunks_mentioning(b))
        if new_weight <= 0:
            await client.command(
                db, "DELETE FROM CO_OCCURS_WITH WHERE @rid = :rid", params={"rid": edge["@rid"]})
            deleted += 1
        else:
            await client.command(
                db, "UPDATE CO_OCCURS_WITH SET weight = :w WHERE @rid = :rid",
                params={"w": new_weight, "rid": edge["@rid"]})
    return deleted


async def delete_by_version(client: ArcadeDBClient, branch_id: str, version_id: str) -> dict:
    """Delete everything indexed from one asset version (idempotent).

    Removes the asset's Chunks (cascading their MENTIONS/OBSERVED_IN/HAS_CHUNK
    edges), its Source vertex (cascading its MENTIONED_IN/HAS_CHUNK edges),
    and the RELATION edges + TemporalFacts tagged with this
    ``source_version_id``. Shared Entities are kept -- except ones left with
    no real content connecting them any more, which are pruned. Returns
    delete counts.

    CO_OCCURS_WITH deliberately does NOT count as "real content" in the
    orphan check below: two entities that only still share a CO_OCCURS_WITH
    edge with EACH OTHER, both otherwise fully disconnected from any
    surviving Chunk/Source, are leftover graph debris, not real data. Found
    live this session as 23 leftover Entity vertices the previous version of
    this check (mention_count=0 AND both().size()=0, i.e. zero edges of ANY
    type) never caught -- both() counts CO_OCCURS_WITH too, so entities that
    had lost all their real content but still pointed at each other via
    CO_OCCURS_WITH never looked "orphaned enough" to prune. Explicitly
    checking only the real-content edge types (MENTIONED_IN/RELATION/
    HAS_TEMPORAL_FACT) instead fixes this in one pass -- no iteration
    needed, since the check for each entity no longer depends on whether
    other entities in the same batch also get deleted.

    CO_OCCURS_WITH itself IS now cleaned up (was a real gap before this
    comment was added: this edge type has no source_version_id -- it can't,
    weight accumulates across every version that ever co-mentioned the
    pair -- so deleting a version used to leave every CO_OCCURS_WITH edge it
    contributed to sitting at its old inflated weight forever, orphaned
    edges included). See _recompute_co_occurs_with(): weight is recomputed
    from scratch as "how many Chunks currently mention both entities",
    matching exactly how index_co_occurrence() built it up in the first
    place (one increment per moment where the pair was found together) --
    not decremented, since nothing recorded how much any one version
    originally contributed. Edges that recompute to 0 are deleted.
    """
    db = await _ensure_ready(client, branch_id)

    # entities connected to this version's content -- MENTIONS (per-moment,
    # via the Chunks) and MENTIONED_IN (document-level, via the Source, from
    # summary-level extraction) both count. Missing MENTIONED_IN here was
    # itself a real gap: an entity extracted only from the summary, whose
    # name never literally appears in any moment's own text, would have a
    # MENTIONED_IN edge but no MENTIONS edge at all -- invisible to the old
    # MENTIONS-only scan, so it would never get recomputed/considered here.
    affected: set[str] = set()
    for r in await client.query(
            db, "SELECT out('MENTIONS').entity_id AS ids FROM Chunk WHERE version_id = :v",
            params={"v": version_id}):
        for eid in (r.get("ids") or []):
            if eid:
                affected.add(eid)
    for r in await client.query(
            db, "SELECT in('MENTIONED_IN').entity_id AS ids FROM Source WHERE version_id = :v",
            params={"v": version_id}):
        for eid in (r.get("ids") or []):
            if eid:
                affected.add(eid)

    chunks = _count(await client.command(
        db, "DELETE VERTEX FROM Chunk WHERE version_id = :v", params={"v": version_id}))
    rels = _count(await client.command(
        db, "DELETE FROM RELATION WHERE source_version_id = :v", params={"v": version_id}))
    facts = _count(await client.command(
        db, "DELETE VERTEX FROM TemporalFact WHERE source_version_id = :v", params={"v": version_id}))
    sources = _count(await client.command(
        db, "DELETE VERTEX FROM Source WHERE version_id = :v", params={"v": version_id}))
    # P12: the source's temporal job goes too (a running one sees the source gone and cancels).
    # Imported lazily: temporal_jobs imports from this module.
    from app.services.temporal_jobs import delete_jobs
    jobs = await delete_jobs(client, db, version_id)

    # People layer (dev plan v1.4 §4.2 cleanup rules). Imported lazily:
    # identity_store imports from this module.
    from app.services.identity_store import delete_face_data
    face = await delete_face_data(client, db, version_id)

    orphans = 0
    co_occurs_pruned = 0
    if affected:
        ids = list(affected)
        for eid in ids:                       # refresh mention_count from surviving edges
            cnt = await client.query(
                db, "SELECT count(*) AS c FROM MENTIONS WHERE inV().entity_id = :e",
                params={"e": eid})
            await client.command(
                db, "UPDATE Entity SET mention_count = :c WHERE entity_id = :e",
                params={"c": cnt[0]["c"] if cnt else 0, "e": eid})
        co_occurs_pruned = await _recompute_co_occurs_with(client, db, ids)
        orphans = _count(await client.command(
            db, "DELETE VERTEX FROM Entity WHERE entity_id IN :ids "
                "AND mention_count = 0 "
                "AND out('MENTIONED_IN').size() = 0 "
                "AND both('RELATION').size() = 0 "
                "AND both('HAS_TEMPORAL_FACT').size() = 0", params={"ids": ids}))

    result = {"version_id": version_id, "chunks": chunks, "relationships": rels,
              "temporal_facts": facts, "temporal_jobs": jobs, "sources": sources, "orphan_entities": orphans,
              "co_occurs_pruned": co_occurs_pruned, **face}
    logger.info("[indexer] delete-by-version %s → %s", version_id, result)
    # P21: physical deletion of an inline source
    from app.services.source_expiry import mark_deleted
    await mark_deleted(client, db, version_id)
    return result
