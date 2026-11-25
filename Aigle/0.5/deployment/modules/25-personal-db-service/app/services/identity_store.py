"""People / identity data layer (dev plan v1.4 §4.2, §4.5).

Module 11's face-detection service does detection, tracking and intra-video
clustering, then sends one video's result here. This module stores it and links
each FaceGroup to a cross-video Identity:

  FaceTrace -TRACE_IN_GROUP-> FaceGroup -GROUP_OF_IDENTITY-> Identity
  FaceTrace -HAS_KEY_FACE-> Face (representative)     Identity -HAS_PROFILE_FACE-> Face
  FaceTrace -TRACE_IN_SOURCE-> Source (the content-indexing node for the video)

Cross-video matching follows 1.4 §4.5 step 5: the best weighted hit among other
videos' FaceGroups and the user's seed photos. Below the threshold the group stays
without an Identity (1.4/1.5: "stored as unnamed, added to the pending list"). A hit
on a group that has an Identity reuses it; a hit on a group without one -- a case
neither plan covers -- creates a pending Identity (source='auto') holding both groups,
so someone seen in several videos becomes one person before anyone names them, while
someone seen in a single video doesn't add an entry to the people list. Each
GROUP_OF_IDENTITY edge records its score and that it was created automatically, so
thresholds can be re-tuned later against stored embeddings.

Writes for a video replace any earlier write of that video (re-analysis after a
failure must not leave stale traces behind); Identities survive, except pending ones
the system created that no longer hold any group.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.models.face import (
    FaceGroupIn,
    FaceTracesRequest,
    FaceTracesResponse,
    GroupMatch,
)
from app.models.identity import (
    BindGroupResult,
    BindResult,
    DeleteIdentityResult,
    FaceImage,
    GroupOut,
    HistoryReplay,
    IdentityList,
    IdentityOut,
    IdentityPatch,
    IdentityTraces,
    IdentityUpload,
    MergeResult,
    PendingIdentityOut,
    PendingPersonIn,
    PendingPersonOut,
    PendingPersons,
    PeopleInVideosResponse,
    PeopleMatch,
    PeopleSearchRequest,
    PeopleSearchResponse,
    PeopleSegment,
    ProfileImageIn,
    TraceOut,
)
from app.services import searcher
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import _count, _edge_exists, _ensure_ready
from app.services.schema_init import initialize_face_schema

logger = logging.getLogger("personal_db.identity")

# databases whose face schema has been applied by this process
_face_schema_ready: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def ensure_face_schema(client: ArcadeDBClient, db: str) -> None:
    """Apply the face layer once per database per process (idempotent DDL)."""
    if db not in _face_schema_ready:
        await initialize_face_schema(client, db)
        _face_schema_ready.add(db)


async def face_schema_present(client: ArcadeDBClient, db: str) -> bool:
    """Whether this database has the face layer at all -- databases created before it
    existed and never written to by face indexing don't, and any query naming a
    FaceTrace/FaceGroup type there would fail."""
    if db in _face_schema_ready:
        return True
    rows = await client.query(db, "SELECT name FROM schema:types WHERE name = 'FaceTrace'")
    return bool(rows)


async def _link(client: ArcadeDBClient, db: str, etype: str,
                from_type: str, from_key: str, from_val: str,
                to_type: str, to_key: str, to_val: str,
                props: dict | None = None) -> None:
    """Check-then-create an edge (ArcadeDB edges have no native UPSERT)."""
    if await _edge_exists(client, db, etype, from_key, from_val, to_key, to_val):
        return
    params = {"f": from_val, "t": to_val}
    set_clause = ""
    if props:
        set_clause = " SET " + ", ".join(f"{k} = :p_{k}" for k in props)
        params.update({f"p_{k}": v for k, v in props.items()})
    await client.command(
        db, f"CREATE EDGE {etype} FROM (SELECT FROM {from_type} WHERE {from_key} = :f) "  # nosec B608 -- every _link() call site passes hardcoded literals for etype/from_type/from_key/to_type/to_key
            f"TO (SELECT FROM {to_type} WHERE {to_key} = :t){set_clause}", params=params)


# ───────────────────────────────────────────────────────────── write path

async def write_face_traces(client: ArcadeDBClient, branch_id: str,
                            req: FaceTracesRequest) -> FaceTracesResponse:
    db = await _ensure_ready(client, branch_id)
    await ensure_face_schema(client, db)

    replaced = await delete_face_data(client, db, req.version_id)

    trace_frames = {t.trace_id: len(t.frames) for t in req.traces}
    for t in req.traces:
        frames_json = json.dumps([f.model_dump() for f in t.frames], separators=(",", ":"))
        avg_conf = sum(f.confidence for f in t.frames) / len(t.frames)
        await client.command(
            db, "UPDATE FaceTrace SET trace_id = :id, version_id = :v, start_sec = :s, end_sec = :e, "
                "frame_count = :n, avg_confidence = :c, embedding = :emb, frames_json = :fj "
                "UPSERT WHERE trace_id = :id",
            params={"id": t.trace_id, "v": req.version_id, "s": t.start_sec, "e": t.end_sec,
                    "n": len(t.frames), "c": avg_conf, "emb": t.embedding, "fj": frames_json})
        rep = t.representative
        face_id = _rep_face_id(t.trace_id)
        await client.command(
            db, "UPDATE Face SET face_id = :id, kind = 'representative', version_id = :v, "
                "image_path = :p, image_version_id = :pv, frame_no = :fn, timestamp_sec = :ts, "
                "bbox = :b, quality = :q, min_side = :m UPSERT WHERE face_id = :id",
            params={"id": face_id, "v": req.version_id, "p": rep.image_path, "pv": rep.image_version_id,
                    "fn": rep.frame_no, "ts": rep.timestamp_sec, "b": rep.bbox, "q": rep.quality,
                    "m": rep.min_side})
        await _link(client, db, "HAS_KEY_FACE", "FaceTrace", "trace_id", t.trace_id, "Face", "face_id", face_id)

    # Link to the video's Source only if content indexing already created it. A Source
    # is deliberately NOT created here: version_exists() reads "has a Source" as "this
    # version was indexed", so a face-only shell would make a video whose content
    # pipeline failed look analysed. Traces written first get linked when content
    # indexing creates the Source (link_face_traces_to_source(), called from indexer.py).
    await link_face_traces_to_source(client, db, req.version_id)

    reps = {t.trace_id: t.representative for t in req.traces}
    matches: list[GroupMatch] = []
    for g in req.groups:
        await client.command(
            db, "UPDATE FaceGroup SET group_id = :id, version_id = :v, embedding = :emb, "
                "frame_count = :n, created_at = :now UPSERT WHERE group_id = :id",
            params={"id": g.group_id, "v": req.version_id, "emb": g.embedding,
                    "n": sum(trace_frames[tid] for tid in g.trace_ids), "now": _now()})
        for tid in g.trace_ids:
            await _link(client, db, "TRACE_IN_GROUP", "FaceTrace", "trace_id", tid, "FaceGroup", "group_id", g.group_id)
        best_trace = max(g.trace_ids, key=lambda tid: reps[tid].quality)
        matches.append(await _match_group(client, db, req.version_id, g, reps[best_trace].min_side))

    result = FaceTracesResponse(version_id=req.version_id, traces_written=len(req.traces),
                                groups_written=len(req.groups), replaced=replaced, matches=matches)
    logger.info("[identity] face-traces %s → traces=%d groups=%d linked=%d new_identities=%d unassigned=%d",
                req.version_id, len(req.traces), len(req.groups),
                sum(bool(m.identity_id) and not m.created_identity for m in matches),
                sum(m.created_identity for m in matches), sum(m.identity_id is None for m in matches))
    return result


def _rep_face_id(trace_id: str) -> str:
    return f"rep_{trace_id}"


async def link_face_traces_to_source(client: ArcadeDBClient, db: str, version_id: str) -> int:
    """Create TRACE_IN_SOURCE for this version's traces that don't have it yet, if its
    Source exists. Called from both sides -- after a face write, and when content
    indexing creates a Source -- so the link appears whichever lands first."""
    if not await face_schema_present(client, db):
        return 0
    source = await client.query(db, "SELECT @rid FROM Source WHERE version_id = :v LIMIT 1", params={"v": version_id})
    if not source:
        return 0
    rows = await client.query(
        db, "SELECT trace_id FROM FaceTrace WHERE version_id = :v AND out('TRACE_IN_SOURCE').size() = 0",
        params={"v": version_id})
    for r in rows:
        await _link(client, db, "TRACE_IN_SOURCE", "FaceTrace", "trace_id", r["trace_id"],
                    "Source", "version_id", version_id)
    return len(rows)


# ───────────────────────────────────────────────────────────── cross-video matching

async def _match_group(client: ArcadeDBClient, db: str, version_id: str, group: FaceGroupIn,
                       rep_min_side: float) -> GroupMatch:
    if rep_min_side < settings.face_match_min_face_px:
        return GroupMatch(group_id=group.group_id, identity_id=None, created_identity=False,
                          skipped_small_face=True)

    best = await _best_match(client, db, version_id, group.embedding)
    if not best or best[0] < settings.face_match_threshold:
        return GroupMatch(group_id=group.group_id, identity_id=None, created_identity=False,
                          score=best[0] if best else None)

    score, identity_id, other_group_id = best
    created = identity_id is None
    if created:
        # the other video's group has no Identity yet: the two groups become one pending person
        identity_id = await _create_pending_identity(
            client, db, await _best_face(client, db, [other_group_id, group.group_id]))
        await _link_group(client, db, other_group_id, identity_id, score=score)
    await _link_group(client, db, group.group_id, identity_id, score=score)
    await _fix_avatar(client, db, identity_id)
    await client.command(db, "UPDATE Identity SET updated_at = :now WHERE identity_id = :i",
                         params={"now": _now(), "i": identity_id})
    return GroupMatch(group_id=group.group_id, identity_id=identity_id, created_identity=created, score=score)


async def _best_match(client: ArcadeDBClient, db: str, version_id: str,
                      embedding: list[float]) -> tuple[float, str | None, str | None] | None:
    """Highest weighted similarity among other videos' FaceGroups and seed photos, as
    (score, identity_id, group_id): a group hit carries the group and its Identity (None
    if it has none), a seed hit only its Identity. None if there's no candidate."""
    k = settings.face_match_candidates
    candidates: list[tuple[float, str | None, str | None]] = []

    # vectorNeighbors treats an empty filter subquery as "no filter" (see
    # searcher._rid_filter_matches), so each search only runs when its filter matches.
    group_filter = "version_id <> :v"
    if await client.query(db, f"SELECT @rid FROM FaceGroup WHERE {group_filter} LIMIT 1", params={"v": version_id}):  # nosec B608 -- group_filter is the hardcoded literal above
        rows = await client.query(
            db, "SELECT group_id, distance "
                f"FROM (SELECT expand(vectorNeighbors('FaceGroup[embedding]', :q, {k}, "
                f"{{'filter': (SELECT @rid AS rid FROM FaceGroup WHERE {group_filter}).rid}})))",  # nosec B608 -- k is settings.face_match_candidates (fixed config), group_filter is the hardcoded literal above
            params={"q": embedding, "v": version_id})
        # Graph traversal (out('GROUP_OF_IDENTITY')) doesn't resolve on vectorNeighbors'
        # expanded rows -- verified live it comes back null -- so the hits' identities
        # are looked up with a plain SELECT afterwards.
        sims = {r["group_id"]: 1.0 - float(r["distance"]) for r in rows}
        if sims:
            owners = await client.query(
                db, "SELECT group_id, out('GROUP_OF_IDENTITY').identity_id AS ids FROM FaceGroup WHERE group_id IN :g",
                params={"g": sorted(sims)})
            for r in owners:
                ids = r.get("ids") or [None]
                candidates.append((sims[r["group_id"]] * settings.face_group_match_weight, ids[0], r["group_id"]))

    # a deleted Identity keeps its seed photos (for undo) but they must not take up the k hits
    seed_filter = ("kind = 'seed' AND identity_id IS NOT NULL AND identity_id NOT IN "
                   "(SELECT identity_id FROM Identity WHERE identity_status = 'deleted').identity_id")
    if await client.query(db, f"SELECT @rid FROM Face WHERE {seed_filter} LIMIT 1"):  # nosec B608 -- seed_filter is the hardcoded literal above
        rows = await client.query(
            db, "SELECT identity_id, distance "
                f"FROM (SELECT expand(vectorNeighbors('Face[embedding]', :q, {k}, "
                f"{{'filter': (SELECT @rid AS rid FROM Face WHERE {seed_filter}).rid}})))",  # nosec B608 -- k is settings.face_match_candidates, seed_filter is the hardcoded literal above
            params={"q": embedding})
        for r in rows:
            candidates.append(((1.0 - float(r["distance"])) * settings.face_seed_match_weight, r["identity_id"], None))

    # a merged Identity's groups were moved to its target, and a deleted one was let go of;
    # never attach new ones to either -- a deleted Identity's seed photos stay but no longer match
    ids = sorted({c[1] for c in candidates if c[1]})
    merged = {r["identity_id"] for r in await client.query(
        db, "SELECT identity_id FROM Identity WHERE identity_id IN :ids AND identity_status IN ['merged', 'deleted']",
        params={"ids": ids})} if ids else set()
    live = [c for c in candidates if c[1] not in merged]
    return max(live, key=lambda c: c[0]) if live else None


async def _best_face(client: ArcadeDBClient, db: str, group_ids: list[str]) -> str | None:
    """The best-quality representative frame among these groups' traces."""
    rows = await client.query(
        db, "SELECT face_id FROM (SELECT expand(in('TRACE_IN_GROUP').out('HAS_KEY_FACE')) "
            "FROM FaceGroup WHERE group_id IN :g) ORDER BY quality DESC LIMIT 1", params={"g": group_ids})
    return rows[0]["face_id"] if rows else None


async def _best_face_of_largest(client: ArcadeDBClient, db: str, group_ids: list[str]) -> str | None:
    """The best-quality representative frame of the largest of these groups (most traces;
    groups tied for largest are searched together). A stray one-trace group -- a trace
    bound by mistake, or the one a pending person was started from -- never supplies it
    while the person holds a bigger one."""
    if not group_ids:
        return None
    rows = await client.query(db, "SELECT group_id, in('TRACE_IN_GROUP').size() AS n FROM FaceGroup "
                                  "WHERE group_id IN :g", params={"g": group_ids})
    if not rows:
        return None
    most = max(r.get("n") or 0 for r in rows)
    return await _best_face(client, db, [r["group_id"] for r in rows if (r.get("n") or 0) == most])


async def _create_pending_identity(client: ArcadeDBClient, db: str, avatar_face_id: str | None) -> str:
    """A pending Identity the system creates (source='auto') when groups in two videos
    match before anyone named them. name is '' rather than null: Identity.name has a
    full-text index, which rejects null keys."""
    identity_id = uuid.uuid4().hex
    now = _now()
    await client.command(
        db, "UPDATE Identity SET identity_id = :id, name = '', aliases_json = '{}', identity_type = 'person', "
            "source = 'auto', identity_status = 'pending', created_at = :now, updated_at = :now "
            "UPSERT WHERE identity_id = :id",
        params={"id": identity_id, "now": now})
    if avatar_face_id:
        await _link(client, db, "HAS_PROFILE_FACE", "Identity", "identity_id", identity_id, "Face", "face_id", avatar_face_id)
    return identity_id


async def _link_group(client: ArcadeDBClient, db: str, group_id: str, identity_id: str,
                      score: float | None) -> None:
    await _link(client, db, "GROUP_OF_IDENTITY", "FaceGroup", "group_id", group_id,
                "Identity", "identity_id", identity_id,
                props={"score": score, "auto": True, "created_at": _now()})


# ───────────────────────────────────────────────────────────── clone path

async def clone_face_data(client: ArcadeDBClient, db: str, source_version_id: str,
                          target_version_id: str, new_id) -> dict:
    """Clone a source video's face data onto a cloned target version (module 04's
    dedup optimization -- see indexer.clone_version(), which calls this via a local
    import the same way it calls temporal_jobs.clone_temporal(), to avoid a circular
    import: this module already imports from indexer.py).

    FaceGroup is per-video (this module's own docstring), so unlike Entity it can't
    just be shared -- it's cloned with a new id, same as Chunk is. GROUP_OF_IDENTITY
    is recreated pointing at the SAME Identity the source group already points at
    (never cloned, same reasoning as Entity in clone_version()) -- carrying that
    match over is the whole point: the target video shouldn't need its own re-run of
    cross-video matching to reach the conclusion the source already did. A group with
    no Identity yet (pending/unmatched) clones to another ownerless group, unchanged,
    so it still shows up in the target's pending-persons. Each representative Face is
    cloned too (new face_id, version_id=target) but keeps the source's own
    image_path/image_version_id -- that crop file isn't re-derived, same as how an
    image resource elsewhere in clone_version() is shared rather than duplicated.

    *new_id(target_version, old_id)*: the clone's id scheme (indexer._clone_id).
    """
    zero = {"face_traces": 0, "faces": 0, "face_groups": 0}
    if not await face_schema_present(client, db):
        return zero

    trace_id_map: dict[str, str] = {}
    faces_cloned = 0
    for row in await client.query(db, "SELECT FROM FaceTrace WHERE version_id = :v", params={"v": source_version_id}):
        old_trace_id = row["trace_id"]
        new_trace_id = new_id(target_version_id, old_trace_id)
        trace_id_map[old_trace_id] = new_trace_id
        fields = {k: v for k, v in row.items() if not str(k).startswith("@")}
        fields["trace_id"] = new_trace_id
        fields["version_id"] = target_version_id
        set_parts = ", ".join(f"{k} = :{k}" for k in fields)
        await client.command(
            db, f"UPDATE FaceTrace SET {set_parts} UPSERT WHERE trace_id = :trace_id", params=fields)  # nosec B608 -- fields' keys are an existing FaceTrace row's own column names, not request-derived

        rep_ids = await client.query(
            db, "SELECT out('HAS_KEY_FACE').face_id AS f FROM FaceTrace WHERE trace_id = :t",
            params={"t": old_trace_id})
        old_face_id = ((rep_ids[0].get("f") if rep_ids else None) or [None])[0]
        rep = await client.query(db, "SELECT FROM Face WHERE face_id = :f", params={"f": old_face_id}) if old_face_id else []
        if rep:
            face_fields = {k: v for k, v in rep[0].items() if not str(k).startswith("@")}
            new_face_id = _rep_face_id(new_trace_id)
            face_fields["face_id"] = new_face_id
            face_fields["version_id"] = target_version_id
            fset = ", ".join(f"{k} = :{k}" for k in face_fields)
            await client.command(
                db, f"UPDATE Face SET {fset} UPSERT WHERE face_id = :face_id", params=face_fields)  # nosec B608 -- face_fields' keys are an existing Face row's own column names, not request-derived
            await _link(client, db, "HAS_KEY_FACE", "FaceTrace", "trace_id", new_trace_id,
                       "Face", "face_id", new_face_id)
            faces_cloned += 1

    groups_cloned = 0
    for row in await client.query(db, "SELECT FROM FaceGroup WHERE version_id = :v", params={"v": source_version_id}):
        old_group_id = row["group_id"]
        new_group_id = new_id(target_version_id, old_group_id)
        fields = {k: v for k, v in row.items() if not str(k).startswith("@")}
        fields["group_id"] = new_group_id
        fields["version_id"] = target_version_id
        set_parts = ", ".join(f"{k} = :{k}" for k in fields)
        await client.command(
            db, f"UPDATE FaceGroup SET {set_parts} UPSERT WHERE group_id = :group_id", params=fields)  # nosec B608 -- fields' keys are an existing FaceGroup row's own column names, not request-derived
        groups_cloned += 1

        members = await client.query(
            db, "SELECT in('TRACE_IN_GROUP').trace_id AS ids FROM FaceGroup WHERE group_id = :g", params={"g": old_group_id})
        for old_tid in ((members[0].get("ids") if members else None) or []):
            new_tid = trace_id_map.get(old_tid)
            if new_tid:
                await _link(client, db, "TRACE_IN_GROUP", "FaceTrace", "trace_id", new_tid,
                           "FaceGroup", "group_id", new_group_id)

        owners = await client.query(
            db, "SELECT out('GROUP_OF_IDENTITY').identity_id AS ids, out('GROUP_OF_IDENTITY').score AS scores, "
                "out('GROUP_OF_IDENTITY').auto AS autos FROM FaceGroup WHERE group_id = :g", params={"g": old_group_id})
        ids = (owners[0].get("ids") if owners else None) or []
        if ids and ids[0]:
            scores = owners[0].get("scores") or [None]
            autos = owners[0].get("autos") or [None]
            await _link(client, db, "GROUP_OF_IDENTITY", "FaceGroup", "group_id", new_group_id,
                       "Identity", "identity_id", ids[0],
                       props={"score": scores[0], "auto": bool(autos[0]), "created_at": _now()})

    await link_face_traces_to_source(client, db, target_version_id)
    return {"face_traces": len(trace_id_map), "faces": faces_cloned, "face_groups": groups_cloned}


# ───────────────────────────────────────────────────────────── delete path

async def delete_face_data(client: ArcadeDBClient, db: str, version_id: str) -> dict:
    """Remove one video's face data (1.4 §4.2 cleanup rules). Used both when the video's
    index is deleted and before re-writing it.

      - the video's FaceTraces, their representative Faces and its FaceGroups go
        (groups are per-video, so all of them are left empty); seed photos never do
      - Identities stay -- deleting one is an explicit user action -- except pending
        ones the system created (source='auto') that no longer hold any group, which
        would otherwise pile up as empty shells nobody ever touched. One left holding a
        single group stays: it was a cross-video match when it was made
      - an Identity whose avatar was one of the deleted frames is repointed to its best
        remaining representative frame, or left without one
    """
    zero = {"face_traces": 0, "faces": 0, "face_groups": 0, "auto_identities_removed": 0, "avatars_repointed": 0}
    if not await face_schema_present(client, db):
        return zero

    touched = await _ids(client, db, "SELECT out('GROUP_OF_IDENTITY').identity_id AS ids FROM FaceGroup WHERE version_id = :v", version_id)
    avatar_owners = await _ids(client, db, "SELECT in('HAS_PROFILE_FACE').identity_id AS ids FROM Face "
                                           "WHERE version_id = :v AND kind = 'representative'", version_id)

    faces = _count(await client.command(
        db, "DELETE VERTEX FROM Face WHERE version_id = :v AND kind = 'representative'", params={"v": version_id}))
    traces = _count(await client.command(
        db, "DELETE VERTEX FROM FaceTrace WHERE version_id = :v", params={"v": version_id}))
    groups = _count(await client.command(
        db, "DELETE VERTEX FROM FaceGroup WHERE version_id = :v", params={"v": version_id}))

    removed = 0
    if touched:
        removed = _count(await client.command(
            db, "DELETE VERTEX FROM Identity WHERE identity_id IN :ids AND source = 'auto' "
                "AND identity_status = 'pending' AND in('GROUP_OF_IDENTITY').size() = 0",
            params={"ids": sorted(touched)}))

    repointed = 0
    for identity_id in sorted(avatar_owners):
        repointed += await _fix_avatar(client, db, identity_id)

    return {"face_traces": traces, "faces": faces, "face_groups": groups,
            "auto_identities_removed": removed, "avatars_repointed": repointed}


# GD-11: every vertex type of the face layer, in the order purge_face_layer deletes them.
FACE_LAYER = (("IdentityHistory", "identity_history"), ("Identity", "identities"), ("FaceGroup", "face_groups"),
               ("FaceTrace", "face_traces"), ("Face", "faces"))


async def purge_face_layer(client: ArcadeDBClient, db: str) -> dict:
    """Delete the whole face layer of one user database (GD-11, face_detection consent
    withdrawn): every Face (video representatives and seed photos, with their
    embeddings), FaceTrace, FaceGroup, Identity -- named ones too, since each is
    built from face matches and holds its avatar -- and IdentityHistory, whose
    snapshots would let an undo bring them back. Their edges go with them, including
    TRACE_IN_SOURCE; the rest of the database (Chunk, Source, Entity, ...) is untouched.

    The face JPEGs themselves (Face.image_path) are files in Module 04, not here.
    Returns {kind: vertices deleted}; all zeros when the database never had the layer."""
    if not await face_schema_present(client, db):
        return {key: 0 for _, key in FACE_LAYER}
    return {key: _count(await client.command(db, f"DELETE VERTEX FROM {vtype}"))  # nosec B608 -- vtype from the FACE_LAYER literal above
            for vtype, key in FACE_LAYER}


async def _ids(client: ArcadeDBClient, db: str, sql: str, version_id: str) -> set[str]:
    out: set[str] = set()
    for r in await client.query(db, sql, params={"v": version_id}):
        out.update(i for i in (r.get("ids") or []) if i)
    return out


async def avatar_choice(client: ArcadeDBClient, db: str, identity_id: str) -> tuple[str | None, str | None] | None:
    """(current, chosen) avatar face_id of an Identity, or None when it isn't re-chosen:
    the Identity is gone, or its avatar is a seed photo (which always stays). Chosen is
    the best-quality frame of its largest group (_best_face_of_largest), None if it
    holds no faces."""
    rows = await client.query(
        db, "SELECT out('HAS_PROFILE_FACE').face_id AS f, out('HAS_PROFILE_FACE').kind AS k "
            "FROM Identity WHERE identity_id = :i", params={"i": identity_id})
    if not rows:
        return None
    current, kinds = rows[0].get("f") or [], rows[0].get("k") or []
    if "seed" in kinds:
        return None
    best = await _best_face_of_largest(client, db, list((await _owned_items(client, db, identity_id))[0]))
    return (current[0] if current else None), best


async def _fix_avatar(client: ArcadeDBClient, db: str, identity_id: str) -> int:
    """Re-choose an Identity's avatar after its groups changed (avatar_choice), so the
    frame a person happened to be started from -- often a poor one -- doesn't stay
    their picture once better faces join. Dropped if it holds no faces. Returns 1 if a
    new avatar was set."""
    choice = await avatar_choice(client, db, identity_id)
    if choice is None or choice[0] == choice[1]:
        return 0
    current, best = choice
    if current:
        await client.command(db, "DELETE FROM HAS_PROFILE_FACE WHERE outV().identity_id = :i", params={"i": identity_id})
    if not best:
        return 0
    await _link(client, db, "HAS_PROFILE_FACE", "Identity", "identity_id", identity_id, "Face", "face_id", best)
    return 1


# ───────────────────────────────────────────────────────────── Identity API (1.4 §4.5)
#
# Identity management, creation, pending persons, bind/unbind, delete/merge with
# 24-hour undo/redo.

class IdentityNotFoundError(LookupError):
    """The Identity, trace or video named in the request doesn't exist."""


class IdentityConflictError(Exception):
    """The request is valid but not allowed in the Identity's current state."""


_IDENTITY_FIELDS = ("identity_id, name, aliases_json, identity_type, source, identity_status, origin_version_id, "
                    "created_at, updated_at, out('HAS_PROFILE_FACE').face_id AS avatar_ids, "
                    "in('GROUP_OF_IDENTITY').version_id AS group_versions")


async def _face_db(client: ArcadeDBClient, branch_id: str) -> str:
    db = await _ensure_ready(client, branch_id)
    await ensure_face_schema(client, db)
    return db


async def _faces(client: ArcadeDBClient, db: str, face_ids) -> dict[str, dict]:
    ids = sorted({f for f in face_ids if f})
    if not ids:
        return {}
    rows = await client.query(
        db, "SELECT face_id, kind, image_path, image_version_id, quality FROM Face WHERE face_id IN :ids",
        params={"ids": ids})
    return {r["face_id"]: r for r in rows}


def _face_image(row: dict | None) -> FaceImage | None:
    if not row:
        return None
    return FaceImage(face_id=row["face_id"], kind=row.get("kind") or "representative",
                     image_path=row.get("image_path"), image_version_id=row.get("image_version_id"))


async def _identities(client: ArcadeDBClient, db: str, where: str, params: dict, tail: str = "") -> list[IdentityOut]:
    rows = await client.query(db, f"SELECT {_IDENTITY_FIELDS} FROM Identity WHERE {where} {tail}", params=params)  # nosec B608 -- _IDENTITY_FIELDS is a fixed constant; every _identities() caller passes a hardcoded where/tail literal
    faces = await _faces(client, db, [(r.get("avatar_ids") or [None])[0] for r in rows])
    out = []
    for r in rows:
        versions = [v for v in (r.get("group_versions") or []) if v]
        out.append(IdentityOut(
            identity_id=r["identity_id"], name=r.get("name") or "",
            aliases=json.loads(r.get("aliases_json") or "{}"), identity_type=r.get("identity_type"),
            source=r.get("source"), identity_status=r.get("identity_status") or "active",
            origin_version_id=r.get("origin_version_id"), created_at=r.get("created_at"),
            updated_at=r.get("updated_at"), avatar=_face_image(faces.get((r.get("avatar_ids") or [None])[0])),
            group_count=len(versions), video_count=len(set(versions))))
    return out


async def _identity(client: ArcadeDBClient, db: str, identity_id: str) -> IdentityOut:
    found = await _identities(client, db, "identity_id = :i", {"i": identity_id})
    if not found:
        raise IdentityNotFoundError(f"identity {identity_id} not found")
    return found[0]


def _merge_aliases(current: dict[str, str], changes: dict[str, str | None] | None) -> dict[str, str]:
    merged = dict(current)
    for locale, value in (changes or {}).items():
        if value is None:
            merged.pop(locale, None)
        else:
            merged[locale] = value
    return merged


async def _upsert_identity(client: ArcadeDBClient, db: str, identity_id: str, *, name: str, aliases: dict[str, str],
                           identity_type: str, source: str, status: str,
                           origin_version_id: str | None = None) -> None:
    now = _now()
    await client.command(
        db, "UPDATE Identity SET identity_id = :id, name = :n, aliases_json = :a, identity_type = :t, source = :s, "
            "identity_status = :st, origin_version_id = :o, created_at = :now, updated_at = :now "
            "UPSERT WHERE identity_id = :id",
        params={"id": identity_id, "n": name, "a": json.dumps(aliases, ensure_ascii=False), "t": identity_type,
                "s": source, "st": status, "o": origin_version_id, "now": now})


async def _touch(client: ArcadeDBClient, db: str, identity_ids) -> None:
    ids = sorted({i for i in identity_ids if i})
    if ids:
        await client.command(db, "UPDATE Identity SET updated_at = :now WHERE identity_id IN :ids",
                             params={"now": _now(), "ids": ids})


# ── identity management ──────────────────────────────────────────────────

async def list_identities(client: ArcadeDBClient, branch_id: str, limit: int, offset: int,
                          status: str | None = None) -> IdentityList:
    """Most recently updated first. Merged and deleted Identities are left out unless asked for by status."""
    db = await _face_db(client, branch_id)
    # Not "NOT IN ['merged', 'deleted']": ArcadeDB 26.6 answers that with no rows at all.
    where, params = (("identity_status = :s", {"s": status}) if status
                     else ("identity_status <> 'merged' AND identity_status <> 'deleted'", {}))
    total = await client.query(db, f"SELECT count(*) AS c FROM Identity WHERE {where}", params=params)  # nosec B608 -- where is the hardcoded literal above
    items = await _identities(client, db, where, params,
                              f"ORDER BY updated_at DESC SKIP {int(offset)} LIMIT {int(limit)}")  # nosec B608 -- offset/limit explicitly int()-coerced right here
    return IdentityList(total=total[0]["c"] if total else 0, items=items)


async def get_identity(client: ArcadeDBClient, branch_id: str, identity_id: str) -> IdentityOut:
    return await _identity(client, await _face_db(client, branch_id), identity_id)


async def patch_identity(client: ArcadeDBClient, branch_id: str, identity_id: str, req: IdentityPatch) -> IdentityOut:
    """name / aliases / pending -> active. 'merged' is only ever set by merge, 'deleted' by delete."""
    db = await _face_db(client, branch_id)
    ident = await _live_identity(client, db, identity_id)
    sets, params = ["updated_at = :now"], {"i": identity_id, "now": _now()}
    if req.name is not None:
        sets.append("name = :n")
        params["n"] = req.name
    if req.aliases is not None:
        sets.append("aliases_json = :a")
        params["a"] = json.dumps(_merge_aliases(ident.aliases, req.aliases), ensure_ascii=False)
    if req.identity_status is not None:
        sets.append("identity_status = :st")         # pending -> active, or already active
        params["st"] = req.identity_status
    await client.command(db, f"UPDATE Identity SET {', '.join(sets)} WHERE identity_id = :i", params=params)  # nosec B608 -- sets built above from hardcoded "x = :x"-shaped literals only
    return await _identity(client, db, identity_id)


async def upload_identity(client: ArcadeDBClient, branch_id: str, req: IdentityUpload) -> tuple[IdentityOut, bool]:
    """With identity_id: update that Identity, or create it under that ID (a retried
    request never makes a second one). Without: create a new one. A new Identity is
    active -- someone named it on purpose. Returns (identity, created)."""
    db = await _face_db(client, branch_id)
    identity_id = req.identity_id or uuid.uuid4().hex
    existing = await _identities(client, db, "identity_id = :i", {"i": identity_id})
    if existing:
        ident = existing[0]
        _check_live(ident)
        await client.command(
            db, "UPDATE Identity SET name = :n, aliases_json = :a, identity_type = :t, updated_at = :now "
                "WHERE identity_id = :i",
            params={"i": identity_id, "n": req.name, "t": req.identity_type, "now": _now(),
                    "a": json.dumps(_merge_aliases(ident.aliases, req.aliases), ensure_ascii=False)})
        return await _identity(client, db, identity_id), False
    await _upsert_identity(client, db, identity_id, name=req.name, aliases=_merge_aliases({}, req.aliases),
                           identity_type=req.identity_type, source=req.source, status="active")
    return await _identity(client, db, identity_id), True


# ── history: delete, merge, undo, redo (1.4 §4.5 "Merge & Recovery") ────
#
# Delete and merge each record two snapshots of every item they change, in one format:
#   identities: {identity_id: fields | None}        None = doesn't exist
#   groups:     {group_id: {"identity_id": id | None, "edge": {score, auto, created_at}}}
#   seed_faces: {face_id: seed Face fields | None}
#   avatars:    {identity_id: face_id | None}
# Running the operation is applying before -> after. Undo applies after -> before and
# redo before -> after, item by item, and only where the current value still equals
# the side being left -- anything changed again since is skipped and reported.

_COMPARED_FIELDS = ("name", "aliases_json", "identity_type", "source", "identity_status", "origin_version_id")
_MISSING = object()


async def _identity_fields(client: ArcadeDBClient, db: str, identity_id: str) -> dict | None:
    rows = await client.query(db, "SELECT * FROM Identity WHERE identity_id = :i", params={"i": identity_id})
    return {k: v for k, v in rows[0].items() if not k.startswith("@")} if rows else None


async def _owned_items(client: ArcadeDBClient, db: str, identity_id: str) -> tuple[dict, dict, str | None]:
    """(groups, seed_faces, avatar) as they stand for this Identity, in snapshot form."""
    edges = await client.query(
        db, "SELECT outV().group_id AS group_id, score, auto, created_at FROM GROUP_OF_IDENTITY "
            "WHERE inV().identity_id = :i", params={"i": identity_id})
    groups = {e["group_id"]: {"identity_id": identity_id,
                              "edge": {"score": e.get("score"), "auto": e.get("auto"), "created_at": e.get("created_at")}}
              for e in edges}
    seeds = {r["face_id"]: {k: r.get(k) for k in ("face_id", "kind", "identity_id", "image_path", "image_version_id",
                                                   "embedding")}
             for r in await client.query(db, "SELECT * FROM Face WHERE kind = 'seed' AND identity_id = :i",
                                         params={"i": identity_id})}
    avatar = await client.query(db, "SELECT out('HAS_PROFILE_FACE').face_id AS f FROM Identity WHERE identity_id = :i",
                                params={"i": identity_id})
    return groups, seeds, ((avatar[0].get("f") or [None])[0] if avatar else None)


async def _current(client: ArcadeDBClient, db: str, snap: dict) -> dict:
    """The live values of every item a snapshot names."""
    identities = {i: await _identity_fields(client, db, i) for i in snap["identities"]}
    groups = {}
    for gid in snap["groups"]:
        rows = await client.query(db, "SELECT out('GROUP_OF_IDENTITY').identity_id AS ids FROM FaceGroup WHERE group_id = :g",
                                  params={"g": gid})
        groups[gid] = ((rows[0].get("ids") or [None])[0]) if rows else _MISSING   # the video may be gone
    seeds = {}
    for fid in snap["seed_faces"]:
        rows = await client.query(db, "SELECT identity_id FROM Face WHERE face_id = :f", params={"f": fid})
        seeds[fid] = rows[0].get("identity_id") if rows else None
    avatars = {}
    for i in snap["avatars"]:
        rows = await client.query(db, "SELECT out('HAS_PROFILE_FACE').face_id AS f FROM Identity WHERE identity_id = :i",
                                  params={"i": i})
        avatars[i] = ((rows[0].get("f") or [None])[0]) if rows else None
    return {"identities": identities, "groups": groups, "seed_faces": seeds, "avatars": avatars}


async def _apply(client: ArcadeDBClient, db: str, frm: dict, to: dict) -> tuple[list[str], list[str]]:
    """Move every item from its `frm` value to its `to` value, where it still holds `frm`.
    Returns (applied, skipped) as readable item names."""
    cur = await _current(client, db, frm)
    applied: list[str] = []
    skipped: list[str] = []

    def note(ok: bool, item: str) -> bool:
        (applied if ok else skipped).append(item)
        return ok

    # 1. identities that come into being, and field changes
    for i, want in to["identities"].items():
        had, now = frm["identities"][i], cur["identities"][i]
        if want is None:
            continue                                      # removals go last
        if had is None:
            if note(now is None, f"identity {i}"):
                await _upsert_identity(client, db, i, name=want.get("name") or "",
                                       aliases=json.loads(want.get("aliases_json") or "{}"),
                                       identity_type=want.get("identity_type") or "person",
                                       source=want.get("source") or "manual",
                                       status=want.get("identity_status") or "active",
                                       origin_version_id=want.get("origin_version_id"))
                await client.command(db, "UPDATE Identity SET created_at = :c WHERE identity_id = :i",
                                     params={"i": i, "c": want.get("created_at") or _now()})
            continue
        for f in _COMPARED_FIELDS:
            if had.get(f) == want.get(f):
                continue
            if note(now is not None and now.get(f) == had.get(f), f"identity {i} {f}"):
                await client.command(db, f"UPDATE Identity SET {f} = :v WHERE identity_id = :i",  # nosec B608 -- f iterates _COMPARED_FIELDS, a fixed tuple
                                     params={"i": i, "v": want.get(f)})

    # 2. group ownership
    for gid, want in to["groups"].items():
        had = frm["groups"][gid]["identity_id"]
        if had == want["identity_id"]:
            continue
        now = cur["groups"][gid]
        target = want["identity_id"]
        ok = now is not _MISSING and now == had and (
            target is None or await _identity_fields(client, db, target) is not None)
        if note(ok, f"group {gid}"):
            await client.command(db, "DELETE FROM GROUP_OF_IDENTITY WHERE outV().group_id = :g", params={"g": gid})
            if target:
                await _link(client, db, "GROUP_OF_IDENTITY", "FaceGroup", "group_id", gid, "Identity", "identity_id",
                            target, props=want.get("edge") or {"score": None, "auto": False, "created_at": _now()})

    # 3. seed photos: whose they are, or whether they exist
    for fid, want in to["seed_faces"].items():
        had = frm["seed_faces"][fid]
        had_owner = had["identity_id"] if had else None
        want_owner = want["identity_id"] if want else None
        if had_owner == want_owner:
            continue
        if note(cur["seed_faces"][fid] == had_owner, f"seed photo {fid}"):
            if want is None:
                await client.command(db, "DELETE VERTEX FROM Face WHERE face_id = :f", params={"f": fid})
            elif had is None:
                await client.command(
                    db, "INSERT INTO Face SET face_id = :f, kind = 'seed', identity_id = :i, image_path = :p, "
                        "image_version_id = :pv, embedding = :e",
                    params={"f": fid, "i": want_owner, "p": want.get("image_path"),
                            "pv": want.get("image_version_id"), "e": want.get("embedding")})
            else:
                await client.command(db, "UPDATE Face SET identity_id = :i WHERE face_id = :f",
                                     params={"f": fid, "i": want_owner})

    # 4. avatars
    for i, want in to["avatars"].items():
        had = frm["avatars"][i]
        if had == want or to["identities"].get(i, True) is None:
            continue
        exists = await _identity_fields(client, db, i) is not None
        face_ok = want is None or bool(await _faces(client, db, [want]))
        if note(exists and face_ok and cur["avatars"][i] == had, f"avatar of {i}"):
            await client.command(db, "DELETE FROM HAS_PROFILE_FACE WHERE outV().identity_id = :i", params={"i": i})
            if want:
                await _link(client, db, "HAS_PROFILE_FACE", "Identity", "identity_id", i, "Face", "face_id", want)

    # 5. identities that go away
    for i, want in to["identities"].items():
        if want is not None or frm["identities"][i] is None:
            continue
        if note(cur["identities"][i] is not None, f"identity {i}"):
            await client.command(db, "DELETE VERTEX FROM Identity WHERE identity_id = :i", params={"i": i})

    await _touch(client, db, [i for i in to["identities"] if to["identities"][i] is not None])
    return applied, skipped


async def _record(client: ArcadeDBClient, db: str, operation: str, before: dict, after: dict) -> tuple[str, str]:
    history_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    deadline = (now + timedelta(hours=24)).isoformat()
    await client.command(
        db, "INSERT INTO IdentityHistory SET history_id = :h, operation = :o, before_snapshot_json = :b, "
            "after_snapshot_json = :a, history_state = 'applied', undo_deadline = :d, created_at = :c",
        params={"h": history_id, "o": operation, "b": json.dumps(before, ensure_ascii=False),
                "a": json.dumps(after, ensure_ascii=False), "d": deadline, "c": now.isoformat()})
    return history_id, deadline


async def delete_identity(client: ArcadeDBClient, branch_id: str, identity_id: str) -> DeleteIdentityResult:
    """1.4 §4.5: remove only the Identity's GROUP_OF_IDENTITY edges. Its groups, traces
    and representative frames stay, as unassigned groups back on their videos' pending
    lists. The Identity node, its seed photos and its avatar stay too; the node is marked
    'deleted', which leaves it out of listings and People Search, refuses changes, and
    stops its seed photos from matching later videos. The snapshot holds the status and
    each group's edge, for undo within 24 hours."""
    db = await _face_db(client, branch_id)
    _check_live(await _identity(client, db, identity_id))
    fields = await _identity_fields(client, db, identity_id)
    groups, _, _ = await _owned_items(client, db, identity_id)
    before = {"identities": {identity_id: fields}, "groups": groups, "seed_faces": {}, "avatars": {}}
    after = {"identities": {identity_id: {**fields, "identity_status": "deleted"}},
             "groups": {g: {"identity_id": None} for g in groups},
             "seed_faces": {}, "avatars": {}}
    history_id, deadline = await _record(client, db, "delete", before, after)
    await _apply(client, db, before, after)
    logger.info("[identity] deleted %s (history %s): %d groups unassigned",
                identity_id, history_id, len(groups))
    return DeleteIdentityResult(identity_id=identity_id, history_id=history_id, undo_deadline=deadline,
                                groups_unassigned=sorted(groups))


async def merge_identity(client: ArcadeDBClient, branch_id: str, source_id: str, target_id: str,
                         keep_source: bool) -> MergeResult:
    """Fold source into target: source's groups and seed photos move to target, source's
    aliases fill the locales target doesn't have yet (never overwriting). Target's avatar:
    its own seed photo, else source's, else the best frame of the largest group it now
    holds (as _fix_avatar). Source is kept as 'merged' or deleted."""
    db = await _face_db(client, branch_id)
    if source_id == target_id:
        raise IdentityConflictError("cannot merge an identity into itself")
    source = await _identity_fields(client, db, source_id)
    target = await _identity_fields(client, db, target_id)
    if source is None or target is None:
        raise IdentityNotFoundError(f"identity {source_id if source is None else target_id} not found")
    for i, f in ((source_id, source), (target_id, target)):
        if f.get("identity_status") == "merged":
            raise IdentityConflictError(f"identity {i} was already merged into another identity")
        if f.get("identity_status") == "deleted":
            raise IdentityConflictError(f"identity {i} was deleted")

    groups, seeds, src_avatar = await _owned_items(client, db, source_id)
    tgt_groups, tgt_seeds, tgt_avatar = await _owned_items(client, db, target_id)
    if tgt_avatar in tgt_seeds or src_avatar in seeds:
        avatar = tgt_avatar if tgt_avatar in tgt_seeds else src_avatar
    else:
        avatar = await _best_face_of_largest(client, db, [*tgt_groups, *groups]) or tgt_avatar or src_avatar
    target_aliases = json.loads(target.get("aliases_json") or "{}")
    aliases = {**json.loads(source.get("aliases_json") or "{}"), **target_aliases}

    before = {"identities": {source_id: source, target_id: target}, "groups": groups, "seed_faces": seeds,
              "avatars": {target_id: tgt_avatar, source_id: src_avatar}}
    after = {
        "identities": {source_id: {**source, "identity_status": "merged"} if keep_source else None,
                       target_id: {**target, "aliases_json": json.dumps(aliases, ensure_ascii=False)}},
        "groups": {g: {"identity_id": target_id, "edge": {**v["edge"], "auto": False, "created_at": _now()}}
                   for g, v in groups.items()},
        "seed_faces": {f: {**v, "identity_id": target_id} for f, v in seeds.items()},
        "avatars": {target_id: avatar, source_id: src_avatar if keep_source else None},
    }
    history_id, deadline = await _record(client, db, "merge", before, after)
    await _apply(client, db, before, after)
    logger.info("[identity] merged %s into %s (history %s): %d groups, %d seed photos",
                source_id, target_id, history_id, len(groups), len(seeds))
    return MergeResult(merge_id=history_id, undo_deadline=deadline, source_id=source_id,
                       source_kept=keep_source, groups_moved=sorted(groups),
                       target=await _identity(client, db, target_id))


async def replay_history(client: ArcadeDBClient, branch_id: str, history_id: str, direction: str) -> HistoryReplay:
    """undo: after -> before; redo: before -> after. 409 past the deadline, or when the
    record isn't in the state this direction starts from. When nothing can be applied --
    every item was changed again since -- everything is reported skipped and the record
    keeps its state (MTR-23): the operation was not undone (or redone), so it is not
    marked as if it had been."""
    db = await _face_db(client, branch_id)
    rows = await client.query(db, "SELECT * FROM IdentityHistory WHERE history_id = :h", params={"h": history_id})
    if not rows:
        raise IdentityNotFoundError(f"history {history_id} not found")
    h = rows[0]
    if datetime.now(timezone.utc) > datetime.fromisoformat(h["undo_deadline"]):
        raise IdentityConflictError(f"history {history_id} expired at {h['undo_deadline']}")
    start, end = ("applied", "reverted") if direction == "undo" else ("reverted", "applied")
    if h["history_state"] != start:
        raise IdentityConflictError(f"history {history_id} is {h['history_state']}; {direction} needs {start}")
    before, after = json.loads(h["before_snapshot_json"]), json.loads(h["after_snapshot_json"])
    frm, to = (after, before) if direction == "undo" else (before, after)
    applied, skipped = await _apply(client, db, frm, to)
    if not applied:
        end = start                                  # nothing changed: the record stays as it was
    await client.command(db, "UPDATE IdentityHistory SET history_state = :s WHERE history_id = :h",
                         params={"s": end, "h": history_id})
    logger.info("[identity] %s %s %s: %d applied, %d skipped", direction, h["operation"], history_id,
                len(applied), len(skipped))
    return HistoryReplay(history_id=history_id, operation=h["operation"], history_state=end,
                         undo_deadline=h["undo_deadline"], applied=applied, skipped=skipped)


# ── seed photo (1.4 §4.5 profile-image) ─────────────────────────────────

def _seed_face_id(identity_id: str) -> str:
    return f"seed_{identity_id}"


async def set_profile_image(client: ArcadeDBClient, branch_id: str, identity_id: str,
                            req: ProfileImageIn) -> IdentityOut:
    """Store the Identity's seed photo and make it the avatar. From then on its
    embedding is in Face[embedding], so later videos match this person by it. One
    seed photo per Identity: a new one replaces the old (the old file stays in
    Module 04 for object-storage lifecycle cleanup, 1.4 §4.5)."""
    db = await _face_db(client, branch_id)
    await _live_identity(client, db, identity_id)
    face_id = _seed_face_id(identity_id)
    await client.command(
        db, "UPDATE Face SET face_id = :f, kind = 'seed', identity_id = :i, image_path = :p, "
            "image_version_id = :pv, embedding = :e UPSERT WHERE face_id = :f",
        params={"f": face_id, "i": identity_id, "p": req.image_path, "pv": req.image_version_id,
                "e": req.embedding})
    await client.command(db, "DELETE FROM HAS_PROFILE_FACE WHERE outV().identity_id = :i", params={"i": identity_id})
    await _link(client, db, "HAS_PROFILE_FACE", "Identity", "identity_id", identity_id, "Face", "face_id", face_id)
    await _touch(client, db, [identity_id])
    return await _identity(client, db, identity_id)


async def get_profile_image(client: ArcadeDBClient, branch_id: str, identity_id: str) -> FaceImage:
    """The Identity's avatar: its seed photo, or a representative frame."""
    ident = await _identity(client, await _face_db(client, branch_id), identity_id)
    if ident.avatar is None:
        raise IdentityNotFoundError(f"identity {identity_id} has no picture")
    return ident.avatar


# ── traces and groups ────────────────────────────────────────────────────

async def _traces(client: ArcadeDBClient, db: str, where: str, params: dict) -> tuple[list[TraceOut], dict[str, float]]:
    """Traces with their group, representative frame and the video's Source.status.
    Also returns each representative's quality, for picking a group's picture."""
    rows = await client.query(
        db, "SELECT trace_id, version_id, start_sec, end_sec, avg_confidence, "
            "out('TRACE_IN_GROUP').group_id AS g, out('HAS_KEY_FACE').face_id AS f, "
            f"out('TRACE_IN_SOURCE').status AS ss FROM FaceTrace WHERE {where} ORDER BY start_sec", params=params)  # nosec B608 -- every _traces() caller passes a hardcoded where literal
    faces = await _faces(client, db, [(r.get("f") or [None])[0] for r in rows])
    traces, quality = [], {}
    for r in rows:
        face = faces.get((r.get("f") or [None])[0])
        if face:
            quality[r["trace_id"]] = face.get("quality") or 0.0
        traces.append(TraceOut(
            trace_id=r["trace_id"], version_id=r["version_id"], group_id=(r.get("g") or [None])[0],
            start_sec=r["start_sec"], end_sec=r["end_sec"], avg_confidence=r.get("avg_confidence"),
            representative=_face_image(face), source_status=(r.get("ss") or [None])[0]))
    return traces, quality


def _group_out(group: dict, traces: list[TraceOut], quality: dict[str, float]) -> GroupOut:
    members = [t for t in traces if t.group_id == group["group_id"]]
    best = max(members, key=lambda t: quality.get(t.trace_id, -1.0), default=None)
    return GroupOut(group_id=group["group_id"], version_id=group["version_id"], frame_count=group.get("frame_count"),
                    traces=members, representative=best.representative if best else None)


async def pending_persons(client: ArcadeDBClient, branch_id: str, version_id: str) -> PendingPersons:
    """1.4 §4.5: ① this video's groups without an Identity ② pending Identities for this
    video -- holding one of its groups, or created from it via pending-person ③ active
    (confirmed) Identities already holding one of its groups, so a caller showing every
    group of a video (not just the unresolved ones) doesn't need a second call. One
    video has a few dozen groups at most, so everything is read at once and sorted
    out here."""
    db = await _face_db(client, branch_id)
    traces, quality = await _traces(client, db, "version_id = :v", {"v": version_id})
    groups = await client.query(
        db, "SELECT group_id, version_id, frame_count, out('GROUP_OF_IDENTITY').identity_id AS ids "
            "FROM FaceGroup WHERE version_id = :v", params={"v": version_id})
    unassigned = [_group_out(g, traces, quality) for g in groups if not g.get("ids")]

    by_identity: dict[str, list[GroupOut]] = {}
    for g in groups:
        for identity_id in g.get("ids") or []:
            by_identity.setdefault(identity_id, []).append(_group_out(g, traces, quality))
    ids = sorted(by_identity)
    where = "identity_status = 'pending' AND (origin_version_id = :v" + (" OR identity_id IN :ids)" if ids else ")")
    pending = await _identities(client, db, where, {"v": version_id, "ids": ids}, "ORDER BY created_at")
    active = (await _identities(client, db, "identity_status = 'active' AND identity_id IN :ids",
                                {"ids": ids}, "ORDER BY name")) if ids else []
    first_seen = lambda g: min((t.start_sec for t in g.traces), default=0.0)
    return PendingPersons(
        version_id=version_id,
        unassigned_groups=sorted(unassigned, key=first_seen),
        pending_identities=[PendingIdentityOut(**p.model_dump(), groups=by_identity.get(p.identity_id, []))
                            for p in pending],
        active_identities=[PendingIdentityOut(**p.model_dump(), groups=by_identity.get(p.identity_id, []))
                           for p in active])


async def identity_traces(client: ArcadeDBClient, branch_id: str, identity_id: str) -> IdentityTraces:
    """Every trace under this Identity, across videos (1.4 §4.5 GET /{id}/traces)."""
    db = await _face_db(client, branch_id)
    await _identity(client, db, identity_id)
    group_ids = [r["group_id"] for r in await client.query(
        db, "SELECT group_id FROM FaceGroup WHERE out('GROUP_OF_IDENTITY').identity_id CONTAINS :i",
        params={"i": identity_id})]
    traces: list[TraceOut] = []
    for gid in group_ids:
        traces += (await _traces(client, db, "out('TRACE_IN_GROUP').group_id CONTAINS :g", {"g": gid}))[0]
    traces.sort(key=lambda t: (t.version_id, t.start_sec))
    return IdentityTraces(identity_id=identity_id, traces=traces)


async def _group_identity(client: ArcadeDBClient, db: str, group_id: str | None) -> str | None:
    if not group_id:
        return None
    rows = await client.query(db, "SELECT out('GROUP_OF_IDENTITY').identity_id AS ids FROM FaceGroup WHERE group_id = :g",
                              params={"g": group_id})
    return ((rows[0].get("ids") or [None])[0]) if rows else None


async def _refresh_group(client: ArcadeDBClient, db: str, group_id: str, result: BindResult) -> None:
    """After a group gained or lost a trace: recompute its embedding (frame-weighted
    mean of its traces, as Module 11 computes it) or, if it's now empty, delete it --
    and with it a system-created pending Identity left holding nothing."""
    members = await client.query(
        db, "SELECT embedding, frame_count FROM FaceTrace WHERE out('TRACE_IN_GROUP').group_id CONTAINS :g",
        params={"g": group_id})
    if members:
        total = [0.0] * len(members[0]["embedding"])
        for m in members:
            w = m.get("frame_count") or 1
            total = [a + w * b for a, b in zip(total, m["embedding"])]
        norm = math.sqrt(sum(x * x for x in total)) or 1.0
        await client.command(
            db, "UPDATE FaceGroup SET embedding = :e, frame_count = :n WHERE group_id = :g",
            params={"g": group_id, "e": [x / norm for x in total], "n": sum(m.get("frame_count") or 0 for m in members)})
        return
    owner = await _group_identity(client, db, group_id)
    await client.command(db, "DELETE VERTEX FROM FaceGroup WHERE group_id = :g", params={"g": group_id})
    result.removed_groups.append(group_id)
    if owner and _count(await client.command(
            db, "DELETE VERTEX FROM Identity WHERE identity_id = :i AND source = 'auto' "
                "AND identity_status = 'pending' AND in('GROUP_OF_IDENTITY').size() = 0", params={"i": owner})):
        result.removed_identities.append(owner)


async def _trace_group(client: ArcadeDBClient, db: str, version_id: str, trace_id: str) -> str | None:
    rows = await client.query(
        db, "SELECT out('TRACE_IN_GROUP').group_id AS g FROM FaceTrace WHERE trace_id = :t AND version_id = :v",
        params={"t": trace_id, "v": version_id})
    if not rows:
        raise IdentityNotFoundError(f"trace {trace_id} not found in video {version_id}")
    return (rows[0].get("g") or [None])[0]


def _check_live(ident: IdentityOut) -> None:
    """Merged and deleted Identities only change back through undo."""
    if ident.identity_status == "merged":
        raise IdentityConflictError(f"identity {ident.identity_id} was merged into another identity")
    if ident.identity_status == "deleted":
        raise IdentityConflictError(f"identity {ident.identity_id} was deleted")


async def _live_identity(client: ArcadeDBClient, db: str, identity_id: str) -> IdentityOut:
    ident = await _identity(client, db, identity_id)
    _check_live(ident)
    return ident


async def _bind(client: ArcadeDBClient, db: str, identity_id: str, version_id: str, trace_id: str) -> BindResult:
    previous = await _trace_group(client, db, version_id, trace_id)
    result = BindResult(identity_id=identity_id, version_id=version_id, trace_id=trace_id,
                        group_id=previous, previous_group_id=previous)
    previous_owner = await _group_identity(client, db, previous)
    if previous_owner == identity_id:
        return result                                  # already this identity's

    target = await client.query(
        db, "SELECT group_id FROM FaceGroup WHERE version_id = :v AND out('GROUP_OF_IDENTITY').identity_id CONTAINS :i "
            "ORDER BY created_at LIMIT 1", params={"v": version_id, "i": identity_id})
    if target:
        group_id = target[0]["group_id"]
    else:
        group_id = uuid.uuid4().hex
        await client.command(
            db, "INSERT INTO FaceGroup SET group_id = :g, version_id = :v, frame_count = 0, created_at = :now",
            params={"g": group_id, "v": version_id, "now": _now()})
        await _link(client, db, "GROUP_OF_IDENTITY", "FaceGroup", "group_id", group_id, "Identity", "identity_id",
                    identity_id, props={"score": None, "auto": False, "created_at": _now()})

    await client.command(db, "DELETE FROM TRACE_IN_GROUP WHERE outV().trace_id = :t", params={"t": trace_id})
    await _link(client, db, "TRACE_IN_GROUP", "FaceTrace", "trace_id", trace_id, "FaceGroup", "group_id", group_id)
    result.group_id = group_id
    await _refresh_group(client, db, group_id, result)
    if previous:
        await _refresh_group(client, db, previous, result)
    for owner in (identity_id, previous_owner):
        if owner and owner not in result.removed_identities:
            await _fix_avatar(client, db, owner)
    await _touch(client, db, [identity_id, previous_owner])
    return result


async def bind_trace(client: ArcadeDBClient, branch_id: str, identity_id: str, version_id: str,
                     trace_id: str) -> BindResult:
    """1.4 §4.5 bind/trace: move one trace into this Identity's group for its video
    (created if there's none). The group it left is recomputed, or deleted if empty."""
    db = await _face_db(client, branch_id)
    await _live_identity(client, db, identity_id)
    return await _bind(client, db, identity_id, version_id, trace_id)


async def bind_group(client: ArcadeDBClient, branch_id: str, identity_id: str, version_id: str,
                     group_id: str) -> BindGroupResult:
    """Put a whole group -- every trace Module 11 clustered together in one video -- under
    this Identity in one step, instead of bind/trace once per trace. The group is kept as
    it is (its traces, embedding and ID); only its GROUP_OF_IDENTITY edge points here now,
    as a cross-video match would link it (1.4 §4.2 "promoting a cluster"). A group another
    person held moves; a system-created pending person left with no groups is deleted, as
    bind/trace does. Not undoable, like bind/trace: bind the group back to undo it."""
    db = await _face_db(client, branch_id)
    await _live_identity(client, db, identity_id)
    await _check_group(client, db, version_id, group_id)
    return await _bind_group(client, db, identity_id, version_id, group_id)


async def _check_group(client: ArcadeDBClient, db: str, version_id: str, group_id: str) -> None:
    if not await client.query(db, "SELECT group_id FROM FaceGroup WHERE group_id = :g AND version_id = :v",
                              params={"g": group_id, "v": version_id}):
        raise IdentityNotFoundError(f"group {group_id} not found in version {version_id}")


async def _bind_group(client: ArcadeDBClient, db: str, identity_id: str, version_id: str,
                      group_id: str) -> BindGroupResult:
    previous = await _group_identity(client, db, group_id)
    result = BindGroupResult(identity_id=identity_id, version_id=version_id, group_id=group_id,
                             previous_identity_id=previous)
    if previous == identity_id:
        return result                                  # already this identity's

    await client.command(db, "DELETE FROM GROUP_OF_IDENTITY WHERE outV().group_id = :g", params={"g": group_id})
    await _link(client, db, "GROUP_OF_IDENTITY", "FaceGroup", "group_id", group_id, "Identity", "identity_id",
                identity_id, props={"score": None, "auto": False, "created_at": _now()})
    if previous and _count(await client.command(
            db, "DELETE VERTEX FROM Identity WHERE identity_id = :i AND source = 'auto' "
                "AND identity_status = 'pending' AND in('GROUP_OF_IDENTITY').size() = 0", params={"i": previous})):
        result.removed_identities.append(previous)
    for owner in (identity_id, previous):
        if owner and owner not in result.removed_identities:
            await _fix_avatar(client, db, owner)
    await _touch(client, db, [identity_id, previous])
    return result


async def unbind_trace(client: ArcadeDBClient, branch_id: str, identity_id: str, version_id: str,
                       trace_id: str) -> BindResult:
    """1.4 §4.5 unbind: take the trace out of its group, into a fresh group of its own --
    same as a trace that was never bound, so it's still reachable (pending-persons) afterward."""
    db = await _face_db(client, branch_id)
    await _live_identity(client, db, identity_id)
    previous = await _trace_group(client, db, version_id, trace_id)
    if await _group_identity(client, db, previous) != identity_id:
        raise IdentityConflictError(f"trace {trace_id} is not bound to identity {identity_id}")

    group_id = uuid.uuid4().hex
    await client.command(
        db, "INSERT INTO FaceGroup SET group_id = :g, version_id = :v, frame_count = 0, created_at = :now",
        params={"g": group_id, "v": version_id, "now": _now()})
    await client.command(db, "DELETE FROM TRACE_IN_GROUP WHERE outV().trace_id = :t", params={"t": trace_id})
    await _link(client, db, "TRACE_IN_GROUP", "FaceTrace", "trace_id", trace_id, "FaceGroup", "group_id", group_id)

    result = BindResult(identity_id=identity_id, version_id=version_id, trace_id=trace_id,
                        group_id=group_id, previous_group_id=previous)
    await _refresh_group(client, db, group_id, result)
    await _refresh_group(client, db, previous, result)
    if identity_id not in result.removed_identities:
        await _fix_avatar(client, db, identity_id)
        await _touch(client, db, [identity_id])
    return result


async def create_pending_person(client: ArcadeDBClient, branch_id: str, version_id: str,
                                req: PendingPersonIn) -> PendingPersonOut:
    """1.4 §4.5 pending-person: a pending Identity the user starts from a video
    (source='manual'), optionally with one of its traces, a list of them, or a whole
    group bound right away. Every trace or the group is checked before anything is
    created, so a bad ID creates nothing."""
    db = await _face_db(client, branch_id)
    trace_ids = list(dict.fromkeys(req.trace_ids or ([req.trace_id] if req.trace_id else [])))
    for trace_id in trace_ids:
        await _trace_group(client, db, version_id, trace_id)          # 404 before creating anything
    if req.group_id:
        await _check_group(client, db, version_id, req.group_id)
    identity_id = uuid.uuid4().hex
    await _upsert_identity(client, db, identity_id, name=req.name, aliases={}, identity_type="person",
                           source="manual", status="pending", origin_version_id=version_id)
    binds = [await _bind(client, db, identity_id, version_id, t) for t in trace_ids]
    group = await _bind_group(client, db, identity_id, version_id, req.group_id) if req.group_id else None
    return PendingPersonOut(identity=await _identity(client, db, identity_id),
                            bind=binds[0] if req.trace_id else None,
                            binds=binds if req.trace_ids else [], bind_group=group)


# ───────────────────────────────────────────────────────────── People Search (1.4 §4.6)

def _name_match(query: str, names: list[str]) -> tuple[str, bool] | None:
    """The best of these names/aliases for the query: (name, exact). Case-insensitive;
    a partial match is a substring of the name."""
    q = query.strip().casefold()
    partial = None
    for n in names:
        if not n:
            continue
        if n.casefold() == q:
            return n, True
        if partial is None and q in n.casefold():
            partial = n
    return (partial, False) if partial else None


_GONE = ("merged", "deleted")


async def _people_matches(client: ArcadeDBClient, db: str, query: str) -> list[PeopleMatch]:
    """MTR-24: the Identities whose name or alias matches the query, exact matches first.

    - Names go through the Identity[name] full-text index (a substring LIKE for a lone
      CJK character, which the index can never match -- searcher._bm25_condition).
      The index's bigram match is over-inclusive, so each hit is checked again in
      Python, as searcher._resolve_entity_id does.
    - Aliases have no index: every live Identity's aliases_json is checked in Python."""
    params: dict = {}
    cond = searcher._bm25_condition("Identity", "name", query, "q", params)
    sql = f"SELECT identity_id, name, identity_status FROM Identity WHERE {cond}"  # nosec B608 -- cond holds constants; the query is bound
    by_name = await client.query(db, sql, params=params)
    with_aliases = await client.query(db, "SELECT identity_id, name, aliases_json, identity_status FROM Identity "
                                          "WHERE aliases_json IS NOT NULL AND aliases_json <> '{}'")
    best: dict[str, PeopleMatch] = {}
    for r, names in ([(r, [r.get("name") or ""]) for r in by_name]
                     + [(r, list(json.loads(r["aliases_json"]).values())) for r in with_aliases]):
        if r.get("identity_status") in _GONE:
            continue
        hit = _name_match(query, names)
        known = best.get(r["identity_id"])
        if hit and (known is None or (hit[1] and not known.exact)):
            best[r["identity_id"]] = PeopleMatch(identity_id=r["identity_id"], name=r.get("name") or "",
                                                 matched=hit[0], exact=hit[1])
    return sorted(best.values(), key=lambda m: (not m.exact, m.name))


async def search_people(client: ArcadeDBClient, branch_id: str, req: PeopleSearchRequest) -> PeopleSearchResponse:
    """Identities whose name or alias matches (_people_matches), and every segment they
    appear in, from videos whose Source is active -- the same rule as content search, so
    an unregistered/archived video, or one not indexed yet, never shows up."""
    db = await _face_db(client, branch_id)
    matches = await _people_matches(client, db, req.query)

    segments: list[PeopleSegment] = []
    sources: dict[str, dict | None] = {}
    for m in matches:
        for t in (await identity_traces(client, branch_id, m.identity_id)).traces:
            if t.version_id not in sources:
                src = await client.query(db, "SELECT filename, asset_path, status FROM Source WHERE version_id = :v",
                                         params={"v": t.version_id})
                sources[t.version_id] = src[0] if src else None
            src = sources[t.version_id]
            if not src or src.get("status") not in (None, "active"):
                continue
            segments.append(PeopleSegment(
                identity_id=m.identity_id, identity_name=m.name, version_id=t.version_id, trace_id=t.trace_id,
                first_sec=t.start_sec, last_sec=t.end_sec, confidence=t.avg_confidence, keyframe=t.representative,
                keyframe_path=t.representative.image_path if t.representative else None,
                asset_path=src.get("asset_path"), filename=src.get("filename")))
    return PeopleSearchResponse(query=req.query, identities=matches, total=len(segments),
                                results=segments[:req.limit])


async def people_in_videos(client: ArcadeDBClient, branch_id: str, identity_ids: list[str],
                           version_ids: list[str]) -> PeopleInVideosResponse:
    """Every trace for any of these identities, restricted to these specific videos --
    the scoped counterpart to search_people(). Bounded by (identities) x (videos), not
    by an identity's total appearance count system-wide: a video_search caller already
    knows which videos it cares about (its own RRF-ranked candidates) and which identity
    a query's name matched (search_people()'s own `identities`), so unlike
    search_people()'s flat, limit-truncated segment list, this can never crowd a popular
    identity's own video out of its own correct result just because they're also in many
    other videos."""
    if not identity_ids or not version_ids:
        return PeopleInVideosResponse(results=[])
    db = await _face_db(client, branch_id)
    groups = await client.query(
        db, "SELECT group_id, out('GROUP_OF_IDENTITY').identity_id AS ids FROM FaceGroup WHERE version_id IN :versions",
        params={"versions": version_ids})
    group_identity: dict[str, str] = {}
    for g in groups:
        for iid in g.get("ids") or []:
            if iid in identity_ids:
                group_identity[g["group_id"]] = iid
    if not group_identity:
        return PeopleInVideosResponse(results=[])
    names_coro = client.query(
        db, "SELECT identity_id, name FROM Identity WHERE identity_id IN :ids",
        params={"ids": list(set(group_identity.values()))})
    # One _traces() call per matched group, same query _each already makes
    # sequentially elsewhere in this file (identity_traces()) -- but here
    # group_identity's groups are all independent of each other, so there's
    # no reason to wait for one before starting the next. Run them (and the
    # names lookup) concurrently instead of in a for-loop.
    trace_coros = [_traces(client, db, "out('TRACE_IN_GROUP').group_id CONTAINS :g", {"g": group_id})
                  for group_id in group_identity]
    names_rows, *trace_results = await asyncio.gather(names_coro, *trace_coros)
    names = {r["identity_id"]: r["name"] for r in names_rows}
    segments: list[PeopleSegment] = []
    for (group_id, identity_id), (traces, _) in zip(group_identity.items(), trace_results):
        for t in traces:
            segments.append(PeopleSegment(
                identity_id=identity_id, identity_name=names.get(identity_id, ""), version_id=t.version_id,
                trace_id=t.trace_id, first_sec=t.start_sec, last_sec=t.end_sec, confidence=t.avg_confidence,
                keyframe=t.representative, keyframe_path=t.representative.image_path if t.representative else None))
    return PeopleInVideosResponse(results=segments)
