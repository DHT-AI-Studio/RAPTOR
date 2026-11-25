"""People / identity layer (dev plan v1.4 §4.2, §4.5): face trace/group writes,
cross-video matching, Source linking and delete consistency.

Integration tests -- they hit a real ArcadeDB (see conftest.py) and skip when it
isn't reachable. The first two tests are pure and always run.

    PD_ARCADEDB_URL=http://localhost:12480 PD_ARCADEDB_PASSWORD=... pytest tests/test_face_identity.py
"""
from __future__ import annotations

import math

import pytest
from app.core.config import settings
from app.models.face import FaceTracesRequest
from app.models.index import ChunkIndexRequest, SourceSummaryIndexRequest
from app.services import graph_query, schema_init
from app.services.arcadedb_client import db_name_for
from app.services.identity_store import delete_face_data, write_face_traces
from app.services.indexer import (
    delete_by_version,
    index_chunk,
    index_source_summary,
    version_exists,
)

from tests.integration.conftest import fake_vector

pytestmark = pytest.mark.asyncio

DIM = settings.face_vector_dim
FACE_EDGES = ("TRACE_IN_SOURCE", "HAS_KEY_FACE", "TRACE_IN_GROUP", "GROUP_OF_IDENTITY", "HAS_PROFILE_FACE")


def unit(cos_to_x: float, axis: int = 1) -> list[float]:
    """A unit vector whose cosine similarity with e0 is exactly cos_to_x."""
    v = [0.0] * DIM
    v[0] = cos_to_x
    v[axis] = math.sqrt(max(0.0, 1.0 - cos_to_x ** 2))
    return v


def trace(trace_id: str, emb: list[float], start: float = 0.0, min_side: float = 120.0, quality: float = 0.5) -> dict:
    frames = [{"frame_no": i, "timestamp_sec": start + i / 8, "bbox": [10, 10, 10 + min_side, 10 + min_side],
               "confidence": 0.9} for i in range(8)]
    return {"trace_id": trace_id, "start_sec": start, "end_sec": start + 7 / 8, "embedding": emb, "frames": frames,
            "representative": {"frame_no": 3, "timestamp_sec": start + 3 / 8, "bbox": [10, 10, 10 + min_side, 10 + min_side],
                               "quality": quality, "min_side": min_side,
                               "image_path": f"u1/videos/x/face_{trace_id}.jpg", "image_version_id": "c1"}}


def video(version_id: str, groups: dict[str, list[dict]]) -> FaceTracesRequest:
    """groups: group_id -> its traces; the group embedding is the first trace's."""
    traces = [t for ts in groups.values() for t in ts]
    return FaceTracesRequest(
        version_id=version_id, asset_path=f"u1/videos/{version_id}", filename=f"{version_id}.mp4", traces=traces,
        groups=[{"group_id": gid, "embedding": ts[0]["embedding"], "trace_ids": [t["trace_id"] for t in ts]}
                for gid, ts in groups.items()])


async def count(client, db: str, sql: str, params: dict | None = None) -> int:
    rows = await client.query(db, f"SELECT count(*) AS c FROM {sql}", params=params)
    return rows[0]["c"] if rows else 0


# ── pure ─────────────────────────────────────────────────────────────────

async def test_face_edges_stay_out_of_content_subgraph_queries():
    """1.4 §4.2 constraint 2: face data must never surface in GraphRAG/TKG subgraphs."""
    subgraph_edges = set(graph_query._SUBGRAPH_EDGE_TYPES.split("|"))
    assert not subgraph_edges & set(FACE_EDGES)


async def test_face_vertices_have_no_status_property():
    """1.4 §4.2 constraint 1: the subgraph filter drops paths through any node whose
    `status` isn't null/active, so the face layer never declares a `status` property."""
    ddl = schema_init._face_ddl(DIM, "COSINE")
    assert not [s for s in ddl if s.startswith("CREATE PROPERTY") and ".status " in s]
    assert any(".identity_status " in s for s in ddl)


# ── schema ───────────────────────────────────────────────────────────────

async def test_face_schema_is_idempotent(make_db, client):
    db = db_name_for(await make_db("ittest_face_schema"))
    await schema_init.initialize_face_schema(client, db)        # already applied by make_db; again must not raise
    for vtype in ("FaceTrace", "Face", "FaceGroup", "Identity", "IdentityHistory"):
        assert await count(client, db, vtype) == 0


async def test_database_created_before_the_face_layer(arcade, client):
    """Existing users' databases only have the content schema: deleting a video there
    must still work, and the first face write adds the face layer itself."""
    branch = "ittest_face_legacy"
    db = db_name_for(branch)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    await arcade.create_database(db)
    try:
        for stmt in schema_init._ddl(settings.vector_dim, settings.vector_similarity):
            await arcade.command(db, stmt, ignore_exists=True)
        assert (await delete_by_version(client, branch, "v1"))["face_traces"] == 0
        res = await write_face_traces(client, branch, video("v1", {"g1": [trace("t1", unit(1.0))]}))
        assert res.traces_written == 1
    finally:
        await arcade.drop_database(db)


# ── write ────────────────────────────────────────────────────────────────

async def test_write_stores_traces_and_an_unassigned_group(make_db, client):
    branch = await make_db("ittest_face_write")
    db = db_name_for(branch)
    res = await write_face_traces(client, branch, video("v1", {"g1": [trace("t1", unit(1.0)), trace("t2", unit(1.0), 5)]}))

    assert (res.traces_written, res.groups_written) == (2, 1)
    (m,) = res.matches
    assert m.identity_id is None and not m.created_identity and m.score is None   # nothing to match against yet
    assert await count(client, db, "FaceTrace") == 2
    assert await count(client, db, "Face WHERE kind = 'representative'") == 2
    assert await count(client, db, "HAS_KEY_FACE") == 2
    assert await count(client, db, "TRACE_IN_GROUP") == 2
    assert await count(client, db, "FaceGroup") == 1
    assert await count(client, db, "Identity") == 0            # seen in one video only: stays on the pending list


async def test_rewriting_a_video_replaces_it(make_db, client):
    """Re-analysis after a failure: same payload twice gives the same state, and a
    smaller result leaves no stale traces behind."""
    branch = await make_db("ittest_face_rewrite")
    db = db_name_for(branch)
    full = video("v1", {"g1": [trace("t1", unit(1.0)), trace("t2", unit(1.0), 5)], "g2": [trace("t3", unit(0.0))]})
    await write_face_traces(client, branch, full)
    res = await write_face_traces(client, branch, full)
    assert res.replaced["face_traces"] == 3
    assert await count(client, db, "FaceTrace") == 3
    assert await count(client, db, "FaceGroup") == 2

    await write_face_traces(client, branch, video("v1", {"g1": [trace("t1", unit(1.0))]}))
    assert await count(client, db, "FaceTrace") == 1
    assert await count(client, db, "FaceGroup") == 1


async def test_rewriting_a_video_does_not_pile_up_auto_identities(make_db, client):
    branch = await make_db("ittest_face_rewrite_ids")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    v2 = video("v2", {"b": [trace("t2", unit(0.9))]})
    await write_face_traces(client, branch, v2)
    await write_face_traces(client, branch, v2)
    assert await count(client, db, "Identity") == 1
    assert await count(client, db, "GROUP_OF_IDENTITY") == 2


# ── cross-video matching ─────────────────────────────────────────────────

async def test_same_person_in_another_video_becomes_one_pending_identity(make_db, client):
    branch = await make_db("ittest_face_match")
    db = db_name_for(branch)
    first = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))).matches[0]
    second = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(0.9))]}))).matches[0]
    third = (await write_face_traces(client, branch, video("v3", {"c": [trace("t3", unit(0.95))]}))).matches[0]

    assert first.identity_id is None
    # v2 matches v1's unassigned group: one new pending person now holds both
    assert second.created_identity and second.identity_id
    assert second.score == pytest.approx(0.9 * settings.face_group_match_weight, abs=1e-3)
    ident = (await client.query(db, "SELECT * FROM Identity"))[0]
    assert (ident["identity_id"], ident["source"], ident["identity_status"], ident["name"]) == \
        (second.identity_id, "auto", "pending", "")
    owners = await client.query(db, "SELECT group_id, out('GROUP_OF_IDENTITY').identity_id AS ids FROM FaceGroup")
    assert {r["group_id"]: r["ids"] for r in owners} == {"a": [second.identity_id], "b": [second.identity_id],
                                                        "c": [second.identity_id]}
    assert all(e["auto"] is True for e in await client.query(db, "SELECT auto FROM GROUP_OF_IDENTITY"))
    # v3 joins the existing identity instead of making another
    assert not third.created_identity and third.identity_id == second.identity_id
    assert await count(client, db, "Identity") == 1


async def test_new_pending_identity_takes_the_better_picture(make_db, client):
    branch = await make_db("ittest_face_match_avatar")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0), quality=0.9)]}))
    m = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(0.9), quality=0.3)]}))).matches[0]
    avatar = await client.query(db, "SELECT out('HAS_PROFILE_FACE').face_id AS f FROM Identity WHERE identity_id = :i",
                                params={"i": m.identity_id})
    assert avatar[0]["f"] == ["rep_t1"]


async def test_different_person_stays_unassigned(make_db, client):
    branch = await make_db("ittest_face_nomatch")
    first = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))).matches[0]
    # a cosine right at the threshold falls short once the group weight (0.95) is applied: not the same person
    cos = settings.face_match_threshold
    other = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(cos))]}))).matches[0]
    assert first.identity_id is None and other.identity_id is None and not other.created_identity
    assert other.score == pytest.approx(cos * settings.face_group_match_weight, abs=1e-3)
    assert await count(client, db_name_for(branch), "Identity") == 0


async def test_groups_within_one_video_never_match_each_other(make_db, client):
    branch = await make_db("ittest_face_samevideo")
    res = await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))], "b": [trace("t2", unit(1.0), 5)]}))
    assert all(m.identity_id is None for m in res.matches)     # intra-video clustering is Module 11's call
    assert await count(client, db_name_for(branch), "Identity") == 0


async def test_small_faces_are_not_auto_linked(make_db, client):
    branch = await make_db("ittest_face_small")
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    small = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(1.0), min_side=30)]}))).matches[0]
    assert small.skipped_small_face and small.identity_id is None and not small.created_identity
    assert await count(client, db_name_for(branch), "Identity") == 0


async def test_seed_photo_match_uses_its_identity(make_db, client):
    branch = await make_db("ittest_face_seed")
    db = db_name_for(branch)
    await client.command(db, "INSERT INTO Identity SET identity_id = 'mom', name = 'Mom', aliases_json = '{}', "
                             "identity_type = 'person', source = 'manual', identity_status = 'active'")
    await client.command(db, "INSERT INTO Face SET face_id = 'seed_mom', kind = 'seed', identity_id = 'mom', "
                             "embedding = :e", params={"e": unit(1.0)})
    m = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(0.9))]}))).matches[0]
    assert not m.created_identity and m.identity_id == "mom"
    assert m.score == pytest.approx(0.9 * settings.face_seed_match_weight, abs=1e-3)


async def test_merged_identities_are_never_a_target(make_db, client):
    branch = await make_db("ittest_face_merged")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    first = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(0.95))]}))).matches[0]
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i",
                         params={"i": first.identity_id})
    m = (await write_face_traces(client, branch, video("v3", {"c": [trace("t3", unit(0.95))]}))).matches[0]
    assert m.identity_id != first.identity_id


# ── Source linking ───────────────────────────────────────────────────────

async def test_face_write_never_creates_a_source(make_db, client):
    """A Source is what version_exists() reads as "this video was indexed"; a face-only
    shell would make a video whose content pipeline failed look analysed."""
    branch = await make_db("ittest_face_nosource")
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    assert await count(client, db_name_for(branch), "Source") == 0
    assert not await version_exists(client, branch, "v1")


async def test_traces_written_first_link_when_content_indexing_creates_the_source(make_db, client):
    branch = await make_db("ittest_face_linklater")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0)), trace("t2", unit(1.0), 5)]}))
    assert await count(client, db, "TRACE_IN_SOURCE") == 0

    await index_chunk(client, branch, ChunkIndexRequest(
        chunk_id="v1_m0", type="videos", embedding_type="text", text="hello", version_id="v1",
        embedding=fake_vector("hello")))
    assert await count(client, db, "TRACE_IN_SOURCE") == 2


async def test_traces_written_after_the_source_link_immediately(make_db, client):
    branch = await make_db("ittest_face_linknow")
    db = db_name_for(branch)
    await index_source_summary(client, branch, SourceSummaryIndexRequest(
        version_id="v1", summary="a talk", embedding=fake_vector("a talk"), media_type="videos"))
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    assert await count(client, db, "TRACE_IN_SOURCE") == 1


# ── delete consistency ───────────────────────────────────────────────────

async def test_deleting_a_video_cleans_its_face_data_only(make_db, client):
    branch = await make_db("ittest_face_delete")
    db = db_name_for(branch)
    a = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))], "x": [trace("tx", unit(0.0), 5)]}))).matches
    b = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(0.9))]}))).matches[0]
    assert b.created_identity                                    # one person across both videos
    assert await count(client, db, "GROUP_OF_IDENTITY") == 2

    assert a[1].identity_id is None                              # "x" only appeared in v1

    res = await delete_by_version(client, branch, "v1")
    assert (res["face_traces"], res["faces"], res["face_groups"]) == (2, 2, 2)
    assert res["auto_identities_removed"] == 0                   # still holds v2's group
    assert await count(client, db, "FaceTrace") == 1             # v2 untouched
    assert await count(client, db, "Identity WHERE identity_id = :i", {"i": b.identity_id}) == 1

    res = await delete_by_version(client, branch, "v2")
    assert res["auto_identities_removed"] == 1                   # now empty, and nobody named it
    assert await count(client, db, "Identity") == 0


async def test_named_identities_survive_losing_all_their_groups(make_db, client):
    branch = await make_db("ittest_face_keepnamed")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    m = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(0.9))]}))).matches[0]
    await client.command(db, "UPDATE Identity SET name = 'Mom', identity_status = 'active' WHERE identity_id = :i",
                         params={"i": m.identity_id})
    await delete_by_version(client, branch, "v1")
    await delete_by_version(client, branch, "v2")
    assert await count(client, db, "Identity WHERE identity_id = :i", {"i": m.identity_id}) == 1
    assert await count(client, db, "HAS_PROFILE_FACE") == 0      # its only picture is gone; frontend shows a default


async def test_avatar_moves_to_a_remaining_frame_when_its_video_is_deleted(make_db, client):
    branch = await make_db("ittest_face_avatar")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0), quality=0.9)]}))
    m = (await write_face_traces(client, branch, video("v2", {"b": [trace("t2", unit(0.9))]}))).matches[0]
    avatar = await client.query(db, "SELECT out('HAS_PROFILE_FACE').face_id AS f FROM Identity WHERE identity_id = :i",
                                params={"i": m.identity_id})
    assert avatar[0]["f"] == ["rep_t1"]

    res = await delete_by_version(client, branch, "v1")
    assert res["avatars_repointed"] == 1
    avatar = await client.query(db, "SELECT out('HAS_PROFILE_FACE').face_id AS f FROM Identity WHERE identity_id = :i",
                                params={"i": m.identity_id})
    assert avatar[0]["f"] == ["rep_t2"]


async def test_seed_photos_are_never_deleted_with_a_video(make_db, client):
    branch = await make_db("ittest_face_seedkeep")
    db = db_name_for(branch)
    await client.command(db, "INSERT INTO Face SET face_id = 'seed_x', kind = 'seed', identity_id = 'x', "
                             "embedding = :e", params={"e": unit(1.0)})
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(0.9))]}))
    await delete_face_data(client, db, "v1")
    assert await count(client, db, "Face WHERE kind = 'seed'") == 1


# ── isolation (MTR-10 / VIE01-397) ───────────────────────────────────────

_FACE_TYPES = ("FaceTrace", "Face", "FaceGroup", "Identity", "IdentityHistory")


def _face_leaks(response, face_markers: list[str]) -> list[str]:
    """Every face id / face image path / face vertex type found anywhere in a search response."""
    import json
    dump = json.dumps(response.model_dump(), default=str, ensure_ascii=False)
    leaks = [m for m in face_markers if m in dump]
    for node in [*getattr(response, "nodes", []), *getattr(response, "subgraph_nodes", []),
                 *getattr(response, "entities", [])]:
        labels = node.get("labels") or [node.get("@type") or node.get("type")]
        leaks += [f"node labelled {lbl}" for lbl in labels if lbl in _FACE_TYPES]
    return leaks


async def test_content_search_never_returns_face_data(make_db, client, mock_embed):
    """MTR-10: face data never appears in content GraphRAG / TKG / search results --
    "film noir" -> zero face nodes, a TKG query -> zero face intervals -- while People
    Search still finds the person. Content and faces share one video (and its Source),
    and the person is named after the query, so any leak would match."""
    from app.models.graph_index import (
        EntityIndexRequest,
        RelationshipIndexRequest,
        TemporalFactIndexRequest,
    )
    from app.models.graph_search import GraphRAGRequest, GraphSearchRequest, TKGRequest
    from app.models.identity import IdentityUpload, PeopleSearchRequest
    from app.models.search import SearchRequest
    from app.services import graph_indexer, identity_store, searcher

    from tests.integration.test_index import _chunk

    branch = await make_db("ittest_face_isolation")
    db = db_name_for(branch)
    vid = "v-noir"
    text = "A film noir scene: Detective Marlowe waits in a rain-soaked alley at midnight."

    # content of the video
    await index_chunk(client, branch, _chunk("c-noir", text, version_id=vid, asset_path=f"u1/videos/{vid}",
                                             filename=f"{vid}.mp4"))
    for eid, name, etype in (("film-noir", "Film Noir", "GENRE"), ("marlowe", "Detective Marlowe", "PERSON")):
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(
            entity_id=eid, name=name, type=etype, source_chunk_id="c-noir"))
    await graph_indexer.index_relationship(client, branch, RelationshipIndexRequest(
        from_entity_id="marlowe", to_entity_id="film-noir", relation="appears_in", source_version_id=vid))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="f-noir", entity="Film Noir", entity_id="film-noir", relation="popular_during",
        value="1940s Hollywood", time_start="1941", time_end="1958", chunk_id="c-noir", source_version_id=vid))

    # faces in the same video, linked to its Source, with a person named after the query
    await write_face_traces(client, branch, video(vid, {"g-noir": [trace("t-noir-1", unit(1.0)),
                                                                   trace("t-noir-2", unit(1.0), 4.0)]}))
    person, _ = await identity_store.upload_identity(client, branch, IdentityUpload(name="Film Noir Fan"))
    for tid in ("t-noir-1", "t-noir-2"):
        await identity_store.bind_trace(client, branch, person.identity_id, vid, tid)
    assert await count(client, db, "TRACE_IN_SOURCE") == 2

    # the face data is there: People Search finds the person in this video
    people = await identity_store.search_people(client, branch, PeopleSearchRequest(query="Film Noir"))
    assert {s.trace_id for s in people.results} == {"t-noir-1", "t-noir-2"}

    groups = await client.query(db, "SELECT group_id FROM FaceGroup")
    faces = await client.query(db, "SELECT face_id FROM Face")
    markers = ["t-noir-1", "t-noir-2", person.identity_id, "Film Noir Fan", "face_t-noir-",
               *(g["group_id"] for g in groups), *(f["face_id"] for f in faces)]

    graphrag = await searcher.graphrag_search(client, branch, GraphRAGRequest(
        query="film noir", strategy="literal", score_threshold=0))
    tkg = await searcher.tkg_search(client, branch, TKGRequest(query="film noir", score_threshold=0))
    graph = await searcher.graph_search(client, branch, GraphSearchRequest(entity_name="Film Noir"))
    searches = {"graphrag": graphrag, "tkg": tkg, "graph": graph}
    for name, fn in (("hybrid", searcher.hybrid_search), ("vector", searcher.vector_search),
                     ("bm25", searcher.bm25_search)):
        searches[name] = await fn(client, branch, SearchRequest(query="film noir", top_k=20))
    searches["bm25_person_name"] = await searcher.bm25_search(
        client, branch, SearchRequest(query="Film Noir Fan", top_k=20))

    # not vacuous: content search does find the content ...
    assert graphrag.nodes and tkg.temporal_facts and graph.entities
    assert [f["fact_id"] for f in tkg.temporal_facts] == ["f-noir"]
    assert searches["bm25"].results
    # ... and never any face data
    assert {name: leaks for name, resp in searches.items() if (leaks := _face_leaks(resp, markers))} == {}
