"""Identity API (dev plan v1.4 §4.5): management, creation, pending persons, bind/unbind.

Integration tests against a real ArcadeDB (see conftest.py); they skip when it isn't
reachable.

    PD_ARCADEDB_URL=http://localhost:12480 PD_ARCADEDB_PASSWORD=... pytest tests/test_identity_api.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx
import pytest
from app.models.identity import (
    IdentityPatch,
    IdentityUpload,
    PendingPersonIn,
    PeopleSearchRequest,
    ProfileImageIn,
)
from app.services import identity_store as store
from app.services.arcadedb_client import db_name_for
from app.services.identity_store import write_face_traces

from tests.integration.test_face_identity import count, trace, unit, video

pytestmark = pytest.mark.asyncio


async def two_videos(client, branch):
    """P is in v1 (group a: t1, t2) and v2 (group b: t3) -> one auto pending identity.
    X is only in v1 (group x: tx) -> unassigned."""
    await write_face_traces(client, branch, video("v1", {
        "a": [trace("t1", unit(1.0), quality=0.4), trace("t2", unit(1.0), 5, quality=0.8)],
        "x": [trace("tx", unit(0.0), 10)]}))
    m = (await write_face_traces(client, branch, video("v2", {"b": [trace("t3", unit(0.9))]}))).matches[0]
    assert m.created_identity
    return m.identity_id


async def named(client, branch, name="Mom"):
    ident, _ = await store.upload_identity(client, branch, IdentityUpload(name=name))
    return ident.identity_id


async def owner_of(client, db, trace_id):
    rows = await client.query(db, "SELECT out('TRACE_IN_GROUP').group_id AS g FROM FaceTrace WHERE trace_id = :t",
                              params={"t": trace_id})
    gid = (rows[0].get("g") or [None])[0]
    if not gid:
        return None, None
    ids = await client.query(db, "SELECT out('GROUP_OF_IDENTITY').identity_id AS i FROM FaceGroup WHERE group_id = :g",
                             params={"g": gid})
    return gid, (ids[0].get("i") or [None])[0]


# ── management ───────────────────────────────────────────────────────────

async def test_list_pages_and_leaves_out_merged(make_db, client):
    branch = await make_db("ittest_ident_list")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    ids = [await named(client, branch, n) for n in ("A", "B", "C")]
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": ids[0]})

    page = await store.list_identities(client, branch, limit=2, offset=0)
    assert page.total == 3                                       # B, C and the auto one; A is merged
    rest = await store.list_identities(client, branch, limit=2, offset=2)
    listed = [i.identity_id for i in page.items + rest.items]
    assert sorted(listed) == sorted([auto, ids[1], ids[2]])
    assert listed[0] == ids[2]                                   # most recently updated first

    merged = await store.list_identities(client, branch, limit=50, offset=0, status="merged")
    assert [i.identity_id for i in merged.items] == [ids[0]]

    p = (await store.list_identities(client, branch, limit=50, offset=0, status="pending")).items[0]
    assert (p.identity_id, p.group_count, p.video_count) == (auto, 2, 2)
    assert p.avatar.face_id == "rep_t2" and p.avatar.image_path.endswith("face_t2.jpg")   # best frame of both groups


async def test_get_unknown_identity(make_db, client):
    branch = await make_db("ittest_ident_get")
    with pytest.raises(store.IdentityNotFoundError):
        await store.get_identity(client, branch, "nope")


async def test_patch_renames_merges_aliases_and_confirms(make_db, client):
    branch = await make_db("ittest_ident_patch")
    auto = await two_videos(client, branch)

    p = await store.patch_identity(client, branch, auto, IdentityPatch(name="Peter", aliases={"zh-TW": "彼得"}))
    assert (p.name, p.aliases, p.identity_status) == ("Peter", {"zh-TW": "彼得"}, "pending")
    p = await store.patch_identity(client, branch, auto, IdentityPatch(aliases={"en": "Pete", "ja": "ピーター"}))
    assert p.aliases == {"zh-TW": "彼得", "en": "Pete", "ja": "ピーター"}   # untouched locales kept
    p = await store.patch_identity(client, branch, auto, IdentityPatch(aliases={"ja": None}, identity_status="active"))
    assert p.aliases == {"zh-TW": "彼得", "en": "Pete"}
    assert p.identity_status == "active" and p.name == "Peter"


async def test_patch_rejects_bad_input(make_db, client):
    with pytest.raises(ValueError):
        IdentityPatch(aliases={"fr": "Pierre"})
    with pytest.raises(ValueError):
        IdentityPatch(identity_status="merged")
    with pytest.raises(ValueError):
        IdentityPatch(name="")

    branch = await make_db("ittest_ident_patch_merged")
    db = db_name_for(branch)
    i = await named(client, branch)
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": i})
    with pytest.raises(store.IdentityConflictError):
        await store.patch_identity(client, branch, i, IdentityPatch(name="x"))


async def test_upload_creates_then_updates_the_same_identity(make_db, client):
    branch = await make_db("ittest_ident_upload")
    db = db_name_for(branch)
    ident, created = await store.upload_identity(client, branch, IdentityUpload(
        identity_id="mom", name="Mom", aliases={"zh-TW": "媽媽"}))
    assert created and (ident.identity_status, ident.source) == ("active", "manual")
    ident, created = await store.upload_identity(client, branch, IdentityUpload(
        identity_id="mom", name="Mother", aliases={"en": "Mum"}))
    assert not created
    assert (ident.name, ident.aliases) == ("Mother", {"zh-TW": "媽媽", "en": "Mum"})
    assert await count(client, db, "Identity") == 1


# ── delete ───────────────────────────────────────────────────────────────

async def test_delete_unassigns_groups_and_records_history(make_db, client):
    branch = await make_db("ittest_ident_delete")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    await store.patch_identity(client, branch, auto, IdentityPatch(name="Peter", identity_status="active"))
    await client.command(db, "INSERT INTO Face SET face_id = 'seed_p', kind = 'seed', identity_id = :i, "
                             "image_path = 'identity/b/p.jpg', embedding = :e", params={"i": auto, "e": unit(1.0)})

    res = await store.delete_identity(client, branch, auto)
    assert sorted(res.groups_unassigned) == ["a", "b"]
    assert await count(client, db, "GROUP_OF_IDENTITY") == 0     # only the group edges go (MTR-21 AC)
    assert await count(client, db, "FaceTrace") == 4             # traces, groups and frames stay
    assert await count(client, db, "FaceGroup") == 3
    assert await count(client, db, "Face WHERE kind = 'representative'") == 4
    assert await count(client, db, "Face WHERE kind = 'seed'") == 1          # so do seed photos
    p = await store.get_identity(client, branch, auto)                       # and the node, marked
    assert (p.name, p.identity_status, p.group_count) == ("Peter", "deleted", 0)
    assert p.avatar is not None                                              # its avatar edge stays

    h = (await client.query(db, "SELECT * FROM IdentityHistory WHERE history_id = :h", params={"h": res.history_id}))[0]
    assert (h["operation"], h["history_state"], h["undo_deadline"]) == ("delete", "applied", res.undo_deadline)
    before = json.loads(h["before_snapshot_json"])
    assert before["identities"][auto]["name"] == "Peter" and before["identities"][auto]["identity_status"] == "active"
    assert sorted(before["groups"]) == ["a", "b"] and before["seed_faces"] == {} and before["avatars"] == {}
    assert json.loads(h["after_snapshot_json"])["identities"][auto]["identity_status"] == "deleted"

    pending = await store.pending_persons(client, branch, "v1")
    assert sorted(g.group_id for g in pending.unassigned_groups) == ["a", "x"]


async def test_deleted_identity_is_not_matched_again_by_its_photo(make_db, client):
    branch = await make_db("ittest_ident_delete_seed")
    db = db_name_for(branch)
    i = await named(client, branch)
    await client.command(db, "INSERT INTO Face SET face_id = 'seed_m', kind = 'seed', identity_id = :i, "
                             "embedding = :e", params={"i": i, "e": unit(1.0)})
    await store.delete_identity(client, branch, i)
    m = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(0.95))]}))).matches[0]
    assert m.identity_id is None
    assert await count(client, db, "Face WHERE kind = 'seed'") == 1          # kept, but no longer matches


async def test_deleted_seed_photos_do_not_crowd_out_live_ones(make_db, client, monkeypatch):
    branch = await make_db("ittest_ident_delete_crowd")
    db = db_name_for(branch)
    gone, live = await named(client, branch, "Gone"), await named(client, branch, "Live")
    await add_seed(client, db, gone, "seed_g", unit(1.0))                    # nearer to the new face
    await add_seed(client, db, live, "seed_l", unit(0.7))
    await store.delete_identity(client, branch, gone)
    monkeypatch.setattr(store.settings, "face_match_candidates", 1)          # one hit per index
    m = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(0.95))]}))).matches[0]
    assert m.identity_id == live


async def test_deleted_identity_is_hidden_and_refuses_changes(make_db, client):
    branch = await make_db("ittest_ident_delete_hidden")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(1.0))]}))
    mom, other = await named(client, branch), await named(client, branch, "Other")
    await store.delete_identity(client, branch, mom)

    listed = await store.list_identities(client, branch, limit=50, offset=0)
    assert [i.identity_id for i in listed.items] == [other] and listed.total == 1
    deleted = await store.list_identities(client, branch, limit=50, offset=0, status="deleted")
    assert [i.identity_id for i in deleted.items] == [mom]
    assert (await store.search_people(client, branch, PeopleSearchRequest(query="Mom"))).identities == []

    for call in (lambda: store.patch_identity(client, branch, mom, IdentityPatch(name="x")),
                 lambda: store.upload_identity(client, branch, IdentityUpload(identity_id=mom, name="x")),
                 lambda: store.bind_trace(client, branch, mom, "v1", "t1"),
                 lambda: store.set_profile_image(client, branch, mom, photo(unit(1.0))),
                 lambda: store.merge_identity(client, branch, mom, other, keep_source=True),
                 lambda: store.merge_identity(client, branch, other, mom, keep_source=True),
                 lambda: store.delete_identity(client, branch, mom)):
        with pytest.raises(store.IdentityConflictError, match="deleted"):
            await call()
    assert (await store.get_identity(client, branch, mom)).name == "Mom"     # untouched
    assert (await owner_of(client, db, "t1"))[1] is None


# ── pending persons ──────────────────────────────────────────────────────

async def test_pending_persons_lists_both_kinds(make_db, client):
    branch = await make_db("ittest_ident_pending")
    auto = await two_videos(client, branch)
    manual = (await store.create_pending_person(client, branch, "v1", PendingPersonIn(name="someone"))).identity
    await named(client, branch, "Active one")                    # active: never on a pending list

    p = await store.pending_persons(client, branch, "v1")
    (x,) = p.unassigned_groups
    assert x.group_id == "x" and [t.trace_id for t in x.traces] == ["tx"]
    assert x.representative.face_id == "rep_tx"
    by_id = {i.identity_id: i for i in p.pending_identities}
    assert set(by_id) == {auto, manual.identity_id}
    (a,) = by_id[auto].groups                                    # only its v1 group
    assert a.group_id == "a" and [t.trace_id for t in a.traces] == ["t1", "t2"]
    assert a.representative.face_id == "rep_t2"                 # best quality of the group
    assert by_id[manual.identity_id].groups == [] and by_id[manual.identity_id].origin_version_id == "v1"

    p2 = await store.pending_persons(client, branch, "v2")
    assert p2.unassigned_groups == [] and [i.identity_id for i in p2.pending_identities] == [auto]


async def test_pending_persons_also_lists_active_identities_holding_a_group(make_db, client):
    branch = await make_db("ittest_ident_pending_active")
    await two_videos(client, branch)                              # group a (t1,t2) in v1, auto-pending
    bob = await named(client, branch, "Bob")
    await store.bind_group(client, branch, bob, "v1", "a")        # confirmed person now owns group a

    p = await store.pending_persons(client, branch, "v1")
    assert [i.identity_id for i in p.pending_identities] == []     # the auto-pending one is gone, folded into Bob
    (by_bob,) = p.active_identities
    assert by_bob.identity_id == bob and by_bob.identity_status == "active"
    (a,) = by_bob.groups
    assert a.group_id == "a" and [t.trace_id for t in a.traces] == ["t1", "t2"]
    (x,) = p.unassigned_groups                                    # untouched by the bind
    assert x.group_id == "x"


async def test_pending_person_with_a_trace_takes_it(make_db, client):
    branch = await make_db("ittest_ident_pending_bind")
    db = db_name_for(branch)
    await two_videos(client, branch)
    out = await store.create_pending_person(client, branch, "v1", PendingPersonIn(trace_id="tx"))
    assert (out.identity.identity_status, out.identity.source) == ("pending", "manual")
    assert out.bind.previous_group_id == "x" and out.bind.removed_groups == ["x"]   # x held only tx
    assert (await owner_of(client, db, "tx"))[1] == out.identity.identity_id
    assert out.identity.avatar.face_id == "rep_tx"

    with pytest.raises(store.IdentityNotFoundError):
        await store.create_pending_person(client, branch, "v1", PendingPersonIn(trace_id="nope"))
    assert await count(client, db, "Identity WHERE source = 'manual'") == 1   # nothing created for the bad request


async def test_a_poor_first_frame_does_not_stay_the_avatar(make_db, client):
    branch = await make_db("ittest_ident_avatar_upgrade")
    await write_face_traces(client, branch, video("v1", {
        "a": [trace("t1", unit(1.0), quality=0.5), trace("t2", unit(1.0), 5, quality=0.9)],
        "x": [trace("tx", unit(0.0), 10, quality=0.1)]}))
    out = await store.create_pending_person(client, branch, "v1", PendingPersonIn(trace_id="tx"))
    p = out.identity.identity_id
    assert out.identity.avatar.face_id == "rep_tx"               # the only face it has
    await store.bind_trace(client, branch, p, "v1", "t2")
    assert (await store.get_identity(client, branch, p)).avatar.face_id == "rep_t2"   # a better one joined


async def test_the_avatar_comes_from_the_largest_group(make_db, client):
    branch = await make_db("ittest_ident_avatar_largest")
    await write_face_traces(client, branch, video("v1", {"a": [
        trace("t1", unit(1.0), quality=0.5), trace("t2", unit(1.0), 5, quality=0.7), trace("t5", unit(1.0), 9, quality=0.6)]}))
    m = (await write_face_traces(client, branch, video("v2", {"b": [trace("t3", unit(0.9), quality=0.95)]}))).matches[0]
    ident = await store.get_identity(client, branch, m.identity_id)
    assert ident.avatar.face_id == "rep_t2"                      # not v2's lone, sharper frame


# ── bind / unbind ────────────────────────────────────────────────────────

async def test_bind_moves_traces_and_cleans_up_behind(make_db, client):
    branch = await make_db("ittest_ident_bind")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    mom = await named(client, branch)

    r = await store.bind_trace(client, branch, mom, "v1", "t2")
    assert r.previous_group_id == "a" and r.group_id not in ("a", None) and r.removed_groups == []
    assert (await owner_of(client, db, "t2"))[1] == mom
    a = (await client.query(db, "SELECT embedding, frame_count FROM FaceGroup WHERE group_id = 'a'"))[0]
    assert a["frame_count"] == 8 and a["embedding"][0] == pytest.approx(1.0)   # recomputed from t1 alone
    ident = await store.get_identity(client, branch, mom)
    assert ident.avatar.face_id == "rep_t2" and ident.group_count == 1
    assert (await store.get_identity(client, branch, auto)).avatar.face_id != "rep_t2"   # moved off the frame it lost

    r = await store.bind_trace(client, branch, mom, "v1", "t1")  # same video: joins the group made above
    assert r.removed_groups == ["a"] and r.removed_identities == []          # auto still holds v2's group
    assert (await store.get_identity(client, branch, mom)).group_count == 1

    r = await store.bind_trace(client, branch, mom, "v2", "t3")
    assert r.removed_groups == ["b"] and r.removed_identities == [auto]      # system-made and now empty
    assert (await store.get_identity(client, branch, mom)).video_count == 2

    again = await store.bind_trace(client, branch, mom, "v2", "t3")
    assert again.group_id == again.previous_group_id and again.removed_groups == []   # already hers


async def test_bind_group_puts_every_trace_under_the_person_in_one_step(make_db, client):
    branch = await make_db("ittest_ident_bind_group")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)                  # auto holds a (v1) and b (v2); x is unassigned
    mom = await named(client, branch)

    r = await store.bind_group(client, branch, mom, "v1", "x")
    assert (r.group_id, r.previous_identity_id, r.removed_identities) == ("x", None, [])
    assert await owner_of(client, db, "tx") == ("x", mom)    # the group itself is kept, now hers
    assert (await store.get_identity(client, branch, mom)).avatar.face_id == "rep_tx"

    r = await store.bind_group(client, branch, mom, "v1", "a")
    assert r.previous_identity_id == auto and r.removed_identities == []    # auto still holds b
    assert [await owner_of(client, db, t) for t in ("t1", "t2")] == [("a", mom), ("a", mom)]
    assert (await store.get_identity(client, branch, mom)).avatar.face_id == "rep_t2"   # a is her largest group
    assert (await store.get_identity(client, branch, auto)).avatar.face_id == "rep_t3"  # re-chosen from what it kept

    r = await store.bind_group(client, branch, mom, "v2", "b")
    assert r.removed_identities == [auto]                    # system-made and now empty
    again = await store.bind_group(client, branch, mom, "v2", "b")
    assert again.previous_identity_id == mom and again.removed_identities == []   # already hers


async def test_bind_group_refuses_unknown_groups_and_merged_identities(make_db, client):
    branch = await make_db("ittest_ident_bind_group_bad")
    db = db_name_for(branch)
    await two_videos(client, branch)
    mom = await named(client, branch)
    with pytest.raises(store.IdentityNotFoundError):
        await store.bind_group(client, branch, mom, "v1", "nope")
    with pytest.raises(store.IdentityNotFoundError):
        await store.bind_group(client, branch, mom, "v2", "x")                # x is in v1
    assert await owner_of(client, db, "tx") == ("x", None)                    # nothing moved
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": mom})
    with pytest.raises(store.IdentityConflictError):
        await store.bind_group(client, branch, mom, "v1", "x")


async def test_pending_person_from_a_group_or_a_list_of_traces(make_db, client):
    branch = await make_db("ittest_ident_pending_many")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)

    out = await store.create_pending_person(client, branch, "v1", PendingPersonIn(group_id="x"))
    assert out.bind_group.group_id == "x" and await owner_of(client, db, "tx") == ("x", out.identity.identity_id)

    out = await store.create_pending_person(client, branch, "v1", PendingPersonIn(trace_ids=["t1", "t2"]))
    p = out.identity.identity_id
    assert [b.trace_id for b in out.binds] == ["t1", "t2"] and out.bind is None
    (g1, o1), (g2, o2) = [await owner_of(client, db, t) for t in ("t1", "t2")]
    assert o1 == o2 == p and g1 == g2                          # one group for this person in v1
    assert out.identity.avatar.face_id == "rep_t2"
    assert out.binds[-1].removed_groups == ["a"]               # a held only t1 and t2
    assert (await store.get_identity(client, branch, auto)).group_count == 1   # auto keeps v2's b

    before = await count(client, db, "Identity")
    with pytest.raises(store.IdentityNotFoundError):
        await store.create_pending_person(client, branch, "v2", PendingPersonIn(trace_ids=["t3", "nope"]))
    with pytest.raises(store.IdentityNotFoundError):
        await store.create_pending_person(client, branch, "v1", PendingPersonIn(group_id="nope"))
    assert await count(client, db, "Identity") == before       # a bad ID creates nothing
    assert (await owner_of(client, db, "t3"))[1] == auto       # nor moves the good ones

    with pytest.raises(ValueError):
        PendingPersonIn(trace_id="t1", group_id="x")           # one of trace_id, trace_ids, group_id


async def test_bind_refuses_merged_identities_and_unknown_traces(make_db, client):
    branch = await make_db("ittest_ident_bind_bad")
    db = db_name_for(branch)
    await two_videos(client, branch)
    mom = await named(client, branch)
    with pytest.raises(store.IdentityNotFoundError):
        await store.bind_trace(client, branch, mom, "v1", "nope")
    with pytest.raises(store.IdentityNotFoundError):
        await store.bind_trace(client, branch, mom, "v2", "t1")             # t1 is in v1
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": mom})
    with pytest.raises(store.IdentityConflictError):
        await store.bind_trace(client, branch, mom, "v1", "t1")


async def test_unbind_leaves_the_trace_in_a_fresh_unassigned_group(make_db, client):
    branch = await make_db("ittest_ident_unbind")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    mom = await named(client, branch)
    with pytest.raises(store.IdentityConflictError):
        await store.unbind_trace(client, branch, mom, "v1", "t1")           # not hers

    await store.bind_trace(client, branch, mom, "v1", "tx")
    r = await store.unbind_trace(client, branch, mom, "v1", "tx")
    assert r.group_id is not None and r.group_id != r.previous_group_id and len(r.removed_groups) == 1
    assert await owner_of(client, db, "tx") == (r.group_id, None)   # back in a group of its own, unassigned
    ident = await store.get_identity(client, branch, mom)
    assert ident.group_count == 0 and ident.avatar is None       # a named identity stays, without a picture
    pending = await store.pending_persons(client, branch, "v1")
    assert r.group_id in {g.group_id for g in pending.unassigned_groups}   # reachable again, not orphaned

    r = await store.unbind_trace(client, branch, auto, "v2", "t3")
    assert r.removed_groups == ["b"] and r.removed_identities == []          # auto still holds v1's group


async def test_identity_traces_span_videos(make_db, client):
    branch = await make_db("ittest_ident_traces")
    auto = await two_videos(client, branch)
    out = await store.identity_traces(client, branch, auto)
    assert [(t.version_id, t.trace_id, t.group_id) for t in out.traces] == [
        ("v1", "t1", "a"), ("v1", "t2", "a"), ("v2", "t3", "b")]
    assert out.traces[0].representative.image_path.endswith("face_t1.jpg")
    assert out.traces[0].source_status is None                   # no content indexed in this test


# ── clone (module 04 dedup fast path) ───────────────────────────────────

async def test_clone_version_carries_face_data_to_target(make_db, client):
    """A dedup-cloned asset (module 04: identical checksum, archived elsewhere) must
    still have People/face data -- previously indexer.clone_version() never touched
    Face/FaceTrace/FaceGroup/Identity at all, so a cloned video always came up with
    zero faces. See identity_store.clone_face_data()."""
    from app.models.index import ChunkIndexRequest, CloneVersionRequest
    from app.services import indexer

    branch = await make_db("ittest_face_clone")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)   # v1: group a (t1,t2, -> auto) + group x (tx, unassigned); v2: group b (t3)
    await indexer.index_chunk(client, branch, ChunkIndexRequest(
        chunk_id="c1", type="videos", embedding_type="text", text="hello",
        version_id="v1", asset_path="u1/videos/v1"))   # clone_version requires a Source to clone from

    await indexer.clone_version(client, branch, CloneVersionRequest(
        source_version_id="v1", source_asset_path="u1/videos/v1",
        target_version_id="v9", target_asset_path="u1/videos/v9-copy"))

    # the identity-linked group (a) cloned onto v9, still pointing at the SAME identity
    traces_after = await store.identity_traces(client, branch, auto)
    assert {"v1", "v2", "v9"} <= {t.version_id for t in traces_after.traces}

    # the unassigned group (x) also cloned, reachable via pending-persons -- the bug
    pending = await store.pending_persons(client, branch, "v9")
    assert len(pending.unassigned_groups) == 1
    [cloned] = pending.unassigned_groups[0].traces
    assert cloned.trace_id != "tx"                              # distinct id, a real clone, not a shared vertex
    assert cloned.representative.image_path == "u1/videos/x/face_tx.jpg"  # shares the source's own crop file

    # idempotent, like the rest of clone_version (module 04 can retry Register)
    await indexer.clone_version(client, branch, CloneVersionRequest(
        source_version_id="v1", source_asset_path="u1/videos/v1",
        target_version_id="v9", target_asset_path="u1/videos/v9-copy"))
    assert await count(client, db, "FaceGroup WHERE version_id = 'v9'") == 2   # not duplicated


# ── HTTP ─────────────────────────────────────────────────────────────────

async def test_http_routes_and_status_codes(make_db, client):
    from app.main import app

    branch = await make_db("ittest_ident_http")
    auto = await two_videos(client, branch)
    h = {"X-Branch-ID": branch}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        r = await http.get("/personal/identity/identities", headers=h)
        assert r.status_code == 200 and r.json()["total"] == 1   # not swallowed by /{identity_id}
        assert (await http.get("/personal/identity/identities")).status_code == 400
        assert (await http.get("/personal/identity/nope", headers=h)).status_code == 404

        r = await http.post("/personal/identity/upload", headers=h, json={"identity_id": "mom", "name": "Mom"})
        assert r.status_code == 201
        r = await http.post("/personal/identity/upload", headers=h, json={"identity_id": "mom", "name": "Mother"})
        assert r.status_code == 200 and r.json()["name"] == "Mother"

        assert (await http.patch(f"/personal/identity/{auto}", headers=h,
                                 json={"identity_status": "merged"})).status_code == 422
        assert (await http.patch(f"/personal/identity/{auto}", headers=h,
                                 json={"aliases": {"fr": "x"}})).status_code == 422

        r = await http.post("/personal/identity/file/v1/pending-person", headers=h, json={"trace_id": "tx"})
        assert r.status_code == 201 and r.json()["bind"]["removed_groups"] == ["x"]
        r = await http.get("/personal/identity/file/v1/pending-persons", headers=h)
        assert r.status_code == 200 and len(r.json()["pending_identities"]) == 2

        r = await http.post("/personal/identity/mom/bind/trace", headers=h, json={"version_id": "v1", "trace_id": "t1"})
        assert r.status_code == 200
        assert (await http.post("/personal/identity/mom/unbind", headers=h,
                                json={"version_id": "v2", "trace_id": "t3"})).status_code == 409
        r = await http.get("/personal/identity/mom/traces", headers=h)
        assert [t["trace_id"] for t in r.json()["traces"]] == ["t1"]

        r = await http.delete("/personal/identity/mom", headers=h)
        assert r.status_code == 200 and r.json()["history_id"]
        r = await http.get("/personal/identity/mom", headers=h)
        assert r.status_code == 200 and r.json()["identity_status"] == "deleted"   # the node stays
        assert (await http.patch("/personal/identity/mom", headers=h, json={"name": "x"})).status_code == 409
        assert (await http.delete("/personal/identity/mom", headers=h)).status_code == 409
        r = await http.get("/personal/identity/identities", headers=h, params={"status": "deleted"})
        assert [i["identity_id"] for i in r.json()["items"]] == ["mom"]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        r = await http.get("/personal/identity/identities", headers={"X-Branch-ID": "ittest_ident_no_db"})
        assert r.status_code == 404


# ── merge, undo, redo ────────────────────────────────────────────────────

async def add_seed(client, db, identity_id, face_id, emb):
    await client.command(db, "INSERT INTO Face SET face_id = :f, kind = 'seed', identity_id = :i, "
                             "image_path = 'identity/b/x.jpg', embedding = :e",
                         params={"f": face_id, "i": identity_id, "e": emb})


async def test_merge_moves_groups_photos_and_fills_aliases(make_db, client):
    branch = await make_db("ittest_ident_merge")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    await store.patch_identity(client, branch, auto, IdentityPatch(aliases={"en": "Pete", "ja": "ピーター"}))
    await add_seed(client, db, auto, "seed_p", unit(0.0, axis=2))
    mom, _ = await store.upload_identity(client, branch, IdentityUpload(name="Peter", aliases={"en": "Peter P."}))

    res = await store.merge_identity(client, branch, auto, mom.identity_id, keep_source=True)
    assert sorted(res.groups_moved) == ["a", "b"] and res.source_kept
    t = res.target
    assert (t.name, t.group_count, t.video_count) == ("Peter", 2, 2)
    assert t.aliases == {"en": "Peter P.", "ja": "ピーター"}       # target's own locale is never overwritten
    assert t.avatar.face_id == "rep_t2"                         # had none: takes the source's
    src = await store.get_identity(client, branch, auto)
    assert src.identity_status == "merged" and src.group_count == 0
    assert (await client.query(db, "SELECT identity_id FROM Face WHERE face_id = 'seed_p'"))[0]["identity_id"] == mom.identity_id
    assert auto not in [i.identity_id for i in (await store.list_identities(client, branch, 50, 0)).items]

    # MTR-23: exactly one IdentityHistory record, written for this merge
    rows = await client.query(db, "SELECT * FROM IdentityHistory")
    assert len(rows) == 1
    h = rows[0]
    assert (h["history_id"], h["operation"], h["history_state"]) == (res.merge_id, "merge", "applied")
    assert h["undo_deadline"] == res.undo_deadline
    made = datetime.fromisoformat(h["created_at"])
    assert datetime.fromisoformat(h["undo_deadline"]) - made == timedelta(hours=24)
    before, after = json.loads(h["before_snapshot_json"]), json.loads(h["after_snapshot_json"])
    assert set(before["identities"]) == set(after["identities"]) == {auto, mom.identity_id}   # both identities
    assert before["identities"][auto]["identity_status"] == "pending"
    assert after["identities"][auto]["identity_status"] == "merged"
    assert {g: v["identity_id"] for g, v in before["groups"].items()} == {"a": auto, "b": auto}   # group ownership
    assert {g: v["identity_id"] for g, v in after["groups"].items()} == {"a": mom.identity_id, "b": mom.identity_id}

    # the moved photo keeps matching new videos, now for the target
    m = (await write_face_traces(client, branch, video("v3", {"c": [trace("t4", unit(0.0, axis=2))]}))).matches[0]
    assert m.identity_id == mom.identity_id


async def test_merge_can_delete_the_source(make_db, client):
    branch = await make_db("ittest_ident_merge_drop")
    auto = await two_videos(client, branch)
    mom = await named(client, branch)
    res = await store.merge_identity(client, branch, auto, mom, keep_source=False)
    assert not res.source_kept
    with pytest.raises(store.IdentityNotFoundError):
        await store.get_identity(client, branch, auto)


async def test_merge_gives_the_target_its_best_face_and_undo_restores_the_old_one(make_db, client):
    branch = await make_db("ittest_ident_merge_avatar")
    auto = await two_videos(client, branch)
    mom = (await store.create_pending_person(client, branch, "v1", PendingPersonIn(trace_id="tx"))).identity
    assert mom.avatar.face_id == "rep_tx"
    res = await store.merge_identity(client, branch, auto, mom.identity_id, keep_source=True)
    assert res.target.avatar.face_id == "rep_t2"                 # the largest group's best, not the frame it started from
    await store.replay_history(client, branch, res.merge_id, "undo")
    assert (await store.get_identity(client, branch, mom.identity_id)).avatar.face_id == "rep_tx"


async def test_merge_refuses_self_merged_and_unknown(make_db, client):
    branch = await make_db("ittest_ident_merge_bad")
    db = db_name_for(branch)
    a, b = await named(client, branch, "A"), await named(client, branch, "B")
    with pytest.raises(store.IdentityConflictError):
        await store.merge_identity(client, branch, a, a, True)
    with pytest.raises(store.IdentityNotFoundError):
        await store.merge_identity(client, branch, a, "nope", True)
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": b})
    with pytest.raises(store.IdentityConflictError):
        await store.merge_identity(client, branch, a, b, True)


async def test_undo_and_redo_a_merge(make_db, client):
    branch = await make_db("ittest_ident_merge_undo")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    await add_seed(client, db, auto, "seed_p", unit(0.0, axis=2))
    mom = await named(client, branch)
    res = await store.merge_identity(client, branch, auto, mom, keep_source=False)

    u = await store.replay_history(client, branch, res.merge_id, "undo")
    assert (u.operation, u.history_state, u.skipped) == ("merge", "reverted", [])
    src = await store.get_identity(client, branch, auto)          # rebuilt under the same ID
    assert (src.identity_status, src.source, src.group_count) == ("pending", "auto", 2)
    assert src.avatar.face_id == "rep_t2"
    assert (await store.get_identity(client, branch, mom)).group_count == 0
    assert (await client.query(db, "SELECT identity_id FROM Face WHERE face_id = 'seed_p'"))[0]["identity_id"] == auto

    with pytest.raises(store.IdentityConflictError):
        await store.replay_history(client, branch, res.merge_id, "undo")   # already undone

    r = await store.replay_history(client, branch, res.merge_id, "redo")
    assert (r.history_state, r.skipped) == ("applied", [])
    assert (await store.get_identity(client, branch, mom)).group_count == 2
    with pytest.raises(store.IdentityNotFoundError):
        await store.get_identity(client, branch, auto)


async def test_undo_with_everything_changed_since_skips_all_and_keeps_the_state(make_db, client):
    """MTR-23: an undo that finds every item already different from the merge's result applies nothing,
    reports everything skipped, and leaves the record's state as it was -- no data is touched."""
    branch = await make_db("ittest_ident_merge_idem")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    mom = await named(client, branch)
    res = await store.merge_identity(client, branch, auto, mom, keep_source=True)
    first = await store.replay_history(client, branch, res.merge_id, "undo")
    assert first.applied and first.history_state == "reverted"

    # every item is now back at its before value -- e.g. the first undo's state write was lost
    await client.command(db, "UPDATE IdentityHistory SET history_state = 'applied' WHERE history_id = :h",
                         params={"h": res.merge_id})
    again = await store.replay_history(client, branch, res.merge_id, "undo")
    assert again.applied == [] and sorted(again.skipped) == sorted(first.applied)
    assert again.history_state == "applied"                                         # state unchanged
    row = (await client.query(db, "SELECT history_state FROM IdentityHistory WHERE history_id = :h",
                              params={"h": res.merge_id}))[0]
    assert row["history_state"] == "applied"
    src = await store.get_identity(client, branch, auto)                           # data untouched
    assert (src.identity_status, src.group_count) == ("pending", 2)
    assert (await store.get_identity(client, branch, mom)).group_count == 0


async def test_undo_leaves_later_edits_alone(make_db, client):
    branch = await make_db("ittest_ident_merge_later")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    await store.patch_identity(client, branch, auto, IdentityPatch(aliases={"ja": "ピーター"}))
    mom = await named(client, branch)
    res = await store.merge_identity(client, branch, auto, mom, keep_source=True)

    # after the merge: the user edits the target's aliases, and v2's group is gone (video deleted)
    await store.patch_identity(client, branch, mom, IdentityPatch(aliases={"en": "Mum"}))
    await client.command(db, "DELETE VERTEX FROM FaceGroup WHERE group_id = 'b'")

    u = await store.replay_history(client, branch, res.merge_id, "undo")
    assert sorted(u.skipped) == ["group b", f"identity {mom} aliases_json"]
    assert "group a" in u.applied and f"identity {auto} identity_status" in u.applied
    assert (await store.get_identity(client, branch, mom)).aliases == {"ja": "ピーター", "en": "Mum"}   # kept
    assert (await store.get_identity(client, branch, auto)).identity_status == "pending"

    again = await store.replay_history(client, branch, res.merge_id, "redo")
    assert "group b" in again.skipped                            # still gone


async def test_undo_and_redo_a_delete(make_db, client):
    branch = await make_db("ittest_ident_delete_undo")
    db = db_name_for(branch)
    auto = await two_videos(client, branch)
    await store.patch_identity(client, branch, auto, IdentityPatch(name="Peter", identity_status="active"))
    await add_seed(client, db, auto, "seed_p", unit(0.0, axis=2))
    await client.command(db, "DELETE FROM HAS_PROFILE_FACE WHERE outV().identity_id = :i", params={"i": auto})
    await client.command(db, "CREATE EDGE HAS_PROFILE_FACE FROM (SELECT FROM Identity WHERE identity_id = :i) "
                             "TO (SELECT FROM Face WHERE face_id = 'seed_p')", params={"i": auto})
    res = await store.delete_identity(client, branch, auto)

    # meanwhile group a is given to someone else; b is left alone
    mom = await named(client, branch)
    await client.command(db, "CREATE EDGE GROUP_OF_IDENTITY FROM (SELECT FROM FaceGroup WHERE group_id = 'a') "
                             "TO (SELECT FROM Identity WHERE identity_id = :i)", params={"i": mom})

    u = await store.replay_history(client, branch, res.history_id, "undo")
    assert u.skipped == ["group a"]
    p = await store.get_identity(client, branch, auto)           # same ID, same fields
    assert (p.name, p.identity_status, p.source, p.group_count) == ("Peter", "active", "auto", 1)
    assert p.avatar.face_id == "seed_p"
    seed = (await client.query(db, "SELECT identity_id, embedding FROM Face WHERE face_id = 'seed_p'"))[0]
    assert seed["identity_id"] == auto and len(seed["embedding"]) == 512
    assert (await owner_of(client, db, "t1"))[1] == mom          # the later assignment wins

    r = await store.replay_history(client, branch, res.history_id, "redo")
    assert r.skipped == ["group a"]
    p = await store.get_identity(client, branch, auto)
    assert (p.identity_status, p.group_count) == ("deleted", 0)
    assert await count(client, db, "Face WHERE kind = 'seed'") == 1


async def test_undo_after_the_deadline_is_refused(make_db, client):
    branch = await make_db("ittest_ident_undo_late")
    db = db_name_for(branch)
    res = await store.delete_identity(client, branch, await named(client, branch))
    await client.command(db, "UPDATE IdentityHistory SET undo_deadline = '2000-01-01T00:00:00+00:00' "
                             "WHERE history_id = :h", params={"h": res.history_id})
    with pytest.raises(store.IdentityConflictError, match="2000-01-01"):
        await store.replay_history(client, branch, res.history_id, "undo")
    with pytest.raises(store.IdentityNotFoundError):
        await store.replay_history(client, branch, "nope", "undo")


async def test_http_merge_undo_redo(make_db, client):
    from app.main import app

    branch = await make_db("ittest_ident_http_merge")
    auto = await two_videos(client, branch)
    mom = await named(client, branch)
    h = {"X-Branch-ID": branch}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        r = await http.post(f"/personal/identity/{auto}/mergeinto", headers=h, json={"target_id": mom})
        assert r.status_code == 200 and r.json()["target"]["group_count"] == 2
        merge_id = r.json()["merge_id"]
        assert (await http.post(f"/personal/identity/{auto}/mergeinto", headers=h,
                                json={"target_id": mom})).status_code == 409
        r = await http.post(f"/personal/identity/merge/{merge_id}/undo", headers=h)
        assert r.status_code == 200 and r.json()["history_state"] == "reverted"
        assert (await http.post(f"/personal/identity/merge/{merge_id}/undo", headers=h)).status_code == 409
        r = await http.post(f"/personal/identity/merge/{merge_id}/redo", headers=h)
        assert r.status_code == 200 and r.json()["history_state"] == "applied"
        assert (await http.post("/personal/identity/merge/nope/undo", headers=h)).status_code == 404



# ── seed photo ───────────────────────────────────────────────────────────

def photo(emb, ref="c9"):
    return ProfileImageIn(image_path="identity/b/mom.jpg", image_version_id=ref, embedding=emb)


async def test_seed_photo_becomes_the_avatar_and_matches_later_videos(make_db, client):
    branch = await make_db("ittest_ident_photo")
    db = db_name_for(branch)
    mom = await named(client, branch)
    with pytest.raises(store.IdentityNotFoundError):
        await store.get_profile_image(client, branch, mom)       # no picture yet

    ident = await store.set_profile_image(client, branch, mom, photo(unit(1.0)))
    assert (ident.avatar.face_id, ident.avatar.kind, ident.avatar.image_version_id) == (f"seed_{mom}", "seed", "c9")
    assert (await store.get_profile_image(client, branch, mom)).image_path == "identity/b/mom.jpg"

    # the first video she appears in already matches her, by the photo alone
    m = (await write_face_traces(client, branch, video("v1", {"a": [trace("t1", unit(0.9))]}))).matches[0]
    assert m.identity_id == mom and not m.created_identity
    assert (await store.get_identity(client, branch, mom)).avatar.kind == "seed"   # the photo stays her picture

    await store.set_profile_image(client, branch, mom, photo(unit(0.0, axis=2), ref="c10"))
    seeds = await client.query(db, "SELECT image_version_id FROM Face WHERE kind = 'seed' AND identity_id = :i",
                               params={"i": mom})
    assert [r["image_version_id"] for r in seeds] == ["c10"]      # replaced, not added


async def test_seed_photo_refused_for_merged_or_unknown(make_db, client):
    branch = await make_db("ittest_ident_photo_bad")
    db = db_name_for(branch)
    with pytest.raises(store.IdentityNotFoundError):
        await store.set_profile_image(client, branch, "nope", photo(unit(1.0)))
    mom = await named(client, branch)
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": mom})
    with pytest.raises(store.IdentityConflictError):
        await store.set_profile_image(client, branch, mom, photo(unit(1.0)))


async def test_http_profile_image(make_db, client):
    from app.main import app

    branch = await make_db("ittest_ident_http_photo")
    mom = await named(client, branch)
    h = {"X-Branch-ID": branch}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        assert (await http.get(f"/personal/identity/{mom}/profile-image", headers=h)).status_code == 404
        body = {"image_path": "identity/b/mom.jpg", "image_version_id": "c9", "embedding": unit(1.0)}
        r = await http.post(f"/personal/identity/{mom}/profile-image", headers=h, json=body)
        assert r.status_code == 200 and r.json()["avatar"]["kind"] == "seed"
        r = await http.get(f"/personal/identity/{mom}/profile-image", headers=h)
        assert r.status_code == 200 and r.json()["image_version_id"] == "c9"
        assert (await http.post(f"/personal/identity/{mom}/profile-image", headers=h,
                                json={**body, "embedding": [0.1] * 3})).status_code == 422
