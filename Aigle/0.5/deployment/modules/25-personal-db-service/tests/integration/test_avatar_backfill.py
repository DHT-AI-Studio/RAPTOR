"""avatar_backfill: re-choosing existing avatars once. Integration tests against a real ArcadeDB (see
conftest.py); they skip when it isn't reachable."""
from __future__ import annotations

import pytest
from app.models.identity import IdentityUpload, ProfileImageIn
from app.services import identity_store as store
from app.services.arcadedb_client import db_name_for
from app.services.avatar_backfill import backfill_avatars
from app.services.identity_store import write_face_traces

from tests.integration.test_face_identity import trace, unit, video

pytestmark = pytest.mark.asyncio


async def person(client, branch, name, group_id, avatar_face_id):
    """A person holding one group, with the given avatar -- the state an old pick left behind."""
    db = db_name_for(branch)
    ident, _ = await store.upload_identity(client, branch, IdentityUpload(name=name))
    await store._link(client, db, "GROUP_OF_IDENTITY", "FaceGroup", "group_id", group_id, "Identity", "identity_id",
                      ident.identity_id, props={"score": None, "auto": False, "created_at": store._now()})
    await store._link(client, db, "HAS_PROFILE_FACE", "Identity", "identity_id", ident.identity_id, "Face", "face_id",
                      avatar_face_id)
    return ident.identity_id


async def avatar(client, branch, identity_id):
    return (await store.get_identity(client, branch, identity_id)).avatar.face_id


async def test_backfill_rechooses_stale_avatars_and_leaves_seeds_and_merged_people(make_db, client):
    branch = await make_db("ittest_avatar_backfill")
    db = db_name_for(branch)
    await write_face_traces(client, branch, video("v1", {
        "a": [trace("t1", unit(1.0), quality=0.3), trace("t2", unit(1.0), 5, quality=0.9)],
        "x": [trace("tx", unit(0.0), 10, quality=0.5)],
        "s": [trace("ts", unit(0.0, axis=2), 20, quality=0.4), trace("tu", unit(0.0, axis=2), 25, quality=0.8)]}))
    lead = await person(client, branch, "Lead", "a", "rep_t1")                   # a poor first pick
    seeded = await person(client, branch, "Seeded", "x", "rep_tx")
    await store.set_profile_image(client, branch, seeded, ProfileImageIn(
        image_path="identity/b/seed.jpg", image_version_id="c9", embedding=unit(0.0, axis=3)))
    gone = await person(client, branch, "Gone", "s", "rep_ts")
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": gone})

    dry = await backfill_avatars(client, db, dry_run=True)
    assert (dry.people, dry.changed, dry.unchanged, dry.skipped) == (3, 1, 0, 2)
    assert await avatar(client, branch, lead) == "rep_t1"                      # a dry run changes nothing

    done = await backfill_avatars(client, db)
    assert done.changed == 1
    assert await avatar(client, branch, lead) == "rep_t2"                      # the best frame of its group
    assert await avatar(client, branch, seeded) == f"seed_{seeded}"            # a seed photo stays
    assert await avatar(client, branch, gone) == "rep_ts"                      # merged: kept for undo

    again = await backfill_avatars(client, db)
    assert (again.changed, again.unchanged) == (0, 1)                          # re-runnable
