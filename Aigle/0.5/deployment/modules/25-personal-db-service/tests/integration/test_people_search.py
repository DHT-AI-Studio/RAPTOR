"""People Search (dev plan v1.4 §4.6): POST /personal/search/people.

Integration tests against a real ArcadeDB (see conftest.py); they skip when it isn't
reachable.

    PD_ARCADEDB_URL=http://localhost:12480 PD_ARCADEDB_PASSWORD=... pytest tests/test_people_search.py
"""
from __future__ import annotations

import httpx
import pytest
from app.models.identity import IdentityPatch, IdentityUpload, PeopleSearchRequest
from app.models.index import SourceSummaryIndexRequest
from app.services import identity_store as store
from app.services.arcadedb_client import db_name_for
from app.services.indexer import index_source_summary, set_status_by_version

from tests.integration.conftest import fake_vector
from tests.integration.test_identity_api import two_videos

pytestmark = pytest.mark.asyncio


async def indexed(client, branch, version_id):
    """The video's content index, as content indexing would have written it."""
    await index_source_summary(client, branch, SourceSummaryIndexRequest(
        version_id=version_id, summary=f"{version_id} summary", embedding=fake_vector(version_id),
        media_type="videos", filename=f"{version_id}.mp4", asset_path=f"u1/videos/{version_id}"))


async def peter(client, branch):
    """Two videos, both content-indexed; P is named Peter Parker, alias 彼得."""
    pid = await two_videos(client, branch)
    await store.patch_identity(client, branch, pid, IdentityPatch(name="Peter Parker", aliases={"zh-TW": "彼得"}))
    for v in ("v1", "v2"):
        await indexed(client, branch, v)
    return pid


async def search(client, branch, q, limit=50):
    return await store.search_people(client, branch, PeopleSearchRequest(query=q, limit=limit))


async def test_finds_a_person_and_every_segment(make_db, client):
    branch = await make_db("ittest_people_find")
    pid = await peter(client, branch)

    res = await search(client, branch, "Peter Parker")
    (m,) = res.identities
    assert (m.identity_id, m.matched, m.exact) == (pid, "Peter Parker", True)
    assert [(s.version_id, s.trace_id) for s in res.results] == [("v1", "t1"), ("v1", "t2"), ("v2", "t3")]
    s = res.results[0]
    assert (s.identity_name, s.first_sec, s.last_sec) == ("Peter Parker", 0.0, 7 / 8)
    assert (s.asset_path, s.filename) == ("u1/videos/v1", "v1.mp4")
    assert s.keyframe.image_path.endswith("face_t1.jpg") and s.confidence == pytest.approx(0.9)
    assert s.keyframe_path == s.keyframe.image_path
    assert res.total == 3


async def test_matches_part_of_a_name_or_alias_in_any_case(make_db, client):
    branch = await make_db("ittest_people_partial")
    pid = await peter(client, branch)
    for q, matched in (("peter", "Peter Parker"), ("PARKER", "Peter Parker"), ("彼得", "彼得"), ("彼", "彼得")):
        (m,) = (await search(client, branch, q)).identities
        assert (m.identity_id, m.matched) == (pid, matched), q
    assert (await search(client, branch, "彼得")).identities[0].exact
    assert (await search(client, branch, "Mary")).identities == []


async def test_names_go_through_the_full_text_index_and_its_false_positives_are_dropped(make_db, client):
    """The Identity[name] index matches 金門 for 金門豬肉 (a shared bigram); the Python check drops it.
    A lone CJK character can't match the index, so it is a substring scan instead."""
    branch = await make_db("ittest_people_index")
    pid = await peter(client, branch)
    await store.patch_identity(client, branch, pid, IdentityPatch(name="金門豬肉", aliases={}))
    other = (await store.upload_identity(client, branch, IdentityUpload(name="金門")))[0].identity_id
    assert [m.identity_id for m in (await search(client, branch, "金門豬肉")).identities] == [pid]
    assert [(m.identity_id, m.exact) for m in (await search(client, branch, "金門")).identities] == [
        (other, True), (pid, False)]
    assert {m.identity_id for m in (await search(client, branch, "豬")).identities} == {pid}


async def test_exact_matches_come_first(make_db, client):
    branch = await make_db("ittest_people_order")
    await peter(client, branch)
    await store.upload_identity(client, branch, IdentityUpload(name="Peter"))
    res = await search(client, branch, "peter")
    assert [(m.name, m.exact) for m in res.identities] == [("Peter", True), ("Peter Parker", False)]


async def test_taken_down_and_unindexed_videos_are_left_out(make_db, client):
    branch = await make_db("ittest_people_status")
    await peter(client, branch)
    await set_status_by_version(client, branch, "v2", "unregistered")
    assert [s.version_id for s in (await search(client, branch, "Peter")).results] == ["v1", "v1"]
    await set_status_by_version(client, branch, "v2", "active")
    assert len((await search(client, branch, "Peter")).results) == 3            # back after recovery

    branch2 = await make_db("ittest_people_noindex")
    await two_videos(client, branch2)                            # faces written, content never indexed
    pid = (await store.list_identities(client, branch2, 50, 0)).items[0].identity_id
    await store.patch_identity(client, branch2, pid, IdentityPatch(name="Peter"))
    res = await search(client, branch2, "Peter")
    assert len(res.identities) == 1 and res.results == []


async def test_merged_and_unnamed_identities_are_not_found(make_db, client):
    branch = await make_db("ittest_people_merged")
    db = db_name_for(branch)
    pid = await peter(client, branch)
    await client.command(db, "UPDATE Identity SET identity_status = 'merged' WHERE identity_id = :i", params={"i": pid})
    assert (await search(client, branch, "Peter")).identities == []
    await client.command(db, "UPDATE Identity SET identity_status = 'pending', name = '', aliases_json = '{}' "
                             "WHERE identity_id = :i", params={"i": pid})
    assert (await search(client, branch, "a")).identities == []


async def test_limit_caps_segments_not_the_total(make_db, client):
    branch = await make_db("ittest_people_limit")
    await peter(client, branch)
    res = await search(client, branch, "Peter", limit=2)
    assert res.total == 3 and len(res.results) == 2


async def test_http_route(make_db, client):
    from app.main import app

    branch = await make_db("ittest_people_http")
    await peter(client, branch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        r = await http.post("/personal/search/people", headers={"X-Branch-ID": branch}, json={"query": "彼得"})
        assert r.status_code == 200 and r.json()["total"] == 3
        assert (await http.post("/personal/search/people", headers={"X-Branch-ID": branch},
                                json={"query": ""})).status_code == 422
        assert (await http.post("/personal/search/people", json={"query": "x"})).status_code == 400


async def test_a_user_without_a_database_finds_no_one(client):
    """Fail-open: not initialized yet is an empty result, not an error."""
    from app.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        r = await http.post("/personal/search/people", headers={"X-Branch-ID": "ittest_people_never_created"},
                            json={"query": "Peter"})
    assert r.status_code == 200
    assert r.json() == {"query": "Peter", "identities": [], "total": 0, "results": []}
