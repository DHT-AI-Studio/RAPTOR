"""P21 against a REAL ArcadeDB: an inline source is archived once its expiry passes -- left out of
queries, never reactivated, archived again if content arrives late -- and marked deleted with it."""
from __future__ import annotations

import time

import httpx
import pytest
from app.models.graph_index import TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, indexer, searcher, source_expiry
from app.services.arcadedb_client import db_name_for

from tests.integration.test_index import _chunk

pytestmark = pytest.mark.asyncio


async def _source(client, branch, version="v-inline", chunk="c1"):
    await indexer.index_chunk(client, branch, _chunk(chunk, "Google acquired Android in July 2005.",
                                                     version_id=version, filename="inline.txt"))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id=f"f-{chunk}", entity="Google", relation="acquired", value="Android Inc.", time_start="2005-07",
        time_end="2005-07", source_version_id=version))


async def _statuses(client, db, version="v-inline"):
    out = {}
    for vtype, col in (("Chunk", "version_id"), ("Source", "version_id"), ("TemporalFact", "source_version_id")):
        rows = await client.query(db, f"SELECT status FROM {vtype} WHERE {col} = :v", params={"v": version})
        out[vtype] = sorted({r.get("status") or "active" for r in rows})
    return out


def _ms(seconds_from_now: float) -> int:
    return int((time.time() + seconds_from_now) * 1000)


async def test_expired_source_is_archived_and_stays_archived(client, make_db):
    branch = await make_db("ittest_p21_expiry")
    db = db_name_for(branch)
    await _source(client, branch)
    reg = await source_expiry.register(client, branch, "v-inline", _ms(3600))
    assert (reg["state"], reg["expired"]) == ("pending", False)
    assert await source_expiry.sweep_database(client, db) == 0                       # not yet

    await source_expiry.register(client, branch, "v-inline", _ms(-1))                 # its 7 days are up
    assert await source_expiry.sweep_database(client, db) == 1
    assert await _statuses(client, db) == {"Chunk": ["archived"], "Source": ["archived"], "TemporalFact": ["archived"]}
    assert (await source_expiry.get(client, branch, "v-inline"))["state"] == "expired"
    tkg = await searcher.tkg_search(client, branch, TKGRequest(query="Google", entity="Google", score_threshold=0))
    assert tkg.temporal_facts == []                                                   # left out of queries

    await indexer.set_status_by_version(client, branch, "v-inline", "active")         # no resurrection
    assert await _statuses(client, db) == {"Chunk": ["archived"], "Source": ["archived"], "TemporalFact": ["archived"]}
    again = await source_expiry.register(client, branch, "v-inline", _ms(3600))       # an expiry is not lifted
    assert again["state"] == "expired" and again["expired"]


async def test_content_arriving_after_expiry_is_archived_by_the_next_sweep(client, make_db):
    branch = await make_db("ittest_p21_late")
    db = db_name_for(branch)
    await source_expiry.register(client, branch, "v-inline", _ms(-1))                 # registered, not indexed yet
    assert await source_expiry.sweep_database(client, db) == 0
    await _source(client, branch, chunk="c-late")                                     # a late worker
    assert await source_expiry.sweep_database(client, db) == 1
    assert (await _statuses(client, db))["Chunk"] == ["archived"]
    assert await source_expiry.sweep_database(client, db) == 0                       # nothing left visible


async def test_deletion_is_recorded_and_other_sources_are_untouched(client, make_db):
    branch = await make_db("ittest_p21_deleted")
    db = db_name_for(branch)
    await _source(client, branch)
    await _source(client, branch, version="v-upload", chunk="c2")                     # an upload: no expiry
    await source_expiry.register(client, branch, "v-inline", _ms(-1))
    await source_expiry.sweep_database(client, db)
    await indexer.delete_by_version(client, branch, "v-inline")                       # Module 04's destroy
    assert (await source_expiry.get(client, branch, "v-inline"))["state"] == "deleted"
    assert (await _statuses(client, db, "v-upload"))["Chunk"] == ["active"]
    assert await source_expiry.sweep(client) >= 0                                     # every tenant; old ones skipped


async def test_over_http(client, make_db):
    from app.main import app
    branch = await make_db("ittest_p21_http")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        h = {"X-Branch-ID": branch}
        assert (await http.get("/personal/temporal/expiry/v9", headers=h)).status_code == 404
        r = await http.put("/personal/temporal/expiry/v9", headers=h, json={"expires_at": "2030-01-01T00:00:00Z"})
        assert r.status_code == 200 and r.json()["state"] == "pending" and not r.json()["expired"]
        assert (await http.get("/personal/temporal/expiry/v9", headers=h)).json()["expires_at"] == 1893456000000
        for bad in ("next week", "2030-01-01T00:00:00"):                              # unreadable / no zone
            r = await http.put("/personal/temporal/expiry/v9", headers=h, json={"expires_at": bad})
            assert r.status_code == 400 and r.json()["detail"]["error"] == "INVALID_EXPIRES_AT"
        caps = (await http.get("/personal/temporal/capabilities")).json()
        assert caps["operations"]["expiry"] == "/personal/temporal/expiry/{version_id}"
