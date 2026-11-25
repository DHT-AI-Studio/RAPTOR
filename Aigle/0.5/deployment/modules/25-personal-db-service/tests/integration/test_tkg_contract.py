"""P08 / P09 against a REAL ArcadeDB: TKG contract v2 pages an entity's whole result (v1 does not change),
ranks by relevance over every match, refuses unsupported fields with a structured 400, and capabilities
answers over HTTP."""
from __future__ import annotations

import httpx
import pytest
from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher

pytestmark = pytest.mark.asyncio

YEARS = range(2001, 2008)                                  # seven Android facts


async def _corpus(client, make_db, name):
    branch = await make_db(name)
    for eid, n in [("e1", "Android"), ("e2", "Nexus"), ("e3", "Pixel")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=n, type="PRODUCT"))
    for y in YEARS:
        await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
            fact_id=f"a{y}", entity="Android", entity_id="e1", relation="released", value=f"version {y}",
            time_start=str(y), time_end=str(y), source_sentence=f"Android released version {y}."))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="n2002", entity="Nexus", entity_id="e2", relation="shipped", value="with Android",
        time_start="2002", time_end="2002", source_sentence="Nexus shipped with Android."))
    return branch


async def test_v1_is_unchanged_v2_pages_the_whole_result(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p08_pages")
    v1 = await searcher.tkg_search(client, branch, TKGRequest(query="Android", entity="Android", limit=3,
                                                               score_threshold=0))
    assert (v1.version, v1.cursor, v1.total) == (1, None, None)
    assert {f"a{y}" for y in YEARS} <= {f["fact_id"] for f in v1.temporal_facts}   # entity facts uncapped, as before

    req, seen, pages = TKGRequest(query="Android", entity="Android", limit=3, score_threshold=0, version=2), [], 0
    while True:
        resp = await searcher.tkg_search(client, branch, req)
        pages += 1
        seen += [f["fact_id"] for f in resp.temporal_facts]
        assert len(resp.temporal_facts) <= 3 and resp.complete == (resp.cursor is None)
        if resp.cursor is None:
            break
        req = req.model_copy(update={"cursor": resp.cursor})
    assert seen == ["a2001", "a2002", "n2002", "a2003", "a2004", "a2005", "a2006", "a2007"]   # mentions included
    assert (pages, resp.total, resp.version, resp.sort) == (3, 8, 2, "time")
    assert resp.temporal_facts[-1]["source_sentence"] == "Android released version 2007."   # evidence travels


async def test_v2_relevance_and_window(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p08_relevance")
    resp = await searcher.tkg_search(client, branch, TKGRequest(
        query="Nexus shipped", keywords=["Android"], limit=2, score_threshold=0, version=2, sort="relevance",
        window={"start": "2002", "end": "2002"}, match="overlaps"))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["n2002", "a2002"] and resp.complete


async def test_v2_refusals_are_structured_400s(client, make_db):
    from app.main import app
    branch = await _corpus(client, make_db, "ittest_p09_refuse")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        r = await http.post("/personal/search/tkg", headers={"X-Branch-ID": branch},
                            json={"query": "Android", "version": 2, "as_of": "2026-01-01"})
        assert r.status_code == 400
        assert r.json()["detail"] == {"error": "UNSUPPORTED_FEATURES", "features": ["as_of"],
                                      "see": "/personal/temporal/capabilities"}
        r = await http.post("/personal/search/tkg", headers={"X-Branch-ID": branch},
                            json={"query": "Android", "as_of": "2026-01-01"})               # v1: ignored
        assert r.status_code == 200
        r = await http.post("/personal/search/tkg", headers={"X-Branch-ID": branch},
                            json={"query": "Android", "version": 2, "reference": {"text": "x"}})
        assert r.status_code == 400 and r.json()["detail"]["error"] == "INCOMPLETE_REFERENCE"   # P16
        caps = (await http.get("/personal/temporal/capabilities")).json()
        assert caps["contract_versions"] == [1, 2] and caps["tkg"]["unsupported"] == ["as_of"]
