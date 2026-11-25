"""P17 against a REAL ArcadeDB: /personal/temporal/verify checks an answer's claims against the tenant's
facts, with the real date grammar ("July 2005", "2005年7月")."""
from __future__ import annotations

import httpx
import pytest
from app.models.graph_index import TemporalFactIndexRequest
from app.services import graph_indexer, searcher

from tests.integration.test_tkg_reference import _corpus

pytestmark = pytest.mark.asyncio
EVIDENCE = ["f-acq", "f-staff", "f-g1", "f-insta"]


async def test_dates_orderings_and_scope(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p17_verify")
    out = await searcher.verify_claims(client, branch, [
        {"kind": "date", "fact": "f-acq", "date": "July 2005"},
        {"kind": "date", "fact": "f-acq", "date": "2005年7月"},
        {"kind": "date", "fact": "f-acq", "date": "12 July 2005"},             # finer than the evidence
        {"kind": "ordering", "subject": "f-acq", "object": "f-g1", "relations": ["before"]},
        {"kind": "ordering", "subject": "f-staff", "object": "f-acq", "relations": ["after"]},
        {"kind": "date", "fact": "f-found", "date": "2003"},                   # a real fact, not in the evidence
        {"kind": "absence"},
    ], EVIDENCE, complete=False)
    assert [v["verified"] for v in out["verdicts"]] == [True, True, False, True, False, False, False]
    assert out["verdicts"][2]["evidence_when"] == "2005-07"
    assert out["verdicts"][4]["reason"].startswith("not certain")
    assert not out["all_verified"]
    assert out["facts"]["f-acq"]["when"] == "2005-07" and "f-found" not in out["facts"]


async def test_an_archived_fact_establishes_nothing(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p17_archived")
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="f-old", entity="Google", relation="acquired", value="Android Inc.", time_start="2005-07",
        time_end="2005-07"))
    from app.services.arcadedb_client import db_name_for
    await client.command(db_name_for(branch), "UPDATE TemporalFact SET status = 'archived' WHERE fact_id = 'f-old'")
    out = await searcher.verify_claims(client, branch, [{"kind": "date", "fact": "f-old", "date": "2005"}],
                                       ["f-old"], complete=True)
    assert "not an active fact" in out["verdicts"][0]["reason"]


async def test_over_http(client, make_db):
    from app.main import app
    branch = await _corpus(client, make_db, "ittest_p17_http")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        r = await http.post("/personal/temporal/verify", headers={"X-Branch-ID": branch}, json={
            "claims": [{"kind": "ordering", "subject": "f-g1", "object": "f-insta", "relations": ["before"]}],
            "evidence": ["f-g1", "f-insta"], "complete": True})
        assert r.status_code == 200 and r.json()["all_verified"]
        r = await http.post("/personal/temporal/verify", headers={"X-Branch-ID": branch},
                            json={"claims": [], "evidence": [], "timezone": "Mars/Olympus"})
        assert r.status_code == 400 and r.json()["detail"]["error"] == "INVALID_TIMEZONE"
        caps = (await http.get("/personal/temporal/capabilities")).json()
        assert caps["operations"]["verify"] == "/personal/temporal/verify"
