"""P16 against a REAL ArcadeDB: TKG v2 reference-event queries resolve the event (or say why not) and keep
the facts that have the requested Allen relations to it."""
from __future__ import annotations

import httpx
import pytest
from app.models.graph_index import TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher

pytestmark = pytest.mark.asyncio

FACTS = [  # fact_id, entity, relation, value, date, sentence
    ("f-found", "Android Inc.", "was founded", "in Palo Alto", "2003-10", "Android Inc. was founded in October 2003."),
    ("f-danger", "Andy Rubin", "left", "Danger", "2003", "Andy Rubin left Danger in 2003."),
    ("f-acq", "Google", "acquired", "Android Inc.", "2005-07", "Google acquired Android Inc. in July 2005."),
    ("f-staff", "key employees", "joined", "Google", "2005-07",
     "Its key employees joined Google in July 2005, as part of the acquisition."),
    ("f-oha", "Open Handset Alliance", "was unveiled", "with Android", "2007-11-05",
     "The Open Handset Alliance was unveiled on November 5, 2007."),
    ("f-g1", "T-Mobile G1", "was released", "as the first Android phone", "2008-10-22",
     "The T-Mobile G1 was released on October 22, 2008."),
    ("f-insta", "Facebook", "acquired", "Instagram", "2012-04", "Facebook acquired Instagram in April 2012."),
]


async def _corpus(client, make_db, name):
    branch = await make_db(name)
    for fid, entity, relation, value, date, sentence in FACTS:
        await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
            fact_id=fid, entity=entity, relation=relation, value=value, time_start=date, time_end=date,
            date_text=date, source_sentence=sentence, extent="point"))
    return branch


def _req(**kw):
    return TKGRequest(query="what happened", score_threshold=0, version=2, limit=50, **kw)


async def test_after_a_text_reference(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p16_after")
    resp = await searcher.tkg_search(client, branch, _req(
        reference={"text": "Google acquired Android"}, reference_relations=["after"]))
    assert resp.reference["status"] == "resolved" and resp.reference["fact"]["fact_id"] == "f-acq"
    assert resp.reference["fact"]["when"] == "2005-07"
    got = {f["fact_id"]: f["allen"]["reference"]["certainty"] for f in resp.temporal_facts}
    # the same month as the acquisition may be after it; the event itself and what came before are not
    assert got == {"f-staff": "possible", "f-oha": "certain", "f-g1": "certain", "f-insta": "certain"}

    certain = await searcher.tkg_search(client, branch, _req(
        reference={"text": "Google acquired Android"}, reference_relations=["after"], certainty="certain"))
    assert [f["fact_id"] for f in certain.temporal_facts] == ["f-oha", "f-g1", "f-insta"]

    before = await searcher.tkg_search(client, branch, _req(
        reference={"fact_id": "f-acq"}, reference_relations=["before"], certainty="certain"))
    assert [f["fact_id"] for f in before.temporal_facts] == ["f-danger", "f-found"]


async def test_a_reference_and_a_window_both_apply(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p16_window")
    resp = await searcher.tkg_search(client, branch, _req(
        reference={"text": "Google acquired Android"}, reference_relations=["after"],
        window={"start": "2007", "end": "2008"}, match="overlaps"))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["f-oha", "f-g1"]
    assert resp.temporal_facts[0]["allen"]["vs_window"] and resp.temporal_facts[0]["allen"]["reference"]


async def test_an_ambiguous_or_missing_reference_returns_no_facts(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p16_ambiguous")
    resp = await searcher.tkg_search(client, branch, _req(
        reference={"text": "the acquisition"}, reference_relations=["after"]))
    assert resp.reference["status"] == "ambiguous" and resp.temporal_facts == []
    assert {c["fact_id"] for c in resp.reference["candidates"]} >= {"f-acq", "f-insta"}
    assert "fact" not in resp.reference

    for ref in ({"text": "Apple bought Beats"}, {"fact_id": "nope"}):
        resp = await searcher.tkg_search(client, branch, _req(reference=ref, reference_relations=["after"]))
        assert resp.reference["status"] == "not_found" and resp.temporal_facts == [] and resp.complete


async def test_over_http(client, make_db):
    from app.main import app
    branch = await _corpus(client, make_db, "ittest_p16_http")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        r = await http.post("/personal/search/tkg", headers={"X-Branch-ID": branch}, json={
            "query": "what happened", "version": 2, "score_threshold": 0,
            "reference": {"text": "Facebook acquired Instagram"}, "reference_relations": ["before"],
            "certainty": "certain", "window": {"start": "2007"}})
        assert r.status_code == 200
        body = r.json()
        assert body["reference"]["status"] == "resolved"
        assert [f["fact_id"] for f in body["temporal_facts"]] == ["f-oha", "f-g1"]


async def test_a_source_filter_scopes_facts_and_the_reference(client, make_db):
    branch = await _corpus(client, make_db, "ittest_p21_scope")
    from app.services.arcadedb_client import db_name_for
    for fid, version in (("f-acq", "v-a"), ("f-g1", "v-a"), ("f-oha", "v-b"), ("f-insta", "v-b")):
        await client.command(db_name_for(branch), "UPDATE TemporalFact SET source_version_id = :v WHERE fact_id = :f",
                             params={"v": version, "f": fid})
    only_a = await searcher.tkg_search(client, branch, _req(source_filter=["v-a"]))
    assert {f["fact_id"] for f in only_a.temporal_facts} == {"f-acq", "f-g1"}
    # the reference is resolved inside the scope too: Facebook / Instagram is not in v-a
    resp = await searcher.tkg_search(client, branch, _req(
        source_filter=["v-a"], reference={"text": "Facebook acquired Instagram"}, reference_relations=["before"]))
    assert resp.reference["status"] == "not_found" and resp.temporal_facts == []
    resp = await searcher.tkg_search(client, branch, _req(
        source_filter=["v-a"], reference={"text": "the acquisition"}, reference_relations=["after"]))
    assert resp.reference["status"] == "resolved"                      # unambiguous inside v-a
    assert [f["fact_id"] for f in resp.temporal_facts] == ["f-g1"]
