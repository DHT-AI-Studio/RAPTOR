"""
P16: a reference event travels from Module 18's plan to Module 25 as TKG contract v2, a resolved one
leads the triples, and an unresolved one is explained instead of summarised.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_tkg_reference.py -v
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import pipeline
import tkg_agent
from models import TKGRequest


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _module25(monkeypatch, reply):
    sent = []

    class Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            sent.append((url, json))
            return _Response({} if url.endswith("/internal/db/init") else reply)

    monkeypatch.setattr(tkg_agent.httpx, "Client", Client)
    return sent


def test_module18_reference_is_kept_on_the_plan(monkeypatch):
    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return _Response({"modes": ["TKG"], "reference": "Google acquired Android",
                              "reference_relations": ["after"]})

    monkeypatch.setattr(pipeline.httpx, "AsyncClient", Client)
    plan = asyncio.run(pipeline._build_plan_from_module18("q"))
    assert (plan.reference, plan.reference_relations) == ("Google acquired Android", ["after"])


ACQ = {"fact_id": "f-acq", "entity": "Google", "relation": "acquired", "value": "Android Inc.",
       "time_start": "2005-07-01", "when": "2005-07"}


def test_a_resolved_reference_is_sent_as_v2_and_leads_the_triples(monkeypatch):
    sent = _module25(monkeypatch, {"subgraph_edges": [], "reference": {"status": "resolved", "fact": ACQ},
                                   "temporal_facts": [
                                       {"entity": "T-Mobile G1", "relation": "was released", "value": "phone",
                                        "when": "2008-10-22", "allen": {"reference": {"relations": ["after"],
                                                                                      "certainty": "certain"}}},
                                       {"entity": "key employees", "relation": "joined", "value": "Google",
                                        "when": "2005-07", "allen": {"reference": {"relations": ["before", "after"],
                                                                                   "certainty": "possible"}}}]})
    prompts = []
    monkeypatch.setattr(tkg_agent, "_summarise", lambda q, t: prompts.append(t) or "summary")
    result = tkg_agent._do_tkg(TKGRequest(query="what happened after Google acquired Android", branch_id="b1",
                                          reference="Google acquired Android", reference_relations=["after"]))
    body = next(b for url, b in sent if url.endswith("/personal/search/tkg"))
    assert (body["version"], body["reference"], body["reference_relations"]) == (
        2, {"text": "Google acquired Android"}, ["after"])
    assert result.triples[0]["reference"] and result.triples[0]["when"] == "2005-07"
    assert [tkg_agent._maybe(t) for t in result.triples] == ["", "", " [possibly]"]


def test_without_a_reference_the_request_stays_v1(monkeypatch):
    sent = _module25(monkeypatch, {"subgraph_edges": [], "temporal_facts": []})
    tkg_agent._do_tkg(TKGRequest(query="after 2011", time_start="2012-01-01", branch_id="b1",
                                 reference="x", reference_relations=None))
    body = next(b for url, b in sent if url.endswith("/personal/search/tkg"))
    assert "version" not in body and "reference" not in body


def test_an_ambiguous_reference_is_explained_not_summarised(monkeypatch):
    _module25(monkeypatch, {"subgraph_edges": [], "temporal_facts": [], "reference": {
        "status": "ambiguous", "text": "the acquisition",
        "candidates": [{"entity": "Google", "relation": "acquired", "value": "Android Inc."},
                       {"entity": "Facebook", "relation": "acquired", "value": "Instagram"}]}})
    monkeypatch.setattr(tkg_agent, "_summarise", lambda q, t: (_ for _ in ()).throw(AssertionError("no summary")))
    result = tkg_agent._do_tkg(TKGRequest(query="after the acquisition", branch_id="b1",
                                          reference="the acquisition", reference_relations=["after"]))
    assert result.triples == []
    assert "several different events" in result.summary and "Facebook acquired Instagram" in result.summary
    _module25(monkeypatch, {"subgraph_edges": [], "temporal_facts": [],
                            "reference": {"status": "not_found", "text": "Apple bought Beats"}})
    assert "was not found" in tkg_agent._do_tkg(TKGRequest(
        query="q", branch_id="b1", reference="Apple bought Beats", reference_relations=["after"])).summary
