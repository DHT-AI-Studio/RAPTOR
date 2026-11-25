"""
P07: the TKG Allen match mode travels from Module 18's plan to Module 25, and Module 25's
`allen` labels come back on the facts.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_tkg_match.py -v
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


def _module18_plan(monkeypatch, reply):
    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return _Response(reply)

    monkeypatch.setattr(pipeline.httpx, "AsyncClient", Client)
    return asyncio.run(pipeline._build_plan_from_module18("q"))


def test_module18_match_is_kept_on_the_plan(monkeypatch):
    plan = _module18_plan(monkeypatch, {"modes": ["TKG"], "match": "overlaps",
                                        "time_from": "2010-01-01T00:00:00.000+08:00",
                                        "time_to": "2010-12-31T23:59:59.999+08:00"})
    assert plan.match == "overlaps" and plan.time_range["start"].startswith("2010-01-01")
    assert _module18_plan(monkeypatch, {"modes": ["TKG"], "time_from": "2012-01-01T00:00:00.000+08:00"}).match is None


def test_llm_plan_uses_the_same_rule():
    closed = pipeline._build_plan_from_llm({"time_range": {"start": "2010-01-01", "end": "2010-12-31"}})
    open_ = pipeline._build_plan_from_llm({"time_range": {"start": "2012-01-01", "end": None}})
    none = pipeline._build_plan_from_llm({})
    assert (closed.match, open_.match, none.match) == ("overlaps", "inside", None)


def test_tkg_agent_forwards_match_and_keeps_allen(monkeypatch):
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
            if url.endswith("/internal/db/init"):
                return _Response({})
            return _Response({"subgraph_edges": [], "temporal_facts": [
                {"entity": "Hugo Barra", "relation": "served_as", "value": "spokesperson",
                 "time_start": "2008-01-01", "time_end": "2013-12-31",
                 "allen": {"vs_window": ["contains"], "certainty": "certain"}},
                {"entity": "Samsung", "relation": "released", "value": "Galaxy S", "time_start": "2010-06-01"}]})

    monkeypatch.setattr(tkg_agent.httpx, "Client", Client)
    monkeypatch.setattr(tkg_agent, "_summarise", lambda q, t: "")
    result = tkg_agent._do_tkg(TKGRequest(query="Hugo Barra", time_start="2010-01-01", time_end="2010-12-31",
                                          match="overlaps", branch_id="b1"))
    body = next(b for url, b in sent if url.endswith("/personal/search/tkg"))
    assert body["match"] == "overlaps"
    assert result.triples[0]["allen"] == {"vs_window": ["contains"], "certainty": "certain"}
    assert "allen" not in result.triples[1]                       # an older Module 25 / no window: no label

    sent.clear()
    tkg_agent._do_tkg(TKGRequest(query="after 2011", time_start="2012-01-01", branch_id="b1"))
    assert "match" not in next(b for url, b in sent if url.endswith("/personal/search/tkg"))
