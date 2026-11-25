"""
P17: the TKG summary is built from verified claims -- Module 07 proposes them (structured output),
Module 25 verifies them, a template renders them -- or falls back to the facts as recorded.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_tkg_grounded.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import grounded_claims
import httpx
import tkg_agent
from models import TKGRequest

ACQ = {"fact_id": "f-acq", "entity": "Google", "relation": "acquired", "value": "Android Inc.", "when": "2005-07"}
G1 = {"fact_id": "f-g1", "entity": "T-Mobile G1", "relation": "was released", "value": "", "when": "2008-10-22"}


class _Response:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(self.status_code))

    def json(self):
        return self._data


def _services(monkeypatch, facts, claims, verify, *, infer_status=200, rerank=None):
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
            if url.endswith("/personal/search/tkg"):
                return _Response({"subgraph_edges": [], "temporal_facts": facts, "complete": True})
            if url.endswith("/inference/infer"):
                return _Response({"result": {"parsed": {"claims": claims}}}, infer_status)
            if url.endswith("/personal/temporal/verify"):
                return _Response(verify)
            if url.endswith("/personal/search/rerank"):
                return _Response({"results": rerank or []})
            return _Response({})

    monkeypatch.setattr(tkg_agent.httpx, "Client", Client)
    monkeypatch.setattr(grounded_claims.httpx, "Client", Client)
    monkeypatch.setattr(tkg_agent, "_summarise", lambda q, t: (_ for _ in ()).throw(AssertionError("no prose")))
    return sent


def _post(sent, suffix):
    return next(b for url, b in sent if url.endswith(suffix))


def test_verified_claims_are_rendered_by_template(monkeypatch):
    claims = [{"kind": "date", "fact": "f-acq", "date": "2005-07"},
              {"kind": "ordering", "subject": "f-acq", "object": "f-g1", "relations": ["before"]}]
    verify = {"all_verified": True, "facts": {"f-acq": ACQ, "f-g1": G1},
              "verdicts": [{"index": 0, "verified": True, "reason": None},
                           {"index": 1, "verified": True, "reason": None, "relations": ["before"]}]}
    sent = _services(monkeypatch, [ACQ, G1], claims, verify)
    result = tkg_agent._do_tkg(TKGRequest(query="When did Google buy Android?", branch_id="b1"))
    infer = _post(sent, "/inference/infer")
    assert infer["options"]["format"] == grounded_claims.claims_schema(["f-acq", "f-g1"])
    assert infer["options"]["temperature"] == 0.0
    assert infer["options"]["max_tokens"] == 1024 and grounded_claims.CLAIMS_SCHEMA["properties"]["claims"]["maxItems"] == 12
    assert "[f-acq] Google acquired Android Inc. (2005-07)" in infer["data"]["messages"][1]["content"]
    assert _post(sent, "/personal/temporal/verify")["evidence"] == ["f-acq", "f-g1"]
    assert result.summary == ("Verified against the temporal facts:\n"
                              "- Google acquired Android Inc.: 2005-07.\n"
                              "- Google acquired Android Inc. (2005-07) was before T-Mobile G1 was released (2008-10-22).")
    assert [c["verified"] for c in result.claims] == [True, True]


def test_one_failed_claim_falls_back_to_the_cited_facts_with_their_status(monkeypatch):
    claims = [{"kind": "date", "fact": "f-acq", "date": "2005-07-12"}]
    verify = {"all_verified": False, "facts": {"f-acq": ACQ}, "verdicts": [
        {"index": 0, "verified": False, "reason": "not entailed: the evidence gives 2005-07"}]}
    _services(monkeypatch, [ACQ, G1], claims, verify)
    result = tkg_agent._do_tkg(TKGRequest(query="q", branch_id="b1"))
    assert "2005-07-12" not in result.summary                         # the model's unverified date never appears
    assert "- Google acquired Android Inc. (2005-07)" in result.summary
    assert "claim 1: not verified (not entailed: the evidence gives 2005-07)" in result.summary
    assert result.claims == [{"claim": claims[0], "verified": False,
                              "reason": "not entailed: the evidence gives 2005-07"}]


def test_module07_down_lists_the_facts_as_recorded(monkeypatch):
    sent = _services(monkeypatch, [ACQ, G1], [], {}, infer_status=503)
    result = tkg_agent._do_tkg(TKGRequest(query="q", branch_id="b1"))
    assert result.summary.startswith("Temporal facts as recorded")
    assert "- T-Mobile G1 was released (2008-10-22)" in result.summary and result.claims == []
    assert not any(url.endswith("/personal/temporal/verify") for url, _ in sent)


def test_evidence_is_ranked_by_the_reranker_within_the_budget(monkeypatch):
    facts = [{"fact_id": f"f{i}", "entity": f"E{i}", "relation": "did", "value": "x", "when": "2001"}
             for i in range(5)]
    monkeypatch.setattr(grounded_claims, "EVIDENCE_BUDGET", 2)
    sent = _services(monkeypatch, facts, [], {}, rerank=[{"id": "f3", "score": 0.9}, {"id": "f1", "score": 0.5}])
    tkg_agent._do_tkg(TKGRequest(query="q", branch_id="b1"))
    assert _post(sent, "/personal/search/rerank")["top_k"] == 2
    content = _post(sent, "/inference/infer")["data"]["messages"][1]["content"]
    assert content.index("[f3]") < content.index("[f1]") and "[f0]" not in content


def test_possible_orderings_are_worded_as_possible():
    claim = {"kind": "ordering", "subject": "a", "object": "b", "relations": ["after"], "wording": "possible"}
    facts = {"a": {"entity": "Staff", "relation": "joined", "value": "Google", "when": "2005-07"}, "b": ACQ}
    line = grounded_claims._render_claim(claim, {"relations": ["before", "after", "equals"]}, facts)
    assert line == "Staff joined Google (2005-07) was possibly after Google acquired Android Inc. (2005-07)."


# ── a fact id where a fact id belongs; a fact's only date ───────────────────

GALAXY = {"fact_id": "f-gs", "entity": "Samsung", "relation": "released", "value": "Galaxy S phone",
          "time_start": "2010-06-01T00:00:00+08:00", "time_end": None, "when": "2010-06 (no end date stated)"}
UNTIL = {"fact_id": "f-until", "entity": "Android", "relation": "milestone", "value": "x",
         "time_start": None, "time_end": "2014-12-31T23:59:59+08:00", "when": "until 2014 (no start date stated)"}
RANGE = {"fact_id": "f-barra", "entity": "Hugo Barra", "relation": "served_as", "value": "spokesperson",
         "time_start": "2008-01-01T00:00:00+08:00", "time_end": "2013-12-31T23:59:59+08:00", "when": "2008 to 2013"}


def test_claims_may_only_cite_this_questions_evidence():
    schema = grounded_claims.claims_schema(["f-gs", "f-barra"])
    props = schema["properties"]["claims"]["items"]["properties"]
    assert all(props[k] == {"type": "string", "enum": ["f-gs", "f-barra"]} for k in ("fact", "subject", "object"))
    assert grounded_claims.CLAIMS_SCHEMA["properties"]["claims"]["items"]["properties"]["fact"] == {"type": "string"}
    assert "never a year or a date" in grounded_claims._SYSTEM and 'endpoint "start"' in grounded_claims._SYSTEM


def test_a_whole_date_on_a_fact_with_only_a_start_is_claimed_for_its_start():
    claims = [{"kind": "date", "fact": "f-gs", "date": "2010-06", "endpoint": "whole"},
              {"kind": "date", "fact": "f-until", "date": "2014"},
              {"kind": "date", "fact": "f-barra", "date": "2008", "endpoint": "whole"},
              {"kind": "date", "fact": "f-barra", "date": "2008", "endpoint": "start"}]
    out = grounded_claims.normalise(claims, [GALAXY, UNTIL, RANGE])
    assert [c.get("endpoint") for c in out] == ["start", "end", "whole", "start"]
    assert claims[0]["endpoint"] == "whole"                              # the model's own claim is left as it was
    self_order = [{"kind": "ordering", "subject": "f-barra", "object": "f-barra", "relations": ["during"]},
                  {"kind": "ordering", "subject": "f-gs", "object": "f-barra", "relations": ["during"]}]
    assert grounded_claims.normalise(self_order, [GALAXY, RANGE]) == self_order[1:]
    assert "only when the question asks how events are ordered" in grounded_claims._SYSTEM


def test_the_normalised_claim_is_what_module_25_verifies_and_reads_as_a_one_off_event(monkeypatch):
    claims = [{"kind": "date", "fact": "f-gs", "date": "2010-06", "endpoint": "whole"}]
    shown = {k: GALAXY[k] for k in ("fact_id", "entity", "relation", "value", "when")}   # as Module 25 returns it
    verify = {"all_verified": True, "facts": {"f-gs": shown}, "verdicts": [{"index": 0, "verified": True, "reason": None}]}
    sent = _services(monkeypatch, [GALAXY], claims, verify)
    result = tkg_agent._do_tkg(TKGRequest(query="What happened in 2010?", branch_id="b1"))
    assert _post(sent, "/personal/temporal/verify")["claims"] == [
        {"kind": "date", "fact": "f-gs", "date": "2010-06", "endpoint": "start"}]
    assert result.summary == "Verified against the temporal facts:\n- Samsung released Galaxy S phone: 2010-06."
    began = grounded_claims._render_claim({"kind": "date", "fact": "f-barra", "date": "2008", "endpoint": "start"},
                                          {}, {"f-barra": RANGE}, {"f-barra": None})
    assert began == "Hugo Barra served_as spokesperson began: 2008."         # a span that has an end does begin


def test_slow_claims_fall_back_to_the_facts_as_recorded_inside_the_pipelines_wait(monkeypatch):
    """Module 07 past GROUNDED_CLAIMS_TIMEOUT: the TKG agent still answers, with the facts as recorded."""
    timeouts = []

    class Client:
        def __init__(self, *a, timeout=None, **kw):
            timeouts.append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            if url.endswith("/personal/search/tkg"):
                return _Response({"subgraph_edges": [], "temporal_facts": [ACQ, G1], "complete": True})
            if url.endswith("/inference/infer"):
                raise httpx.ReadTimeout("Module 07 is busy")
            return _Response({"results": []})

    monkeypatch.setattr(tkg_agent.httpx, "Client", Client)
    monkeypatch.setattr(grounded_claims.httpx, "Client", Client)
    result = tkg_agent._do_tkg(TKGRequest(query="q", branch_id="b1"))
    assert grounded_claims.CLAIMS_TIMEOUT in timeouts and grounded_claims.CLAIMS_TIMEOUT < 90
    assert result.summary.startswith("Temporal facts as recorded")
    assert "- Google acquired Android Inc. (2005-07)" in result.summary and result.claims == []
