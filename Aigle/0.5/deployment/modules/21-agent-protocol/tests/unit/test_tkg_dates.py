"""
D8: the TKG agent's summary states dates at the precision Module 25 gives them (`when`), not the
padded days of valid_from / valid_to ("2024-07" stored as 2024-07-01 became "1 July 2024").

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_tkg_dates.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import tkg_agent
from models import TKGRequest


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _fake_httpx(monkeypatch, facts):
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
                return _Response({"subgraph_edges": [], "temporal_facts": facts})
            if url.endswith("/v1/chat/completions"):
                return _Response({"choices": [{"message": {"content": "summary"}}]})
            return _Response({})

    monkeypatch.setattr(tkg_agent.httpx, "Client", Client)
    monkeypatch.setenv("TKG_GROUNDED_ANSWERS", "false")      # the free-text summary path (P17 keeps it as a fallback)
    return sent


def test_summary_prompt_uses_when_not_the_padded_day(monkeypatch):
    sent = _fake_httpx(monkeypatch, [
        {"entity": "Samsung", "relation": "strike", "value": "union", "time_start": "2024-07-01",
         "when": "2024-07 (no end date stated)", "precision": "month"},
        {"entity": "Hugo Barra", "relation": "served_as", "value": "spokesperson",
         "time_start": "2008-01-01", "time_end": "2013-12-31", "when": "2008 to 2013"}])
    result = tkg_agent._do_tkg(TKGRequest(query="What happened?", branch_id="b1"))
    assert [t.get("when") for t in result.triples] == ["2024-07 (no end date stated)", "2008 to 2013"]
    assert result.triples[0]["valid_from"] == "2024-07-01"                    # still there for v1 readers
    prompt = next(b for url, b in sent if url.endswith("/v1/chat/completions"))["messages"][0]["content"]
    assert "Samsung --[strike]--> union (2024-07 (no end date stated))" in prompt
    assert "2024-07-01" not in prompt and "2013-12-31" not in prompt
    assert "not 1 July" in prompt                                             # told not to add a day


def test_older_module25_without_when_keeps_the_padded_days(monkeypatch):
    sent = _fake_httpx(monkeypatch, [{"entity": "Samsung", "relation": "released", "value": "Galaxy S",
                                      "time_start": "2010-06-01", "time_end": None}])
    result = tkg_agent._do_tkg(TKGRequest(query="q", branch_id="b1"))
    assert "when" not in result.triples[0]
    prompt = next(b for url, b in sent if url.endswith("/v1/chat/completions"))["messages"][0]["content"]
    assert "Samsung --[released]--> Galaxy S (2010-06-01 ~ None)" in prompt
