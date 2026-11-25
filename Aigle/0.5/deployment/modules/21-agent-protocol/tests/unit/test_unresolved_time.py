"""
TMP-7: when Module 18 cannot read a question's time condition (Module 25's `parse` timed out, failed, or
cannot read the date) the plan is UNRESOLVED, and the pipeline answers nothing: no agent is asked, no LLM
writes an answer, so no unfiltered fact is ever presented as the answer to a temporal question.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_unresolved_time.py -v
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import pipeline
from models import QueryPlan


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


def test_module18s_unresolved_time_condition_is_kept_on_the_plan(monkeypatch):
    plan = _module18_plan(monkeypatch, {"modes": ["TKG"], "temporal_status": "UNRESOLVED",
                                        "temporal_reason": "parse_failure", "time_from": None, "time_to": None})
    assert (plan.temporal_status, plan.temporal_reason, plan.time_range) == ("UNRESOLVED", "parse_failure", None)


def test_an_unresolved_question_is_not_answered_and_nothing_is_retrieved(monkeypatch):
    async def plan(question, inference_url):
        return QueryPlan(modes={"vector", "keyword", "tkg"}, temporal_status="UNRESOLVED",
                         temporal_reason="parse_failure")

    async def never(*a, **kw):
        raise AssertionError("nothing may be retrieved or generated for an unresolved time condition")

    monkeypatch.setattr(pipeline, "build_query_plan", plan)
    monkeypatch.setattr(pipeline, "_call_agent", never)
    result = asyncio.run(pipeline.run_rag_pipeline("what happened after 2011", user_id="u1", branch_id="u1"))
    assert (result.answer, result.sources, result.temporal_claims, result.graph_context) == (None, [], [], "")
    assert (result.temporal_status, result.temporal_reason, result.temporal_constraint_applied) == (
        "UNRESOLVED", "parse_failure", False)
