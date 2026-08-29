"""Unit tests for the scoring engine (BM-5). LLM judge + embeddings mocked."""
from __future__ import annotations

import pytest

from app.models.schema import ScoringDimension, ScoringMethod, ScoringSchema, TestCase
from app.services import judge, scorer


def _tc(**kw):
    base = {"id": "tc1", "input": {"message": "hi"}}
    base.update(kw)
    return TestCase(**base)


async def test_keyword_match_ratio():
    tc = _tc(expected_keywords=["alpha", "beta", "gamma"])
    schema = ScoringSchema(dimensions=[ScoringDimension(name="kw", weight=1.0,
                                                        method=ScoringMethod.keyword_match)])
    out = await scorer.score(tc, "ALPHA and beta appear", 10.0, schema)
    assert out["per_dimension"]["kw"] == pytest.approx(2 / 3)


async def test_keyword_match_no_keywords_is_zero():
    tc = _tc()
    schema = ScoringSchema(dimensions=[ScoringDimension(name="kw", weight=1.0,
                                                        method=ScoringMethod.keyword_match)])
    out = await scorer.score(tc, "whatever", 10.0, schema)
    assert out["per_dimension"]["kw"] == 0.0


async def test_latency_threshold_pass_and_fail():
    schema = ScoringSchema(dimensions=[ScoringDimension(name="lat", weight=1.0,
                                                        method=ScoringMethod.latency_threshold,
                                                        max_ms=5000)])
    assert (await scorer.score(_tc(), "x", 4000.0, schema))["per_dimension"]["lat"] == 1.0
    assert (await scorer.score(_tc(), "x", 6000.0, schema))["per_dimension"]["lat"] == 0.0


async def test_regex_match():
    schema = ScoringSchema(dimensions=[ScoringDimension(name="re", weight=1.0,
                                                        method=ScoringMethod.regex_match,
                                                        pattern=r"\d{3}")])
    assert (await scorer.score(_tc(), "id 123 ok", 1.0, schema))["per_dimension"]["re"] == 1.0
    assert (await scorer.score(_tc(), "no digits", 1.0, schema))["per_dimension"]["re"] == 0.0


async def test_cosine_similarity_mocked(monkeypatch):
    async def fake_compare(text_a, text_b, model=None):
        assert (text_a, text_b) == ("out", "exp")
        return 1.0

    monkeypatch.setattr(judge, "embed_and_compare", fake_compare)
    tc = _tc(expected_answer="exp")
    schema = ScoringSchema(dimensions=[ScoringDimension(name="cos", weight=1.0,
                                                        method=ScoringMethod.cosine_similarity)])
    out = await scorer.score(tc, "out", 1.0, schema)
    assert out["per_dimension"]["cos"] == pytest.approx(1.0)


async def test_llm_judge_mocked_and_weighted_aggregate(monkeypatch):
    async def fake_eval(**kwargs):
        return 1.0

    monkeypatch.setattr(judge, "evaluate", fake_eval)
    tc = _tc(expected_keywords=["hit"])
    schema = ScoringSchema(dimensions=[
        ScoringDimension(name="j", weight=0.5, method=ScoringMethod.llm_judge, rubric="grade"),
        ScoringDimension(name="lat", weight=0.3, method=ScoringMethod.latency_threshold, max_ms=100),
        ScoringDimension(name="kw", weight=0.2, method=ScoringMethod.keyword_match),
    ])
    out = await scorer.score(tc, "a hit here", 50.0, schema)
    # j=1.0*0.5 + lat=1.0*0.3 + kw=1.0*0.2 = 1.0
    assert out["aggregate"] == pytest.approx(1.0)
    assert set(out["per_dimension"]) == {"j", "lat", "kw"}
