"""Tests for the pluggable scorer registry + new built-in strategies."""
from __future__ import annotations

import pytest

from app.models.schema import ScoringDimension, ScoringSchema, TestCase
from app.services import scorer
from app.services.scoring import is_registered, list_scorers, register_scorer
from app.services.scoring.registry import ScoringContext


def _tc(**kw):
    base = {"id": "tc1", "input": {"message": "hi"}}
    base.update(kw)
    return TestCase(**base)


def test_builtins_registered():
    for name in ["keyword_match", "latency_threshold", "regex_match",
                 "cosine_similarity", "llm_judge", "exact_match",
                 "contains_all", "contains_any", "numeric_tolerance"]:
        assert is_registered(name), f"{name} not registered"


def test_unknown_method_rejected_by_model():
    with pytest.raises(ValueError):
        ScoringDimension(name="x", weight=1.0, method="does_not_exist")


def test_legacy_flat_params_fold_into_params():
    dim = ScoringDimension(name="lat", weight=1.0, method="latency_threshold", max_ms=5000)
    assert dim.params["max_ms"] == 5000


async def test_exact_match_via_params():
    schema = ScoringSchema(dimensions=[
        ScoringDimension(name="em", weight=1.0, method="exact_match",
                         params={"expected": "Paris"}),
    ])
    hit = await scorer.score(_tc(), "  paris ", 1.0, schema)
    miss = await scorer.score(_tc(), "London", 1.0, schema)
    assert hit["per_dimension"]["em"] == 1.0
    assert miss["per_dimension"]["em"] == 0.0


async def test_numeric_tolerance():
    schema = ScoringSchema(dimensions=[
        ScoringDimension(name="num", weight=1.0, method="numeric_tolerance",
                         params={"expected": 42, "tolerance": 1}),
    ])
    ok = await scorer.score(_tc(), "the answer is 42.5", 1.0, schema)
    bad = await scorer.score(_tc(), "the answer is 50", 1.0, schema)
    assert ok["per_dimension"]["num"] == 1.0
    assert bad["per_dimension"]["num"] == 0.0


async def test_custom_scorer_registration_and_use():
    @register_scorer("shouty")
    async def shouty(ctx: ScoringContext) -> float:
        return 1.0 if ctx.output.isupper() else 0.0

    assert "shouty" in list_scorers()
    schema = ScoringSchema(dimensions=[
        ScoringDimension(name="s", weight=1.0, method="shouty"),
    ])
    yes = await scorer.score(_tc(), "HELLO", 1.0, schema)
    no = await scorer.score(_tc(), "hello", 1.0, schema)
    assert yes["per_dimension"]["s"] == 1.0
    assert no["per_dimension"]["s"] == 0.0
