"""Scoring engine (BM-5).

Dispatches each scoring dimension through the pluggable scorer registry and
computes a weighted aggregate in [0, 1] — the benchmark equivalent of
AutoResearch's ``val_bpb``. New scoring strategies are added in
``app/services/scoring/`` without touching this file.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.models.schema import ScoringSchema, TestCase
from app.services.scoring import ScoringContext, get_scorer

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


async def score(
    test_case: TestCase,
    output: str,
    latency_ms: float,
    scoring_schema: ScoringSchema,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one test-case output; return {per_dimension, aggregate}."""
    per_dimension: Dict[str, float] = {}
    aggregate = 0.0

    for dim in scoring_schema.dimensions:
        scorer_fn = get_scorer(dim.method)
        if scorer_fn is None:
            logger.warning("no scorer registered for method '%s'", dim.method)
            raw = 0.0
        else:
            ctx = ScoringContext(
                output=output,
                latency_ms=latency_ms,
                expected_keywords=test_case.expected_keywords,
                expected_answer=test_case.expected_answer,
                params=dict(dim.params),
                score_range=scoring_schema.score_range,
                judge_model=judge_model,
            )
            raw = _clamp01(await scorer_fn(ctx))

        per_dimension[dim.name] = raw
        aggregate += dim.weight * raw

    return {"per_dimension": per_dimension, "aggregate": _clamp01(aggregate)}
