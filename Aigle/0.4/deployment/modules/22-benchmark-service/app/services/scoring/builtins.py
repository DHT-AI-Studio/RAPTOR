"""Built-in scoring strategies.

Each function is registered by name and reads its configuration from
``ctx.params``. To add a new strategy, write a function here (or anywhere that
gets imported) and decorate it with ``@register_scorer("name")`` — the schema
model and run manager pick it up automatically.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.services import judge
from app.services.scoring.registry import ScoringContext, register_scorer

logger = logging.getLogger(__name__)


def _keywords(ctx: ScoringContext) -> Optional[List[str]]:
    return ctx.params.get("keywords") or ctx.expected_keywords


# ── Lexical / overlap ────────────────────────────────────────────────

@register_scorer("keyword_match")
async def keyword_match(ctx: ScoringContext) -> float:
    """Fraction of expected keywords found in the output (case-insensitive)."""
    kws = _keywords(ctx)
    if not kws:
        return 0.0
    hay = ctx.output.lower()
    return sum(1 for kw in kws if kw.lower() in hay) / len(kws)


@register_scorer("contains_all")
async def contains_all(ctx: ScoringContext) -> float:
    """1.0 if every expected keyword appears, else 0.0."""
    kws = _keywords(ctx)
    if not kws:
        return 0.0
    hay = ctx.output.lower()
    return 1.0 if all(kw.lower() in hay for kw in kws) else 0.0


@register_scorer("contains_any")
async def contains_any(ctx: ScoringContext) -> float:
    """1.0 if any expected keyword appears, else 0.0."""
    kws = _keywords(ctx)
    if not kws:
        return 0.0
    hay = ctx.output.lower()
    return 1.0 if any(kw.lower() in hay for kw in kws) else 0.0


@register_scorer("exact_match")
async def exact_match(ctx: ScoringContext) -> float:
    """1.0 if the output equals the expected answer (normalized), else 0.0."""
    expected = ctx.params.get("expected")
    if expected is None:
        expected = ctx.expected_answer
    if expected is None:
        return 0.0
    a, b = ctx.output, str(expected)
    if ctx.params.get("strip", True):
        a, b = a.strip(), b.strip()
    if not ctx.params.get("case_sensitive", False):
        a, b = a.lower(), b.lower()
    return 1.0 if a == b else 0.0


# ── Structural ───────────────────────────────────────────────────────

@register_scorer("regex_match")
async def regex_match(ctx: ScoringContext) -> float:
    """1.0 if the regex matches the output, else 0.0."""
    pattern = ctx.params.get("pattern")
    if not pattern:
        return 0.0
    try:
        return 1.0 if re.search(pattern, ctx.output) else 0.0
    except re.error as exc:
        logger.warning("invalid regex %r: %s", pattern, exc)
        return 0.0


# ── Numeric ──────────────────────────────────────────────────────────

@register_scorer("numeric_tolerance")
async def numeric_tolerance(ctx: ScoringContext) -> float:
    """1.0 if the first number in the output is within tolerance of expected."""
    expected = ctx.params.get("expected")
    if expected is None:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", ctx.output)
    if not match:
        return 0.0
    tol = float(ctx.params.get("tolerance", 0.0))
    return 1.0 if abs(float(match.group()) - float(expected)) <= tol else 0.0


# ── Performance ──────────────────────────────────────────────────────

@register_scorer("latency_threshold")
async def latency_threshold(ctx: ScoringContext) -> float:
    """1.0 if response latency is within max_ms, else 0.0."""
    max_ms = ctx.params.get("max_ms")
    if max_ms is None:
        return 0.0
    return 1.0 if ctx.latency_ms <= float(max_ms) else 0.0


# ── Semantic ─────────────────────────────────────────────────────────

@register_scorer("cosine_similarity")
async def cosine_similarity(ctx: ScoringContext) -> float:
    """Cosine similarity between output and expected answer embeddings (Ollama bge-m3)."""
    expected = ctx.params.get("expected_answer") or ctx.expected_answer
    if not expected or not ctx.output:
        return 0.0
    try:
        return await judge.embed_and_compare(ctx.output, expected)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cosine_similarity embedding failed: %s", exc)
        return 0.0


# ── Qualitative (open-ended) ─────────────────────────────────────────

@register_scorer("llm_judge")
async def llm_judge(ctx: ScoringContext) -> float:
    """Grade the output against a free-text rubric via the LLM judge (Module 07)."""
    return await judge.evaluate(
        rubric=ctx.params.get("rubric") or "Score the quality of the output.",
        output=ctx.output,
        expected_answer=ctx.expected_answer,
        score_range=ctx.score_range,
        model=ctx.judge_model,
    )
