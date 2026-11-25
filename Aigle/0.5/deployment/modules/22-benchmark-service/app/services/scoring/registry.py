"""Pluggable scoring-strategy registry.

A scoring method is just an async function registered by name. New strategies
are added by writing a function and decorating it with ``@register_scorer``;
no changes to the dispatch, the schema model, or the run manager are needed —
this is what makes the benchmark service's scoring open-ended rather than a
fixed enum of a handful of methods.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringContext:
    """Everything a scorer needs to grade one test-case output.

    ``params`` carries the dimension's method-specific configuration
    (e.g. ``max_ms``, ``pattern``, ``rubric``, ``expected``, ``tolerance``).
    """

    output: str
    latency_ms: float
    expected_keywords: list[str] | None = None
    expected_answer: str | None = None
    labels: list[str] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    score_range: Sequence[float] = (1, 5)
    judge_model: str | None = None


# A scorer takes a context and returns a normalized score in [0, 1].
Scorer = Callable[[ScoringContext], Awaitable[float]]

_REGISTRY: dict[str, Scorer] = {}


def register_scorer(name: str) -> Callable[[Scorer], Scorer]:
    """Decorator: register ``fn`` as the scorer for ``name``."""

    def _decorator(fn: Scorer) -> Scorer:
        if name in _REGISTRY:
            raise ValueError(f"scorer '{name}' is already registered")
        _REGISTRY[name] = fn
        return fn

    return _decorator


def get_scorer(name: str) -> Scorer | None:
    return _REGISTRY.get(name)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def list_scorers() -> list[str]:
    return sorted(_REGISTRY)
