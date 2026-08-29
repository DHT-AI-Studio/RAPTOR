"""Pluggable scoring-strategy registry.

A scoring method is just an async function registered by name. New strategies
are added by writing a function and decorating it with ``@register_scorer``;
no changes to the dispatch, the schema model, or the run manager are needed —
this is what makes the benchmark service's scoring open-ended rather than a
fixed enum of a handful of methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence


@dataclass
class ScoringContext:
    """Everything a scorer needs to grade one test-case output.

    ``params`` carries the dimension's method-specific configuration
    (e.g. ``max_ms``, ``pattern``, ``rubric``, ``expected``, ``tolerance``).
    """

    output: str
    latency_ms: float
    expected_keywords: Optional[List[str]] = None
    expected_answer: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    score_range: Sequence[float] = (1, 5)
    judge_model: Optional[str] = None


# A scorer takes a context and returns a normalized score in [0, 1].
Scorer = Callable[[ScoringContext], Awaitable[float]]

_REGISTRY: Dict[str, Scorer] = {}


def register_scorer(name: str) -> Callable[[Scorer], Scorer]:
    """Decorator: register ``fn`` as the scorer for ``name``."""

    def _decorator(fn: Scorer) -> Scorer:
        if name in _REGISTRY:
            raise ValueError(f"scorer '{name}' is already registered")
        _REGISTRY[name] = fn
        return fn

    return _decorator


def get_scorer(name: str) -> Optional[Scorer]:
    return _REGISTRY.get(name)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def list_scorers() -> List[str]:
    return sorted(_REGISTRY)
