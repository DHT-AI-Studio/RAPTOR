"""Scoring-strategy registry package.

Importing this package registers all built-in scorers. Consumers use
``get_scorer`` / ``is_registered`` / ``list_scorers`` to dispatch, and
``register_scorer`` to add new strategies.
"""
from app.services.scoring.registry import (
    ScoringContext,
    get_scorer,
    is_registered,
    list_scorers,
    register_scorer,
)

# Importing builtins registers the shipped scorers as a side effect.
from app.services.scoring import builtins as _builtins  # noqa: F401,E402

__all__ = [
    "ScoringContext",
    "get_scorer",
    "is_registered",
    "list_scorers",
    "register_scorer",
]
