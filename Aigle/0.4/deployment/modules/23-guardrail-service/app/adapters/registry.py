"""Guard-adapter registry — a lightweight, self-registering lookup table.

Adding a new guard model = a new file under app/adapters/models/ that calls
register() at import time, plus one import line in app/adapters/__init__.py.
No file outside app/adapters/ needs to change.
"""
from __future__ import annotations

from app.adapters.base import BaseGuardAdapter


class UnknownGuardModelError(Exception):
    """Raised when a configured model name doesn't match any registered
    adapter — fails fast instead of silently misclassifying it as some
    other family's model."""


_adapters: dict[str, BaseGuardAdapter] = {}
_matchers: list[tuple[str, list[str]]] = []   # (family, substrings), checked in registration order


def register(adapter: BaseGuardAdapter, match_substrings: list[str]) -> None:
    _adapters[adapter.family] = adapter
    _matchers.append((adapter.family, match_substrings))


def get_adapter(model: str) -> BaseGuardAdapter:
    m = model.lower()
    for family, substrings in _matchers:
        if any(s in m for s in substrings):
            return _adapters[family]
    raise UnknownGuardModelError(
        f"No guard adapter matches model '{model}'. Add one under "
        f"app/adapters/models/ and register it in app/adapters/__init__.py."
    )
