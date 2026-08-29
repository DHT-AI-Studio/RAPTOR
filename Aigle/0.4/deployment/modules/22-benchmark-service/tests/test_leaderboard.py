"""Tests for the leaderboard (top-N runs by aggregate score)."""
from __future__ import annotations

import pytest

from app.services import run_manager


class _FakePool:
    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, query, *args):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self.pool = _FakePool(rows)


async def test_leaderboard_maps_and_ranks(monkeypatch):
    # Rows come back already score-ordered (ORDER BY in SQL).
    rows = [
        {"id": "r1", "aggregate_score": 0.9, "config_override": {"model_path": "A"},
         "scores_per_dimension": {"q": 0.9}, "created_at": None},
        {"id": "r2", "aggregate_score": 0.7, "config_override": {"model_path": "B"},
         "scores_per_dimension": {"q": 0.7}, "created_at": None},
    ]
    monkeypatch.setattr(run_manager, "db", _FakeDB(rows))
    out = await run_manager.leaderboard("11111111-1111-1111-1111-111111111111", limit=5)

    assert [o["rank"] for o in out] == [1, 2]
    assert out[0]["run_id"] == "r1"
    assert out[0]["aggregate_score"] == 0.9
    assert out[0]["config_override"] == {"model_path": "A"}


async def test_leaderboard_invalid_schema_id_returns_empty():
    assert await run_manager.leaderboard("not-a-uuid") == []
