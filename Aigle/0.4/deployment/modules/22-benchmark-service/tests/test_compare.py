"""Unit tests for run comparison (BM-8) with two fixed-score fixture runs."""
from __future__ import annotations

import pytest

from app.services import run_manager
from app.services.run_manager import CompareError


def _run(run_id, schema_id, status, aggregate, dims, cases):
    return {
        "id": run_id,
        "schema_id": schema_id,
        "status": status,
        "aggregate_score": aggregate,
        "scores_per_dimension": dims,
        "scores_per_case": cases,
        "created_at": None,
    }


class _FakePool:
    async def fetchrow(self, *args, **kwargs):
        return {"name": "Fixture Schema"}


class _FakeDB:
    pool = _FakePool()


def _install(monkeypatch, run_a, run_b):
    async def fake_fetch(run_id):
        return {"a": run_a, "b": run_b}.get(run_id)

    monkeypatch.setattr(run_manager, "_fetch_full_run", fake_fetch)
    monkeypatch.setattr(run_manager, "db", _FakeDB())


async def test_compare_known_scores(monkeypatch):
    run_a = _run("a", "s1", "completed", 0.60,
                 {"relevance": 0.5, "latency": 1.0},
                 [{"case_id": "tc1", "aggregate": 0.4}, {"case_id": "tc2", "aggregate": 0.8}])
    run_b = _run("b", "s1", "completed", 0.75,
                 {"relevance": 0.7, "latency": 1.0},
                 [{"case_id": "tc1", "aggregate": 0.6}, {"case_id": "tc2", "aggregate": 0.9}])
    _install(monkeypatch, run_a, run_b)

    result = await run_manager.compare_runs("a", "b")

    assert result.delta_aggregate == pytest.approx(0.15)
    dims = {d.name: d for d in result.dimensions}
    assert dims["relevance"].delta == pytest.approx(0.2)
    assert dims["latency"].delta == pytest.approx(0.0)
    cases = {c.case_id: c for c in result.cases}
    assert cases["tc1"].delta == pytest.approx(0.2)
    assert cases["tc2"].delta == pytest.approx(0.1)
    assert result.metadata["schema_id"] == "s1"


async def test_compare_missing_run_404(monkeypatch):
    _install(monkeypatch, None, None)
    with pytest.raises(CompareError) as exc:
        await run_manager.compare_runs("a", "b")
    assert exc.value.status_code == 404


async def test_compare_cross_schema_422(monkeypatch):
    run_a = _run("a", "s1", "completed", 0.6, {}, [])
    run_b = _run("b", "s2", "completed", 0.7, {}, [])
    _install(monkeypatch, run_a, run_b)
    with pytest.raises(CompareError) as exc:
        await run_manager.compare_runs("a", "b")
    assert exc.value.status_code == 422


async def test_compare_incomplete_run_409(monkeypatch):
    run_a = _run("a", "s1", "completed", 0.6, {}, [])
    run_b = _run("b", "s1", "running", None, None, None)
    _install(monkeypatch, run_a, run_b)
    with pytest.raises(CompareError) as exc:
        await run_manager.compare_runs("a", "b")
    assert exc.value.status_code == 409


async def test_pairwise_summary_with_position_debias(monkeypatch):
    # tc1: B is clearly better; tc2: both equal → tie.
    row_a = {"scores_per_case": [
        {"case_id": "tc1", "output": "bad"}, {"case_id": "tc2", "output": "GOOD"}]}
    row_b = {"scores_per_case": [
        {"case_id": "tc1", "output": "GOOD answer"}, {"case_id": "tc2", "output": "GOOD"}]}

    async def fake_inputs(schema_id):
        return {"tc1": "q1", "tc2": "q2"}

    async def fake_pairwise(task, a, b, criteria=None, model=None):
        a_good, b_good = "GOOD" in a, "GOOD" in b
        if a_good and not b_good:
            return "A"
        if b_good and not a_good:
            return "B"
        return "TIE"

    monkeypatch.setattr(run_manager, "_case_input_map", fake_inputs)
    monkeypatch.setattr(run_manager.judge, "pairwise", fake_pairwise)

    summary = await run_manager._pairwise_summary(row_a, row_b, "s1")
    assert summary["b_wins"] == 1
    assert summary["a_wins"] == 0
    assert summary["ties"] == 1
    assert summary["total"] == 2
    assert summary["b_win_rate"] == 0.5


async def test_debiased_pairwise_disagreement_is_tie(monkeypatch):
    # Judge always says the FIRST shown answer is better → pure position bias.
    async def biased(task, a, b, criteria=None, model=None):
        return "A"

    monkeypatch.setattr(run_manager.judge, "pairwise", biased)
    verdict = await run_manager._debiased_pairwise("q", "out_a", "out_b")
    assert verdict == "TIE"  # swap disagreement → tie
