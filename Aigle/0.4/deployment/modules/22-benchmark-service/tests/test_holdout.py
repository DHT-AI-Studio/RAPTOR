"""Tests for held-out eval split (anti-overfit)."""
from __future__ import annotations

from app.services.autotune import orchestrator
from app.services.schema_store import partition_cases


def _cases(n):
    return [{"id": f"q{i}", "input": {"inputs": str(i)}} for i in range(n)]


def test_partition_splits_dev_and_holdout():
    dev, holdout = partition_cases(_cases(10), holdout_ratio=0.3, seed=1)
    assert len(holdout) == 3
    assert len(dev) == 7
    ids = {c["id"] for c in dev} | {c["id"] for c in holdout}
    assert ids == {f"q{i}" for i in range(10)}          # no case lost
    assert not ({c["id"] for c in dev} & {c["id"] for c in holdout})  # no overlap


def test_partition_is_deterministic():
    a = partition_cases(_cases(10), seed=42)
    b = partition_cases(_cases(10), seed=42)
    assert [c["id"] for c in a[0]] == [c["id"] for c in b[0]]


def test_partition_keeps_at_least_one_each():
    dev, holdout = partition_cases(_cases(2), holdout_ratio=0.9)
    assert len(dev) == 1 and len(holdout) == 1


def test_partition_too_few_cases_returns_none():
    assert partition_cases(_cases(1)) is None
    assert partition_cases([]) is None


async def test_heldout_eval_reuses_best_checkpoint(monkeypatch):
    async def fake_get_run(run_id):
        assert run_id == "best-run"
        return {"config_override": {"model_path": "/best", "lora_r": 16, "_reason": "x"}}

    captured = {}

    async def fake_evaluate(schema_id, model_path, knobs, reason=None):
        captured.update(schema_id=schema_id, model_path=model_path, knobs=knobs)
        return 0.75, "holdout-run"

    monkeypatch.setattr(orchestrator.run_manager, "get_run", fake_get_run)
    monkeypatch.setattr(orchestrator, "_evaluate", fake_evaluate)

    score = await orchestrator._heldout_eval("holdout-schema", "best-run", {"lora_r": 16})
    assert score == 0.75
    assert captured["schema_id"] == "holdout-schema"      # scored on the held-out schema
    assert captured["model_path"] == "/best"              # reused the winning checkpoint


async def test_heldout_eval_none_when_no_model_path(monkeypatch):
    async def fake_get_run(run_id):
        return {"config_override": {"lora_r": 16}}  # no model_path
    monkeypatch.setattr(orchestrator.run_manager, "get_run", fake_get_run)
    assert await orchestrator._heldout_eval("s", "r", {}) is None
