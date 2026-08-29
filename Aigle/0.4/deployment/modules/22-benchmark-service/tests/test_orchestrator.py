"""Tests for the optimization loop (AUTOTUNE Phase B).

The loop is exercised with fakes for training (Module 16) and evaluation
(run_manager), so no GPU / DB is touched — we validate the *control logic*:
budget cap, early-stop, failure tolerance, and cooperative stop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from app.models.experiment import Plan
from app.services.autotune import orchestrator
from app.services.autotune.training_client import TrainingError

EXP_ID = "11111111-1111-1111-1111-111111111111"
SCHEMA_ID = "22222222-2222-2222-2222-222222222222"


def _plan_dict() -> Dict[str, Any]:
    return Plan(
        task_type="instruction",
        select_multiple_gpus=False,
        vram_budget_gb=12,
        base_training_config={"model_name_or_path": "/m", "lora_config": {"r": 8}},
        dataset_config={"dataset_name_or_path": "ds", "column_mapping": {"messages": "messages"}},
        search_space={"learning_rate": {"type": "float", "min": 1e-5, "max": 5e-4, "log": True}},
        eval_schema_id=SCHEMA_ID,
    ).model_dump(mode="json")


class ScriptedProposer:
    """Feed a fixed sequence of configs; optionally stop after N proposals."""

    def __init__(self, configs: List[Dict[str, Any]], stop_after: Optional[int] = None) -> None:
        self.configs = configs
        self.stop_after = stop_after
        self.i = 0

    async def propose(self, plan: Plan, history: List[Dict[str, Any]]):
        if self.stop_after is not None and self.i >= self.stop_after:
            return {}, True, None
        cfg = self.configs[self.i % len(self.configs)]
        self.i += 1
        return dict(cfg), False, None


class Harness:
    """Stateful fakes for experiment_store + run_manager + training_client."""

    def __init__(self, budget: Dict[str, Any], scores: List[Optional[float]],
                 fail_on: Optional[set] = None, stop_on_call: Optional[int] = None,
                 baseline: Optional[List[Dict[str, Any]]] = None) -> None:
        self.baseline = baseline or []
        self.record = {
            "experiment_id": EXP_ID, "plan": _plan_dict(), "budget": budget,
            "iterations_done": 0, "status": "running", "best_run_id": None,
            "best_score": None, "best_config": None, "eval_schema_id": SCHEMA_ID,
        }
        self.scores = list(scores)
        self.fail_on = fail_on or set()        # 1-based proposal index to fail training
        self.stop_on_call = stop_on_call       # flip status to 'stopped' on Nth get_experiment
        self.status_calls: List[str] = []
        self.best_updates: List[tuple] = []
        self.get_calls = 0
        self.train_calls = 0
        self.run_counter = 0

    # experiment_store ---------------------------------------------------
    async def get_experiment(self, exp_id: str):
        self.get_calls += 1
        if self.stop_on_call is not None and self.get_calls >= self.stop_on_call:
            self.record["status"] = "stopped"
        return dict(self.record)

    async def update_status(self, exp_id, status, error=None):
        self.status_calls.append(status.value if hasattr(status, "value") else status)
        self.record["status"] = status.value if hasattr(status, "value") else status

    async def set_iterations(self, exp_id, n):
        self.record["iterations_done"] = n

    async def update_best(self, exp_id, run_id, score, config):
        self.best_updates.append((run_id, score, config))
        self.record["best_score"] = score

    # run_manager --------------------------------------------------------
    async def leaderboard(self, schema_id, limit=5):
        return self.baseline

    async def create_run(self, schema_id, config_override=None):
        self.run_counter += 1
        return {"run_id": f"run{self.run_counter}", "status": "queued"}

    async def execute_run(self, run_id):
        return None

    async def get_run(self, run_id):
        score = self.scores.pop(0) if self.scores else None
        status = "completed" if score is not None else "failed"
        return {"run_id": run_id, "status": status, "aggregate_score": score}

    # training_client ----------------------------------------------------
    async def train_and_wait(self, plan, knobs, minutes, on_job_id=None):
        self.train_calls += 1
        if on_job_id is not None:
            await on_job_id(f"job-{self.train_calls}")
        if self.train_calls in self.fail_on:
            raise TrainingError("boom")
        return f"/app/tmp/models/candidate-{self.train_calls}"


def _wire(monkeypatch, h: Harness):
    monkeypatch.setattr(orchestrator.experiment_store, "get_experiment", h.get_experiment)
    monkeypatch.setattr(orchestrator.experiment_store, "update_status", h.update_status)
    monkeypatch.setattr(orchestrator.experiment_store, "set_iterations", h.set_iterations)
    monkeypatch.setattr(orchestrator.experiment_store, "update_best", h.update_best)
    monkeypatch.setattr(orchestrator.run_manager, "leaderboard", h.leaderboard)
    monkeypatch.setattr(orchestrator.run_manager, "create_run", h.create_run)
    monkeypatch.setattr(orchestrator.run_manager, "execute_run", h.execute_run)
    monkeypatch.setattr(orchestrator.run_manager, "get_run", h.get_run)
    monkeypatch.setattr(orchestrator.training_client, "train_and_wait", h.train_and_wait)


async def test_loop_runs_until_max_experiments(monkeypatch):
    h = Harness(budget={"max_experiments": 3, "minutes_per_experiment": 1, "early_stop_patience": 10},
                scores=[0.5, 0.7, 0.6])
    _wire(monkeypatch, h)
    proposer = ScriptedProposer([{"learning_rate": 1e-4}])
    await orchestrator.run_experiment(EXP_ID, proposer)

    assert h.record["iterations_done"] == 3
    assert h.status_calls[-1] == "completed"
    assert h.record["best_score"] == 0.7          # iter 2 was best


async def test_early_stop_after_patience(monkeypatch):
    h = Harness(budget={"max_experiments": 10, "minutes_per_experiment": 1, "early_stop_patience": 2},
                scores=[0.8, 0.5, 0.5])
    _wire(monkeypatch, h)
    proposer = ScriptedProposer([{"learning_rate": 1e-4}])
    await orchestrator.run_experiment(EXP_ID, proposer)

    assert h.record["iterations_done"] == 3       # best, +2 non-improving → stop
    assert h.status_calls[-1] == "completed"
    assert h.record["best_score"] == 0.8


async def test_training_failure_is_tolerated(monkeypatch):
    # 2nd candidate's training fails → recorded as a miss, loop continues.
    h = Harness(budget={"max_experiments": 3, "minutes_per_experiment": 1, "early_stop_patience": 10},
                scores=[0.6, 0.9], fail_on={2})
    _wire(monkeypatch, h)
    proposer = ScriptedProposer([{"learning_rate": 1e-4}])
    await orchestrator.run_experiment(EXP_ID, proposer)

    assert h.record["iterations_done"] == 3
    assert h.train_calls == 3
    assert h.record["best_score"] == 0.9          # 3rd candidate wins
    assert h.status_calls[-1] == "completed"


async def test_proposer_convergence_stops_loop(monkeypatch):
    h = Harness(budget={"max_experiments": 10, "minutes_per_experiment": 1, "early_stop_patience": 10},
                scores=[0.6, 0.7])
    _wire(monkeypatch, h)
    proposer = ScriptedProposer([{"learning_rate": 1e-4}], stop_after=2)
    await orchestrator.run_experiment(EXP_ID, proposer)

    assert h.record["iterations_done"] == 2
    assert h.status_calls[-1] == "completed"


async def test_initial_best_persisted_from_baseline(monkeypatch):
    # A pre-existing baseline tops the leaderboard; no iteration beats it.
    # The experiment must still report that baseline as best (not null).
    baseline = [{"run_id": "base", "aggregate_score": 0.9,
                 "config_override": {"model_path": "/m", "lora_r": 32}}]
    h = Harness(budget={"max_experiments": 1, "minutes_per_experiment": 1, "early_stop_patience": 5},
                scores=[0.5], baseline=baseline)
    _wire(monkeypatch, h)
    proposer = ScriptedProposer([{"learning_rate": 1e-4}])
    await orchestrator.run_experiment(EXP_ID, proposer)

    assert h.record["best_score"] == 0.9          # baseline persisted, not None
    assert h.best_updates[0] == ("base", 0.9, {"lora_r": 32})  # model_path stripped
    assert h.record["iterations_done"] == 1


async def test_cooperative_stop(monkeypatch):
    # Flip to 'stopped' on the 3rd get_experiment (start + iter1 top + iter2 top).
    h = Harness(budget={"max_experiments": 10, "minutes_per_experiment": 1, "early_stop_patience": 10},
                scores=[0.6, 0.7, 0.8], stop_on_call=3)
    _wire(monkeypatch, h)
    proposer = ScriptedProposer([{"learning_rate": 1e-4}])
    await orchestrator.run_experiment(EXP_ID, proposer)

    assert h.record["iterations_done"] == 1       # only one iteration ran before stop
    assert "completed" not in h.status_calls      # stopped, not completed
