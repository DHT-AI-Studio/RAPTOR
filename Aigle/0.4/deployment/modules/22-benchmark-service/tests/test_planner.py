"""Tests for the planner (AUTOTUNE Phase A): NL goal → Plan."""
from __future__ import annotations

import json

import pytest

from app.services.autotune import planner as planner_mod
from app.services.autotune.planner import (
    Planner,
    PlannerError,
    _read_dataset_rows,
    _row_to_case,
    sanitize_budget,
    sanitize_eval,
    sanitize_search_space,
    sanitize_training,
)


@pytest.fixture(autouse=True)
def _no_network_catalog(monkeypatch):
    """Default: empty local catalog (no Module 07 call) unless a test overrides."""
    async def _empty():
        return [], []
    monkeypatch.setattr(planner_mod, "_fetch_catalog", _empty)


def test_sanitize_clamps_and_drops():
    proposed = {
        "learning_rate": {"type": "float", "min": 1e-9, "max": 9.9, "log": True},  # out of bounds
        "lora_r": {"type": "categorical", "choices": [8, 16, 999]},                # 999 not allowed
        "max_epochs": {"type": "int", "min": 1, "max": 50},                        # capped at 8
        "danger_flag": {"type": "categorical", "choices": [True]},                 # unknown → dropped
    }
    ss = sanitize_search_space(proposed)
    assert set(ss) == {"learning_rate", "lora_r", "max_epochs"}   # danger_flag dropped
    assert ss["learning_rate"].min >= 1e-6 and ss["learning_rate"].max <= 2e-3
    assert ss["lora_r"].choices == [8, 16]                        # 999 removed
    assert ss["max_epochs"].max == 8                              # capped


def test_sanitize_drops_categorical_with_no_valid_choices():
    ss = sanitize_search_space({"lora_r": {"type": "categorical", "choices": [3, 5, 7]}})
    assert ss == {}


def test_sanitize_keeps_new_categorical_knobs():
    # target_modules (preset names) and lr_scheduler_type are whitelisted; invalid
    # choices are intersected away, valid ones kept.
    ss = sanitize_search_space({
        "target_modules": {"type": "categorical", "choices": ["attn", "attn_mlp", "bogus"]},
        "lr_scheduler_type": {"type": "categorical", "choices": ["cosine", "linear", "nope"]},
        "optimizer_type": {"type": "categorical", "choices": ["adamw", "lion", "sgd", "nope"]},
        "lora_variant": {"type": "categorical", "choices": ["plain", "rslora", "dora", "nope"]},
        "gradient_accumulation_steps": {"type": "categorical", "choices": [1, 2, 4, 8, 999]},
    })
    assert ss["target_modules"].choices == ["attn", "attn_mlp"]      # bogus dropped
    assert ss["lr_scheduler_type"].choices == ["cosine", "linear"]   # nope dropped
    assert ss["optimizer_type"].choices == ["adamw", "lion", "sgd"]  # nope dropped
    assert ss["lora_variant"].choices == ["plain", "rslora", "dora"]  # nope dropped
    assert ss["gradient_accumulation_steps"].choices == [1, 2, 4, 8]  # 999 dropped


def test_dataset_resolves_to_hub_id_and_cache():
    from app.services.autotune.planner import _hub_id, _resolve_dataset

    # sanitized dir name -> Hub id (first underscore becomes the slash)
    assert _hub_id("philschmid_dolly-15k-oai-style") == "philschmid/dolly-15k-oai-style"
    assert _hub_id("philschmid/dolly-15k-oai-style") == "philschmid/dolly-15k-oai-style"

    ds = [{"name": "philschmid_dolly-15k-oai-style",
           "path": "/app/tmp/datasets/philschmid_dolly-15k-oai-style"}]
    # LLM even gave the dir name → reconstructed to Hub id + cache_dir, no download
    hub, cache, dl = _resolve_dataset("philschmid_dolly-15k-oai-style", ds, "EP")
    assert hub == "philschmid/dolly-15k-oai-style"
    assert cache == "/app/tmp/datasets/philschmid_dolly-15k-oai-style"
    assert dl is None
    # not local → keep Hub id, flag download
    hub2, cache2, dl2 = _resolve_dataset("org/other", [], "EP")
    assert hub2 == "org/other" and cache2 is None and dl2["kind"] == "dataset"


def _complete_returning(text: str):
    async def _fn(prompt: str) -> str:
        return text
    return _fn


_GOOD = """{
  "model_name_or_path": "/app/tmp/models/google_gemma-3-270m-it",
  "dataset_name_or_path": "philschmid/dolly-15k-oai-style",
  "task_type": "instruction",
  "search_space": {"learning_rate": {"type":"float","min":1e-5,"max":5e-4,"log":true},
                   "lora_r": {"type":"categorical","choices":[8,16,32]}},
  "test_cases": [{"id":"q1","input":{"inputs":"Capital of France?"},"expected_keywords":["Paris"]},
                 {"id":"q2","input":{"inputs":"2+2?"},"expected_keywords":["4"]}],
  "scoring_schema": {"dimensions":[{"name":"keywords","weight":1.0,"method":"keyword_match"}],"aggregate":"weighted_sum"}
}"""


async def test_plan_resolves_local_and_no_download(monkeypatch):
    async def fake_create_schema(schema):
        assert schema.target_pipeline.value == "local_infer"
        return {"id": "11111111-1111-1111-1111-111111111111"}

    async def catalog():
        return ([{"name": "google_gemma-3-270m-it", "path": "/app/tmp/models/google_gemma-3-270m-it"}],
                [{"name": "philschmid_dolly-15k-oai-style",
                  "path": "/app/tmp/datasets/philschmid_dolly-15k-oai-style"}])

    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)
    monkeypatch.setattr(planner_mod, "_fetch_catalog", catalog)
    # This test is about model/dataset *resolution*; feed the sampler real rows so
    # the eval builds (the fake catalog path need not exist on the test machine).
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]} for i in range(20)]
    monkeypatch.setattr(planner_mod, "_read_dataset_rows", lambda *a, **k: rows)

    plan = await Planner(complete=_complete_returning(_GOOD)).plan("fine-tune gemma on dolly")

    assert plan.eval_schema_id == "11111111-1111-1111-1111-111111111111"
    assert plan.base_training_config["quantization_config"]["load_in_4bit"] is True  # safe template
    assert set(plan.search_space) == {"learning_rate", "lora_r"}
    # both present locally → resolved to local path, nothing to download
    assert plan.base_training_config["model_name_or_path"] == "/app/tmp/models/google_gemma-3-270m-it"
    assert plan.dataset_config["cache_dir"] == "/app/tmp/datasets/philschmid_dolly-15k-oai-style"
    assert plan.needs_download == []


async def test_plan_flags_download_when_not_local(monkeypatch):
    # autouse fixture gives an EMPTY catalog → model + dataset both missing
    plan = await Planner(complete=_complete_returning(_GOOD)).plan("fine-tune gemma on dolly")

    kinds = {n["kind"] for n in plan.needs_download}
    assert kinds == {"model", "dataset"}
    assert all("download_endpoint" in n for n in plan.needs_download)
    # dataset absent → eval is NOT fabricated; it's deferred until download
    assert plan.eval_schema_id is None and plan.holdout_schema_id is None


async def test_plan_defers_eval_when_dataset_absent(monkeypatch):
    """Dataset not local → flag needs_download and defer the eval (no LLM invention)."""
    called = False

    async def fake_create_schema(schema):
        nonlocal called
        called = True
        return {"id": "x"}

    async def catalog():  # model local, dataset NOT
        return ([{"name": "google_gemma-3-270m-it", "path": "/app/tmp/models/google_gemma-3-270m-it"}], [])

    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)
    monkeypatch.setattr(planner_mod, "_fetch_catalog", catalog)

    plan = await Planner(complete=_complete_returning(_GOOD)).plan("fine-tune gemma on dolly")
    assert plan.eval_schema_id is None            # no eval built
    assert not called                             # no schema persisted
    assert {n["kind"] for n in plan.needs_download} == {"dataset"}


async def test_plan_errors_when_local_dataset_unadaptable(monkeypatch, tmp_path):
    """Dataset is local but no row fits a known layout → error, don't invent cases."""
    d = tmp_path / "weird"
    d.mkdir()
    (d / "train.json").write_text("\n".join('{"foo": "bar", "baz": 1}' for _ in range(5)),
                                  encoding="utf-8")

    async def catalog():
        return ([{"name": "google_gemma-3-270m-it", "path": "/app/tmp/models/google_gemma-3-270m-it"}],
                [{"name": "weird", "path": str(d)}])

    monkeypatch.setattr(planner_mod, "_fetch_catalog", catalog)
    bad = _GOOD.replace("philschmid/dolly-15k-oai-style", "weird")
    with pytest.raises(PlannerError):
        await Planner(complete=_complete_returning(bad)).plan("goal")


# ── Budget from the goal ─────────────────────────────────────────────

def test_sanitize_budget_clamps_partial():
    b = sanitize_budget({"max_experiments": 5, "minutes_per_experiment": 10})
    assert b.max_experiments == 5 and b.minutes_per_experiment == 10
    assert b.early_stop_patience == 3  # untouched → Budget default


def test_sanitize_budget_clamps_out_of_range():
    b = sanitize_budget({"max_experiments": 9999, "minutes_per_experiment": -3,
                         "early_stop_patience": 0})
    assert b.max_experiments == 200          # capped
    assert b.minutes_per_experiment == 30.0  # -3 invalid → dropped → default
    assert b.early_stop_patience == 1        # floored


def test_sanitize_budget_none_when_empty_or_bad():
    assert sanitize_budget(None) is None
    assert sanitize_budget({}) is None
    assert sanitize_budget("5 runs") is None
    assert sanitize_budget({"max_experiments": "lots"}) is None


_GOOD_WITH_BUDGET = _GOOD[:-1] + ',"budget":{"max_experiments":3,"minutes_per_experiment":8}}'


async def test_plan_extracts_budget_from_goal(monkeypatch):
    async def fake_create_schema(schema):
        return {"id": "55555555-5555-5555-5555-555555555555"}
    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)

    plan = await Planner(complete=_complete_returning(_GOOD_WITH_BUDGET)).plan("... run 3 experiments ...")
    assert plan.budget is not None
    assert plan.budget.max_experiments == 3 and plan.budget.minutes_per_experiment == 8


async def test_plan_budget_none_when_goal_silent(monkeypatch):
    async def fake_create_schema(schema):
        return {"id": "66666666-6666-6666-6666-666666666666"}
    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)

    plan = await Planner(complete=_complete_returning(_GOOD)).plan("fine-tune gemma on dolly")
    assert plan.budget is None  # goal said nothing → request/default budget applies


# ── Eval sizing (dev / held-out counts) from the goal ────────────────

def test_sanitize_eval_defaults_and_overrides():
    assert sanitize_eval(None, 30, 15) == (30, 15)          # nothing → defaults
    assert sanitize_eval({}, 30, 15) == (30, 15)
    assert sanitize_eval({"dev_cases": 50}, 30, 15) == (50, 15)   # partial override
    assert sanitize_eval({"dev_cases": 40, "holdout_cases": 20}, 30, 15) == (40, 20)


def test_sanitize_eval_clamps():
    assert sanitize_eval({"dev_cases": 9999, "holdout_cases": 0}, 30, 15) == (200, 1)  # capped / floored
    assert sanitize_eval({"dev_cases": 1}, 30, 15) == (4, 15)      # floored to 4
    assert sanitize_eval({"dev_cases": "many"}, 30, 15) == (30, 15)  # bad → default


async def test_plan_extracts_eval_sizes_from_goal(monkeypatch, tmp_path):
    d = tmp_path / "philschmid_dolly-15k-oai-style"
    d.mkdir()
    (d / "train.json").write_text(
        "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"gold {i}"}) for i in range(60)),
        encoding="utf-8")

    calls = []

    async def fake_create_schema(schema):
        calls.append(schema)
        return {"id": f"bbbbbbbb-0000-0000-0000-00000000000{len(calls)}"}

    async def catalog():
        return ([], [{"name": "philschmid_dolly-15k-oai-style", "path": str(d)}])

    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)
    monkeypatch.setattr(planner_mod, "_fetch_catalog", catalog)

    goal_json = _GOOD[:-1] + ',"eval":{"dev_cases":10,"holdout_cases":6}}'
    plan = await Planner(complete=_complete_returning(goal_json)).plan("... use 10 questions ...")

    dev, holdout = calls
    assert len(dev.test_cases) == 10 and len(holdout.test_cases) == 6  # goal-driven sizes
    assert plan.holdout_schema_id is not None


# ── Training subset size from the goal ───────────────────────────────

def test_sanitize_training_defaults_and_clamp():
    assert sanitize_training(None, 30, 6) == (30, 6)
    assert sanitize_training({"train_size": 200}, 30, 6) == (200, 6)
    assert sanitize_training({"train_size": 99999, "val_size": 0}, 30, 6) == (5000, 1)  # capped/floored
    assert sanitize_training({"train_size": "lots"}, 30, 6) == (30, 6)  # bad → default


async def test_plan_extracts_training_size_from_goal(monkeypatch, tmp_path):
    d = tmp_path / "philschmid_dolly-15k-oai-style"
    d.mkdir()
    (d / "train.json").write_text(
        "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"gold {i}"}) for i in range(60)),
        encoding="utf-8")

    async def fake_create_schema(schema):
        return {"id": "cccccccc-0000-0000-0000-000000000001"}

    async def catalog():
        return ([], [{"name": "philschmid_dolly-15k-oai-style", "path": str(d)}])

    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)
    monkeypatch.setattr(planner_mod, "_fetch_catalog", catalog)

    goal_json = _GOOD[:-1] + ',"training":{"train_size":200,"val_size":20}}'
    plan = await Planner(complete=_complete_returning(goal_json)).plan("... train on 200 rows ...")
    assert plan.dataset_config["train_size"] == 200
    assert plan.dataset_config["val_size"] == 20


# ── Real-data eval: sample rows straight from the local dataset ──────

def test_row_to_case_messages():
    row = {"messages": [{"role": "user", "content": "When did X start?"},
                        {"role": "assistant", "content": "In 2000."}]}
    case = _row_to_case(row, "ds_0")
    assert case == {"id": "ds_0", "input": {"inputs": "When did X start?"},
                    "expected_answer": "In 2000."}


def test_row_to_case_instruction_and_prompt_styles():
    a = _row_to_case({"instruction": "Sum", "input": "2+2", "output": "4"}, "a")
    assert a["input"]["inputs"] == "Sum\n2+2" and a["expected_answer"] == "4"
    b = _row_to_case({"prompt": "hi", "completion": "hello"}, "b")
    assert b["input"]["inputs"] == "hi" and b["expected_answer"] == "hello"
    c = _row_to_case({"question": "Q?", "answer": "A"}, "c")
    assert c["input"]["inputs"] == "Q?" and c["expected_answer"] == "A"


def test_row_to_case_drops_incomplete():
    assert _row_to_case({"messages": [{"role": "user", "content": "only user"}]}, "x") is None
    assert _row_to_case({"instruction": "no output"}, "x") is None
    assert _row_to_case("not a dict", "x") is None


def test_read_dataset_rows_jsonl(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "train.json").write_text(
        "\n".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                           {"role": "assistant", "content": f"a{i}"}]})
                  for i in range(20)),
        encoding="utf-8")
    rows = _read_dataset_rows(str(d), n=5, seed=7)
    assert len(rows) == 5 and all("messages" in r for r in rows)
    # deterministic under a fixed seed
    assert _read_dataset_rows(str(d), n=5, seed=7) == rows


def test_read_dataset_rows_prefers_holdout_split(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "train.json").write_text(json.dumps({"prompt": "train", "completion": "x"}), encoding="utf-8")
    (d / "test.json").write_text(json.dumps({"prompt": "test", "completion": "y"}), encoding="utf-8")
    rows = _read_dataset_rows(str(d), n=5, seed=7)
    assert rows == [{"prompt": "test", "completion": "y"}]  # test.json wins over train.json


def test_read_dataset_rows_missing_dir():
    assert _read_dataset_rows("/no/such/dir", n=5, seed=7) == []


async def test_plan_builds_cosine_eval_from_real_rows(monkeypatch, tmp_path):
    d = tmp_path / "philschmid_dolly-15k-oai-style"
    d.mkdir()
    (d / "train.json").write_text(
        "\n".join(json.dumps({"messages": [{"role": "user", "content": f"question {i}?"},
                                           {"role": "assistant", "content": f"gold {i}"}]})
                  for i in range(60)),
        encoding="utf-8")

    calls = []  # one entry per persisted schema (dev, then held-out)

    async def fake_create_schema(schema):
        calls.append(schema)
        return {"id": f"aaaaaaaa-0000-0000-0000-00000000000{len(calls)}"}

    async def catalog():
        return ([{"name": "google_gemma-3-270m-it", "path": "/app/tmp/models/google_gemma-3-270m-it"}],
                [{"name": "philschmid_dolly-15k-oai-style", "path": str(d)}])

    monkeypatch.setattr(planner_mod.schema_store, "create_schema", fake_create_schema)
    monkeypatch.setattr(planner_mod, "_fetch_catalog", catalog)

    plan = await Planner(complete=_complete_returning(_GOOD)).plan("fine-tune gemma on dolly")

    # planner sampled dev + held-out as two independent, disjoint real-row schemas
    assert len(calls) == 2
    dev, holdout = calls
    assert plan.eval_schema_id == "aaaaaaaa-0000-0000-0000-000000000001"      # dev
    assert plan.holdout_schema_id == "aaaaaaaa-0000-0000-0000-000000000002"  # held-out
    # defaults 30 dev / 15 held-out; 60 rows available → both filled
    assert len(dev.test_cases) == 30 and len(holdout.test_cases) == 15
    # both are REAL rows (gold answers), disjoint, scored by cosine_similarity
    all_answers = [c.expected_answer for c in dev.test_cases + holdout.test_cases]
    assert all(a and a.startswith("gold") for a in all_answers)
    assert len(set(all_answers)) == 45                        # no overlap dev↔held-out
    assert dev.scoring_schema.dimensions[0].method == "cosine_similarity"
    assert holdout.scoring_schema.dimensions[0].method == "cosine_similarity"


async def test_plan_rejects_no_valid_knobs(monkeypatch):
    bad = ('{"model_name_or_path":"m","dataset_name_or_path":"d",'
           '"search_space":{"nonsense":{"type":"float","min":0,"max":1}}}')
    p = Planner(complete=_complete_returning(bad))
    with pytest.raises(PlannerError):
        await p.plan("goal")
