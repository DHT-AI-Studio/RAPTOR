"""Tests for the Module 16 training client's config merge (AUTOTUNE §B4)."""
from __future__ import annotations

from app.models.experiment import Plan
from app.services.autotune.training_client import build_submission, build_training_config


def test_build_training_config_top_level_and_aliases():
    base = {
        "model_name_or_path": "/app/tmp/models/g",
        "learning_rate": 2e-5,
        "lora_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    }
    merged = build_training_config(base, {"learning_rate": 3e-4, "lora_r": 32})
    assert merged["learning_rate"] == 3e-4            # top-level override
    assert merged["lora_config"]["r"] == 32           # alias → nested
    assert merged["lora_config"]["lora_alpha"] == 16  # untouched
    # base must not be mutated (deep copy)
    assert base["learning_rate"] == 2e-5
    assert base["lora_config"]["r"] == 8


def test_build_training_config_dotted_path():
    merged = build_training_config({"lora_config": {"r": 8}}, {"lora_config.lora_alpha": 64})
    assert merged["lora_config"]["lora_alpha"] == 64
    assert merged["lora_config"]["r"] == 8


def test_target_modules_preset_is_expanded():
    # The search space proposes a hashable preset NAME; the client expands it to
    # the real module list before it reaches Module 16.
    base = {"lora_config": {"r": 8, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]}}
    merged = build_training_config(base, {"target_modules": "attn_mlp"})
    tm = merged["lora_config"]["target_modules"]
    assert isinstance(tm, list)
    assert "gate_proj" in tm and "up_proj" in tm and "down_proj" in tm  # MLP projections added
    assert "q_proj" in tm

    attn = build_training_config(base, {"target_modules": "attn"})
    assert attn["lora_config"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_lora_variant_expands_to_peft_flags():
    base = {"lora_config": {"r": 8}}
    dora = build_training_config(base, {"lora_variant": "dora"})
    assert dora["lora_config"]["use_dora"] is True and dora["lora_config"]["use_rslora"] is False
    rslora = build_training_config(base, {"lora_variant": "rslora"})
    assert rslora["lora_config"]["use_rslora"] is True and rslora["lora_config"]["use_dora"] is False
    plain = build_training_config(base, {"lora_variant": "plain"})
    assert plain["lora_config"]["use_rslora"] is False and plain["lora_config"]["use_dora"] is False


def test_build_submission_shape():
    plan = Plan(
        task_type="instruction",
        select_multiple_gpus=False,
        vram_budget_gb=12,
        base_training_config={"model_name_or_path": "/m", "lora_config": {"r": 8}},
        dataset_config={"dataset_name_or_path": "ds", "column_mapping": {"messages": "messages"}},
        search_space={"lora_r": {"type": "categorical", "choices": [8, 16, 32]}},
        eval_schema_id="00000000-0000-0000-0000-000000000000",
    )
    body = build_submission(plan, {"lora_r": 16})
    assert body["task_type"] == "instruction"
    assert body["vram_budget_gb"] == 12
    assert body["training_config"]["lora_config"]["r"] == 16
    assert body["dataset_config"]["dataset_name_or_path"] == "ds"
