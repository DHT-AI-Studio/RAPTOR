"""Tests for the LLM proposer (AUTOTUNE Phase 2).

The LLM call is injected (``complete``), so these run offline and exercise the
prompt-independent logic: JSON parsing, clamping, stop, and the retry -> random
fallback that keeps the loop from stalling on bad model output.
"""
from __future__ import annotations

from app.models.experiment import Plan
from app.services.autotune.proposer import LLMProposer, _build_prompt


def _plan() -> Plan:
    return Plan(
        task_type="instruction",
        select_multiple_gpus=False,
        vram_budget_gb=12,
        base_training_config={"model_name_or_path": "/m", "lora_config": {"r": 8}},
        dataset_config={"dataset_name_or_path": "ds", "column_mapping": {"messages": "messages"}},
        search_space={
            "learning_rate": {"type": "float", "min": 1e-5, "max": 5e-4, "log": True},
            "lora_r": {"type": "categorical", "choices": [8, 16, 32]},
        },
        eval_schema_id="00000000-0000-0000-0000-000000000000",
    )


def _complete_returning(text: str):
    async def _fn(prompt: str) -> str:
        return text
    return _fn


def _complete_sequence(*texts: str):
    """Return each text in turn on successive calls (last one repeats)."""
    calls = {"i": 0}

    async def _fn(prompt: str) -> str:
        i = min(calls["i"], len(texts) - 1)
        calls["i"] += 1
        return texts[i]
    return _fn


async def test_valid_proposal_is_returned_with_reason():
    # LLM returns an in-range config + reason wrapped in chatty text + markdown.
    raw = ('Sure! ```json\n{"config": {"learning_rate": 3e-4, "lora_r": 16}, '
           '"stop": false, "reason": "higher rank looked promising"}\n```')
    p = LLMProposer(complete=_complete_returning(raw))
    config, stop, reason = await p.propose(_plan(), history=[])
    assert stop is False
    assert config["learning_rate"] == 3e-4
    assert config["lora_r"] == 16
    assert reason == "higher rank looked promising"


async def test_out_of_range_values_are_clamped():
    raw = '{"config": {"learning_rate": 9.9, "lora_r": 999}, "stop": false}'
    p = LLMProposer(complete=_complete_returning(raw))
    config, stop, _ = await p.propose(_plan(), history=[])
    assert config["learning_rate"] == 5e-4     # clamped to max
    assert config["lora_r"] == 8               # invalid choice -> first choice


async def test_trailing_comma_is_repaired():
    # Invalid strict JSON (trailing comma) — repair pass should recover it.
    raw = '{"config": {"learning_rate": 2e-4, "lora_r": 8,}, "stop": false,}'
    p = LLMProposer(complete=_complete_returning(raw))
    config, stop, _ = await p.propose(_plan(), history=[])
    assert stop is False
    assert config["learning_rate"] == 2e-4
    assert config["lora_r"] == 8


async def test_truncated_output_is_recovered():
    # LLM output cut off before the final brace (long reason ran past the token
    # limit) — the brace-balancing extractor should still recover the config.
    raw = ('{"config": {"learning_rate": 0.00045, "lora_r": 32}, "stop": false, '
           '"reason": "increasing lr and expanding rank to add capacity')
    p = LLMProposer(complete=_complete_returning(raw))
    config, stop, reason = await p.propose(_plan(), history=[])
    assert stop is False
    assert config["learning_rate"] == 0.00045
    assert config["lora_r"] == 32
    assert reason and reason.startswith("increasing lr")   # not the random fallback


async def test_stop_signal():
    p = LLMProposer(complete=_complete_returning('{"stop": true, "reason": "converged"}'))
    config, stop, reason = await p.propose(_plan(), history=[])
    assert stop is True
    assert config == {}
    assert reason == "converged"


async def test_bad_output_falls_back_to_random():
    # Always returns garbage -> both attempts fail -> random sample in range.
    p = LLMProposer(complete=_complete_returning("not json at all"))
    config, stop, reason = await p.propose(_plan(), history=[])
    assert stop is False
    assert 1e-5 <= config["learning_rate"] <= 5e-4
    assert config["lora_r"] in (8, 16, 32)
    assert reason and "fallback" in reason


async def test_empty_config_falls_back_to_random():
    # Valid JSON but no usable knobs -> treated as failure -> random fallback.
    p = LLMProposer(complete=_complete_returning('{"config": {"unknown": 1}, "stop": false}'))
    config, stop, _ = await p.propose(_plan(), history=[])
    assert set(config).issubset({"learning_rate", "lora_r"})
    assert config  # non-empty (random fallback filled it)


async def test_duplicate_proposal_is_nudged_then_accepted():
    # History already has {lr:1e-4, lora_r:16}. The LLM first re-proposes that exact
    # config; the nudge should make it try a DIFFERENT one, which is accepted.
    history = [{"aggregate_score": 0.75,
                "config_override": {"learning_rate": 1e-4, "lora_r": 16}}]
    dup = '{"config": {"learning_rate": 1e-4, "lora_r": 16}, "stop": false, "reason": "re-run to confirm"}'
    new = '{"config": {"learning_rate": 3e-4, "lora_r": 32}, "stop": false, "reason": "explore higher rank"}'
    p = LLMProposer(complete=_complete_sequence(dup, new))
    config, stop, reason = await p.propose(_plan(), history)
    assert stop is False
    assert config == {"learning_rate": 3e-4, "lora_r": 32}
    assert reason == "explore higher rank"


async def test_persistent_duplicate_converges_to_stop():
    # The LLM keeps re-proposing the same already-tried config even after the nudge
    # → treat as convergence and stop (instead of burning budget on a re-run).
    history = [{"aggregate_score": 0.75,
                "config_override": {"learning_rate": 1e-4, "lora_r": 16}}]
    dup = '{"config": {"learning_rate": 1e-4, "lora_r": 16}, "stop": false, "reason": "re-run to confirm"}'
    p = LLMProposer(complete=_complete_returning(dup))
    config, stop, _ = await p.propose(_plan(), history)
    assert stop is True
    assert config == {}


def test_prompt_includes_space_history_and_reason():
    plan = _plan()
    history = [{"aggregate_score": 0.8,
                "config_override": {"learning_rate": 1e-4, "model_path": "/m",
                                    "_reason": "lowered lr to stabilize"}}]
    prompt = _build_prompt(plan.search_space, history)
    assert "learning_rate" in prompt
    assert "lora_r" in prompt
    assert "0.8" in prompt
    assert "model_path" not in prompt              # stripped from the shown config
    assert "lowered lr to stabilize" in prompt     # past reason fed back to the model
