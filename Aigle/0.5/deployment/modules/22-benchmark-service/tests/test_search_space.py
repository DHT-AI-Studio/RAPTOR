"""Tests for search-space sampling + clamping (AUTOTUNE §B)."""
from __future__ import annotations

import random

from app.models.experiment import SearchDimension
from app.services.autotune.search_space import clamp_to_search_space, sample_random


def _space():
    return {
        "learning_rate": SearchDimension(type="float", min=1e-5, max=5e-4, log=True),
        "warmup_ratio": SearchDimension(type="float", min=0.0, max=0.1),
        "max_epochs": SearchDimension(type="int", min=1, max=3),
        "lora_r": SearchDimension(type="categorical", choices=[8, 16, 32]),
    }


def test_sample_random_stays_in_range():
    space = _space()
    rng = random.Random(42)
    for _ in range(200):
        cfg = sample_random(space, rng)
        assert 1e-5 <= cfg["learning_rate"] <= 5e-4
        assert 0.0 <= cfg["warmup_ratio"] <= 0.1
        assert cfg["max_epochs"] in (1, 2, 3)
        assert isinstance(cfg["max_epochs"], int)
        assert cfg["lora_r"] in (8, 16, 32)


def test_clamp_pulls_out_of_range_values_back():
    space = _space()
    clamped = clamp_to_search_space(
        {"learning_rate": 9.9, "warmup_ratio": -1.0, "max_epochs": 99, "lora_r": 999},
        space,
    )
    assert clamped["learning_rate"] == 5e-4          # above max → max
    assert clamped["warmup_ratio"] == 0.0            # below min → min
    assert clamped["max_epochs"] == 3                # above max → max, int
    assert clamped["lora_r"] == 8                    # invalid choice → first choice


def test_clamp_drops_keys_outside_search_space():
    space = _space()
    clamped = clamp_to_search_space(
        {"learning_rate": 1e-4, "some_dangerous_flag": True, "model_name_or_path": "/etc/passwd"},
        space,
    )
    assert "some_dangerous_flag" not in clamped
    assert "model_name_or_path" not in clamped
    assert clamped["learning_rate"] == 1e-4


def test_clamp_int_coercion_and_unparseable_fallback():
    space = {"max_epochs": SearchDimension(type="int", min=1, max=5)}
    assert clamp_to_search_space({"max_epochs": 2.7}, space)["max_epochs"] == 3
    # Un-parseable value falls back to the range midpoint (safe default).
    assert clamp_to_search_space({"max_epochs": "abc"}, space)["max_epochs"] == 3


def test_categorical_valid_choice_preserved():
    space = {"lora_r": SearchDimension(type="categorical", choices=[8, 16, 32])}
    assert clamp_to_search_space({"lora_r": 16}, space)["lora_r"] == 16
