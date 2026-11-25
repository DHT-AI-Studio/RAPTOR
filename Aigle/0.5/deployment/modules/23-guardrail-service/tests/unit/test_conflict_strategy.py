"""Unit tests for GUARD_CONFLICT_STRATEGY (app/core/config.py) as applied in
app/services/guard_classifier.py::combine() — the actual multi-guard-model
merge function (there is no `_merge_guard_results` anywhere in this codebase;
`checker.py::evaluate()` only ever sees one already-combined verdict, produced
by `classify()` -> `combine()`, which is what these tests exercise directly).

`combine()` reads the strategy from `get_settings()`, so these tests
monkeypatch `guard_classifier.get_settings` the same way
tests/unit/test_guard_classifier.py's `FakeSettings` does. `model` fields use real,
already-registered model names (see app/adapters/models/*.py's `register()`
calls) so `combine()`'s own `get_adapter()` calls resolve without needing to
fake the registry too:
  - llama-guard3:8b        -> priority 0
  - gpt-oss-safeguard:20b  -> priority 1
  - granite4.1-guardian:8b -> priority 2
"""
from dataclasses import dataclass

import pytest
from app.services import guard_classifier
from app.services.guard_classifier import ModelVerdict, combine

pytestmark = pytest.mark.asyncio

_LLAMA_GUARD = "llama-guard3:8b"
_GPT_OSS = "gpt-oss-safeguard:20b"
_GRANITE = "granite4.1-guardian:8b"


@dataclass
class FakeSettings:
    guard_conflict_strategy: str


def _verdict(model: str, safe: bool) -> ModelVerdict:
    return ModelVerdict(model=model, safe=safe, categories=[] if safe else ["S1"],
                         category_names={}, raw="safe" if safe else "unsafe")


def _combine_with_strategy(monkeypatch, results, strategy):
    monkeypatch.setattr(guard_classifier, "get_settings", lambda: FakeSettings(guard_conflict_strategy=strategy))
    return combine(results)


# ── 3-model permutations ─────────────────────────────────────────────────────────

_ALL_SAFE = [_verdict(_LLAMA_GUARD, True), _verdict(_GPT_OSS, True), _verdict(_GRANITE, True)]
_ALL_UNSAFE = [_verdict(_LLAMA_GUARD, False), _verdict(_GPT_OSS, False), _verdict(_GRANITE, False)]
_ONE_UNSAFE_TWO_SAFE = [_verdict(_LLAMA_GUARD, False), _verdict(_GPT_OSS, True), _verdict(_GRANITE, True)]
_TWO_UNSAFE_ONE_SAFE = [_verdict(_LLAMA_GUARD, False), _verdict(_GPT_OSS, False), _verdict(_GRANITE, True)]


@pytest.mark.parametrize("strategy", ["strict", "majority", "permissive"])
async def test_all_safe_is_safe_under_every_strategy(monkeypatch, strategy):
    result = _combine_with_strategy(monkeypatch, _ALL_SAFE, strategy)
    assert result.safe is True


@pytest.mark.parametrize("strategy", ["strict", "majority", "permissive"])
async def test_all_unsafe_is_unsafe_under_every_strategy(monkeypatch, strategy):
    result = _combine_with_strategy(monkeypatch, _ALL_UNSAFE, strategy)
    assert result.safe is False


async def test_one_unsafe_two_safe_strict_is_unsafe(monkeypatch):
    result = _combine_with_strategy(monkeypatch, _ONE_UNSAFE_TWO_SAFE, "strict")
    assert result.safe is False   # any unsafe -> unsafe


async def test_one_unsafe_two_safe_majority_is_safe(monkeypatch):
    result = _combine_with_strategy(monkeypatch, _ONE_UNSAFE_TWO_SAFE, "majority")
    assert result.safe is True   # single dissent doesn't reach a majority


async def test_one_unsafe_two_safe_permissive_is_safe(monkeypatch):
    result = _combine_with_strategy(monkeypatch, _ONE_UNSAFE_TWO_SAFE, "permissive")
    assert result.safe is True   # not all models say unsafe


async def test_two_unsafe_one_safe_strict_is_unsafe(monkeypatch):
    result = _combine_with_strategy(monkeypatch, _TWO_UNSAFE_ONE_SAFE, "strict")
    assert result.safe is False


async def test_two_unsafe_one_safe_majority_is_unsafe(monkeypatch):
    result = _combine_with_strategy(monkeypatch, _TWO_UNSAFE_ONE_SAFE, "majority")
    assert result.safe is False   # 2 of 3 unsafe -> a strict majority


async def test_two_unsafe_one_safe_permissive_is_safe(monkeypatch):
    result = _combine_with_strategy(monkeypatch, _TWO_UNSAFE_ONE_SAFE, "permissive")
    assert result.safe is True   # one model still says safe


# ── conflict flag and primary selection are strategy-independent ───────────────────

@pytest.mark.parametrize("strategy", ["strict", "majority", "permissive"])
async def test_conflict_flag_reports_disagreement_regardless_of_strategy(monkeypatch, strategy):
    result = _combine_with_strategy(monkeypatch, _ONE_UNSAFE_TWO_SAFE, strategy)
    assert result.conflict is True


@pytest.mark.parametrize("strategy", ["strict", "majority", "permissive"])
async def test_no_conflict_when_all_models_agree(monkeypatch, strategy):
    result = _combine_with_strategy(monkeypatch, _ALL_SAFE, strategy)
    assert result.conflict is False


@pytest.mark.parametrize("strategy", ["strict", "majority", "permissive"])
async def test_primary_verdict_selection_unaffected_by_strategy(monkeypatch, strategy):
    # llama-guard3:8b has priority 0 (highest) -> always "primary", regardless
    # of which strategy decided the overall safe/unsafe verdict.
    result = _combine_with_strategy(monkeypatch, _TWO_UNSAFE_ONE_SAFE, strategy)
    assert result.raw == "unsafe"   # _LLAMA_GUARD's verdict in _TWO_UNSAFE_ONE_SAFE


# ── default behavior (no strategy set) ──────────────────────────────────────────────

async def test_default_strategy_from_real_settings_is_strict():
    from app.core.config import get_settings
    assert get_settings().guard_conflict_strategy == "strict"


async def test_no_strategy_override_behaves_like_strict(monkeypatch):
    # Using the real, unmocked get_settings() (default config, no env var set)
    # must reproduce the exact pre-existing hardcoded behavior.
    result = combine(_ONE_UNSAFE_TWO_SAFE)
    assert result.safe is False


# ── edge cases: 2-model and 1-model ─────────────────────────────────────────────────

async def test_two_model_majority_is_equivalent_to_permissive(monkeypatch):
    # With only 2 voters, "more than half unsafe" requires both to be unsafe —
    # mathematically identical to permissive. Documented, not a bug.
    split = [_verdict(_LLAMA_GUARD, False), _verdict(_GPT_OSS, True)]
    majority_result = _combine_with_strategy(monkeypatch, split, "majority")
    permissive_result = _combine_with_strategy(monkeypatch, split, "permissive")
    assert majority_result.safe is True
    assert majority_result.safe == permissive_result.safe


async def test_two_model_both_unsafe_majority_is_unsafe(monkeypatch):
    both_unsafe = [_verdict(_LLAMA_GUARD, False), _verdict(_GPT_OSS, False)]
    result = _combine_with_strategy(monkeypatch, both_unsafe, "majority")
    assert result.safe is False


@pytest.mark.parametrize("strategy", ["strict", "majority", "permissive"])
async def test_single_model_is_a_passthrough_regardless_of_strategy(monkeypatch, strategy):
    result = _combine_with_strategy(monkeypatch, [_verdict(_LLAMA_GUARD, False)], strategy)
    assert result.safe is False
    assert result.conflict is None
    assert result.results is None
