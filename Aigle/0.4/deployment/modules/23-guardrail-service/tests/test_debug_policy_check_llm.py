"""Unit tests for app/routers/debug_policy_check_llm.py's pure helper
functions — confirms they behave identically to check_policy_llm.py's
(deliberately duplicated, not imported — see that file's module docstring),
plus _to_debug_response's CheckResponse-shaped output with the added
per-model `prompt` field, plus router-level tests proving the
is_policy_check_enabled() gate actually drives the HTTP response. No
HTTP/Ollama/DB involved. The gate's own four-way truth table (global x
policy switch) is covered by tests/test_state.py's
test_is_policy_check_enabled_requires_both_switches_on."""
from datetime import datetime, timezone
from uuid import uuid4

from app.adapters.base import PolicyContent
from app.engine.models import PolicyDetail
from app.models.guard import MessageRequest
from app.routers import debug_policy_check_llm
from app.routers.debug_policy_check_llm import (
    _original_policy_content,
    _policy_by_family,
    _to_debug_response,
    _unsafe_category,
)
from app.services.guard_classifier import ModelVerdict


def _policy(**overrides) -> PolicyDetail:
    fields = dict(
        id=uuid4(), name="p", version="1.0", raw_content="original content",
        content_llama_guard=None, content_granite_guardian=None, content_gpt_oss_safeguard=None,
        is_active=True, created_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return PolicyDetail(**fields)


def test_policy_by_family_falls_back_to_original_when_no_overrides_set():
    policy = _policy()
    original = _original_policy_content(policy)
    result = _policy_by_family(policy, original)

    assert result == {
        "llama_guard3": PolicyContent(raw="original content"),
        "granite": PolicyContent(raw="original content"),
        "gpt_oss": PolicyContent(raw="original content"),
    }


def test_policy_by_family_uses_override_when_set():
    policy = _policy(content_llama_guard="llama-specific prompt")
    original = _original_policy_content(policy)
    result = _policy_by_family(policy, original)

    assert result["llama_guard3"] == PolicyContent(raw="llama-specific prompt")
    assert result["granite"] == PolicyContent(raw="original content")


def test_original_policy_content_parses_standard_format_json_array():
    raw = (
        '[{"id": "M1", "name": "n", "description": "d", "severity": "high", '
        '"decision": "block", "criteria": ["c"]}]'
    )
    policy = _policy(raw_content=raw)
    original = _original_policy_content(policy)

    assert original.raw == raw
    assert original.standard_policies is not None
    assert original.standard_policies[0].id == "M1"


def test_original_policy_content_falls_back_for_legacy_free_text():
    policy = _policy(raw_content="Block anything about weapons.")
    original = _original_policy_content(policy)

    assert original.standard_policies is None


def test_unsafe_category_aggregates_and_dedupes_across_unsafe_models_only():
    results = [
        ModelVerdict(model="m1", safe=False, categories=["S1", "S9"], category_names={}, raw=""),
        ModelVerdict(model="m2", safe=True, categories=["S2"], category_names={}, raw=""),
    ]
    assert _unsafe_category(results) == "S1,S9"


def test_unsafe_category_none_when_all_safe():
    results = [ModelVerdict(model="m1", safe=True, categories=[], category_names={}, raw="")]
    assert _unsafe_category(results) is None


def test_to_debug_response_single_model_top_level_matches_verdict_and_results_populated():
    verdict = ModelVerdict(model="m1", safe=False, categories=["S1"], category_names={"S1": "Weapons"}, raw="unsafe\nS1")
    response = _to_debug_response([(verdict, "[system]\nfull prompt text")])

    # single-model: top-level mirrors the one verdict (same as guard_classifier.combine())
    assert response.safe is False
    assert response.categories == ["S1"]
    assert response.category_names == {"S1": "Weapons"}
    assert response.raw == "unsafe\nS1"
    assert response.conflict is None

    # unlike CheckResponse/guard_classifier.combine(), results is never null — prompt must survive
    assert response.results is not None
    assert len(response.results) == 1
    r = response.results[0]
    assert r.model == "m1" and r.safe is False and r.categories == ["S1"] and r.raw == "unsafe\nS1"
    assert r.prompt == "[system]\nfull prompt text"


def test_to_debug_response_multi_model_conflict_and_prompt_per_model(monkeypatch):
    from app.services import guard_classifier

    class FakeAdapter:
        def __init__(self, priority):
            self.priority = priority

    monkeypatch.setattr(guard_classifier, "get_adapter", lambda model: {
        "m-high": FakeAdapter(priority=0), "m-low": FakeAdapter(priority=5),
    }[model])

    safe_verdict = ModelVerdict(model="m-high", safe=True, categories=[], category_names={}, raw="primary-safe")
    unsafe_verdict = ModelVerdict(model="m-low", safe=False, categories=["S9"], category_names={}, raw="secondary-unsafe")
    results = [(safe_verdict, "prompt for m-high"), (unsafe_verdict, "prompt for m-low")]

    response = _to_debug_response(results)

    assert response.safe is False              # conservative: any unsafe -> combined unsafe
    assert response.conflict is True
    assert response.raw == "primary-safe"       # lower-priority-number adapter wins as "primary"
    assert [r.model for r in response.results] == ["m-high", "m-low"]
    assert [r.prompt for r in response.results] == ["prompt for m-high", "prompt for m-low"]


def test_to_debug_response_empty_categories_when_safe():
    verdict = ModelVerdict(model="m1", safe=True, categories=[], category_names={}, raw="safe")
    response = _to_debug_response([(verdict, "some prompt")])
    assert response.categories == []
    assert response.results[0].categories == []


async def test_debug_check_input_llm_short_circuits_when_policy_check_disabled(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not be called when is_policy_check_enabled() is False")

    async def fake_disabled():
        return False

    monkeypatch.setattr(debug_policy_check_llm, "is_policy_check_enabled", fake_disabled)
    monkeypatch.setattr(debug_policy_check_llm.policy_engine, "_get_active_policy", fail_if_called)
    monkeypatch.setattr(debug_policy_check_llm.guard_classifier, "classify_per_model_with_prompt", fail_if_called)

    result = await debug_policy_check_llm.debug_check_input_llm(MessageRequest(content="hello"))

    assert result is debug_policy_check_llm._DISABLED_RESPONSE
    assert result.safe is True


async def test_debug_check_input_llm_proceeds_when_policy_check_enabled(monkeypatch):
    policy = _policy()
    verdict = ModelVerdict(model="m1", safe=True, categories=[], category_names={}, raw="model says safe")

    async def fake_enabled():
        return True

    async def fake_get_active_policy():
        return policy

    async def fake_classify_per_model_with_prompt(*args, **kwargs):
        return [(verdict, "rendered prompt text")]

    monkeypatch.setattr(debug_policy_check_llm, "is_policy_check_enabled", fake_enabled)
    monkeypatch.setattr(debug_policy_check_llm.policy_engine, "_get_active_policy", fake_get_active_policy)
    monkeypatch.setattr(debug_policy_check_llm.guard_classifier, "classify_per_model_with_prompt",
                         fake_classify_per_model_with_prompt)

    result = await debug_policy_check_llm.debug_check_input_llm(MessageRequest(content="hello"))

    # proceeded past the gate: response reflects the faked classify result, not _DISABLED_RESPONSE
    assert result.safe is True
    assert result.raw == "model says safe"
    assert result.results[0].prompt == "rendered prompt text"
