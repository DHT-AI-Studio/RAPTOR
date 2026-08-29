"""Unit tests for the guard-model orchestrator (app/services/guard_classifier.py) —
uses fake adapters, no real registry/Ollama involved."""
from dataclasses import dataclass, field

import pytest

from app.adapters.base import GuardRequest, GuardVerdict
from app.services import guard_classifier

pytestmark = pytest.mark.asyncio


@dataclass
class FakeAdapter:
    family: str
    priority: int
    verdict: GuardVerdict
    conversation_calls: list = field(default_factory=list)
    policy_calls: list = field(default_factory=list)

    def build_request(self, content, role, *, policy=None):
        self.policy_calls.append(policy)
        return GuardRequest(endpoint="chat", payload={"messages": [{"role": role, "content": content}]})

    def build_conversation_request(self, messages, *, policy=None):
        self.conversation_calls.append(messages)
        self.policy_calls.append(policy)
        return GuardRequest(endpoint="chat", payload={"messages": []})

    def parse(self, raw_text, *, used_policy_prompt=False):
        return self.verdict


@dataclass
class FakeSettings:
    active_models: list


def _patch(monkeypatch, adapters_by_model: dict, models: list[str], raw_by_model: dict | None = None):
    monkeypatch.setattr(guard_classifier, "get_adapter", lambda model: adapters_by_model[model])
    monkeypatch.setattr(guard_classifier, "get_settings", lambda: FakeSettings(active_models=models))

    async def fake_ollama_call(model, request, **kwargs):
        return (raw_by_model or {}).get(model, "raw-output")

    monkeypatch.setattr(guard_classifier, "ollama_call", fake_ollama_call)


async def test_classify_single_model_passthrough(monkeypatch):
    adapter = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="safe"))
    _patch(monkeypatch, {"model-a": adapter}, ["model-a"])

    result = await guard_classifier.classify("hello", role="user")

    assert result.safe is True
    assert result.conflict is None
    assert result.results is None


async def test_classify_multi_model_conflict_and_primary_by_priority(monkeypatch):
    high_priority = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="primary-says-safe"))
    low_priority = FakeAdapter(family="b", priority=5, verdict=GuardVerdict(safe=False, categories=["S1"], raw="secondary-says-unsafe"))
    _patch(monkeypatch, {"model-a": high_priority, "model-b": low_priority}, ["model-a", "model-b"])

    result = await guard_classifier.classify("hello", role="user")

    assert result.safe is False               # conservative: any unsafe -> combined unsafe
    assert result.conflict is True
    assert result.raw == "primary-says-safe"   # lower-priority-number adapter wins as "primary"
    assert len(result.results) == 2


async def test_classify_conversation_uses_build_conversation_request(monkeypatch):
    adapter = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="safe"))
    _patch(monkeypatch, {"model-a": adapter}, ["model-a"])

    from app.adapters.base import ChatMessage
    messages = [ChatMessage(role="user", content="hi")]
    await guard_classifier.classify_conversation(messages)

    assert adapter.conversation_calls == [messages]


async def test_classify_raw_returns_per_model_results_uncombined(monkeypatch):
    a = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="A"))
    b = FakeAdapter(family="b", priority=1, verdict=GuardVerdict(safe=False, categories=["S2"], raw="B"))
    _patch(monkeypatch, {"model-a": a, "model-b": b}, ["model-a", "model-b"],
           raw_by_model={"model-a": "raw-A", "model-b": "raw-B"})

    results = await guard_classifier.classify_raw("hello")

    assert [r.model for r in results] == ["model-a", "model-b"]
    assert [r.raw for r in results] == ["A", "B"]   # adapter.parse() output, not the raw Ollama text


async def test_classify_per_model_routes_policy_by_adapter_family(monkeypatch):
    from app.adapters.base import PolicyContent

    llama = FakeAdapter(family="llama_guard3", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="ok"))
    granite = FakeAdapter(family="granite", priority=2, verdict=GuardVerdict(safe=True, categories=[], raw="ok"))
    unknown = FakeAdapter(family="some_future_family", priority=9, verdict=GuardVerdict(safe=True, categories=[], raw="ok"))
    _patch(monkeypatch, {"m-llama": llama, "m-granite": granite, "m-unknown": unknown},
           ["m-llama", "m-granite", "m-unknown"])

    original = PolicyContent(raw="original content")
    per_family = {
        "llama_guard3": PolicyContent(raw="llama override"),
        "granite": PolicyContent(raw="granite override"),
    }

    await guard_classifier.classify_per_model("hello", role="user", policy=original, policy_by_family=per_family)

    assert llama.policy_calls == [PolicyContent(raw="llama override")]
    assert granite.policy_calls == [PolicyContent(raw="granite override")]
    # family not present in policy_by_family falls back to the plain `policy` arg
    assert unknown.policy_calls == [original]


async def test_classify_per_model_without_policy_by_family_uses_plain_policy_for_all(monkeypatch):
    from app.adapters.base import PolicyContent

    a = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="ok"))
    b = FakeAdapter(family="b", priority=1, verdict=GuardVerdict(safe=True, categories=[], raw="ok"))
    _patch(monkeypatch, {"model-a": a, "model-b": b}, ["model-a", "model-b"])

    shared = PolicyContent(raw="shared")
    await guard_classifier.classify_per_model("hello", policy=shared)

    assert a.policy_calls == [shared]
    assert b.policy_calls == [shared]


# ── _render_prompt_text / *_with_prompt (debug support) ─────────────────────────

async def test_render_prompt_text_for_generate_endpoint_returns_raw_prompt_as_is():
    request = GuardRequest(endpoint="generate", payload={"prompt": "the exact raw completion prompt", "raw": True})
    assert guard_classifier._render_prompt_text(request) == "the exact raw completion prompt"


async def test_render_prompt_text_for_chat_endpoint_renders_each_message():
    request = GuardRequest(endpoint="chat", payload={"messages": [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "user content"},
    ]})
    rendered = guard_classifier._render_prompt_text(request)
    assert rendered == "[system]\nsystem instructions\n\n[user]\nuser content"


async def test_render_prompt_text_for_chat_endpoint_with_no_messages_key():
    request = GuardRequest(endpoint="chat", payload={})
    assert guard_classifier._render_prompt_text(request) == ""


async def test_classify_per_model_with_prompt_returns_verdict_and_rendered_prompt(monkeypatch):
    adapter = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=False, categories=["S1"], raw="unsafe\nS1"))
    _patch(monkeypatch, {"model-a": adapter}, ["model-a"])

    results = await guard_classifier.classify_per_model_with_prompt("hello", role="user")

    assert len(results) == 1
    verdict, prompt = results[0]
    assert verdict.model == "model-a" and verdict.safe is False and verdict.categories == ["S1"]
    assert prompt == "[user]\nhello"   # FakeAdapter.build_request echoes role/content into one chat message


async def test_classify_conversation_per_model_with_prompt_uses_build_conversation_request(monkeypatch):
    adapter = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="safe"))
    _patch(monkeypatch, {"model-a": adapter}, ["model-a"])

    from app.adapters.base import ChatMessage
    messages = [ChatMessage(role="user", content="hi")]
    results = await guard_classifier.classify_conversation_per_model_with_prompt(messages)

    assert adapter.conversation_calls == [messages]
    verdict, prompt = results[0]
    assert verdict.safe is True
    assert prompt == ""   # FakeAdapter.build_conversation_request returns an empty messages payload


async def test_classify_per_model_with_prompt_does_not_affect_classify_per_model(monkeypatch):
    # The two code paths must be fully independent — calling the *_with_prompt
    # variant must not change what the plain (non-debug) variant returns.
    adapter = FakeAdapter(family="a", priority=0, verdict=GuardVerdict(safe=True, categories=[], raw="safe"))
    _patch(monkeypatch, {"model-a": adapter}, ["model-a"])

    with_prompt = await guard_classifier.classify_per_model_with_prompt("hello", role="user")
    plain = await guard_classifier.classify_per_model("hello", role="user")

    assert plain == [with_prompt[0][0]]
