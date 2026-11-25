"""Shared fakes for the policy-aware check tests (app/services/policy_check_service.py
and the endpoints built on it: /api/v0.2/check/*, /guardrail/policy/decide).

`policy_env` replaces everything the check touches outside the process — the
global/policy switches (Redis), the active policy (Redis/Postgres), the guard
models (Ollama) and the audit write (Postgres) — with a mutable in-memory
state the test sets up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from app.engine.models import PolicyDetail
from app.services import audit_log, policy_check_service
from app.services.default_policy import DEFAULT_STANDARD_POLICY
from app.services.guard_classifier import ModelVerdict
from fastapi import HTTPException


def make_policy(raw_content: str, version: str = "1.0") -> PolicyDetail:
    return PolicyDetail(
        id=uuid4(), name="p", version=version, raw_content=raw_content,
        content_llama_guard=None, content_granite_guardian=None, content_gpt_oss_safeguard=None,
        is_active=True, created_at=datetime.now(timezone.utc),
    )


def standard_policy(version: str = "1.0", **decisions: str) -> PolicyDetail:
    """The default S1-S13 Standard Policy, with `S1="warn"`-style per-category decision overrides."""
    categories = [dict(c, decision=decisions.get(c["id"], c["decision"])) for c in DEFAULT_STANDARD_POLICY]
    return make_policy(json.dumps(categories), version=version)


def safe_verdict(model: str = "m1") -> ModelVerdict:
    return ModelVerdict(model=model, safe=True, categories=[], category_names={}, raw="safe")


def unsafe_verdict(*categories: str, model: str = "m1") -> ModelVerdict:
    names = {c: f"name-{c}" for c in categories}
    return ModelVerdict(model=model, safe=False, categories=list(categories), category_names=names, raw="unsafe")


@dataclass
class PolicyEnv:
    enabled: bool = True
    policy_enabled: bool = True
    policy: PolicyDetail | None = None
    verdicts: list[ModelVerdict] = field(default_factory=lambda: [safe_verdict()])
    fail: Exception | None = None                         # raised by the guard model call
    classify_calls: list[dict] = field(default_factory=list)
    audits: list[dict] = field(default_factory=list)


@pytest.fixture
def policy_env(monkeypatch) -> PolicyEnv:
    env = PolicyEnv()

    async def _enabled():
        return env.enabled

    async def _policy_enabled():
        return env.policy_enabled

    async def _active_policy():
        if env.policy is None:
            raise HTTPException(status_code=503, detail="No active policy")
        return env.policy

    async def _classify(content, role, *, policy, policy_by_family):
        env.classify_calls.append({"content": content, "role": role, "policy": policy, "family": policy_by_family})
        if env.fail is not None:
            raise env.fail
        return list(env.verdicts)

    async def _classify_with_prompt(content, role, *, policy, policy_by_family):
        return [(v, "prompt") for v in await _classify(content, role, policy=policy, policy_by_family=policy_by_family)]

    async def _classify_conversation(messages, *, policy, policy_by_family):
        return await _classify(messages[-1].content, messages[-1].role, policy=policy, policy_by_family=policy_by_family)

    monkeypatch.setattr(policy_check_service, "is_enabled", _enabled)
    monkeypatch.setattr(policy_check_service, "is_policy_enabled", _policy_enabled)
    monkeypatch.setattr(policy_check_service.policy_engine, "_get_active_policy", _active_policy)
    monkeypatch.setattr(policy_check_service.guard_classifier, "classify_per_model", _classify)
    monkeypatch.setattr(policy_check_service.guard_classifier, "classify_per_model_with_prompt", _classify_with_prompt)
    monkeypatch.setattr(policy_check_service.guard_classifier, "classify_conversation_per_model", _classify_conversation)
    monkeypatch.setattr(audit_log, "record_violation", lambda **kw: env.audits.append(kw))
    return env
