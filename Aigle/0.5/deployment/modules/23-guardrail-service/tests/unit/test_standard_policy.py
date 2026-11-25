"""Unit tests for the Standard Policy Format: the parser (app/models/policy.py),
default content (app/services/default_policy.py), per-model prompt
composition (app/services/policy_translator.py, app/adapters/base.py), and
its wiring into policy activation (app/engine/policy_store.py). Must never
raise on malformed input: anything that isn't valid standard-format JSON
returns None so callers fall back to verbatim/legacy behavior."""
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from app.models.policy import StandardPolicyCategory, parse_standard_policies

_VALID_ARRAY = json.dumps([
    {
        "id": "M1", "name": "Medical Misinformation",
        "description": "Dangerous, unproven medical claims presented as fact.",
        "severity": "high", "decision": "block",
        "criteria": ["Claims a substance cures a disease without evidence."],
        "exceptions": ["Academic discussion of debunked claims."],
        "examples": {
            "violation": ["Drinking bleach cures viral infections."],
            "allowed": ["Bleach is dangerous if ingested."],
        },
    },
    {
        "id": "F2", "name": "Financial Fraud",
        "description": "Content promoting scams or fraudulent schemes.",
        "severity": "critical", "decision": "block",
        "criteria": ["Promotes a guaranteed-return investment scheme."],
    },
])


def test_parses_valid_standard_format_array():
    result = parse_standard_policies(_VALID_ARRAY)
    assert result is not None
    assert len(result) == 2
    assert result[0].id == "M1"
    assert result[0].examples.violation == ["Drinking bleach cures viral infections."]
    assert result[1].id == "F2"
    assert result[1].exceptions == []          # optional field defaults to empty
    assert result[1].examples.violation == []  # optional nested field defaults to empty


def test_returns_none_for_malformed_json():
    assert parse_standard_policies("{not valid json") is None


def test_returns_none_for_plain_text():
    assert parse_standard_policies("Block anything about weapons. Escalate suspicious intent.") is None


def test_returns_none_for_single_object_not_array():
    single = json.dumps({
        "id": "M1", "name": "n", "description": "d",
        "severity": "low", "decision": "allow", "criteria": ["c"],
    })
    assert parse_standard_policies(single) is None


def test_returns_none_for_empty_array():
    assert parse_standard_policies("[]") is None


def test_returns_none_when_required_field_missing():
    missing_criteria = json.dumps([{
        "id": "M1", "name": "n", "description": "d", "severity": "low", "decision": "allow",
    }])
    assert parse_standard_policies(missing_criteria) is None


def test_returns_none_for_invalid_enum_value():
    bad_severity = json.dumps([{
        "id": "M1", "name": "n", "description": "d",
        "severity": "extreme", "decision": "allow", "criteria": ["c"],
    }])
    assert parse_standard_policies(bad_severity) is None


def test_returns_none_for_yaml_that_is_not_json():
    # legacy policies are often YAML — must not accidentally validate as JSON
    assert parse_standard_policies("name: my-policy\nrules:\n  - id: R1\n") is None


def test_optional_fields_default_correctly():
    minimal = json.dumps([{
        "id": "H1", "name": "Hate", "description": "d",
        "severity": "medium", "decision": "warn", "criteria": ["c1", "c2"],
    }])
    result = parse_standard_policies(minimal)
    assert result == [StandardPolicyCategory(
        id="H1", name="Hate", description="d", severity="medium", decision="warn",
        criteria=["c1", "c2"],
    )]


# ── app/services/default_policy.py — the shared 13-category default ────────────────

def test_default_standard_policy_has_13_categories_s1_through_s13():
    from app.services.default_policy import DEFAULT_STANDARD_POLICY

    categories = parse_standard_policies(json.dumps(DEFAULT_STANDARD_POLICY))
    assert categories is not None
    assert len(categories) == 13
    assert [c.id for c in categories] == [f"S{i}" for i in range(1, 14)]
    assert "S14" not in [c.id for c in categories]


# ── app/services/policy_translator.py — compose-at-activation ──────────────────────

_TWO_CATEGORIES = [
    {
        "id": "M1", "name": "Medical Misinformation", "description": "Dangerous unproven medical claims.",
        "severity": "high", "decision": "block",
        "criteria": ["Claims a substance cures a disease without evidence."],
        "exceptions": ["Academic discussion of debunked claims."],
        "examples": {"violation": ["Drinking bleach cures viral infections."], "allowed": ["Bleach is dangerous if ingested."]},
    },
    {
        "id": "F2", "name": "Financial Fraud", "description": "Content promoting scams.",
        "severity": "critical", "decision": "block",
        "criteria": ["Promotes a guaranteed-return investment scheme."],
    },
]


def test_translate_policy_to_prompts_returns_none_for_non_standard_content():
    from app.services.policy_translator import translate_policy_to_prompts

    assert translate_policy_to_prompts("name: my-policy\ninput_guardrail:\n  rules: []\n") is None
    assert translate_policy_to_prompts("plain free text") is None


def test_translate_policy_to_prompts_composes_all_three_models():
    from app.services.policy_translator import translate_policy_to_prompts

    composed = translate_policy_to_prompts(json.dumps(_TWO_CATEGORIES))

    assert composed is not None
    assert set(composed) == {"content_llama_guard", "content_granite_guardian", "content_gpt_oss_safeguard"}
    assert composed["content_llama_guard"].placement == "raw_completion"
    assert composed["content_granite_guardian"].placement == "final_user_message"
    assert composed["content_gpt_oss_safeguard"].placement == "system"
    # policy-derived content is baked in as literal text for all three
    assert "M1: Medical Misinformation" in composed["content_llama_guard"].template
    assert "[M1] Medical Misinformation" in composed["content_granite_guardian"].template
    assert "# Policy M1: Medical Misinformation" in composed["content_gpt_oss_safeguard"].template


def test_translate_policy_llama_guard_template_keeps_check_time_variables():
    from app.services.policy_translator import translate_policy_to_prompts

    composed = translate_policy_to_prompts(json.dumps(_TWO_CATEGORIES))
    template = composed["content_llama_guard"].template
    # Only these two are genuine runtime variables — everything else (the
    # category rules) is already resolved.
    assert "{{ check_role }}" in template
    assert "{{ conversation }}" in template


def test_translate_policy_granite_template_includes_no_think_instruction():
    # Fidelity fix vs. IBM's official docs (https://www.ibm.com/granite/docs/models/guardian) —
    # the pre-existing implementation omitted this mode-instruction line.
    from app.services.policy_translator import translate_policy_to_prompts

    composed = translate_policy_to_prompts(json.dumps(_TWO_CATEGORIES))
    assert "<no-think>" in composed["content_granite_guardian"].template


def test_translate_policy_gpt_oss_template_repeats_output_instruction():
    from app.services.policy_translator import translate_policy_to_prompts

    composed = translate_policy_to_prompts(json.dumps(_TWO_CATEGORIES))
    template = composed["content_gpt_oss_safeguard"].template
    assert template.count("Respond with exactly one character on the first line: 0 or 1.") >= 2


# ── app/adapters/base.py — parse_policy_content / render_composed_template ─────────

def test_parse_policy_content_returns_none_for_unset_column():
    from app.adapters.base import parse_policy_content

    assert parse_policy_content(None) is None
    assert parse_policy_content("") is None


def test_parse_policy_content_falls_back_to_raw_for_legacy_manual_override():
    from app.adapters.base import parse_policy_content

    result = parse_policy_content("You are a strict safety classifier.")
    assert result.raw == "You are a strict safety classifier."
    assert result.composed_template is None


def test_parse_policy_content_falls_back_to_raw_for_arbitrary_json():
    # A manual override that happens to be JSON but isn't our envelope shape
    # must not be misread as a composed template.
    from app.adapters.base import parse_policy_content

    result = parse_policy_content(json.dumps({"foo": "bar"}))
    assert result.composed_template is None
    assert result.raw == json.dumps({"foo": "bar"})


def test_parse_policy_content_round_trips_composed_template_envelope():
    import dataclasses

    from app.adapters.base import ComposedPromptTemplate, parse_policy_content

    envelope = json.dumps(dataclasses.asdict(ComposedPromptTemplate(placement="system", template="hello")))
    result = parse_policy_content(envelope)
    assert result.composed_template == ComposedPromptTemplate(placement="system", template="hello")


def test_render_composed_template_substitutes_check_time_variables():
    from app.adapters.base import ComposedPromptTemplate, render_composed_template

    composed = ComposedPromptTemplate(placement="raw_completion", template="Role: {{ check_role }}, Conv: {{ conversation }}")
    assert render_composed_template(composed, check_role="User", conversation="hi") == "Role: User, Conv: hi"


def test_render_composed_template_is_a_no_op_when_no_variables():
    from app.adapters.base import ComposedPromptTemplate, render_composed_template

    composed = ComposedPromptTemplate(placement="system", template="fixed text, no variables")
    assert render_composed_template(composed) == "fixed text, no variables"


def test_composed_llama_guard_template_survives_injection_attempt_in_category_text():
    # A category's own text containing literal Jinja2 syntax must not be
    # re-evaluated by the second, check-time render.
    from app.adapters.base import render_composed_template
    from app.services.policy_translator import translate_policy_to_prompts

    malicious = [{
        "id": "M1", "name": "Test", "description": "Contains {{ 7*7 }} and {% if true %}x{% endif %}.",
        "severity": "low", "decision": "block", "criteria": ["c"],
    }]
    composed = translate_policy_to_prompts(json.dumps(malicious))
    rendered = render_composed_template(composed["content_llama_guard"], check_role="User", conversation="hi")
    assert "{{ 7*7 }}" in rendered
    assert "49" not in rendered


# ── app/engine/policy_store.py — compose-and-persist wired into activation ─────────

def _policy_row(raw_content, policy_id=None, **overrides):
    row = {
        "id": policy_id or uuid4(), "name": "p", "version": "1.0", "raw_content": raw_content,
        "content_llama_guard": None, "content_granite_guardian": None, "content_gpt_oss_safeguard": None,
        "is_active": True, "created_at": datetime.now(timezone.utc),
    }
    row.update(overrides)
    return row


async def _immediate(value):
    return value


class _FakeRedis:
    async def set(self, key, value, ex=None):
        pass

    async def delete(self, key):
        pass


async def test_activate_policy_composes_and_persists_prompts_for_standard_format(monkeypatch):
    from app.engine import policy_store

    policy_id = uuid4()
    row = _policy_row(json.dumps(_TWO_CATEGORIES), policy_id=policy_id)
    updated_fields: dict = {}

    async def fake_activate(pool, pid):
        return row

    async def fake_update(pool, pid, fields):
        updated_fields.update(fields)
        return {**row, **fields}

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "activate_policy", fake_activate)
    monkeypatch.setattr(policy_store.repo, "update_policy_content", fake_update)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(_FakeRedis()))

    detail = await policy_store.activate_policy(policy_id)

    assert set(updated_fields) == {"content_llama_guard", "content_granite_guardian", "content_gpt_oss_safeguard"}
    assert json.loads(updated_fields["content_gpt_oss_safeguard"])["placement"] == "system"
    assert detail.content_gpt_oss_safeguard == updated_fields["content_gpt_oss_safeguard"]


async def test_activate_policy_never_overwrites_existing_manual_override(monkeypatch):
    from app.engine import policy_store

    policy_id = uuid4()
    row = _policy_row(
        json.dumps(_TWO_CATEGORIES), policy_id=policy_id,
        content_granite_guardian="a hand-written override, keep me untouched",
    )
    updated_fields: dict = {}

    async def fake_activate(pool, pid):
        return row

    async def fake_update(pool, pid, fields):
        updated_fields.update(fields)
        return {**row, **fields}

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "activate_policy", fake_activate)
    monkeypatch.setattr(policy_store.repo, "update_policy_content", fake_update)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(_FakeRedis()))

    await policy_store.activate_policy(policy_id)

    assert "content_granite_guardian" not in updated_fields
    assert set(updated_fields) == {"content_llama_guard", "content_gpt_oss_safeguard"}


async def test_activate_policy_skips_compose_for_checker_rules_format(monkeypatch):
    from app.engine import policy_store

    policy_id = uuid4()
    row = _policy_row("name: my-policy\ninput_guardrail:\n  rules: []\n", policy_id=policy_id)
    update_called = False

    async def fake_activate(pool, pid):
        return row

    async def fake_update(pool, pid, fields):
        nonlocal update_called
        update_called = True
        return row

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "activate_policy", fake_activate)
    monkeypatch.setattr(policy_store.repo, "update_policy_content", fake_update)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(_FakeRedis()))

    detail = await policy_store.activate_policy(policy_id)

    assert update_called is False
    assert detail.raw_content == row["raw_content"]


# ── app/routers/guardrail_policies.py — JSON Schema validation + template ──────────

def test_validate_if_standard_format_accepts_valid_array():
    from app.routers.guardrail_policies import _validate_if_standard_format

    _validate_if_standard_format(json.dumps(_TWO_CATEGORIES), "original")   # must not raise


def test_validate_if_standard_format_rejects_malformed_array():
    from app.routers.guardrail_policies import _validate_if_standard_format
    from fastapi import HTTPException

    malformed = json.dumps([{"id": "M1", "name": "n", "description": "d",
                              "severity": "extreme", "decision": "block", "criteria": ["c"]}])
    with pytest.raises(HTTPException) as exc_info:
        _validate_if_standard_format(malformed, "original")
    assert exc_info.value.status_code == 422


def test_validate_if_standard_format_ignores_non_json_content():
    from app.routers.guardrail_policies import _validate_if_standard_format

    _validate_if_standard_format("name: my-policy\ninput_guardrail:\n  rules: []\n", "original")   # must not raise


def test_validate_if_standard_format_ignores_non_original_targets():
    # Per-model overrides are always free-text verbatim, never schema-checked.
    from app.routers.guardrail_policies import _validate_if_standard_format

    malformed = json.dumps([{"id": "M1"}])   # missing every other required field
    _validate_if_standard_format(malformed, "llama-guard")   # must not raise


async def test_get_policy_template_standard_format_returns_13_categories():
    from app.routers.guardrail_policies import get_policy_template

    result = await get_policy_template(format="standard")
    categories = json.loads(result.example_json)
    assert len(categories) == 13
    assert [c["id"] for c in categories] == [f"S{i}" for i in range(1, 14)]


async def test_get_policy_template_checker_rules_format_unchanged_by_default():
    from app.routers.guardrail_policies import get_policy_template

    result = await get_policy_template()   # no format param -> default, backward compatible
    assert "rules" in result.example_yaml
    assert result.example_yaml != ""


# ── app/main.py — startup: always fall back to the default policy ──────────────────
# Supersedes the previous "seed only on an empty table, otherwise just warn"
# behavior — app.main._ensure_active_policy now simply delegates to
# app.engine.policy_store.ensure_default_policy_active(), the exact same
# fallback a runtime deactivate triggers (see the policy_store tests below).

import logging


async def test_ensure_active_policy_delegates_to_ensure_default_policy_active(monkeypatch):
    from app import main as app_main
    from app.engine import policy_store
    from app.engine.models import PolicyDetail

    fake_policy = PolicyDetail(
        id=uuid4(), name="default-standard-policy", version="1.0",
        raw_content="[]", is_active=True, created_at=datetime.now(timezone.utc),
    )

    async def fake_ensure_default_policy_active():
        return fake_policy

    monkeypatch.setattr(policy_store, "ensure_default_policy_active", fake_ensure_default_policy_active)

    result = await app_main._ensure_active_policy(logging.getLogger("test"))

    assert result is fake_policy


async def test_ensure_active_policy_warns_when_fallback_fails_and_guardrail_enabled(monkeypatch, caplog):
    # GB-13: "if GR_DEFAULT_ENABLED=true and no active policy exists, logs a WARNING"
    # — the only way to still have no active policy after startup is the
    # default-policy fallback itself failing; startup must carry on.
    from app import main as app_main
    from app.engine import policy_store

    async def failing_ensure_default_policy_active():
        raise RuntimeError("database unavailable")

    class _Settings:
        gr_default_enabled = True

    monkeypatch.setattr(policy_store, "ensure_default_policy_active", failing_ensure_default_policy_active)
    monkeypatch.setattr(app_main, "get_settings", lambda: _Settings())

    with caplog.at_level(logging.WARNING, logger="test"):
        result = await app_main._ensure_active_policy(logging.getLogger("test"))

    assert result is None
    # GB-17 / VIE01-245 wording.
    assert ("Guardrail enabled but no active policy configured — run seed_default_policy.py"
            in [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING])


# ── app/engine/policy_store.py — ensure_default_policy_active() ────────────────────

async def test_ensure_default_policy_active_creates_when_missing(monkeypatch):
    from app.engine import policy_store

    created_args = {}

    async def fake_get_by_name_version(pool, name, version):
        return None   # never created before

    async def fake_create_policy(name, version, raw_content, target="original"):
        created_args.update(name=name, version=version, raw_content=raw_content)
        from app.engine.models import PolicyCreateResponse
        return PolicyCreateResponse(id=uuid4(), name=name, version=version, created_at=datetime.now(timezone.utc))

    activated_id = None

    async def fake_activate_policy(policy_id):
        nonlocal activated_id
        activated_id = policy_id
        return _fake_detail(name=created_args["name"], version=created_args["version"], is_active=True)

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "get_policy_by_name_version", fake_get_by_name_version)
    monkeypatch.setattr(policy_store, "create_policy", fake_create_policy)
    monkeypatch.setattr(policy_store, "activate_policy", fake_activate_policy)

    result = await policy_store.ensure_default_policy_active()

    assert created_args["name"] == "default-standard-policy"
    categories = json.loads(created_args["raw_content"])
    assert len(categories) == 13   # the shared DEFAULT_STANDARD_POLICY content, not a stub
    assert result.is_active is True


async def test_ensure_default_policy_active_reuses_existing_without_overwriting_content(monkeypatch):
    # An admin may have hand-edited the default policy's content since it was
    # first created — falling back to it must never clobber that.
    from app.engine import policy_store

    existing_id = uuid4()
    create_policy_called = False

    async def fake_get_by_name_version(pool, name, version):
        return {"id": existing_id, "name": name, "version": version}

    async def fake_create_policy(*a, **kw):
        nonlocal create_policy_called
        create_policy_called = True
        raise AssertionError("must not re-create an already-existing default policy")

    async def fake_activate_policy(policy_id):
        assert policy_id == existing_id
        return _fake_detail(name="default-standard-policy", version="1.0",
                             raw_content="admin's hand-edited content", is_active=True)

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "get_policy_by_name_version", fake_get_by_name_version)
    monkeypatch.setattr(policy_store, "create_policy", fake_create_policy)
    monkeypatch.setattr(policy_store, "activate_policy", fake_activate_policy)

    result = await policy_store.ensure_default_policy_active()

    assert create_policy_called is False
    assert result.raw_content == "admin's hand-edited content"


def _fake_detail(name, version, is_active, raw_content="[]"):
    from app.engine.models import PolicyDetail
    return PolicyDetail(id=uuid4(), name=name, version=version, raw_content=raw_content,
                         is_active=is_active, created_at=datetime.now(timezone.utc))


# ── app/engine/policy_store.py — deactivate_policy() falls back to default ─────────

async def test_deactivate_policy_falls_back_to_default_when_deactivating_a_user_policy(monkeypatch):
    from app.engine import policy_store

    policy_id = uuid4()
    row = _policy_row("some content", policy_id=policy_id, name="my-custom-policy")
    fallback_called = False

    async def fake_deactivate(pool, pid):
        return "ok", row

    async def fake_ensure_default_policy_active():
        nonlocal fallback_called
        fallback_called = True
        return _fake_detail(name="default-standard-policy", version="1.0", is_active=True)

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "deactivate_policy", fake_deactivate)
    monkeypatch.setattr(policy_store, "ensure_default_policy_active", fake_ensure_default_policy_active)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(_FakeRedis()))

    detail = await policy_store.deactivate_policy(policy_id)

    assert fallback_called is True
    assert detail.name == "my-custom-policy"   # reports the policy just deactivated, not the fallback


async def test_deactivate_policy_does_not_fall_back_when_deactivating_the_default_itself(monkeypatch):
    from app.engine import policy_store

    policy_id = uuid4()
    row = _policy_row("[]", policy_id=policy_id, name="default-standard-policy", version="1.0")
    fallback_called = False

    async def fake_deactivate(pool, pid):
        return "ok", row

    async def fake_ensure_default_policy_active():
        nonlocal fallback_called
        fallback_called = True

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "deactivate_policy", fake_deactivate)
    monkeypatch.setattr(policy_store, "ensure_default_policy_active", fake_ensure_default_policy_active)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(_FakeRedis()))

    await policy_store.deactivate_policy(policy_id)

    assert fallback_called is False   # no self-reactivation loop


# ── app/engine/policy_store.py — delete_policy() protects the default ──────────────

async def test_delete_policy_rejects_deleting_the_default_policy(monkeypatch):
    from app.engine import policy_store
    from fastapi import HTTPException

    policy_id = uuid4()

    async def fake_delete_status(pool, pid, protected_name=None):
        assert protected_name == "default-standard-policy"
        return "protected"

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "delete_policy_status", fake_delete_status)

    with pytest.raises(HTTPException) as exc_info:
        await policy_store.delete_policy(policy_id)
    assert exc_info.value.status_code == 409
    assert "default-standard-policy" in exc_info.value.detail


async def test_delete_policy_allows_deleting_other_inactive_policies(monkeypatch):
    from app.engine import policy_store

    policy_id = uuid4()

    async def fake_delete_status(pool, pid, protected_name=None):
        return "ok"

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "delete_policy_status", fake_delete_status)

    await policy_store.delete_policy(policy_id)   # must not raise


# ── app/db/repo.py — delete_policy_status() protected-name check ───────────────────

async def test_repo_delete_policy_status_returns_protected_for_matching_name():
    from app.db import repo as db_repo

    class _NullContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeConn(_NullContext):
        async def fetchrow(self, query, *params):
            return {"is_active": False, "name": "default-standard-policy"}

        async def execute(self, query, *params):
            raise AssertionError("must not execute DELETE for a protected policy")

        def transaction(self):
            return _NullContext()

    class _FakePool:
        def acquire(self):
            class _Acquire(_NullContext):
                async def __aenter__(self):
                    return _FakeConn()
            return _Acquire()

    status = await db_repo.delete_policy_status(_FakePool(), uuid4(), protected_name="default-standard-policy")
    assert status == "protected"
