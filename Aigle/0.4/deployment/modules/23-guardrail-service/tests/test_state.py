"""Unit tests for the guardrail enable/disable switches (app/services/state.py)."""
from dataclasses import dataclass, field

import pytest
from fastapi import HTTPException

from app.core.config import get_settings
from app.services import state

pytestmark = pytest.mark.asyncio


class FakeRedis:
    """In-memory stand-in for redis.asyncio.Redis — just get/set, no TTL tracking."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def get(self, key):
        return self._store.get(key)

    async def set(self, key, value, **kwargs):
        assert "ex" not in kwargs and "px" not in kwargs, "guardrail:enabled must be set with no TTL"
        self._store[key] = value


@dataclass
class FakePolicy:
    name: str
    version: str


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(state.redis_conn, "get_redis", lambda: _immediate(redis))
    return redis


async def _immediate(value):
    return value


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_is_enabled_defaults_true_when_key_absent(fake_redis):
    assert await state.is_enabled() is True


async def test_is_enabled_false_when_key_is_zero(fake_redis):
    fake_redis._store[state.ENABLED_KEY] = "0"
    assert await state.is_enabled() is False


async def test_is_enabled_true_for_any_non_zero_value(fake_redis):
    fake_redis._store[state.ENABLED_KEY] = "1"
    assert await state.is_enabled() is True


async def test_set_enabled_writes_1_and_0_with_no_ttl(fake_redis):
    await state.set_enabled(True)
    assert fake_redis._store[state.ENABLED_KEY] == "1"
    await state.set_enabled(False)
    assert fake_redis._store[state.ENABLED_KEY] == "0"


async def test_is_policy_enabled_defaults_true_when_key_absent(fake_redis):
    assert await state.is_policy_enabled() is True


async def test_is_policy_enabled_false_when_key_is_zero(fake_redis):
    fake_redis._store[state.POLICY_ENABLED_KEY] = "0"
    assert await state.is_policy_enabled() is False


async def test_is_policy_enabled_true_for_any_non_zero_value(fake_redis):
    fake_redis._store[state.POLICY_ENABLED_KEY] = "1"
    assert await state.is_policy_enabled() is True


async def test_set_policy_enabled_writes_1_and_0_with_no_ttl(fake_redis):
    await state.set_policy_enabled(True)
    assert fake_redis._store[state.POLICY_ENABLED_KEY] == "1"
    await state.set_policy_enabled(False)
    assert fake_redis._store[state.POLICY_ENABLED_KEY] == "0"


@pytest.mark.parametrize(
    ("enabled_value", "policy_enabled_value", "expected"),
    [
        ("1", "1", True),   # both on -> policy checks run
        ("1", "0", False),  # global on, policy off -> disabled
        ("0", "1", False),  # global off, policy on -> disabled
        ("0", "0", False),  # both off -> disabled
    ],
)
async def test_is_policy_check_enabled_requires_both_switches_on(
    fake_redis, enabled_value, policy_enabled_value, expected,
):
    fake_redis._store[state.ENABLED_KEY] = enabled_value
    fake_redis._store[state.POLICY_ENABLED_KEY] = policy_enabled_value
    assert await state.is_policy_check_enabled() is expected


async def test_init_default_state_seeds_from_env_when_absent(fake_redis, monkeypatch):
    monkeypatch.setenv("GR_DEFAULT_ENABLED", "false")
    await state.init_default_state()
    assert fake_redis._store[state.ENABLED_KEY] == "0"


async def test_init_default_state_does_not_override_existing_key(fake_redis, monkeypatch):
    fake_redis._store[state.ENABLED_KEY] = "0"
    monkeypatch.setenv("GR_DEFAULT_ENABLED", "true")
    await state.init_default_state()
    assert fake_redis._store[state.ENABLED_KEY] == "0"


async def test_get_status_reports_active_policy(fake_redis, monkeypatch):
    fake_redis._store[state.ENABLED_KEY] = "1"
    fake_redis._store[state.POLICY_ENABLED_KEY] = "1"

    async def fake_get_active_policy():
        return FakePolicy(name="medical-safety", version="2.0")

    monkeypatch.setattr(state.policy_store, "get_active_policy", fake_get_active_policy)
    status = await state.get_status()
    assert status.enabled is True
    assert status.policy_enabled is True
    assert status.active_policy_name == "medical-safety"
    assert status.active_policy_version == "2.0"


async def test_get_status_reports_null_policy_when_none_active(fake_redis, monkeypatch):
    fake_redis._store[state.ENABLED_KEY] = "0"
    fake_redis._store[state.POLICY_ENABLED_KEY] = "0"

    async def fake_get_active_policy():
        raise HTTPException(status_code=404, detail="No active policy")

    monkeypatch.setattr(state.policy_store, "get_active_policy", fake_get_active_policy)
    status = await state.get_status()
    assert status.enabled is False
    assert status.policy_enabled is False
    assert status.active_policy_name is None
    assert status.active_policy_version is None
