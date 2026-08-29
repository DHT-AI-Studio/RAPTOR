"""Unit tests for app/engine/policy_engine._get_active_policy — the lookup
POST /policy/check/llm/* and POST /debug/policy/check/llm/* both use.

Covers the fix for the reported bug where those endpoints kept serving a
stale active policy after PUT /guardrail/policies/{id}/activate: Redis
errors (not just a cache miss) must fall back to Postgres, and every cache
write must carry app.engine.policy_store.ACTIVE_POLICY_CACHE_TTL so a
desynced cache self-heals instead of staying wrong indefinitely."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.engine import policy_engine, policy_store

pytestmark = pytest.mark.asyncio


def _row(policy_id):
    return {
        "id": policy_id, "name": "p", "version": "1.0", "raw_content": "fresh from postgres",
        "content_llama_guard": None, "content_granite_guardian": None, "content_gpt_oss_safeguard": None,
        "is_active": True, "created_at": datetime.now(timezone.utc),
    }


class FakeRedis:
    def __init__(self, get_return=None, fail_get: bool = False, fail_set: bool = False):
        self.get_return = get_return
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.set_calls: list[tuple] = []

    async def get(self, key):
        if self.fail_get:
            raise ConnectionError("redis unavailable")
        return self.get_return

    async def set(self, key, value, ex=None):
        if self.fail_set:
            raise ConnectionError("redis unavailable")
        self.set_calls.append((key, value, ex))


async def _immediate(value):
    return value


def _patch_postgres(monkeypatch, policy_id):
    monkeypatch.setattr(policy_engine.connection, "get_pool", lambda: _immediate(object()))

    async def fake_get_active_policy(pool):
        return _row(policy_id)

    monkeypatch.setattr(policy_engine.repo, "get_active_policy", fake_get_active_policy)


async def test_returns_cached_policy_on_a_clean_hit(monkeypatch):
    policy_id = uuid4()
    cached_detail = policy_store._row_to_detail(_row(policy_id))
    redis = FakeRedis(get_return=cached_detail.model_dump_json())
    monkeypatch.setattr(policy_engine.redis_conn, "get_redis", lambda: _immediate(redis))

    policy = await policy_engine._get_active_policy()

    assert policy.id == policy_id


async def test_falls_back_to_postgres_when_redis_get_raises(monkeypatch):
    """This is the read-side counterpart of the activation bug: even if the
    cache is unreachable (not just empty), the check endpoints must still
    resolve the real active policy instead of erroring out."""
    policy_id = uuid4()
    _patch_postgres(monkeypatch, policy_id)
    redis = FakeRedis(fail_get=True)
    monkeypatch.setattr(policy_engine.redis_conn, "get_redis", lambda: _immediate(redis))

    policy = await policy_engine._get_active_policy()

    assert policy.id == policy_id
    assert policy.raw_content == "fresh from postgres"


async def test_cache_miss_refresh_uses_the_shared_ttl_constant(monkeypatch):
    policy_id = uuid4()
    _patch_postgres(monkeypatch, policy_id)
    redis = FakeRedis(get_return=None)
    monkeypatch.setattr(policy_engine.redis_conn, "get_redis", lambda: _immediate(redis))

    await policy_engine._get_active_policy()

    assert redis.set_calls[0][0] == policy_store.ACTIVE_POLICY_CACHE_KEY
    assert redis.set_calls[0][2] == policy_store.ACTIVE_POLICY_CACHE_TTL


async def test_does_not_raise_when_the_opportunistic_recache_write_fails(monkeypatch):
    policy_id = uuid4()
    _patch_postgres(monkeypatch, policy_id)
    redis = FakeRedis(get_return=None, fail_set=True)
    monkeypatch.setattr(policy_engine.redis_conn, "get_redis", lambda: _immediate(redis))

    policy = await policy_engine._get_active_policy()

    assert policy.id == policy_id   # Postgres result still returned despite the failed re-cache
