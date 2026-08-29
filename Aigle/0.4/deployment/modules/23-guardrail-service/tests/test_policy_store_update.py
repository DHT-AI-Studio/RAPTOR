"""Unit tests for app/engine/policy_store.update_policy_content — verifies
the Redis active-policy cache is refreshed only when the edited policy is
the active one (otherwise a stale cache would keep serving old content).

Also covers the fix for the bug where a failed Redis write after a
successful Postgres commit left the active-policy cache permanently stale
(GET /guardrail/policies/active — Postgres-backed — showed the new policy,
but /policy/check/llm/* and /debug/policy/check/llm/* — Redis-cache-backed —
kept serving the old one, forever, since a stale-but-present cache key was
never re-checked against Postgres): _refresh_active_cache must not raise on
a Redis failure, and must best-effort delete the stale key so the next read
is a clean cache-miss instead of silently serving old content."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.engine import policy_store

pytestmark = pytest.mark.asyncio


def _row(is_active: bool, policy_id):
    return {
        "id": policy_id, "name": "p", "version": "1.0", "raw_content": "updated",
        "content_llama_guard": None, "content_granite_guardian": None, "content_gpt_oss_safeguard": None,
        "is_active": is_active, "created_at": datetime.now(timezone.utc),
    }


class FakeRedis:
    def __init__(self, fail_set: bool = False, fail_delete: bool = False):
        self.set_calls: list[tuple] = []
        self.delete_calls: list[str] = []
        self._fail_set = fail_set
        self._fail_delete = fail_delete

    async def set(self, key, value, ex=None):
        if self._fail_set:
            raise ConnectionError("redis unavailable")
        self.set_calls.append((key, value, ex))

    async def delete(self, key):
        if self._fail_delete:
            raise ConnectionError("redis unavailable")
        self.delete_calls.append(key)


async def _patch(monkeypatch, row, **redis_kwargs):
    async def fake_update(pool, policy_id, fields):
        return row

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "update_policy_content", fake_update)
    redis = FakeRedis(**redis_kwargs)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(redis))
    return redis


async def _immediate(value):
    return value


async def test_update_refreshes_redis_cache_when_policy_is_active(monkeypatch):
    policy_id = uuid4()
    redis = await _patch(monkeypatch, _row(is_active=True, policy_id=policy_id))

    detail = await policy_store.update_policy_content(policy_id, {"raw_content": "updated"})

    assert detail.raw_content == "updated"
    assert len(redis.set_calls) == 1
    assert redis.set_calls[0][0] == policy_store.ACTIVE_POLICY_CACHE_KEY


async def test_update_does_not_touch_redis_when_policy_is_inactive(monkeypatch):
    policy_id = uuid4()
    redis = await _patch(monkeypatch, _row(is_active=False, policy_id=policy_id))

    await policy_store.update_policy_content(policy_id, {"raw_content": "updated"})

    assert redis.set_calls == []


async def test_update_raises_404_when_policy_not_found(monkeypatch):
    async def fake_update(pool, policy_id, fields):
        return None

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "update_policy_content", fake_update)

    with pytest.raises(HTTPException) as exc_info:
        await policy_store.update_policy_content(uuid4(), {"raw_content": "x"})
    assert exc_info.value.status_code == 404


async def test_update_cache_write_uses_a_bounded_ttl(monkeypatch):
    policy_id = uuid4()
    redis = await _patch(monkeypatch, _row(is_active=True, policy_id=policy_id))

    await policy_store.update_policy_content(policy_id, {"raw_content": "updated"})

    assert redis.set_calls[0][2] == policy_store.ACTIVE_POLICY_CACHE_TTL


async def test_update_does_not_raise_when_redis_set_fails(monkeypatch):
    """The bug: Postgres already committed by the time this runs, so a Redis
    write failure here must not turn a successful update into a 500 — and
    must not leave a stale value sitting in the cache either (see the
    delete-on-failure assertion below)."""
    policy_id = uuid4()
    redis = await _patch(monkeypatch, _row(is_active=True, policy_id=policy_id), fail_set=True)

    detail = await policy_store.update_policy_content(policy_id, {"raw_content": "updated"})

    assert detail.raw_content == "updated"          # Postgres-sourced result still returned
    assert redis.delete_calls == [policy_store.ACTIVE_POLICY_CACHE_KEY]   # best-effort clean-miss


async def test_update_does_not_raise_when_both_redis_set_and_delete_fail(monkeypatch):
    policy_id = uuid4()
    await _patch(monkeypatch, _row(is_active=True, policy_id=policy_id), fail_set=True, fail_delete=True)

    detail = await policy_store.update_policy_content(policy_id, {"raw_content": "updated"})

    assert detail.raw_content == "updated"           # still doesn't raise — self-heals via TTL instead


async def test_activate_policy_refreshes_redis_cache_with_ttl(monkeypatch):
    policy_id = uuid4()
    row = _row(is_active=True, policy_id=policy_id)

    async def fake_activate(pool, pid):
        return row

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "activate_policy", fake_activate)
    redis = FakeRedis()
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(redis))

    detail = await policy_store.activate_policy(policy_id)

    assert detail.is_active is True
    assert redis.set_calls[0][0] == policy_store.ACTIVE_POLICY_CACHE_KEY
    assert redis.set_calls[0][2] == policy_store.ACTIVE_POLICY_CACHE_TTL


async def test_activate_policy_does_not_raise_when_redis_set_fails(monkeypatch):
    """Reproduces the reported bug's root cause directly: activation must
    still succeed (Postgres is the source of truth and already committed)
    even when the Redis cache refresh fails, and the stale key must be
    cleared rather than left to keep serving the previous policy forever."""
    policy_id = uuid4()
    row = _row(is_active=True, policy_id=policy_id)

    async def fake_activate(pool, pid):
        return row

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "activate_policy", fake_activate)
    redis = FakeRedis(fail_set=True)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(redis))

    detail = await policy_store.activate_policy(policy_id)

    assert detail.is_active is True
    assert redis.delete_calls == [policy_store.ACTIVE_POLICY_CACHE_KEY]


async def test_deactivate_policy_does_not_raise_when_redis_delete_fails(monkeypatch):
    policy_id = uuid4()
    row = _row(is_active=False, policy_id=policy_id)

    async def fake_deactivate(pool, pid):
        return "ok", row

    monkeypatch.setattr(policy_store.connection, "get_pool", lambda: _immediate(object()))
    monkeypatch.setattr(policy_store.repo, "deactivate_policy", fake_deactivate)
    redis = FakeRedis(fail_delete=True)
    monkeypatch.setattr(policy_store.redis_conn, "get_redis", lambda: _immediate(redis))

    detail = await policy_store.deactivate_policy(policy_id)

    assert detail.id == policy_id   # doesn't raise despite the Redis failure
