"""Health endpoint: a missing guard model must make the service report unhealthy.

Why this matters: when the configured guard model is not on Ollama, every check
returns 502, Module 07's hook is fail-open by design, and inference carries on
with nothing blocked. Both the block and the redact rules stop working, including
the pure-regex ones that never needed the model. Nothing errors and nothing looks
wrong, so the only defence is refusing to report healthy.
"""
from __future__ import annotations

from typing import ClassVar

import pytest
from app import main
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


class _Settings:
    ollama_url = "http://fake-ollama:11434"
    proxy_model = "gemma4:12b"
    proxy_mode = "monitor"
    active_models: ClassVar[list] = ["llama-guard3:8b"]


def _client(monkeypatch, missing, error=None):
    """The real health route on a bare app.

    Mounted standalone rather than using `main.app` because that app's lifespan
    opens Postgres and Redis, neither of which the health route touches.
    """
    from fastapi import FastAPI

    async def _probe(settings):
        return missing, error

    async def _enabled():
        return True

    monkeypatch.setattr(main, "_missing_guard_models", _probe)
    monkeypatch.setattr(main, "get_settings", lambda: _Settings())
    monkeypatch.setattr(main.state, "is_enabled", _enabled)

    bare = FastAPI()
    bare.get("/health")(main.health)
    return TestClient(bare, raise_server_exceptions=False)


async def test_health_is_ok_when_the_guard_model_is_present(monkeypatch):
    r = _client(monkeypatch, missing=[]).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_health_is_503_when_a_guard_model_is_missing(monkeypatch):
    r = _client(monkeypatch, missing=["llama-guard3:8b"]).get("/health")
    assert r.status_code == 503, "a missing guard model still reported healthy"
    body = r.json()
    assert body["status"] == "degraded"
    assert body["missing_guard_models"] == ["llama-guard3:8b"]
    assert "unguarded" in body["reason"], "the consequence is not stated in the reason"


async def test_health_is_503_when_ollama_cannot_be_reached(monkeypatch):
    """Distinguished from a missing model: we could not ask, so we cannot claim
    the guard works."""
    r = _client(monkeypatch, missing=None, error="connection refused").get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert "cannot reach Ollama" in body["reason"]
    assert "missing_guard_models" not in body


async def test_health_reports_guardrail_enabled_state(monkeypatch):
    """`guardrail_enabled` reflects the global switch (app/services/state.py) and
    is placed right after `status` in the response."""
    client = _client(monkeypatch, missing=[])

    async def _disabled():
        return False

    monkeypatch.setattr(main.state, "is_enabled", _disabled)

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["guardrail_enabled"] is False
    keys = list(body.keys())
    assert keys.index("guardrail_enabled") == keys.index("status") + 1


async def test_health_is_503_when_enabled_state_unavailable(monkeypatch):
    """checker.check() calls is_enabled() with no try/except of its own, so a
    Redis outage silently fails every request open (Module 07's hook treats the
    resulting error as fail-open too). /health must not hide that the same way
    it doesn't hide a missing guard model."""
    client = _client(monkeypatch, missing=[])

    async def _broken():
        raise RuntimeError("Redis client not initialized — call init_redis() at startup")

    monkeypatch.setattr(main.state, "is_enabled", _broken)

    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["guardrail_enabled"] is None
    assert "cannot determine guardrail enabled state" in body["reason"]


# ── GB-17 / VIE01-245: guard_model_reachable + active_policy_configured ───────

def _gb17_client(monkeypatch, missing, error=None, active=True):
    client = _client(monkeypatch, missing=missing, error=error)

    async def _active():
        return active

    monkeypatch.setattr(main, "_active_policy_configured", _active)
    return client


async def test_health_reports_both_startup_check_fields_when_ok(monkeypatch):
    r = _gb17_client(monkeypatch, missing=[]).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["guard_model_reachable"] is True
    assert body["active_policy_configured"] is True


async def test_guard_model_reachable_false_when_model_missing(monkeypatch):
    body = _gb17_client(monkeypatch, missing=["llama-guard3:8b"]).get("/health").json()
    assert body["guard_model_reachable"] is False


async def test_guard_model_reachable_false_when_ollama_down(monkeypatch):
    body = _gb17_client(monkeypatch, missing=None, error="connection refused").get("/health").json()
    assert body["guard_model_reachable"] is False


async def test_active_policy_configured_false_keeps_status_unchanged(monkeypatch):
    """The new field only reports; it does not change the existing 200/503 rules."""
    r = _gb17_client(monkeypatch, missing=[], active=False).get("/health")
    assert r.status_code == 200
    assert r.json()["active_policy_configured"] is False
    assert r.json()["status"] == "ok"


async def test_active_policy_configured_reads_postgres(monkeypatch):
    async def _pool():
        return object()

    monkeypatch.setattr(main.db_conn, "get_pool", _pool)

    async def _row(pool):
        return {"id": "x"}

    monkeypatch.setattr(main.policy_repo, "get_active_policy", _row)
    assert await main._active_policy_configured() is True

    async def _none(pool):
        return None

    monkeypatch.setattr(main.policy_repo, "get_active_policy", _none)
    assert await main._active_policy_configured() is False


async def test_active_policy_configured_false_when_db_unreachable(monkeypatch):
    async def _broken():
        raise RuntimeError("pool not initialised")

    monkeypatch.setattr(main.db_conn, "get_pool", _broken)
    assert await main._active_policy_configured() is False
