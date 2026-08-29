"""Health endpoint: a missing guard model must make the service report unhealthy.

Why this matters: when the configured guard model is not on Ollama, every check
returns 502, Module 07's hook is fail-open by design, and inference carries on
with nothing blocked. Both the block and the redact rules stop working, including
the pure-regex ones that never needed the model. Nothing errors and nothing looks
wrong, so the only defence is refusing to report healthy.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main

pytestmark = pytest.mark.asyncio


class _Settings:
    ollama_url = "http://fake-ollama:11434"
    proxy_model = "qwen2.5:7b"
    proxy_mode = "monitor"
    active_models = ["llama-guard3:8b"]


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


async def test_health_guardrail_enabled_is_none_when_state_unavailable(monkeypatch):
    """A Redis hiccup while reading the switch must not take the whole health
    check down with it — same fail-soft philosophy as the Ollama probe above."""
    client = _client(monkeypatch, missing=[])

    async def _broken():
        raise RuntimeError("Redis client not initialized — call init_redis() at startup")

    monkeypatch.setattr(main.state, "is_enabled", _broken)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["guardrail_enabled"] is None
