"""
GD-5 / VIE01-250: unit tests for the consent gateway routes (gdpr.py) and
the benchmark schema-upload owner header (benchmark.py).

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_consent_proxy.py -v

get_current_user and get_http_client are dependency-overridden, so no real
Keycloak or Module 33 is needed -- verifies the gateway's own logic: consent
is always the JWT sub's own (never a caller-chosen id), the end user's
X-Forwarded-For reaches Module 33, and a client can't choose a benchmark
schema's owner.
"""
import os
import sys
from typing import ClassVar

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.api.dependencies import get_current_user, get_http_client
from app.core.config import get_settings
from app.main import app
from app.routers import benchmark as benchmark_router
from fastapi.testclient import TestClient

GDPR_PREFIX = f"/api/{get_settings().api_version}/gdpr"
BENCHMARK_PREFIX = f"/api/{get_settings().api_version}/benchmark"
FAKE_SUB = "jwt-verified-user-42"


class _FakeClient:
    def __init__(self):
        self.calls: list = []
        self.status = 200
        self.body: dict = {}

    async def request(self, method, url, headers=None, timeout=None, json=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "json": json})
        return httpx.Response(self.status, json=self.body, request=httpx.Request(method, url))


@pytest.fixture
def fake():
    return _FakeClient()


@pytest.fixture(autouse=True)
def _overrides(fake):
    async def _user():
        return {"sub": FAKE_SUB}

    saved = {dep: app.dependency_overrides.get(dep) for dep in (get_current_user, get_http_client)}
    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_http_client] = lambda: fake
    yield
    for dep, prev in saved.items():
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


client = TestClient(app)


def test_get_consent_forwards_jwt_sub(fake):
    fake.body = {"purposes": [{"id": "memory", "granted": True, "granted_at": None, "description": "x"}]}
    r = client.get(f"{GDPR_PREFIX}/consent", headers={"Authorization": "Bearer x", "X-User-ID": "someone-else"})
    assert r.status_code == 200
    assert r.json() == fake.body
    call = fake.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith("/consent")
    assert call["headers"]["X-User-ID"] == FAKE_SUB
    assert "X-Internal-Token" in call["headers"]


def test_put_consent_forwards_body_sub_and_forwarded_for(fake):
    fake.body = {"id": "memory", "granted": False, "changed": True, "erasure_job_id": "job-1"}
    r = client.put(
        f"{GDPR_PREFIX}/consent/memory", json={"granted": False},
        headers={"Authorization": "Bearer x", "X-User-ID": "someone-else", "X-Forwarded-For": "203.0.113.7"},
    )
    assert r.status_code == 200
    assert r.json()["erasure_job_id"] == "job-1"
    call = fake.calls[0]
    assert call["method"] == "PUT"
    assert call["url"].endswith("/consent/memory")
    assert call["json"] == {"granted": False}
    assert call["headers"]["X-User-ID"] == FAKE_SUB
    assert call["headers"]["X-Forwarded-For"] == "203.0.113.7"


def test_put_consent_without_proxy_uses_client_address(fake):
    client.put(f"{GDPR_PREFIX}/consent/analytics", json={"granted": True}, headers={"Authorization": "Bearer x"})
    assert fake.calls[0]["headers"]["X-Forwarded-For"] == "testclient"


def test_put_consent_rejects_missing_granted(fake):
    r = client.put(f"{GDPR_PREFIX}/consent/memory", json={}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 422
    assert fake.calls == []


def test_unknown_purpose_error_propagates(fake):
    fake.status = 422
    fake.body = {"detail": "bad purpose"}
    r = client.put(f"{GDPR_PREFIX}/consent/nope", json={"granted": True}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 422


class _FakeBenchmarkClient:
    calls: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, headers=None, **kwargs):
        _FakeBenchmarkClient.calls.append({"url": url, "headers": dict(headers or {})})
        return httpx.Response(201, json={"id": "s1"}, request=httpx.Request(method, url))


@pytest.fixture
def fake_benchmark(monkeypatch):
    _FakeBenchmarkClient.calls = []
    monkeypatch.setattr(benchmark_router.httpx, "AsyncClient", _FakeBenchmarkClient)
    return _FakeBenchmarkClient


_SCHEMA = {
    "name": "s", "target_pipeline": "chat",
    "test_cases": [{"id": "c1", "input": {"q": "x"}}],
    "scoring_schema": {"dimensions": [{"name": "d", "weight": 1.0, "method": "exact"}]},
}


def test_schema_upload_owner_is_jwt_sub_not_client_header(fake_benchmark):
    r = client.post(
        f"{BENCHMARK_PREFIX}/schemas", json=_SCHEMA,
        headers={"Authorization": "Bearer x", "X-User-ID": "spoofed-owner"},
    )
    assert r.status_code == 201
    headers = {k.lower(): v for k, v in fake_benchmark.calls[0]["headers"].items()}
    assert headers["x-user-id"] == FAKE_SUB


def test_other_benchmark_routes_drop_client_user_header(fake_benchmark):
    client.get(f"{BENCHMARK_PREFIX}/schemas", headers={"Authorization": "Bearer x", "X-User-ID": "spoofed"})
    headers = {k.lower() for k in fake_benchmark.calls[0]["headers"]}
    assert "x-user-id" not in headers
