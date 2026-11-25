"""
GD-2 / VIE01-247: the self-export gateway route answers GET (the ticket's
method) as well as POST (what it shipped with), identically.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_export_proxy.py -v

get_current_user and get_http_client are dependency-overridden, so no real
Keycloak or Module 33 is needed.
"""
import os
import sys

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
from fastapi.testclient import TestClient

GDPR_PREFIX = f"/api/{get_settings().api_version}/gdpr"
FAKE_SUB = "jwt-verified-user-42"


class _FakeClient:
    def __init__(self):
        self.calls: list = []

    async def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {})})
        return httpx.Response(200, json={"job_id": "job-1", "status": "pending"}, request=httpx.Request(method, url))


@pytest.fixture
def fake():
    fake = _FakeClient()

    async def _user():
        return {"sub": FAKE_SUB}

    saved = {dep: app.dependency_overrides.get(dep) for dep in (get_current_user, get_http_client)}
    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_http_client] = lambda: fake
    yield fake
    for dep, prev in saved.items():
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


client = TestClient(app)


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_self_export_starts_a_job_for_the_jwt_sub(fake, method):
    r = client.request(method, f"{GDPR_PREFIX}/export",
                       headers={"Authorization": "Bearer x", "X-User-ID": "someone-else"})
    assert r.status_code == 200
    assert r.json() == {"job_id": "job-1", "status": "pending"}
    [call] = fake.calls
    assert call["method"] == "POST" and call["url"].endswith("/export")
    assert call["headers"]["X-User-ID"] == FAKE_SUB  # never the caller-supplied header
