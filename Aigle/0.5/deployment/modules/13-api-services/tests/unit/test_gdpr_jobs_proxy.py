"""
V05-10 / VIE01-364: the gateway routes for the operator job list --
gdpr-admin only, filters passed through to Module 33 unchanged.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_jobs_proxy.py -v
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

GDPR = f"/api/{get_settings().api_version}/gdpr"
SUB = "jwt-verified-user-42"
_user: dict = {}


class _FakeClient:
    def __init__(self):
        self.calls: list = []

    async def request(self, method, url, headers=None, timeout=None, params=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "params": dict(params or {})})
        return httpx.Response(200, json={"total": 0, "items": []}, request=httpx.Request(method, url))


@pytest.fixture
def fake():
    fake = _FakeClient()

    async def _current():
        return _user

    saved = {dep: app.dependency_overrides.get(dep) for dep in (get_current_user, get_http_client)}
    app.dependency_overrides[get_current_user] = _current
    app.dependency_overrides[get_http_client] = lambda: fake
    _user.clear()
    _user.update({"sub": SUB})
    yield fake
    for dep, prev in saved.items():
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


client = TestClient(app)
AUTH = {"Authorization": "Bearer x"}


@pytest.mark.parametrize("path", ["/jobs", "/jobs/summary"])
def test_jobs_routes_are_gdpr_admin_only(fake, path):
    r = client.get(f"{GDPR}{path}", headers=AUTH)
    assert r.status_code == 403 and fake.calls == []
    _user["realm_access"] = {"roles": ["gdpr-admin"]}
    r = client.get(f"{GDPR}{path}", headers=AUTH)
    assert r.status_code == 200
    [call] = fake.calls
    assert call["method"] == "GET" and call["url"].endswith(path)
    assert call["headers"]["X-Internal-Token"] == get_settings().gdpr_internal_token
    # An admin listing names no subject -- no caller id is passed as one.
    assert "X-User-ID" not in call["headers"]


def test_job_list_filters_pass_through(fake):
    _user["realm_access"] = {"roles": ["gdpr-admin"]}
    query = {"type": "erasure", "status": "partial_failure", "user_id": "bob", "stuck_hours": "48", "page": "2"}
    client.get(f"{GDPR}/jobs", params=query, headers=AUTH)
    assert fake.calls[0]["params"] == query


def test_summary_forwards_only_stuck_hours(fake):
    _user["realm_access"] = {"roles": ["gdpr-admin"]}
    client.get(f"{GDPR}/jobs/summary", params={"stuck_hours": "6", "user_id": "bob"}, headers=AUTH)
    assert fake.calls[0]["params"] == {"stuck_hours": "6"}


def test_module_33_errors_pass_through(fake):
    _user["realm_access"] = {"roles": ["gdpr-admin"]}

    async def reject(method, url, headers=None, timeout=None, params=None, **kwargs):
        return httpx.Response(422, json={"detail": "bad status"}, request=httpx.Request(method, url))

    fake.request = reject
    r = client.get(f"{GDPR}/jobs", params={"status": "nope"}, headers=AUTH)
    assert r.status_code == 422
