"""
PRIV-X3 / VIE01-357: the gateway routes for rectification, restriction and
objection -- scoped to the JWT sub (never a caller-supplied id), lift
gdpr-admin only, objection passing the caller's IP on for hashing.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_rights_proxy.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json

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

    async def request(self, method, url, headers=None, timeout=None, json=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "json": json})
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))


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
AUTH = {"Authorization": "Bearer x", "X-User-ID": "someone-else"}


def test_rectify_forwards_for_the_jwt_sub(fake):
    r = client.post(f"{GDPR}/rectify", json={"field": "firstName", "value": "Alicia"}, headers=AUTH)
    assert r.status_code == 200
    [call] = fake.calls
    assert (call["method"], call["url"].endswith("/rectify")) == ("POST", True)
    assert call["headers"]["X-User-ID"] == SUB  # never the caller-supplied header
    assert call["json"] == {"field": "firstName", "value": "Alicia"}


def test_restrict_works_with_or_without_a_reason(fake):
    assert client.post(f"{GDPR}/restrict", headers=AUTH).status_code == 200
    assert client.post(f"{GDPR}/restrict", json={"reason": "disputed"}, headers=AUTH).status_code == 200
    assert [c["json"] for c in fake.calls] == [{"reason": None}, {"reason": "disputed"}]
    assert {c["headers"]["X-User-ID"] for c in fake.calls} == {SUB}


def test_lift_is_gdpr_admin_only(fake):
    r = client.post(f"{GDPR}/restrict/lift", json={"user_id": "alice"}, headers=AUTH)
    assert r.status_code == 403 and fake.calls == []
    _user["realm_access"] = {"roles": ["gdpr-admin"]}
    r = client.post(f"{GDPR}/restrict/lift", json={"user_id": "alice"}, headers=AUTH)
    assert r.status_code == 200
    [call] = fake.calls
    assert call["headers"]["X-Requested-By"] == SUB and "X-User-ID" not in call["headers"]
    assert call["json"] == {"user_id": "alice"}


def test_object_passes_the_callers_ip_for_hashing(fake):
    r = client.post(f"{GDPR}/object", json={"processing_purpose": "analytics"},
                    headers={**AUTH, "X-Forwarded-For": "203.0.113.7"})
    assert r.status_code == 200
    [call] = fake.calls
    assert call["headers"]["X-User-ID"] == SUB and call["headers"]["X-Forwarded-For"] == "203.0.113.7"
    assert call["json"] == {"processing_purpose": "analytics"}


def test_module_33_errors_pass_through(fake):
    async def reject(method, url, headers=None, timeout=None, json=None, **kwargs):
        return httpx.Response(422, json={"detail": "isn't supported yet"}, request=httpx.Request(method, url))
    fake.request = reject
    r = client.post(f"{GDPR}/object", json={"processing_purpose": "face_detection"}, headers=AUTH)
    assert r.status_code == 422 and "isn't supported yet" in json.dumps(r.json())
