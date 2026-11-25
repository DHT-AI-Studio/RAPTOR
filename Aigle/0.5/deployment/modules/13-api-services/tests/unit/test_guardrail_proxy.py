"""
guardrail.py's policy CRUD / system-switch proxy requires the `guardrail-admin`
Keycloak role -- without this, any logged-in user could globally disable
guardrail checking or rewrite the live policy content (confirmed live before
this fix existed).

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_guardrail_proxy.py -v
"""
import os
import sys
from typing import ClassVar
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

GUARDRAIL = f"/api/{get_settings().api_version}/guardrail"
POLICY_ID = str(uuid4())
_user: dict = {}


class _FakeResponse:
    """Stands in for the httpx.Response forward_request() reads (.json(), .content)."""
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.content = b"{}" if json_body is not None else b""

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Replaces the httpx.AsyncClient() guardrail.py's forward_request() constructs
    inline (it does not use the shared get_http_client dependency, unlike the other
    proxy routers), so dependency_overrides can't reach it -- patch the class instead."""
    calls: ClassVar[list] = []
    response = _FakeResponse()

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, method, url, headers=None, params=None, content=None, **kw):
        _FakeAsyncClient.calls.append({"method": method, "url": url, "content": content})
        return _FakeAsyncClient.response


@pytest.fixture
def fake():
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.response = _FakeResponse(200, {"ok": True})

    async def _current():
        return _user

    saved = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = _current
    _user.clear()
    _user.update({"sub": "jwt-verified-user-42"})
    with patch("app.routers.guardrail.httpx.AsyncClient", _FakeAsyncClient):
        yield _FakeAsyncClient
    if saved is None:
        app.dependency_overrides.pop(get_current_user, None)
    else:
        app.dependency_overrides[get_current_user] = saved


client = TestClient(app)
AUTH = {"Authorization": "Bearer x"}


@pytest.mark.parametrize(("method", "path", "kwargs"), [
    ("GET", "/policies", {}),
    ("GET", "/policies/active", {}),
    ("GET", "/policies/template", {}),
    ("GET", f"/policies/{POLICY_ID}", {}),
    ("POST", "/policies", {"params": {"name": "n", "version": "v"}, "content": b"desc"}),
    ("POST", "/policies/upload", {"files": {"file": ("policy.yaml", b"desc", "text/plain")},
                                   "data": {"name": "n", "version": "v"}}),
    ("PUT", f"/{POLICY_ID}", {"json": {"raw_content": "x"}}),
    ("PUT", f"/policies/{POLICY_ID}/activate", {}),
    ("PUT", f"/policies/{POLICY_ID}/deactivate", {}),
    ("DELETE", f"/policies/{POLICY_ID}", {}),
    ("POST", "/system/enable", {}),
    ("POST", "/system/disable", {}),
    ("POST", "/system/policy/enable", {}),
    ("POST", "/system/policy/disable", {}),
    ("GET", "/system/status", {}),
])
def test_every_route_requires_guardrail_admin(fake, method, path, kwargs):
    r = client.request(method, f"{GUARDRAIL}{path}", headers=AUTH, **kwargs)
    assert r.status_code == 403, f"{method} {path} should be blocked without the role, got {r.status_code}"
    assert fake.calls == [], f"{method} {path} must not reach module 23 without the role"

    _user["realm_access"] = {"roles": ["guardrail-admin"]}
    r = client.request(method, f"{GUARDRAIL}{path}", headers=AUTH, **kwargs)
    assert r.status_code < 400, f"{method} {path} with the role should succeed, got {r.status_code}: {r.text}"
    assert len(fake.calls) == 1
    del _user["realm_access"]


def test_disable_actually_forwards_to_module_23(fake):
    """The exact endpoint that was callable by any logged-in user before this fix:
    confirm it's blocked without the role and genuinely forwards with it."""
    r = client.post(f"{GUARDRAIL}/system/disable", headers=AUTH)
    assert r.status_code == 403

    _user["realm_access"] = {"roles": ["guardrail-admin"]}
    fake.response = _FakeResponse(200, {"enabled": False})
    r = client.post(f"{GUARDRAIL}/system/disable", headers=AUTH)
    assert r.status_code == 200 and r.json() == {"enabled": False}
    assert fake.calls[0]["url"].endswith("/guardrail/system/disable")
