"""
PRIV-X4 / VIE01-358: the gateway routes for legal holds -- gdpr-admin only,
forwarded to Module 33 with the admin as X-Requested-By and never an
X-User-ID; Module 33's answers (e.g. 409 for an existing hold) pass through.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_legal_hold_proxy.py -v
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
SUB = "dpo-1"
HOLD = {"user_id": "alice", "reason": "regulatory_audit", "hold_until": "2099-01-01", "authority": "DPO"}
_user: dict = {}


class _FakeClient:
    def __init__(self):
        self.calls: list = []
        self.status = 200

    async def request(self, method, url, headers=None, timeout=None, json=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "json": json})
        body = {"detail": "This user already has an open legal hold"} if self.status != 200 else {"ok": True}
        return httpx.Response(self.status, json=body, request=httpx.Request(method, url))


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


def _admin():
    _user["realm_access"] = {"roles": ["gdpr-admin"]}


def test_both_routes_are_gdpr_admin_only(fake):
    assert client.post(f"{GDPR}/admin/legal-hold", json=HOLD, headers=AUTH).status_code == 403
    assert client.get(f"{GDPR}/admin/legal-holds", headers=AUTH).status_code == 403
    assert fake.calls == []


def test_placing_a_hold_forwards_the_body_with_the_admin_as_requester(fake):
    _admin()
    r = client.post(f"{GDPR}/admin/legal-hold", json=HOLD, headers=AUTH)
    assert r.status_code == 200
    [call] = fake.calls
    assert (call["method"], call["url"].endswith("/admin/legal-hold")) == ("POST", True)
    assert call["json"] == HOLD
    assert call["headers"]["X-Requested-By"] == SUB and "X-User-ID" not in call["headers"]


def test_listing_forwards(fake):
    _admin()
    assert client.get(f"{GDPR}/admin/legal-holds", headers=AUTH).status_code == 200
    [call] = fake.calls
    assert (call["method"], call["url"].endswith("/admin/legal-holds")) == ("GET", True)


def test_an_invalid_date_is_refused_before_module_33(fake):
    _admin()
    r = client.post(f"{GDPR}/admin/legal-hold", json={**HOLD, "hold_until": "next year"}, headers=AUTH)
    assert r.status_code == 422 and fake.calls == []


def test_module_33_errors_pass_through(fake):
    _admin()
    fake.status = 409
    r = client.post(f"{GDPR}/admin/legal-hold", json=HOLD, headers=AUTH)
    assert r.status_code == 409 and "open legal hold" in r.text
