"""
GD-4 / VIE01-249: unit tests for the DSAR gateway routes (gdpr.py).

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_dsar_proxy.py -v

get_current_user and get_http_client are both dependency-overridden, so no
real Keycloak or Module 33 is needed -- verifies the gateway's own logic:
self-DSAR is scoped to the JWT sub (never a caller-chosen id), the admin
route is gdpr-admin gated, Module 33's 200-vs-202 is passed through
unchanged, and the job poll enforces ownership.
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

_current_user: dict = {}


class _FakeClient:
    def __init__(self):
        self.calls: list = []
        self.status = 200
        self.body: dict = {}

    async def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {})})
        return httpx.Response(self.status, json=self.body, request=httpx.Request(method, url))


@pytest.fixture
def fake():
    return _FakeClient()


@pytest.fixture(autouse=True)
def _overrides(fake):
    async def _user():
        return _current_user

    saved = {dep: app.dependency_overrides.get(dep) for dep in (get_current_user, get_http_client)}
    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_http_client] = lambda: fake
    _current_user.clear()
    _current_user.update({"sub": FAKE_SUB})
    yield
    for dep, prev in saved.items():
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


client = TestClient(app)


def _as_admin():
    _current_user["realm_access"] = {"roles": ["gdpr-admin"]}


def test_self_dsar_forwards_jwt_sub_and_returns_report(fake):
    fake.body = {"user_id": FAKE_SUB, "data_categories": {}}
    r = client.get(f"{GDPR_PREFIX}/dsar", headers={"Authorization": "Bearer x", "X-User-ID": "someone-else"})
    assert r.status_code == 200
    assert r.json()["user_id"] == FAKE_SUB
    call = fake.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith("/dsar")
    # Identity comes from the JWT, never from a caller-supplied header.
    assert call["headers"]["X-User-ID"] == FAKE_SUB
    assert "X-Internal-Token" in call["headers"]


def test_self_dsar_passes_through_202_pending(fake):
    fake.status = 202
    fake.body = {"job_id": "job-1", "status": "pending"}
    r = client.get(f"{GDPR_PREFIX}/dsar", headers={"Authorization": "Bearer x"})
    assert r.status_code == 202
    assert r.json() == {"job_id": "job-1", "status": "pending"}


def test_admin_dsar_requires_gdpr_admin(fake):
    r = client.get(f"{GDPR_PREFIX}/dsar/other-user", headers={"Authorization": "Bearer x"})
    assert r.status_code == 403
    assert fake.calls == []


def test_admin_dsar_targets_requested_user(fake):
    _as_admin()
    fake.body = {"user_id": "other-user"}
    r = client.get(f"{GDPR_PREFIX}/dsar/other-user", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    headers = fake.calls[0]["headers"]
    assert headers["X-User-ID"] == "other-user"
    assert headers["X-Requested-By"] == FAKE_SUB


def test_dsar_job_poll_owner_allowed(fake):
    fake.body = {"job_id": "job-1", "user_id": FAKE_SUB, "status": "completed", "result": {}}
    r = client.get(f"{GDPR_PREFIX}/dsar/jobs/job-1", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert fake.calls[0]["url"].endswith("/dsar/jobs/job-1")


def test_dsar_job_poll_other_users_job_forbidden(fake):
    fake.body = {"job_id": "job-1", "user_id": "victim", "status": "completed", "result": {"identity": "..."}}
    r = client.get(f"{GDPR_PREFIX}/dsar/jobs/job-1", headers={"Authorization": "Bearer x"})
    assert r.status_code == 403


def test_dsar_job_poll_admin_may_read_any_job(fake):
    _as_admin()
    fake.body = {"job_id": "job-1", "user_id": "victim", "status": "completed"}
    r = client.get(f"{GDPR_PREFIX}/dsar/jobs/job-1", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200


def test_privacy_service_error_status_propagates(fake):
    fake.status = 503
    fake.body = {"detail": "down"}
    r = client.get(f"{GDPR_PREFIX}/dsar", headers={"Authorization": "Bearer x"})
    assert r.status_code == 503
