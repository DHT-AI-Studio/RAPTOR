"""
GD-9: the gateway routes for the Article 30 Record of Processing Activities
(RoPA) -- gdpr-admin only, no self variant (a RoPA is a controller-level
record), CCPA equivalent is a separate path.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_gdpr_records_proxy.py -v
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

    async def request(self, method, url, headers=None, timeout=None, json=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {})})
        return httpx.Response(200, json={"schema_version": "1.0"}, request=httpx.Request(method, url))


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


@pytest.mark.parametrize("path", ["/processing-records", "/processing-records/ccpa"])
def test_records_is_gdpr_admin_only(fake, path):
    r = client.get(f"{GDPR}{path}", headers=AUTH)
    assert r.status_code == 403 and fake.calls == []
    _user["realm_access"] = {"roles": ["gdpr-admin"]}
    r = client.get(f"{GDPR}{path}", headers=AUTH)
    assert r.status_code == 200
    [call] = fake.calls
    assert call["method"] == "GET" and call["url"].endswith(path)
    assert "X-Internal-Token" in call["headers"]


def test_module_33_errors_pass_through(fake):
    _user["realm_access"] = {"roles": ["gdpr-admin"]}

    async def reject(method, url, headers=None, timeout=None, json=None, **kwargs):
        return httpx.Response(503, json={"detail": "retention engine unavailable"}, request=httpx.Request(method, url))
    fake.request = reject
    r = client.get(f"{GDPR}/processing-records", headers=AUTH)
    assert r.status_code == 503 and "retention engine unavailable" in r.text
