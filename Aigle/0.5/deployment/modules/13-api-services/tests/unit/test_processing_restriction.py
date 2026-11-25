"""
PRIV-X3 / VIE01-357: restriction of processing (GDPR Art. 18) enforced by the
gateway's middleware -- writes 451 while the user's Keycloak attribute
gdpr_processing_restricted is true, reads and GDPR rights untouched.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_processing_restriction.py -v

The real gateway app, middleware and get_current_user run; only the token's
signature/UMA checks are stubbed, and Module 33 (which reads the Keycloak
attribute) is a fake HTTP client.
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.api import dependencies as deps
from app.api.dependencies import get_http_client
from app.core.config import get_settings
from app.main import app as gateway
from app.middlewares import processing_restriction as mw
from app.middlewares.processing_restriction import is_restricted_write
from fastapi.testclient import TestClient

V = get_settings().api_version
SUB = "restricted-user-1"


# ── Which requests count as writes ──────────────────────────────────────────

@pytest.mark.parametrize("method, path", [
    ("POST", f"/api/{V}/chat/chat"),
    ("POST", f"/api/{V}/asset/fileupload"),
    ("POST", f"/api/{V}/processing/process-file"),
    ("DELETE", f"/api/{V}/memory"),
    ("PATCH", f"/api/{V}/identity/abc"),
    ("POST", f"/api/{V}/a2a/query"),
    ("POST", f"/api/{V}/mcp"),
    ("PUT", f"/api/{V}/inference/x"),
    ("POST", "/api/0.4/chat/chat"),  # any version
    ("POST", f"/api/{V}/temporal/ingest"),           # stores new source material
    ("POST", f"/api/{V}/temporal/reason/v1/recompute"),
])
def test_blocked_while_restricted(method, path):
    assert is_restricted_write(method, path)


@pytest.mark.parametrize("method, path", [
    ("GET", f"/api/{V}/asset/users/commits"),
    ("GET", f"/api/{V}/memory/timeline"),
    ("POST", f"/api/{V}/gdpr/export"),
    ("POST", f"/api/{V}/gdpr/erasure"),
    ("POST", f"/api/{V}/gdpr/object"),
    ("PUT", f"/api/{V}/gdpr/consent/memory"),
    ("POST", f"/api/{V}/sso/logout"),
    ("POST", f"/api/{V}/search/hybrid"),
    ("POST", f"/api/{V}/asset/scan"),
    ("POST", f"/api/{V}/memory/multimedia/search"),
    ("POST", f"/api/{V}/memory/compact/evaluate"),
    ("POST", f"/api/{V}/temporal/verify"),           # only checks claims against the facts
    ("POST", f"/api/{V}/temporal/parse"),            # only reads a question's time condition (TMP-7)
])
def test_allowed_while_restricted(method, path):
    assert not is_restricted_write(method, path)


# ── Through the real gateway ────────────────────────────────────────────────

class _FakeClient:
    """Module 33 (GET /restrict/status answers from the restricted set) and
    every other backend (200 {})."""

    def __init__(self, restricted=(), status_error=None):
        self.restricted = set(restricted)
        self.status_error = status_error
        self.status_calls: list = []
        self.calls: list = []

    async def get(self, url, headers=None, timeout=None, **kwargs):
        assert url.endswith("/restrict/status")
        self.status_calls.append(headers)
        if self.status_error is not None:
            raise self.status_error
        return httpx.Response(200, json={"restricted": headers["X-User-ID"] in self.restricted},
                              request=httpx.Request("GET", url))

    async def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url})
        return httpx.Response(200, json={}, request=httpx.Request(method, url))

    async def post(self, url, **kwargs):
        return await self.request("POST", url)


class _Redis:  # the rate limiter's incr/expire, always under the limit
    def pipeline(self):
        class _Pipe:
            async def incr(self, key):
                pass

            async def expire(self, key, seconds):
                pass

            async def execute(self):
                return [1, True]

        return _Pipe()


@pytest.fixture
def gw(monkeypatch):
    async def _no_uma(token, path):
        return None

    user = {"sub": SUB}

    def _verify(request, settings=None):
        if not request.headers.get("Authorization"):
            raise deps.HTTPException(status_code=401, detail="Missing Authorization header")
        return dict(user)

    monkeypatch.setattr(deps, "verify_jwt", _verify)
    monkeypatch.setattr(mw, "verify_jwt", _verify)
    monkeypatch.setattr(deps, "check_uma_permission", _no_uma)
    saved = {k: getattr(gateway.state, k, None) for k in ("redis_client", "http_client")}
    fake = _FakeClient(restricted=[SUB])
    gateway.state.redis_client = _Redis()
    gateway.state.http_client = fake
    gateway.dependency_overrides[get_http_client] = lambda: fake
    yield TestClient(gateway, raise_server_exceptions=False), fake, user
    gateway.dependency_overrides.pop(get_http_client, None)
    for k, v in saved.items():
        setattr(gateway.state, k, v)


AUTH = {"Authorization": "Bearer x"}


def test_restricted_write_gets_451_with_the_tickets_body(gw):
    client, fake, _ = gw
    r = client.post(f"/api/{V}/chat/chat", json={"message": "hi"}, headers=AUTH)
    assert r.status_code == 451
    assert r.json() == {"reason": "processing_restricted"}
    assert fake.calls == []  # never reached the backend
    [headers] = fake.status_calls  # asked Module 33 for the Keycloak attribute, for the JWT sub
    assert headers["X-User-ID"] == SUB and "X-Internal-Token" in headers


def test_legacy_alias_write_is_blocked_too(gw):
    client, _, _ = gw
    r = client.post("/api/0.3/chat/chat", json={"message": "hi"}, headers=AUTH)
    assert r.status_code == 451


def test_restricted_user_keeps_reads_and_gdpr_rights(gw):
    client, fake, _ = gw
    assert client.get(f"/api/{V}/gdpr/consent", headers=AUTH).status_code == 200
    assert client.post(f"/api/{V}/gdpr/object", json={"processing_purpose": "memory"}, headers=AUTH).status_code == 200
    assert client.post(f"/api/{V}/gdpr/export", headers=AUTH).status_code == 200
    assert len(fake.calls) == 3
    assert fake.status_calls == []  # reads and /gdpr/* never trigger the lookup


def test_unrestricted_user_is_not_affected(gw):
    client, fake, user = gw
    user["sub"] = "free-user"
    r = client.post(f"/api/{V}/chat/chat", json={"message": "hi"}, headers=AUTH)
    assert r.status_code != 451
    assert len(fake.status_calls) == 1


def test_no_token_is_left_to_the_routes_own_auth(gw):
    client, fake, _ = gw
    r = client.post(f"/api/{V}/chat/chat", json={"message": "hi"})
    assert r.status_code == 401
    assert fake.status_calls == []


@pytest.mark.parametrize("error", [httpx.ConnectError("down"), httpx.ReadTimeout("slow")])
def test_unreadable_attribute_refuses_the_write(gw, error):
    client, fake, _ = gw
    fake.status_error = error
    r = client.post(f"/api/{V}/chat/chat", json={"message": "hi"}, headers=AUTH)
    assert r.status_code == 503
    assert r.json() == {"reason": "processing_restriction_check_unavailable"}
    assert fake.calls == []
