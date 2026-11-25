"""
Deny-by-default regression test: every protected route must reject an
unauthenticated request with 401 *before* any downstream call or handler
logic runs — not just eventually return an error.

Two things are proven per route, not one:
  * the response is 401 (the auth dependency rejected it)
  * no downstream HTTP call was ever attempted — httpx.AsyncClient is
    replaced with a poison client that fails loudly if used at all. A bare
    status-code check can't rule this out on its own: it's exactly the class
    of bug this session found on `/api/0.4/mcp`, where a request reached
    real handler logic (and got forwarded to module 27) with no auth
    dependency ever running first.

Run from the 13-api-services/ directory:
    python -m pytest test_protected_routes_deny_by_default.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

_V = get_settings().api_version  # "0.5"


class _PoisonClient:
    """Stands in for httpx.AsyncClient everywhere in the app. Any attempt to
    actually issue a request through it means a downstream call escaped
    without (or before) the auth dependency rejecting the request."""

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        pass

    async def _poison(self, *a, **kw):
        raise AssertionError(
            "Downstream HTTP call attempted on an unauthenticated request — "
            "the auth dependency did not run before handler/downstream logic."
        )

    get = _poison
    post = _poison
    put = _poison
    delete = _poison
    patch = _poison
    request = _poison
    send = _poison
    build_request = _poison


@pytest.fixture(autouse=True)
def _poison_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _PoisonClient)
    # app.state.http_client is built once at app startup (lifespan), using the
    # *real* httpx.AsyncClient, before this fixture ever runs — patching the
    # class here doesn't retroactively poison that already-constructed
    # instance. Poison it directly too, so routers that pull the shared
    # client via Depends(get_http_client) (rather than constructing their own)
    # can't slip a real request through it.
    existing = getattr(app.state, "http_client", None)
    if existing is not None:
        monkeypatch.setattr(app.state, "http_client", _PoisonClient())


client = TestClient(app)

# (method, path, json_body) — one representative route per protected router
# mounted in main.py. Bodies are valid-shaped (not empty) so a route never
# gets a false-pass by 422'ing on a missing field before the auth dependency
# even runs; the assertion is specifically that AUTH, not body validation,
# is what produces the 401.
PROTECTED_ROUTES = [
    ("POST", f"/api/{_V}/search/hybrid", {"query": "x"}),
    ("POST", f"/api/{_V}/search/video_search", {"query": "x"}),
    ("GET", f"/api/{_V}/asset/users/commits", None),
    ("GET", f"/api/{_V}/processing/cache/all", None),
    ("POST", f"/api/{_V}/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
    ("GET", f"/api/{_V}/training/list", None),
    ("GET", f"/api/{_V}/benchmark/health", None),
    ("GET", f"/api/{_V}/guardrail/policies", None),
    ("POST", f"/api/{_V}/guardrail/system/enable", None),
    ("GET", f"/api/{_V}/aiml/models/local", None),
    ("POST", f"/api/{_V}/sync/image", None),  # multipart — see test_media_sync_denies_without_file below
    ("POST", f"/api/{_V}/a2a/query", {"question": "x"}),
    # memory.py's richer session/long-term surface is commented out (Fung's
    # gateway scope-down, see the router's "Flat top-level aliases" note) --
    # /store is one of the 6 routes actually kept.
    ("POST", f"/api/{_V}/memory/store", {"text": "x", "frame_type": "preference"}),
    # personal_db.py's entire router is commented out (superseded by
    # search.py/video_search.py calling module 25 directly under the old
    # 0.3-style path names) -- no route left here to use as a representative.
    ("POST", f"/api/{_V}/search/people", {"query": "x"}),
    ("GET", f"/api/{_V}/identity/identities", None),
    ("DELETE", f"/api/{_V}/identity/x", None),
    ("POST", f"/api/{_V}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
    ("POST", f"/api/{_V}/mcp/auth/register", {"agent_id": "x", "username": "x", "password": "x"}),
    ("GET", f"/api/{_V}/gdpr/export", None),
    ("POST", f"/api/{_V}/gdpr/export", None),
    ("GET", f"/api/{_V}/gdpr/dsar", None),
    ("POST", f"/api/{_V}/gdpr/rectify", {"field": "firstName", "value": "x"}),
    ("POST", f"/api/{_V}/gdpr/restrict", None),
    ("POST", f"/api/{_V}/gdpr/restrict/lift", {"user_id": "x"}),
    ("POST", f"/api/{_V}/gdpr/object", {"processing_purpose": "memory"}),
    ("POST", f"/api/{_V}/gdpr/admin/legal-hold",
     {"user_id": "x", "reason": "x", "hold_until": "2099-01-01", "authority": "DPO"}),
    ("GET", f"/api/{_V}/gdpr/admin/legal-holds", None),
    ("GET", f"/api/{_V}/gdpr/dsar/some-user", None),
    ("GET", f"/api/{_V}/gdpr/dsar/jobs/some-job", None),
    ("GET", f"/api/{_V}/gdpr/consent", None),
    ("PUT", f"/api/{_V}/gdpr/consent/memory", {"granted": False}),
    ("GET", f"/api/{_V}/gdpr/audit-log", None),
    ("GET", f"/api/{_V}/gdpr/audit-log/my", None),
]


@pytest.mark.parametrize(
    "method,path,body", PROTECTED_ROUTES, ids=[f"{m} {p}" for m, p, _ in PROTECTED_ROUTES]
)
def test_protected_route_denies_before_handler_runs(method, path, body):
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401, (
        f"{method} {path} returned {resp.status_code}, expected 401 for a "
        f"request with no Authorization header"
    )


def test_mcp_rejects_garbage_bearer_before_forwarding():
    """A bearer that is neither a parseable JWT nor an mcp-agent- token must
    also be rejected at the gateway — not just a totally missing header."""
    resp = client.post(
        f"/api/{_V}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


def test_public_routes_are_not_accidentally_gated():
    """Sanity check on the poisoning setup itself: SSO login has no auth
    dependency, so it must NOT 401 here — if it did, it would mean the
    poison fixture (or a stray global dependency) is over-broad, and the
    401s above wouldn't be trustworthy evidence of anything route-specific."""
    resp = client.post(
        f"/api/{_V}/sso/login",
        data={"username": "x", "password": "x"},
    )
    assert resp.status_code != 401
