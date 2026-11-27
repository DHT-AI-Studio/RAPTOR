"""
Unit tests for the Module 26 memory gateway proxy (memory.py).

Run from the 13-api-services/ directory:
    conda run -n CIE python -m pytest test_memory_proxy.py -v

Uses FastAPI's dependency_overrides to bypass real JWT/Keycloak/UMA (get_current_user
returns a fixed fake sub), and monkeypatches httpx.AsyncClient inside the proxy
module so no real Module 26 is needed — verifies the proxy's own logic: requests
scope to the JWT sub, a caller-supplied X-User-ID never overrides it, and
paths/methods forward correctly.

CF-3 (VIE01-233): IngressHeaderStripMiddleware strips a client-supplied
X-User-ID before routing, so a spoofed value never reaches memory.py's 403
check (`_reject_cross_user_spoofing`) -- the request runs as the JWT sub. The
"spoof" tests below therefore assert the stripped behaviour end to end, and
test_reject_cross_user_spoofing_backstop exercises the 403 check directly; it
stays as defence in depth should the middleware ever be removed or bypassed.

Only covers routes still registered in memory.py's active surface (the 6 flat
routes plus the 3 kept alive for Module 21 — see memory.py's module docstring).
Every other Module 26 route memory.py used to proxy is commented out there,
not deleted, and has no test coverage here for the same reason: testing a 404
for an intentionally-disabled route documents nothing memory.py's own comments
don't already say, and silently drifts out of sync with memory.py if a route
is ever restored.

Paths are built from `get_settings().api_version` rather than a hardcoded
"0.3"/"0.4" string, so this file can't silently drift onto a stale version
prefix again the way it previously did (tests still called /api/0.3/memory/...
after the router was mounted under /api/0.4/memory/...).
"""
import json
import os
import sys
from typing import ClassVar

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import app.routers.memory as memory_module
import httpx
import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

FAKE_SUB = "jwt-verified-user-42"
MEMORY_PREFIX = f"/api/{get_settings().api_version}/memory"


async def _override_current_user():
    return {"sub": FAKE_SUB}


client = TestClient(app)


@pytest.fixture(autouse=True)
def _auth_override():
    """get_current_user is a process-wide key on the shared `app` singleton --
    setting it as a bare module-level statement leaks into every other test
    file collected in the same pytest session. Scope it to this file's tests
    only, save/restore around it, matching test_fileupload_decoupling.py /
    test_register_processing.py's convention."""
    previous = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = _override_current_user
    yield
    if previous is None:
        app.dependency_overrides.pop(get_current_user, None)
    else:
        app.dependency_overrides[get_current_user] = previous


@pytest.fixture(autouse=True)
def _disable_guardrail():
    """This file is about memory-proxy forwarding, not Guardrail -- /multimedia/search
    now runs its own input check (make_route_guardrail_dependency) before reaching this
    router, and GR_ENABLED defaults to true, so leaving it on would make
    test_search_multimedia_forwards_path depend on a real, unmocked network call to
    settings.guardrail_url. (Patching memory_module.httpx.AsyncClient to
    _CapturingAsyncClient, as this file already does, isn't a fix on its own: guardrail.py
    imports the same shared httpx module object, so that patch also replaces the client
    the guardrail dependency itself tries to use, and _CapturingAsyncClient has no .post()
    -- confirmed live: that was this fixture's actual failure mode before it existed.)"""
    settings = get_settings()
    previous = settings.gr_enabled
    settings.gr_enabled = False
    yield
    settings.gr_enabled = previous


class _CapturingAsyncClient:
    """Drop-in replacement for httpx.AsyncClient that records the outbound
    request and returns a canned response instead of hitting Module 26."""

    captured: ClassVar[dict] = {}
    response_json: ClassVar[dict] = {"ok": True}
    response_status: int = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, headers=None, params=None, content=None):
        _CapturingAsyncClient.captured = {
            "method": method, "url": url, "headers": dict(headers or {}),
            "params": dict(params or {}),
            "content": json.loads(content) if content else None,
        }
        return httpx.Response(
            _CapturingAsyncClient.response_status,
            json=_CapturingAsyncClient.response_json,
            request=httpx.Request(method, url),
        )


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch):
    monkeypatch.setattr(memory_module.httpx, "AsyncClient", _CapturingAsyncClient)
    yield


# ── Compact & compat aliases (kept live for Module 21's memory_client.py) ────

def test_evaluate_compact_forwards_to_module26_path():
    r = client.post(
        f"{MEMORY_PREFIX}/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory-service:8026/memory/compact/evaluate"
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_spoofed_x_user_id_is_stripped_and_jwt_sub_used():
    # Caller tries to spoof a different user via X-User-ID -- stripped at
    # ingress (CF-3), so Module 26 only ever sees the JWT sub.
    r = client.post(
        f"{MEMORY_PREFIX}/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB
    assert "attacker-controlled-id" not in _CapturingAsyncClient.captured["headers"].values()


def test_x_user_id_matching_jwt_sub_is_allowed():
    # Caller echoing their own sub back is not spoofing — must pass through.
    r = client.post(
        f"{MEMORY_PREFIX}/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake", "X-User-ID": FAKE_SUB},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB


def test_no_x_user_id_header_uses_jwt_sub():
    # The common case — caller doesn't send X-User-ID at all.
    r = client.post(
        f"{MEMORY_PREFIX}/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB


def test_get_session_summaries_is_get():
    r = client.get(
        f"{MEMORY_PREFIX}/sessions/my_sess/summaries",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "GET"
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory-service:8026/memory/sessions/my_sess/summaries"
    )


def test_upstream_502_on_connection_error(monkeypatch):
    class _FailingClient(_CapturingAsyncClient):
        async def request(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(memory_module.httpx, "AsyncClient", _FailingClient)
    r = client.post(
        f"{MEMORY_PREFIX}/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 502


def test_search_multimedia_forwards_path():
    r = client.post(
        f"{MEMORY_PREFIX}/multimedia/search",
        json={"query": "q"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory-service:8026/memory/multimedia/search"


# ── Management (kept) ────────────────────────────────────────────────────────

def test_delete_all_memory_forwards_bare_path():
    r = client.delete(
        MEMORY_PREFIX,
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory-service:8026/memory"


def test_delete_all_memory_spoofed_x_user_id_is_stripped():
    r = client.delete(
        MEMORY_PREFIX,
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB
    assert "attacker-controlled-id" not in _CapturingAsyncClient.captured["headers"].values()


# ── Flat top-level aliases ───────────────────────────────────────────────────

def test_store_memory_forwards_to_longterm_facts():
    r = client.post(
        f"{MEMORY_PREFIX}/store",
        json={"text": "偏好繁體中文", "frame_type": "preference"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory-service:8026/memory/longterm/facts"
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_retrieve_memory_forwards_to_search_as_post_json():
    r = client.get(
        f"{MEMORY_PREFIX}/retrieve",
        params={"query": "上季營收", "top_k": 3},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory-service:8026/memory/search"
    assert _CapturingAsyncClient.captured["method"] == "POST"
    assert _CapturingAsyncClient.captured["content"] == {"query": "上季營收", "top_k": 3}


def test_retrieve_memory_spoofed_x_user_id_is_stripped():
    r = client.get(
        f"{MEMORY_PREFIX}/retrieve",
        params={"query": "q"},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB
    assert "attacker-controlled-id" not in _CapturingAsyncClient.captured["headers"].values()


def test_timeline_memory_forwards_to_user_timeline():
    r = client.get(
        f"{MEMORY_PREFIX}/timeline",
        params={"page": 2, "page_size": 10},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory-service:8026/memory/timeline"
    assert _CapturingAsyncClient.captured["method"] == "GET"
    assert dict(_CapturingAsyncClient.captured["params"])["page"] == "2"


def test_compact_memory_forwards_to_default_session():
    r = client.post(
        f"{MEMORY_PREFIX}/compact",
        json={"trigger": "manual", "context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory-service:8026/memory/sessions/default/compact"
    )
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_compact_memory_forwards_explicit_session_id():
    r = client.post(
        f"{MEMORY_PREFIX}/compact",
        params={"session_id": "my-conversation-42"},
        json={"trigger": "manual", "context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory-service:8026/memory/sessions/my-conversation-42/compact"
    )


def test_compact_memory_spoofed_x_user_id_is_stripped():
    r = client.post(
        f"{MEMORY_PREFIX}/compact",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB
    assert "attacker-controlled-id" not in _CapturingAsyncClient.captured["headers"].values()


def test_archive_memory_forwards_to_default_session_turns():
    r = client.post(
        f"{MEMORY_PREFIX}/archive",
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory-service:8026/memory/sessions/default/turns"
    )
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_archive_memory_forwards_explicit_session_id():
    r = client.post(
        f"{MEMORY_PREFIX}/archive",
        params={"session_id": "my-conversation-42"},
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory-service:8026/memory/sessions/my-conversation-42/turns"
    )


def test_archive_memory_spoofed_x_user_id_is_stripped():
    r = client.post(
        f"{MEMORY_PREFIX}/archive",
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB
    assert "attacker-controlled-id" not in _CapturingAsyncClient.captured["headers"].values()



def test_reject_cross_user_spoofing_backstop():
    """The 403 check still rejects a mismatched X-User-ID if one ever reaches
    the router (e.g. the ingress middleware is removed or bypassed)."""
    from fastapi import HTTPException
    from starlette.requests import Request

    def _request(value):
        return Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"x-user-id", value)]})

    with pytest.raises(HTTPException) as exc:
        memory_module._reject_cross_user_spoofing(_request(b"attacker-controlled-id"), {"sub": FAKE_SUB})
    assert exc.value.status_code == 403
    memory_module._reject_cross_user_spoofing(_request(FAKE_SUB.encode()), {"sub": FAKE_SUB})


# ── Unauthenticated access ───────────────────────────────────────────────────
# All these routes are mounted with dependencies=[Depends(get_current_user)]
# (see main.py's app.include_router call), and get_current_user's first
# sub-dependency is an OAuth2PasswordBearer(auto_error=True) — it rejects a
# request with no Authorization header before get_current_user's own body
# (JWT verification, UMA check) ever runs. The _auth_override fixture above
# replaces get_current_user entirely, which would also bypass that
# OAuth2PasswordBearer sub-dependency, so each test below removes the
# override just for its own request and restores it afterwards.

def _without_auth_override():
    del app.dependency_overrides[get_current_user]
    return lambda: app.dependency_overrides.__setitem__(get_current_user, _override_current_user)


@pytest.mark.parametrize("method,path,kwargs", [
    ("post", f"{MEMORY_PREFIX}/store", {"json": {"text": "x", "frame_type": "preference"}}),
    ("get", f"{MEMORY_PREFIX}/retrieve", {"params": {"query": "q"}}),
    ("get", f"{MEMORY_PREFIX}/timeline", {}),
    ("post", f"{MEMORY_PREFIX}/compact", {"json": {"trigger": "manual", "context_window": 128000}}),
    ("delete", MEMORY_PREFIX, {}),
    ("post", f"{MEMORY_PREFIX}/archive", {"json": {"user_message": "hi", "assistant_response": "hello"}}),
])
def test_unauthenticated_request_rejected_with_401(method, path, kwargs):
    restore = _without_auth_override()
    try:
        r = getattr(client, method)(path, **kwargs)
        assert r.status_code == 401
    finally:
        restore()


# ── OpenAPI schema ────────────────────────────────────────────────────────────

def test_openapi_schema_includes_every_active_memory_route():
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    expected = {
        f"{MEMORY_PREFIX}/store": {"post"},
        f"{MEMORY_PREFIX}/retrieve": {"get"},
        f"{MEMORY_PREFIX}/timeline": {"get"},
        f"{MEMORY_PREFIX}/compact": {"post"},
        MEMORY_PREFIX: {"delete"},
        f"{MEMORY_PREFIX}/archive": {"post"},
        f"{MEMORY_PREFIX}/compact/evaluate": {"post"},
        f"{MEMORY_PREFIX}/sessions/{{session_id}}/summaries": {"get"},
        f"{MEMORY_PREFIX}/multimedia/search": {"post"},
    }
    for path, methods in expected.items():
        assert path in paths, f"missing path in OpenAPI schema: {path}"
        assert methods <= set(paths[path].keys()), f"missing method(s) for {path}: {methods - set(paths[path].keys())}"


# ── Canonical route (no trailing slash) ──────────────────────────────────────
# DELETE is registered as @router.delete("") under the /api/{version}/memory
# prefix, so the canonical path has no trailing slash. FastAPI's default
# redirect_slashes=True means a request to the slash variant gets a 307
# instead of 404 or 200 — assert that explicitly so a client relying on the
# slash variant is caught by a broken assumption, not a silent redirect.

def test_trailing_slash_variant_redirects_to_canonical_no_slash_path():
    r = client.delete(
        MEMORY_PREFIX + "/",
        headers={"Authorization": "Bearer fake"},
        follow_redirects=False,
    )
    assert r.status_code == 307
    assert r.headers["location"].rstrip("/").endswith(MEMORY_PREFIX)
