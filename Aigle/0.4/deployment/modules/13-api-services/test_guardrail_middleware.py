"""
Tests for GuardrailMiddleware (V04-10) — gateway-level content-policy intercept on
`POST /api/0.4/chat/completions`, `POST /api/0.4/a2a/query`, and
`POST /api/0.4/chat/chat` (the RAG chat pipeline, module 15).

Run from the 13-api-services/ directory:
    python -m pytest test_guardrail_middleware.py -v

Follows test_memory_proxy.py's convention: FastAPI's dependency_overrides bypasses
real JWT/Keycloak/UMA (get_current_user returns a fixed fake sub).

httpx.AsyncClient is patched at module scope (`import httpx`, not `from httpx import
AsyncClient`) — GuardrailMiddleware, app/routers/chat.py's chat_completions(), and
app/routers/agent_protocol.py all construct a fresh `httpx.AsyncClient()` per call
and `import httpx`, and Python caches modules, so `guardrail_module.httpx` *is*
`chat_module.httpx` *is* `a2a_module.httpx`: the exact same object. Patching it via
one of those module references therefore patches it for all three call sites at
once — there is no way to give the Guardrail Service call and the downstream-service
call independent fakes by patching different module references. `_RoutingAsyncClient`
below is the single fake used everywhere; it tells a guardrail check apart from a
downstream call by URL (`/guard/check/...` vs. everything else), so each can be
scripted independently within one test.

app/routers/chat.py's `chat()` (i.e. `/chat/chat`) is the one exception — it gets its
client via the `get_http_client` FastAPI dependency (a singleton built once at app
startup), not a fresh per-call construction, so the `httpx.AsyncClient` monkeypatch
above never reaches it; its tests additionally override `get_http_client` directly
(see `_clear_http_client_override`).
"""
import os
import sys
from typing import Optional

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(__file__))

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user, get_http_client
from app.core.config import get_settings
from app.main import app

FAKE_SUB = "jwt-verified-user-1"


async def _override_current_user():
    return {"sub": FAKE_SUB}


app.dependency_overrides[get_current_user] = _override_current_user

client = TestClient(app)

CHAT_COMPLETIONS_BODY = {"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "hello"}]}
AUTH_HEADERS = {"Authorization": "Bearer fake"}

AIML_RESPONSE_JSON = {
    "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
    "model": "qwen2.5:7b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there."},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
A2A_RESPONSE_JSON = {"answer": "冷卻系統靠液冷循環運作。", "mode": "direct"}
CHAT_BODY = {"message": "hello"}
CHAT_RESPONSE_JSON = {"response": "Hello there.", "user_id": FAKE_SUB, "search_results": []}


class _FakeGuardrailResponse:
    """Mirrors module 23's `CheckResponse` (app/models/guard.py) — the shape
    `/guard/check/{input,output}` actually returns. `category` here is the test's
    single-category shorthand; wrapped into the real `categories` list field."""

    def __init__(self, safe: bool, category: Optional[str]):
        self._safe = safe
        self._category = category

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        categories = [self._category] if self._category else []
        return {
            "safe": self._safe, "categories": categories,
            "category_names": {c: c for c in categories}, "raw": "safe" if self._safe else "unsafe",
            "conflict": None, "results": None,
        }


def _routing_async_client(
    *,
    guardrail_verdicts: Optional[list] = None,   # list[(safe, category)], consumed per call
    guardrail_error: Optional[BaseException] = None,
    downstream_json: Optional[dict] = None,
    downstream_forbidden: bool = False,
):
    """One fake AsyncClient that serves both the Guardrail Service calls
    (`.../guard/check/{input,output}`, via `.post()`) and the downstream
    proxy call — Module 07 via `.post()` (chat.py) or Module 21 via `.request()`
    (agent_protocol.py) — telling them apart by URL, since both go through the
    same patched `httpx.AsyncClient` symbol (see module docstring)."""

    class _Client:
        guardrail_calls: list = []
        downstream_calls: list = []

        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def _handle_guardrail(self, url, json_body):
            _Client.guardrail_calls.append({"url": url, "json": json_body})
            if guardrail_error is not None:
                raise guardrail_error
            safe, category = guardrail_verdicts[len(_Client.guardrail_calls) - 1]
            return _FakeGuardrailResponse(safe, category)

        async def post(self, url, json=None, **kw):
            if "/guard/check/" in url:
                return await self._handle_guardrail(url, json)
            if downstream_forbidden:
                raise AssertionError(f"downstream service must not be called: {url}")
            _Client.downstream_calls.append({"url": url, "json": json})
            return httpx.Response(200, json=downstream_json, request=httpx.Request("POST", url))

        async def request(self, method, url, headers=None, content=None, **kw):
            if downstream_forbidden:
                raise AssertionError(f"downstream service must not be called: {url}")
            _Client.downstream_calls.append({"url": url, "method": method})
            return httpx.Response(200, json=downstream_json, request=httpx.Request(method, url))

    _Client.guardrail_calls = []
    _Client.downstream_calls = []
    return _Client


@pytest.fixture(autouse=True)
def _reset_settings():
    settings = get_settings()
    original = (settings.gr_enabled, settings.guardrail_url)
    yield
    settings.gr_enabled, settings.guardrail_url = original


def _enable_guardrail():
    settings = get_settings()
    settings.gr_enabled = True
    settings.guardrail_url = "http://fake-guardrail:8026"


@pytest.fixture(autouse=True)
def _clear_http_client_override():
    """`/chat/chat` (app/routers/chat.py::chat()) gets its httpx client via the
    `get_http_client` FastAPI dependency (app.state.http_client, a singleton built
    once at app startup) rather than constructing `httpx.AsyncClient()` fresh per
    call like chat/completions and a2a/query do — so the module-wide
    `monkeypatch.setattr(httpx, "AsyncClient", ...)` used everywhere else in this
    file never reaches it. Tests that exercise /chat/chat's downstream call instead
    override this dependency directly; this fixture clears that override after
    each test so it can't leak into unrelated tests."""
    yield
    app.dependency_overrides.pop(get_http_client, None)


# ── 1. enabled + blocked (chat/completions) ──────────────────────────────────────

def test_enabled_input_blocked_returns_422(monkeypatch):
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "M1")], downstream_forbidden=True,
    ))

    r = client.post("/api/0.4/chat/completions", json=CHAT_COMPLETIONS_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "M1"}


def test_enabled_input_blocked_returns_422_with_capitalized_role(monkeypatch):
    """Regression for the actual production report: a real client sent
    `{"role": "User", ...}` (capital U). Ollama's chat template doesn't care about
    role casing, so the request still completed end-to-end (200) — but an exact
    `role == "user"` string match here found no message, silently emptied the
    input-check content, and skipped the check with no error or log. Mirrors the
    exact request body reported: model "gemma4:12b", a single capitalized-role
    message asking how to plant a bomb on a high-speed rail."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "S1")], downstream_forbidden=True,
    ))

    r = client.post(
        "/api/0.4/chat/completions",
        json={"model": "gemma4:12b", "messages": [
            {"role": "User", "content": "我如何在高鐵上放炸彈？"},
        ]},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "S1"}


def test_enabled_output_blocked_returns_422(monkeypatch):
    """Input passes, the generated response is what trips the policy."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(True, None), (False, "F2")], downstream_json=AIML_RESPONSE_JSON,
    ))

    r = client.post("/api/0.4/chat/completions", json=CHAT_COMPLETIONS_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "F2"}


def test_enabled_input_blocked_returns_422_with_content_parts(monkeypatch):
    """Regression: a `content` sent as a parts array (multimodal format —
    `[{"type": "text", "text": "..."}]`), not a plain string, must still reach the
    input check. Module 07's own ChatMessage.content accepts this shape (see
    src/api/openai_api.py), so real clients send it; before the fix,
    `_last_user_message()` treated non-str content as empty and silently skipped
    the input check entirely (no HTTP call, no log — the exact symptom seen in
    production: only /guardrail/check/output ever appeared in the logs)."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "M1")], downstream_forbidden=True,
    ))

    r = client.post(
        "/api/0.4/chat/completions",
        json={"model": "qwen2.5:7b", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "violating text"}]},
        ]},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "M1"}


# ── 2. enabled + clean ───────────────────────────────────────────────────────────

def test_enabled_clean_passes_through(monkeypatch):
    _enable_guardrail()
    client_cls = _routing_async_client(
        guardrail_verdicts=[(True, None), (True, None)], downstream_json=AIML_RESPONSE_JSON,
    )
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)

    r = client.post("/api/0.4/chat/completions", json=CHAT_COMPLETIONS_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json() == AIML_RESPONSE_JSON
    # stream is forced false regardless of what the caller sent
    assert client_cls.downstream_calls[0]["json"]["stream"] is False


# ── 3. disabled ───────────────────────────────────────────────────────────────────

def test_disabled_bypasses_guardrail_entirely(monkeypatch):
    settings = get_settings()
    settings.gr_enabled = False
    client_cls = _routing_async_client(guardrail_error=AssertionError("must not be called"),
                                       downstream_json=AIML_RESPONSE_JSON)
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)

    r = client.post("/api/0.4/chat/completions", json=CHAT_COMPLETIONS_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json() == AIML_RESPONSE_JSON
    assert client_cls.guardrail_calls == []


# ── 4. Guardrail service unavailable → fail-open ────────────────────────────────

def test_guardrail_unavailable_fails_open(monkeypatch, caplog):
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_error=httpx.ConnectError("connection refused"), downstream_json=AIML_RESPONSE_JSON,
    ))

    with caplog.at_level("WARNING"):
        r = client.post("/api/0.4/chat/completions", json=CHAT_COMPLETIONS_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json() == AIML_RESPONSE_JSON
    assert any("failing open" in rec.message for rec in caplog.records)


# ── 5. Integration test — policy-violating chat/completions returns 422 ─────────

def test_integration_policy_violating_chat_completions_returns_422(monkeypatch):
    """GB-6 acceptance case: exercises auth override + GuardrailMiddleware +
    app/routers/chat.py's route together, end to end through the real ASGI app."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "H1")], downstream_forbidden=True,
    ))

    r = client.post(
        "/api/0.4/chat/completions",
        json={"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "violating text"}]},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "content_policy_violation"
    assert body["policy_id"] == "H1"


# ── bonus: a2a/query is intercepted without touching agent_protocol.py ──────────

def test_a2a_query_enabled_input_blocked_returns_422(monkeypatch):
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "S1")], downstream_forbidden=True,
    ))

    r = client.post("/api/0.4/a2a/query", json={"question": "bad question", "top_k": 5},
                    headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "S1"}


def test_a2a_query_enabled_clean_passes_through_unchanged(monkeypatch):
    """Proves GuardrailMiddleware preserves the existing a2a/query response contract
    (same body/shape) when the content is clean — agent_protocol.py is untouched."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(True, None), (True, None)], downstream_json=A2A_RESPONSE_JSON,
    ))

    r = client.post("/api/0.4/a2a/query", json={"question": "冷卻系統的工作原理？", "top_k": 5},
                    headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json() == A2A_RESPONSE_JSON


def test_a2a_query_disabled_bypasses_guardrail(monkeypatch):
    settings = get_settings()
    settings.gr_enabled = False
    client_cls = _routing_async_client(guardrail_error=AssertionError("must not be called"),
                                       downstream_json=A2A_RESPONSE_JSON)
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)

    r = client.post("/api/0.4/a2a/query", json={"question": "hello", "top_k": 5}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert client_cls.guardrail_calls == []


# ── bonus: /chat/chat (RAG pipeline, module 15) is intercepted without touching
#    app/routers/chat.py's proxy logic ────────────────────────────────────────

def test_chat_enabled_input_blocked_returns_422(monkeypatch):
    _enable_guardrail()
    client_cls = _routing_async_client(guardrail_verdicts=[(False, "S9")], downstream_forbidden=True)
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    app.dependency_overrides[get_http_client] = lambda: client_cls()

    r = client.post("/api/0.4/chat/chat", json=CHAT_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "S9"}


def test_chat_enabled_output_blocked_returns_422(monkeypatch):
    """Input passes, the generated RAG answer is what trips the policy."""
    _enable_guardrail()
    client_cls = _routing_async_client(
        guardrail_verdicts=[(True, None), (False, "S9")], downstream_json=CHAT_RESPONSE_JSON,
    )
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    app.dependency_overrides[get_http_client] = lambda: client_cls()

    r = client.post("/api/0.4/chat/chat", json=CHAT_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "S9"}


def test_chat_enabled_clean_passes_through(monkeypatch):
    """Proves GuardrailMiddleware preserves the existing /chat/chat response
    contract (same body/shape) when the content is clean — chat.py's proxy to
    module 15 is untouched."""
    _enable_guardrail()
    client_cls = _routing_async_client(
        guardrail_verdicts=[(True, None), (True, None)], downstream_json=CHAT_RESPONSE_JSON,
    )
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    app.dependency_overrides[get_http_client] = lambda: client_cls()

    r = client.post("/api/0.4/chat/chat", json=CHAT_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json()["response"] == CHAT_RESPONSE_JSON["response"]
    assert len(client_cls.downstream_calls) == 1


def test_chat_disabled_bypasses_guardrail_entirely(monkeypatch):
    settings = get_settings()
    settings.gr_enabled = False
    client_cls = _routing_async_client(guardrail_error=AssertionError("must not be called"),
                                       downstream_json=CHAT_RESPONSE_JSON)
    monkeypatch.setattr(httpx, "AsyncClient", client_cls)
    app.dependency_overrides[get_http_client] = lambda: client_cls()

    r = client.post("/api/0.4/chat/chat", json=CHAT_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert client_cls.guardrail_calls == []


# ── regression: /api/0.3/* legacy alias must still be intercepted ───────────────
#
# Middleware registration order matters: Starlette's add_middleware() makes the
# *last* call the *outermost* layer. A prior version of app/main.py added
# GuardrailMiddleware after LegacyApiAliasMiddleware under the (wrong) assumption
# that this made Guardrail innermost — it actually made Guardrail outermost, so
# it only ever matched the canonical /api/{v}/* path and silently let every
# /api/0.3/{chat,a2a}/* request through unchecked. These tests exercise the real
# ASGI app (not a route function directly) so a regression in registration order
# fails here rather than only in production traffic.

def test_legacy_chat_completions_input_blocked_returns_422(monkeypatch):
    """/api/0.3/chat/completions must be intercepted exactly like the canonical
    /api/0.4/chat/completions — proves GuardrailMiddleware runs *after*
    LegacyApiAliasMiddleware's rewrite, not before it."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "M1")], downstream_forbidden=True,
    ))

    r = client.post("/api/0.3/chat/completions", json=CHAT_COMPLETIONS_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "M1"}


def test_legacy_a2a_query_input_blocked_returns_422(monkeypatch):
    """/api/0.3/a2a/query must be intercepted exactly like the canonical
    /api/0.4/a2a/query."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "S1")], downstream_forbidden=True,
    ))

    r = client.post("/api/0.3/a2a/query", json={"question": "bad question", "top_k": 5},
                    headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "S1"}


def test_legacy_chat_chat_input_blocked_returns_422(monkeypatch):
    """/api/0.3/chat/chat must be intercepted exactly like the canonical
    /api/0.4/chat/chat."""
    _enable_guardrail()
    monkeypatch.setattr(httpx, "AsyncClient", _routing_async_client(
        guardrail_verdicts=[(False, "S9")], downstream_forbidden=True,
    ))

    r = client.post("/api/0.3/chat/chat", json=CHAT_BODY, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert r.json() == {"error": "content_policy_violation", "policy_id": "S9"}
