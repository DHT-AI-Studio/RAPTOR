"""CF-3 / VIE01-233: IngressHeaderStripMiddleware drops client-supplied identity
and internal-trust headers before any router or proxy sees them.

The first half uses a tiny echo app (the middleware in isolation); the second
half goes through the real gateway (`app.main.app`) to prove the middleware is
mounted outermost: a spoofed header never reaches Module 21 through the A2A
proxies, on the canonical and the legacy /api/0.3 path, while the JWT-derived
X-User-ID is still sent.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')
os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://auth:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://keycloak:8080")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app as gateway
from app.middlewares.ingress_header_strip import (
    STRIPPED_HEADERS,
    IngressHeaderStripMiddleware,
)
from app.routers import agent_protocol
from fastapi import FastAPI, Request, WebSocket
from fastapi.testclient import TestClient

SPOOFED = {
    "X-User-ID": "attacker",
    "X-Branch-ID": "attacker-branch",
    "X-Internal": "true",
    "X-A2A-Internal": "true",
    "X-Internal-Token": "guessed-token",
}


@pytest.fixture
def echo_client():
    app = FastAPI()
    app.add_middleware(IngressHeaderStripMiddleware)

    @app.get("/echo")
    @app.get("/admin/maintenance/status")
    @app.get("/admin/maintenance-other")
    async def echo(request: Request):
        return {k.lower(): v for k, v in request.headers.items()}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({k.lower(): v for k, v in websocket.headers.items()})
        await websocket.close()

    return TestClient(app)


def test_strip_list_is_the_five_cf3_headers():
    assert {h.decode() for h in STRIPPED_HEADERS} == {h.lower() for h in SPOOFED}


def test_every_spoofed_header_is_stripped(echo_client):
    seen = echo_client.get("/echo", headers=SPOOFED).json()
    assert not {h.lower() for h in SPOOFED} & seen.keys()


def test_stripping_is_case_insensitive(echo_client):
    seen = echo_client.get("/echo", headers={"x-USER-id": "a", "X-a2a-INTERNAL": "true"}).json()
    assert "x-user-id" not in seen
    assert "x-a2a-internal" not in seen


def test_unrelated_headers_pass_through(echo_client):
    seen = echo_client.get(
        "/echo", headers={**SPOOFED, "Authorization": "Bearer t", "X-Request-ID": "rid-1", "X-User-Agent": "ua"},
    ).json()
    assert seen["authorization"] == "Bearer t"
    assert seen["x-request-id"] == "rid-1"
    assert seen["x-user-agent"] == "ua"


def test_maintenance_path_keeps_only_the_internal_token(echo_client):
    seen = echo_client.get("/admin/maintenance/status", headers=SPOOFED).json()
    assert seen["x-internal-token"] == "guessed-token"
    assert not {"x-user-id", "x-branch-id", "x-internal", "x-a2a-internal"} & seen.keys()


def test_maintenance_exemption_is_an_exact_path_prefix(echo_client):
    seen = echo_client.get("/admin/maintenance-other", headers=SPOOFED).json()
    assert "x-internal-token" not in seen


def test_websocket_handshake_headers_are_stripped(echo_client):
    with echo_client.websocket_connect("/ws", headers={**SPOOFED, "X-Request-ID": "rid-ws"}) as ws:
        seen = ws.receive_json()
    assert not {h.lower() for h in SPOOFED} & seen.keys()
    assert seen["x-request-id"] == "rid-ws"


# ── Through the real gateway ────────────────────────────────────────────────

_V = get_settings().api_version


@pytest.fixture
def upstream(monkeypatch):
    seen: list[httpx.Request] = []
    original = httpx.AsyncClient

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(agent_protocol.httpx, "AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    previous = gateway.dependency_overrides.get(get_current_user)
    gateway.dependency_overrides[get_current_user] = lambda: {"sub": "jwt-user"}
    yield seen
    if previous is None:
        gateway.dependency_overrides.pop(get_current_user, None)
    else:
        gateway.dependency_overrides[get_current_user] = previous


@pytest.mark.parametrize("prefix", [f"/api/{_V}", "/api/0.3"])
def test_a2a_proxy_never_forwards_spoofed_headers(upstream, prefix):
    response = TestClient(gateway).get(f"{prefix}/a2a/agents", headers={"Authorization": "Bearer t", **SPOOFED})
    assert response.status_code == 200
    forwarded = {k.lower() for k in upstream[0].headers}
    assert not {h.lower() for h in SPOOFED} & forwarded
    assert upstream[0].headers["authorization"] == "Bearer t"


def test_rag_query_sends_only_the_jwt_identity(upstream):
    response = TestClient(gateway).post(
        f"/api/{_V}/a2a/query", json={"question": "q"}, headers={"Authorization": "Bearer t", **SPOOFED},
    )
    assert response.status_code == 200
    sent = upstream[0].headers
    assert sent.get_list("x-user-id") == ["jwt-user"]
    assert sent.get_list("x-branch-id") == ["jwt-user"]
    assert "x-a2a-internal" not in sent
    assert "x-internal-token" not in sent
