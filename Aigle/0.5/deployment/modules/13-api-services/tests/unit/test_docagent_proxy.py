"""DocAgent proxy: a slow upstream (LLM inference, Module 12 parsing) must
not time out against the shared app.state.http_client's short
request_timeout, and a timeout must surface as a clean 504, not an
unhandled exception."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document":"documents"}')
os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://auth:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://keycloak:8080")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from app.routers import docagent
from fastapi.testclient import TestClient

PREFIX = f"/api/{get_settings().api_version}/docagent"


@pytest.fixture(autouse=True)
def _auth():
    app.dependency_overrides[get_current_user] = lambda: {"sub": "verified"}
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_success_preserves_bytes_status_and_uses_the_long_timeout(monkeypatch):
    seen_timeouts = []
    original = httpx.AsyncClient

    def handle(request):
        return httpx.Response(200, json={"category": "財務報告"})

    def fake_client(**kw):
        seen_timeouts.append(kw.get("timeout"))
        return original(transport=httpx.MockTransport(handle), **kw)

    monkeypatch.setattr(docagent.httpx, "AsyncClient", fake_client)
    response = TestClient(app).post(f"{PREFIX}/process", content=b"a document",
                                     headers={"Authorization": "Bearer token"})
    assert response.status_code == 200
    assert response.json() == {"category": "財務報告"}
    # Not the shared client's short request_timeout -- this route's own, longer one.
    assert seen_timeouts == [get_settings().upload_timeout]
    assert get_settings().upload_timeout > get_settings().request_timeout


def test_upstream_timeout_is_a_clean_504_not_an_unhandled_exception(monkeypatch):
    original = httpx.AsyncClient

    def handle(request):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(docagent.httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(handle), **kw
    ))
    response = TestClient(app).post(f"{PREFIX}/process", content=b"a document",
                                     headers={"Authorization": "Bearer token"})
    assert response.status_code == 504
    assert response.json() == {"error": "doc_agent_timeout"}


def test_upstream_unreachable_is_503(monkeypatch):
    original = httpx.AsyncClient

    def handle(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(docagent.httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(handle), **kw
    ))
    response = TestClient(app).post(f"{PREFIX}/process", content=b"a document",
                                     headers={"Authorization": "Bearer token"})
    assert response.status_code == 503
    assert response.json() == {"error": "doc_agent_unavailable"}


def test_other_request_errors_are_a_clean_502(monkeypatch):
    original = httpx.AsyncClient

    def handle(request):
        raise httpx.RemoteProtocolError("peer closed connection")

    monkeypatch.setattr(docagent.httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(handle), **kw
    ))
    response = TestClient(app).post(f"{PREFIX}/process", content=b"a document",
                                     headers={"Authorization": "Bearer token"})
    assert response.status_code == 502
    assert response.json() == {"error": "doc_agent_error"}
