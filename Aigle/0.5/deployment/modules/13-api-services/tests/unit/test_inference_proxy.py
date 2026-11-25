"""INF-9: the actual mounted gateway rejects requests before upstream I/O."""
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
from app.main import app
from app.routers import inference
from fastapi.testclient import TestClient


@pytest.mark.parametrize("path", ["", "/text", "/text/complete", "/voice/diarize", "/voice/clone",
                                     "/models", "/health", "/unknown"])
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE", "OPTIONS"])
def test_all_paths_deny_without_bearer_before_http(monkeypatch, path, method):
    def poison(*args, **kwargs):
        raise AssertionError("upstream reached before authentication")
    monkeypatch.setattr(inference.httpx, "AsyncClient", poison)
    response = TestClient(app, follow_redirects=False).request(method, "/api/0.5/inference" + path)
    assert response.status_code == 401


@pytest.mark.parametrize("method,path,target", [
    ("POST", "text/complete", "/inference/text/complete"),
    ("POST", "voice/stt", "/inference/voice/stt"),
    ("POST", "voice/tts", "/inference/voice/tts"),
    ("POST", "voice/clone", "/inference/voice/clone"),
    ("GET", "models/default", "/models/default"),
    ("POST", "models/default", "/models/default"),
    ("GET", "health", "/health"),
    # Module 35 doesn't actually have anything at this path -- not mirrored above (see
    # inference.py's module docstring), so this exercises the catch-all fallback specifically,
    # same auth-first + byte-forwarding contract as every declared route above it.
    ("GET", "something/not/mirrored/yet", "/inference/something/not/mirrored/yet"),
])
def test_authenticated_proxy_preserves_bytes_and_identity(monkeypatch, method, path, target):
    seen = []
    original = httpx.AsyncClient
    def handle(request):
        seen.append(request)
        return httpx.Response(201, stream=httpx.ByteStream(b"audio or SSE bytes"),
                              headers={"x-speaker-id": "abc"})
    monkeypatch.setattr(inference.httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(handle), **kw
    ))
    app.dependency_overrides[get_current_user] = lambda: {"sub": "verified"}
    try:
        response = TestClient(app).request(
            method, f"/api/0.5/inference/{path}?a=1&a=2", content=b"original body",
            headers={"Authorization": "Bearer token", "Content-Type": "audio/wav"},
        )
        assert response.status_code == 201
        assert response.content == b"audio or SSE bytes"
        assert response.headers["x-speaker-id"] == "abc"
        assert seen[0].url.path == target
        assert seen[0].url.params.get_list("a") == ["1", "2"]
        assert seen[0].content == b"original body"
        assert seen[0].headers["x-user-id"] == "verified"
        assert seen[0].headers["authorization"] == "Bearer token"
        # CF-3: a client-supplied x-user-id is stripped at ingress (IngressHeaderStripMiddleware),
        # so it never reaches the 403 backstop -- the call runs as the JWT sub instead.
        spoof = TestClient(app).get("/api/0.5/inference/health", headers={"x-user-id": "someone"})
        assert spoof.status_code == 201
        assert len(seen) == 2
        assert seen[1].headers["x-user-id"] == "verified"
    finally:
        app.dependency_overrides.pop(get_current_user, None)
