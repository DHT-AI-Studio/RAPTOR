"""
Unit tests for the MCP OAuth discovery URLs (RFC 9728) behind a TLS-terminating proxy (CF-3a).

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_mcp_oauth_discovery.py -v

Behind the proxy the gateway sees plain http from the proxy, so the discovery metadata must use
GATEWAY_PUBLIC_URL -- the https address the MCP client actually used -- or the client rejects it.
Without GATEWAY_PUBLIC_URL nothing changes: the request's own base URL is used, as before.
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
from app.api.dependencies import get_http_client
from app.core.config import get_settings
from app.main import app
from app.routers.mcp import require_mcp_credential
from fastapi.testclient import TestClient

V = get_settings().api_version
PUBLIC = "https://raptor.example:8443"
KEYCLOAK = "https://raptor.example:8443"


@pytest.fixture
def gateway(request):
    """A client whose settings carry the given public_url ("" = unset)."""
    settings = get_settings().model_copy(update={"public_url": request.param,
                                                 "keycloak_public_url": KEYCLOAK})
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(   # Module 27's 401, streamed like a real one
        lambda r: httpx.Response(401, stream=httpx.ByteStream(b"{}"))))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_http_client] = lambda: upstream
    app.dependency_overrides[require_mcp_credential] = lambda: None
    try:
        yield TestClient(app, base_url="http://10.0.0.5:8012")   # what the gateway sees from the proxy
    finally:
        for dep in (get_settings, get_http_client, require_mcp_credential):
            app.dependency_overrides.pop(dep, None)


@pytest.mark.parametrize("gateway", [PUBLIC, PUBLIC + "/"], indirect=True)
def test_behind_a_tls_proxy_the_metadata_names_the_public_https_url(gateway):
    meta = gateway.get("/.well-known/oauth-protected-resource").json()
    assert meta == {"resource": f"{PUBLIC}/api/{V}/mcp",
                    "authorization_servers": [f"{KEYCLOAK}/realms/{get_settings().keycloak_realm}"]}


@pytest.mark.parametrize("gateway", [PUBLIC], indirect=True)
def test_a_401_points_the_client_at_the_public_metadata(gateway):
    r = gateway.post(f"/api/{V}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                     headers={"authorization": "Bearer expired", "accept": "application/json, text/event-stream"})
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == (
        f'Bearer error="invalid_token", resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource"')


@pytest.mark.parametrize("gateway", [""], indirect=True)
def test_without_a_public_url_the_request_base_url_is_used_as_before(gateway):
    meta = gateway.get("/.well-known/oauth-protected-resource").json()
    assert meta["resource"] == f"http://10.0.0.5:8012/api/{V}/mcp"
