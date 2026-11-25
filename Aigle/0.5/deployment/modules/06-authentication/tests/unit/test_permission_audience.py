"""
The UMA permission check asks the resource server (KEYCLOAK_CLIENT_ID), not the token's azp.

A token from the public raptor-mcp client (MCP OAuth login, docs/MCP_OAUTH_REQUIREMENTS.md) has
azp=raptor-mcp. That client has no authorization services, so a UMA request with audience=raptor-mcp
gets a 500 from Keycloak, and every MCP call was refused (seen on Keycloak 26.6: the same token is
allowed with audience=raptor). Keycloak is faked here; requests.post records what would be sent.

Run from the 06-authentication/ directory:
    python -m pytest tests/unit/test_permission_audience.py -v
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import jwt
import pytest
from app.core.config import settings
from app.core.security import APIsecurity
from app.keycloak import authentication
from app.routers.auth import permission
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code
        self.text = ""

    def json(self):
        return {}


def _token(azp: str) -> HTTPAuthorizationCredentials:
    claims = {"iss": "https://raptor.example:8443/realms/dhtsolution", "azp": azp, "sub": "u-1"}
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=jwt.encode(claims, "x", algorithm="HS256"))


@pytest.fixture
def uma(monkeypatch):
    sent = []

    def fake_post(url, data, headers, timeout):
        sent.append({"url": url, **data})
        return _Resp(200 if data["audience"] == settings.KEYCLOAK_CLIENT_ID else 500)
    monkeypatch.setattr(authentication.requests, "post", fake_post)
    return sent


@pytest.mark.parametrize("azp", ["raptor-mcp", "raptor"])
def test_permission_endpoint_asks_the_resource_server(uma, azp):
    assert asyncio.run(permission(request_path="/api/0.5/mcp", token=_token(azp))) == {"message": "Permission granted"}
    assert uma[0]["audience"] == settings.KEYCLOAK_CLIENT_ID == "raptor"
    assert uma[0]["permission"] == "/api/0.5/mcp"
    assert uma[0]["url"].endswith("/realms/dhtsolution/protocol/openid-connect/token")


def test_the_api_dependency_asks_the_resource_server(uma):
    class _Req:
        class url:
            path = "/auth/whatever"
    asyncio.run(APIsecurity.authenticate(request=_Req(), token=_token("raptor-mcp")))
    assert uma[0]["audience"] == "raptor"


def test_a_token_without_a_realm_is_still_refused(uma):
    bad = HTTPAuthorizationCredentials(scheme="Bearer", credentials=jwt.encode({"iss": "nowhere"}, "x", algorithm="HS256"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(permission(request_path="/api/0.5/mcp", token=bad))
    assert e.value.status_code == 401 and uma == []
