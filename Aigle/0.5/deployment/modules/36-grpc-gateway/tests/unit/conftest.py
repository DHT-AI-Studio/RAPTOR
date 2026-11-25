from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import grpc
import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dev_paths

dev_paths.setup()

# A known x-api-key allow-list for tests that exercise the x-api-key credential
# path (e.g. test_dummy_call.py::test_api_key_credential_is_also_accepted) --
# AuthenticationInterceptor now checks this for real, so an unconfigured
# (empty) allow-list would reject every key, including a legitimately-testing
# one. Must be set before config.get_settings() is first called anywhere
# (it's @lru_cache'd), so this runs at conftest import time, ahead of every
# test module's own imports.
TEST_API_KEY = "test-key"
os.environ.setdefault("GRPC_GATEWAY_API_KEYS", TEST_API_KEY)

from config import get_settings
from raptor.v1 import raptor_pb2_grpc
from server import build_server

# ── test JWKS: a real RSA key this process signs test tokens with, served to
# IdentityInterceptor through a mocked transport so no real Keycloak is needed.
#
# Before VIE01-383, IdentityInterceptor decoded a Bearer token without checking
# its signature, so any undecodable string (e.g. the old "Bearer x" fixture
# every test in this suite used) fell back to a fixed dev-user identity --
# "authenticated" was trivial to fake. Now a Bearer token's signature is
# verified for real, so every test that wants an authenticated call needs an
# actual RS256 token signed with a key IdentityInterceptor can verify -- that's
# what make_token() below is for.
_TEST_KID = "test-key"
_TEST_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_JWK: dict = json.loads(RSAAlgorithm(RSAAlgorithm.SHA256).to_jwk(_TEST_PRIVATE_KEY.public_key()))
_TEST_JWK.update(kid=_TEST_KID, use="sig")


def _jwks_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/protocol/openid-connect/certs"):
        return httpx.Response(200, json={"keys": [_TEST_JWK]})
    return httpx.Response(404)


_UNSET = object()


def make_token(
    sub: str = "dev-user", *, kid: str | None = _TEST_KID, exp_offset: float = 3600, aud=_UNSET, **extra_claims
) -> str:
    """A real RS256 JWT, signed with this test suite's own key, verifiable against the JWKS
    _jwks_handler serves. `aud` defaults to settings.jwt_audience; pass `aud=None` or a wrong
    value, or `kid=None`/wrong, or a negative `exp_offset`, to hit a specific rejection path."""
    settings = get_settings()
    if aud is _UNSET:
        aud = settings.jwt_audience or None
    payload = {
        "sub": sub,
        "iss": f"{settings.keycloak_url}/realms/{settings.jwt_expected_realm}",
        "exp": int(time.time() + exp_offset),
        **({"aud": aud} if aud is not None else {}),
        **extra_claims,
    }
    headers = {"kid": kid} if kid else None
    return jwt.encode(payload, _TEST_PRIVATE_KEY, algorithm="RS256", headers=headers)


def identity_http_client() -> httpx.AsyncClient:
    """A fresh mocked transport serving this file's test JWKS -- for a test file that builds
    its own server directly (build_server(...)) instead of using the grpc_channel/raptor_stub
    fixtures below, so its own server's IdentityInterceptor can still verify make_token()."""
    return httpx.AsyncClient(transport=httpx.MockTransport(_jwks_handler))


def authz_http_client(grant: bool = True) -> httpx.AsyncClient:
    """A fresh mocked transport standing in for module 06's /auth/permission -- grant=True
    (default) mimics every test token being allowed; grant=False exercises the denial path."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth/permission"):
            return httpx.Response(200 if grant else 403, json={"message": "Permission granted"} if grant else {"error": "access_denied"})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest_asyncio.fixture
async def grpc_channel():
    """Starts the real interceptor-wired server on an ephemeral port and yields a client channel.

    identity_http_client/authz_http_client are mocked to serve this file's test JWKS and
    always grant permission, so a Bearer token made with make_token() verifies and is
    authorized without a real Keycloak or module 06.
    """
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_insecure_port("localhost:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        yield channel
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.fixture
def raptor_stub(grpc_channel):
    return raptor_pb2_grpc.RaptorServiceStub(grpc_channel)
