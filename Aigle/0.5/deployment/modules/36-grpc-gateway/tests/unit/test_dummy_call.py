"""Acceptance test: an authenticated dummy unary call passes through the full
interceptor chain (auth -> identity -> authz -> request-id -> logging ->
metrics -> exception-mapping) without error, and an unauthenticated (or
unverifiable) call is rejected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import TEST_API_KEY, make_token
from raptor.v1 import raptor_pb2


@pytest.mark.asyncio
async def test_authenticated_ping_passes_through_full_chain(raptor_stub):
    call = raptor_stub.Ping(
        raptor_pb2.PingRequest(message="hello"),
        metadata=(("authorization", f"Bearer {make_token()}"),),
    )
    response = await call

    assert response.message == "hello"
    assert response.request_id  # request-id interceptor populated it
    trailing = await call.trailing_metadata()
    assert trailing["x-request-id"] == response.request_id


@pytest.mark.asyncio
async def test_ping_without_credentials_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="hello"))

    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
    # RequestIdInterceptor sets this before ExceptionMappingInterceptor aborts —
    # confirms context.abort() without an explicit trailing_metadata argument
    # does not discard metadata set earlier via context.set_trailing_metadata().
    trailing = exc_info.value.trailing_metadata()
    assert trailing["x-request-id"]


@pytest.mark.asyncio
async def test_api_key_credential_is_also_accepted(raptor_stub):
    response = await raptor_stub.Ping(
        raptor_pb2.PingRequest(message="hi"),
        metadata=(("x-api-key", TEST_API_KEY),),
    )
    assert response.message == "hi"


@pytest.mark.asyncio
async def test_api_key_not_on_the_allow_list_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=(("x-api-key", "wrong-key"),))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["Bearer ", "Bearer    ", "bearer ", "Bearer", "Basic abc", ""])
async def test_authorization_without_a_token_is_unauthenticated(raptor_stub, value):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=(("authorization", value),))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


# VIE01-383: "Bearer " + any non-empty string used to count as a credential --
# AuthenticationInterceptor only checked presence, and IdentityInterceptor
# decoded the JWT without checking its signature, so a garbage string was
# accepted exactly like a real token (docs/_internal/KNOWN_ISSUES.md section 1, now
# resolved). These three malformed/unsigned values must now be rejected, and
# only a token that actually verifies against Keycloak's JWKS is accepted.


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["Bearer abc", "bearer abc", "BEARER abc.def.ghi"])
async def test_authorization_with_an_unverifiable_token_is_rejected(raptor_stub, value):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=(("authorization", value),))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_authorization_with_a_verified_token_is_accepted(raptor_stub):
    resp = await raptor_stub.Ping(
        raptor_pb2.PingRequest(message="x"),
        metadata=(("authorization", f"Bearer {make_token()}"),),
    )
    assert resp.message == "x"
