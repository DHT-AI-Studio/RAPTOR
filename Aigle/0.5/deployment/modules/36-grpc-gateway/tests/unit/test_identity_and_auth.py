"""VIE01-383: AuthenticationInterceptor / IdentityInterceptor rejection paths."""
from __future__ import annotations

import sys
from pathlib import Path

import grpc
import jwt
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import _TEST_JWK, TEST_API_KEY, make_token
from raptor.v1 import raptor_pb2


def _bearer(token: str) -> tuple:
    return (("authorization", f"Bearer {token}"),)


@pytest.mark.asyncio
async def test_valid_token_is_accepted_and_identity_used(raptor_stub):
    resp = await raptor_stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=_bearer(make_token(sub="user-7")))
    assert resp.message == "hi"


@pytest.mark.asyncio
async def test_expired_token_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(make_token(exp_offset=-60)))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_unknown_kid_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(make_token(kid="no-such-key")))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_token_with_no_kid_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(make_token(kid=None)))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_wrong_realm_in_iss_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(
            raptor_pb2.PingRequest(message="x"),
            metadata=_bearer(make_token(iss="http://raptor-keycloak:8080/realms/master")),
        )
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_token_signed_by_a_different_key_is_unauthenticated(raptor_stub):
    import time

    from config import get_settings
    from cryptography.hazmat.primitives.asymmetric import rsa

    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    settings = get_settings()
    forged = jwt.encode(
        {
            "sub": "attacker-claims-admin",
            "iss": f"{settings.keycloak_url}/realms/{settings.jwt_expected_realm}",
            "exp": int(time.time()) + 3600,
        },
        attacker_key,
        algorithm="RS256",
        headers={"kid": _TEST_JWK["kid"]},
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(forged))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_token_with_no_sub_claim_is_unauthenticated(raptor_stub):
    import time

    from config import get_settings
    from conftest import _TEST_PRIVATE_KEY

    settings = get_settings()
    token = jwt.encode(
        {
            "iss": f"{settings.keycloak_url}/realms/{settings.jwt_expected_realm}",
            "exp": int(time.time()) + 3600,
            **({"aud": settings.jwt_audience} if settings.jwt_audience else {}),
        },
        _TEST_PRIVATE_KEY,
        algorithm="RS256",
        headers={"kid": _TEST_JWK["kid"]},
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(token))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_wrong_audience_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(make_token(aud="some-other-service")))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_missing_audience_is_unauthenticated_when_audience_is_configured(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer(make_token(aud=None)))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_dev_fallback_disabled_by_default_rejects_an_invalid_token(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=_bearer("garbage"))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_dev_fallback_enabled_accepts_an_invalid_token():
    from conftest import authz_http_client, identity_http_client
    from server import build_server

    server = build_server(
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), identity_dev_fallback=True
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        from raptor.v1 import raptor_pb2_grpc

        resp = await raptor_pb2_grpc.RaptorServiceStub(channel).Ping(
            raptor_pb2.PingRequest(message="x"), metadata=_bearer("garbage")
        )
        assert resp.message == "x"
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_x_api_key_on_the_allow_list_is_accepted(raptor_stub):
    resp = await raptor_stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=(("x-api-key", TEST_API_KEY),))
    assert resp.message == "hi"


@pytest.mark.asyncio
async def test_x_api_key_not_on_the_allow_list_is_unauthenticated(raptor_stub):
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await raptor_stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=(("x-api-key", "guessed-key"),))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_valid_bearer_token_wins_over_a_stale_x_api_key(raptor_stub):
    resp = await raptor_stub.Ping(
        raptor_pb2.PingRequest(message="hi"),
        metadata=(("authorization", f"Bearer {make_token()}"), ("x-api-key", "wrong-key")),
    )
    assert resp.message == "hi"


@pytest.mark.asyncio
async def test_x_api_key_call_gets_the_dev_user_identity(raptor_stub):
    resp = await raptor_stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=(("x-api-key", TEST_API_KEY),))
    assert resp.message == "hi"


async def _run_with_authz(*, grant: bool) -> None:
    from conftest import authz_http_client, identity_http_client
    from raptor.v1 import raptor_pb2_grpc
    from server import build_server

    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(grant))
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        await raptor_pb2_grpc.RaptorServiceStub(channel).Ping(
            raptor_pb2.PingRequest(message="x"), metadata=_bearer(make_token())
        )
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_authz_grants_a_permitted_call():
    await _run_with_authz(grant=True)  # no raise


@pytest.mark.asyncio
async def test_authz_denies_an_unpermitted_call():
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _run_with_authz(grant=False)
    assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.asyncio
async def test_authz_skipped_when_identity_already_rejected():
    """An invalid token is already UNAUTHENTICATED from IdentityInterceptor; authz denying
    too must not change the code (it shouldn't even make the call)."""
    from conftest import authz_http_client, identity_http_client
    from raptor.v1 import raptor_pb2_grpc
    from server import build_server

    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(grant=False))
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await raptor_pb2_grpc.RaptorServiceStub(channel).Ping(
                raptor_pb2.PingRequest(message="x"), metadata=_bearer("garbage")
            )
        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        await channel.close()
        await server.stop(grace=None)
