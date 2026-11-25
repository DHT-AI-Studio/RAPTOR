"""Tests for TLS/mTLS server credentials: server.py loads grpc.ssl_server_credentials()
from settings.grpc_tls_cert_path/grpc_tls_key_path when grpc_secure=true, and
grpc_mtls_ca_cert_path additionally requires a client certificate signed by
that CA. Certs are generated at test time (self-signed CA + server + client),
never committed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import _build_server_credentials, build_server

AUTH = (("authorization", f"Bearer {make_token()}"),)


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture
def pki(tmp_path):
    """Self-signed CA, a server cert/key (CN=localhost, signed by the CA),
    and a client cert/key (also signed by the CA)."""
    ca_key = tmp_path / "ca.key"
    ca_cert = tmp_path / "ca.crt"
    server_key = tmp_path / "server.key"
    server_csr = tmp_path / "server.csr"
    server_cert = tmp_path / "server.crt"
    client_key = tmp_path / "client.key"
    client_csr = tmp_path / "client.csr"
    client_cert = tmp_path / "client.crt"

    _run(["openssl", "genrsa", "-out", str(ca_key), "2048"])
    _run([
        "openssl", "req", "-x509", "-new", "-key", str(ca_key), "-days", "2",
        "-out", str(ca_cert), "-subj", "/CN=test-ca",
    ])

    _run(["openssl", "genrsa", "-out", str(server_key), "2048"])
    _run([
        "openssl", "req", "-new", "-key", str(server_key), "-out", str(server_csr),
        "-subj", "/CN=localhost",
    ])
    _run([
        "openssl", "x509", "-req", "-in", str(server_csr), "-CA", str(ca_cert), "-CAkey", str(ca_key),
        "-CAcreateserial", "-out", str(server_cert), "-days", "2",
    ])

    _run(["openssl", "genrsa", "-out", str(client_key), "2048"])
    _run([
        "openssl", "req", "-new", "-key", str(client_key), "-out", str(client_csr),
        "-subj", "/CN=test-client",
    ])
    _run([
        "openssl", "x509", "-req", "-in", str(client_csr), "-CA", str(ca_cert), "-CAkey", str(ca_key),
        "-CAcreateserial", "-out", str(client_cert), "-days", "2",
    ])

    return SimpleNamespace(
        ca_cert=ca_cert.read_bytes(),
        server_key=server_key, server_cert=server_cert,
        client_key=client_key.read_bytes(), client_cert=client_cert.read_bytes(),
    )


# ── _build_server_credentials: pure unit tests ──────────────────────────────

def test_build_server_credentials_requires_cert_and_key_paths():
    settings = SimpleNamespace(grpc_tls_cert_path="", grpc_tls_key_path="", grpc_mtls_ca_cert_path="")
    with pytest.raises(RuntimeError):
        _build_server_credentials(settings)


def test_build_server_credentials_builds_tls_creds(pki):
    settings = SimpleNamespace(
        grpc_tls_cert_path=str(pki.server_cert), grpc_tls_key_path=str(pki.server_key), grpc_mtls_ca_cert_path="",
    )
    creds = _build_server_credentials(settings)
    assert isinstance(creds, grpc.ServerCredentials)


# ── end-to-end: real TLS handshake over a real socket ──────────────────────

@pytest.mark.asyncio
async def test_tls_server_accepts_connection_with_matching_ca(pki):
    settings = SimpleNamespace(
        grpc_tls_cert_path=str(pki.server_cert), grpc_tls_key_path=str(pki.server_key), grpc_mtls_ca_cert_path="",
    )
    credentials = _build_server_credentials(settings)

    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_secure_port("localhost:0", credentials)
    await server.start()

    channel_creds = grpc.ssl_channel_credentials(root_certificates=pki.ca_cert)
    channel = grpc.aio.secure_channel(f"localhost:{port}", channel_creds, options=(("grpc.ssl_target_name_override", "localhost"),))
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        resp = await stub.Ping(raptor_pb2.PingRequest(message="over-tls"), metadata=AUTH)
        assert resp.message == "over-tls"
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_tls_client_rejects_server_cert_from_untrusted_ca(pki, tmp_path):
    """A client that doesn't trust the signing CA can't complete the handshake at all."""
    settings = SimpleNamespace(
        grpc_tls_cert_path=str(pki.server_cert), grpc_tls_key_path=str(pki.server_key), grpc_mtls_ca_cert_path="",
    )
    credentials = _build_server_credentials(settings)

    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_secure_port("localhost:0", credentials)
    await server.start()

    # A second, unrelated self-signed CA the client will (wrongly) trust instead.
    other_ca_key = tmp_path / "other_ca.key"
    other_ca_cert = tmp_path / "other_ca.crt"
    _run(["openssl", "genrsa", "-out", str(other_ca_key), "2048"])
    _run([
        "openssl", "req", "-x509", "-new", "-key", str(other_ca_key), "-days", "2",
        "-out", str(other_ca_cert), "-subj", "/CN=other-ca",
    ])

    channel_creds = grpc.ssl_channel_credentials(root_certificates=other_ca_cert.read_bytes())
    channel = grpc.aio.secure_channel(f"localhost:{port}", channel_creds, options=(("grpc.ssl_target_name_override", "localhost"),))
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=AUTH)
        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_mtls_server_requires_client_certificate(pki):
    # pki fixture signs both server and client certs with the same CA --
    # write that CA out for the server to use as its client-verification root.
    ca_path = pki.server_cert.parent / "ca_for_mtls.crt"
    ca_path.write_bytes(pki.ca_cert)
    settings = SimpleNamespace(
        grpc_tls_cert_path=str(pki.server_cert), grpc_tls_key_path=str(pki.server_key),
        grpc_mtls_ca_cert_path=str(ca_path),
    )

    credentials = _build_server_credentials(settings)
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client())
    port = server.add_secure_port("localhost:0", credentials)
    await server.start()
    try:
        # No client cert presented -> handshake fails.
        channel_creds = grpc.ssl_channel_credentials(root_certificates=pki.ca_cert)
        channel = grpc.aio.secure_channel(f"localhost:{port}", channel_creds, options=(("grpc.ssl_target_name_override", "localhost"),))
        try:
            stub = raptor_pb2_grpc.RaptorServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                await stub.Ping(raptor_pb2.PingRequest(message="x"), metadata=AUTH)
            assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE
        finally:
            await channel.close()

        # A valid client cert signed by the trusted CA -> handshake succeeds.
        mtls_creds = grpc.ssl_channel_credentials(
            root_certificates=pki.ca_cert,
            private_key=pki.client_key,
            certificate_chain=pki.client_cert,
        )
        channel2 = grpc.aio.secure_channel(f"localhost:{port}", mtls_creds, options=(("grpc.ssl_target_name_override", "localhost"),))
        try:
            stub2 = raptor_pb2_grpc.RaptorServiceStub(channel2)
            resp = await stub2.Ping(raptor_pb2.PingRequest(message="mtls-ok"), metadata=AUTH)
            assert resp.message == "mtls-ok"
        finally:
            await channel2.close()
    finally:
        await server.stop(grace=None)


# ── the rest of the matrix: what must NOT connect, and streaming over TLS ──
# Found missing while running a real TLS and an mTLS gateway process by hand (VIE01-165): the
# unit tests covered "accepts the right CA" and "mTLS needs a client cert", not these.

def _tls_settings(pki, mtls_ca_path=""):
    return SimpleNamespace(
        grpc_tls_cert_path=str(pki.server_cert), grpc_tls_key_path=str(pki.server_key),
        grpc_mtls_ca_cert_path=str(mtls_ca_path) if mtls_ca_path else "",
    )


async def _serve_tls(pki, mtls_ca_path="", **server_kwargs):
    server = build_server(
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), **server_kwargs
    )
    port = server.add_secure_port("localhost:0", _build_server_credentials(_tls_settings(pki, mtls_ca_path)))
    await server.start()
    return server, port


async def _ping_code(channel):
    try:
        await raptor_pb2_grpc.RaptorServiceStub(channel).Ping(raptor_pb2.PingRequest(message="x"), metadata=AUTH, timeout=5)
    except grpc.aio.AioRpcError as exc:
        return exc.code()
    return grpc.StatusCode.OK


@pytest.mark.asyncio
async def test_mtls_rejects_a_client_certificate_signed_by_another_ca(pki, tmp_path):
    ca_path = tmp_path / "ca_for_mtls.crt"
    ca_path.write_bytes(pki.ca_cert)
    # a client cert from a CA the server was not told to trust
    rogue_key, rogue_ca, rogue_client_key, rogue_client_csr, rogue_client = (
        tmp_path / name for name in ("rogue.key", "rogue.crt", "rc.key", "rc.csr", "rc.crt")
    )
    _run(["openssl", "genrsa", "-out", str(rogue_key), "2048"])
    _run(["openssl", "req", "-x509", "-new", "-key", str(rogue_key), "-days", "2", "-out", str(rogue_ca), "-subj", "/CN=rogue-ca"])
    _run(["openssl", "genrsa", "-out", str(rogue_client_key), "2048"])
    _run(["openssl", "req", "-new", "-key", str(rogue_client_key), "-out", str(rogue_client_csr), "-subj", "/CN=rogue-client"])
    _run([
        "openssl", "x509", "-req", "-in", str(rogue_client_csr), "-CA", str(rogue_ca), "-CAkey", str(rogue_key),
        "-CAcreateserial", "-out", str(rogue_client), "-days", "2",
    ])

    server, port = await _serve_tls(pki, mtls_ca_path=ca_path)
    channel = grpc.aio.secure_channel(
        f"localhost:{port}",
        grpc.ssl_channel_credentials(
            root_certificates=pki.ca_cert, private_key=rogue_client_key.read_bytes(),
            certificate_chain=rogue_client.read_bytes(),
        ),
        options=(("grpc.ssl_target_name_override", "localhost"),),
    )
    try:
        assert await _ping_code(channel) == grpc.StatusCode.UNAVAILABLE
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_tls_server_rejects_a_plaintext_client(pki):
    server, port = await _serve_tls(pki)
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        assert await _ping_code(channel) == grpc.StatusCode.UNAVAILABLE
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_tls_client_rejects_a_server_name_that_does_not_match_the_certificate(pki):
    server, port = await _serve_tls(pki)
    channel = grpc.aio.secure_channel(
        f"localhost:{port}", grpc.ssl_channel_credentials(root_certificates=pki.ca_cert),
        options=(("grpc.ssl_target_name_override", "wrong.example"),),
    )
    try:
        assert await _ping_code(channel) == grpc.StatusCode.UNAVAILABLE
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_a_client_streaming_upload_works_over_tls(pki):
    from test_stream_limit import _FakeRedis, _FakeStorageService

    server, port = await _serve_tls(pki, storage_service=_FakeStorageService(), redis_client=_FakeRedis())
    channel = grpc.aio.secure_channel(
        f"localhost:{port}", grpc.ssl_channel_credentials(root_certificates=pki.ca_cert),
        options=(("grpc.ssl_target_name_override", "localhost"),),
    )
    payload = b"x" * (2 * 1024 * 1024)

    async def messages():
        yield raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(filename="a.bin", total_size=len(payload)))
        for start in range(0, len(payload), 256 * 1024):
            yield raptor_pb2.UploadAssetRequest(chunk_data=payload[start:start + 256 * 1024])

    try:
        reply = await raptor_pb2_grpc.RaptorServiceStub(channel).UploadAsset(messages(), metadata=AUTH, timeout=20)
        assert reply.bytes_received == len(payload)
    finally:
        await channel.close()
        await server.stop(grace=None)
