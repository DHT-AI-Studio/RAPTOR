"""UploadAsset through the REAL StorageService (module 13's class), with only the HTTP
backend (the asset service) faked at the transport level. The other upload tests replace
StorageService itself, so the real `upload_file` -- the code that reads settings, builds the
multipart request and parses the response -- never ran in a test.
"""
from __future__ import annotations

import grpc
import httpx
import pytest
from app.services.storage_service import StorageService
from config import Settings
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server

SUB = "11111111-2222-3333-4444-555555555555"


class _FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def incr(self, key):
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    async def decr(self, key):
        self.store[key] = int(self.store.get(key, 0)) - 1
        return self.store[key]

    async def expire(self, key, ttl):
        pass


def _asset_service(captured: list):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.method == "POST" and request.url.path.endswith("/fileupload"):
            return httpx.Response(200, json={
                "asset_path": "video/mp4/clip", "version_id": "v-abc", "primary_filename": "clip.mp4", "status": "uploaded",
            })
        return httpx.Response(404, json={"detail": "unexpected call in test"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _upload(server_kwargs, requests):
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), **server_kwargs)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = raptor_pb2_grpc.RaptorServiceStub(channel)
        call = stub.UploadAsset(iter_requests(requests), metadata=(("authorization", f"Bearer {make_token(sub=SUB)}"),))
        return await call
    finally:
        await channel.close()
        await server.stop(grace=None)


async def iter_requests(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_upload_goes_through_the_real_storage_service_to_the_asset_service():
    captured: list[httpx.Request] = []
    http_client = _asset_service(captured)
    storage = StorageService(http_client, Settings(_env_file=None))
    payload = b"\x00\x01fake-mp4-bytes" * 1000

    reqs = [
        raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(
            filename="clip.mp4", content_type="video/mp4", total_size=len(payload), idempotency_key="k-1")),
        raptor_pb2.UploadAssetRequest(chunk_data=payload[:5000]),
        raptor_pb2.UploadAssetRequest(chunk_data=payload[5000:]),
    ]
    resp = await _upload({"storage_service": storage, "http_client": http_client, "redis_client": _FakeRedis()}, reqs)

    assert (resp.asset_id, resp.version_id, resp.bytes_received) == ("video/mp4/clip", "v-abc", len(payload))
    (sent,) = [r for r in captured if r.url.path.endswith("/fileupload")]
    assert sent.headers["x-user-id"] == SUB and sent.headers["x-branch-id"] == SUB  # identity from the JWT's sub
    assert b'filename="clip.mp4"' in sent.read() and payload in sent.read()  # the bytes reached the asset service intact
    # the settings StorageService reads were honoured (this is the attribute whose absence broke every upload)
    assert sent.extensions["timeout"]["read"] == Settings(_env_file=None).upload_timeout == 300.0
