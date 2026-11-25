"""DownloadAsset and StreamVideoSegment against the REAL StorageService, with only the asset
service faked (at the HTTP transport), behaving like the real one:

* `GET /filedownload/{asset_path}/{version_id}` knows the platform's asset_path (no file name)
* `GET /presign-url/{key}/{version_id}` wants the object key (asset_path + "/" + file name)
* anything unknown is a 404 with a JSON body.

Field test on a deployed stack found: the identifier the platform hands out (REST, search,
UploadAssetResponse.asset_id) could not be used to download (INTERNAL 404); a missing asset was
INTERNAL with the downstream JSON in the message; bad StreamVideoSegment arguments were only
reported after asking storage; the biggest chunk a default client may ask for was rejected by that
client. The other tests replace StorageService with a fake that accepts any path, so none of that
could show up.
"""
from __future__ import annotations

import re
from urllib.parse import unquote

import grpc
import httpx
import pytest
import pytest_asyncio
from app.services.storage_service import StorageService
from config import Settings
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server

ASSET_PATH = "video/mp4/clip"          # as the platform reports it: no file name
FILENAME = "clip.mp4"
KEY = f"{ASSET_PATH}/{FILENAME}"       # the object key
VERSION = "v-abc123"
PAYLOAD = bytes(range(256)) * 40_000   # 10,240,000 bytes: more than two maximum-size chunks
AUTH = (("authorization", f"Bearer {make_token()}"),)
F = raptor_pb2.VideoStreamFormat


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


class _AssetService:
    """Records every request so a test can assert what storage was (not) asked."""

    def __init__(self):
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = unquote(request.url.path)
        if request.url.host == "storage.test":
            return self._object(request)
        m = re.fullmatch(r"/(filedownload|presign-url)/(.+)/([^/]+)", path)
        if m and m.group(3) == VERSION:
            kind, ident = m.group(1), m.group(2)
            if kind == "filedownload" and ident == ASSET_PATH:
                return httpx.Response(200, json={"metadata": {"asset_path": ASSET_PATH, "version_id": VERSION, "primary_filename": FILENAME}})
            if kind == "presign-url" and ident == KEY:
                return httpx.Response(200, json={"url": "http://storage.test/obj/clip.mp4"})
        return httpx.Response(404, json={"detail": f"File {path} not found"})

    @staticmethod
    def _object(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range", "")
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng)
        start = int(m.group(1)) if m else 0
        end = int(m.group(2)) if m and m.group(2) else len(PAYLOAD) - 1
        body = PAYLOAD[start : end + 1]
        return httpx.Response(206, content=body, headers={"Content-Range": f"bytes {start}-{end}/{len(PAYLOAD)}", "Content-Length": str(len(body))})

    def asked_storage(self) -> bool:
        return any(r.url.host != "storage.test" for r in self.requests)


@pytest_asyncio.fixture
async def stack():
    backend = _AssetService()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    storage = StorageService(http_client, Settings(_env_file=None))
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), storage_service=storage, http_client=http_client, redis_client=_FakeRedis())
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")  # default options: 4 MiB receive limit
    try:
        yield raptor_pb2_grpc.RaptorServiceStub(channel), backend
    finally:
        await channel.close()
        await server.stop(grace=None)


async def _download(stub, **kw):
    header, chunks = None, []
    async for r in stub.DownloadAsset(raptor_pb2.DownloadAssetRequest(version_id=VERSION, **kw), metadata=AUTH):
        if r.HasField("header"):
            header = r.header
        else:
            chunks.append(r.chunk.data)
    return header, chunks


# ── the identifier the platform hands out must work ──────────────────────────


@pytest.mark.asyncio
async def test_download_with_the_asset_path_the_platform_returns(stack):
    stub, _ = stack
    header, chunks = await _download(stub, asset_path=ASSET_PATH)
    assert b"".join(chunks) == PAYLOAD
    assert header.filename == FILENAME  # the real file name, not the last path segment
    assert header.content_length == len(PAYLOAD)


@pytest.mark.asyncio
async def test_download_with_the_full_object_key_still_works(stack):
    stub, _ = stack
    header, chunks = await _download(stub, asset_path=KEY)
    assert b"".join(chunks) == PAYLOAD and header.filename == FILENAME


@pytest.mark.asyncio
async def test_download_range_with_the_platform_asset_path(stack):
    stub, _ = stack
    _, chunks = await _download(stub, asset_path=ASSET_PATH, offset=500_000, length=12_345)
    assert b"".join(chunks) == PAYLOAD[500_000:512_345]


@pytest.mark.asyncio
async def test_stream_video_segment_raw_range_with_the_platform_asset_path(stack):
    stub, _ = stack
    request = raptor_pb2.StreamVideoSegmentRequest(
        asset=raptor_pb2.AssetRef(asset_path=ASSET_PATH, version_id=VERSION),
        format=F.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE, byte_offset=100, byte_length=2000,
    )
    data = b"".join([c.data async for c in stub.StreamVideoSegment(request, metadata=AUTH)])
    assert data == PAYLOAD[100:2100]


# ── a missing asset is NOT_FOUND, with a message meant for the caller ────────


@pytest.mark.asyncio
async def test_download_of_a_missing_asset_is_not_found_without_leaking_the_downstream_body(stack):
    stub, _ = stack
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _download(stub, asset_path="video/mp4/does-not-exist")
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    message = exc_info.value.details()
    assert "does-not-exist" in message
    assert "Asset service error" not in message and "{" not in message


@pytest.mark.asyncio
async def test_stream_video_segment_of_a_missing_asset_is_not_found(stack):
    stub, _ = stack
    request = raptor_pb2.StreamVideoSegmentRequest(
        asset=raptor_pb2.AssetRef(asset_path="video/mp4/nope", version_id=VERSION), format=F.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        _ = [c async for c in stub.StreamVideoSegment(request, metadata=AUTH)]
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


# ── bad arguments are reported before storage is asked anything ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": F.VIDEO_STREAM_FORMAT_MP4_H264, "start_time": 10, "end_time": 2},
        {"format": F.VIDEO_STREAM_FORMAT_UNSPECIFIED},
        {"format": F.VIDEO_STREAM_FORMAT_HLS},
        {"format": F.VIDEO_STREAM_FORMAT_RAW_FILE_RANGE, "byte_offset": -1},
    ],
    ids=["start-after-end", "unspecified-format", "hls", "negative-offset"],
)
async def test_stream_video_segment_rejects_bad_arguments_without_asking_storage(stack, kwargs):
    stub, backend = stack
    request = raptor_pb2.StreamVideoSegmentRequest(
        asset=raptor_pb2.AssetRef(asset_path="video/mp4/does-not-exist", version_id=VERSION), **kwargs
    )
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        _ = [c async for c in stub.StreamVideoSegment(request, metadata=AUTH)]
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT  # not NOT_FOUND / INTERNAL
    assert not backend.asked_storage()


# ── the largest chunk a default client may request must be deliverable ───────


@pytest.mark.asyncio
async def test_largest_requested_chunk_fits_a_default_clients_receive_limit(stack):
    stub, _ = stack
    _, chunks = await _download(stub, asset_path=ASSET_PATH, preferred_chunk_size=64 * 1024 * 1024)
    assert b"".join(chunks) == PAYLOAD
    biggest = max(len(c) for c in chunks)
    assert 4 * 1024 * 1024 - 8192 <= biggest <= 4 * 1024 * 1024 - 1024  # near the cap, but with room for the framing


# ── SubmitJob uses the same platform asset_path (and reports a missing one properly) ──────────


class _Producer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, message):
        self.sent.append((topic, message))


@pytest_asyncio.fixture
async def submit_stack():
    backend = _AssetService()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    producer = _Producer()
    server = build_server(identity_http_client=identity_http_client(), authz_http_client=authz_http_client(), 
        storage_service=StorageService(http_client, Settings(_env_file=None)),
        http_client=http_client, redis_client=_FakeRedis(), kafka_producer=producer,
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        yield raptor_pb2_grpc.RaptorServiceStub(channel), producer
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_submit_job_accepts_the_platform_asset_path(submit_stack):
    stub, producer = submit_stack
    job = await stub.SubmitJob(
        raptor_pb2.SubmitJobRequest(input_asset_ids=[raptor_pb2.AssetRef(asset_path=ASSET_PATH, version_id=VERSION)]),
        metadata=AUTH,
    )
    assert job.state == raptor_pb2.JOB_STATE_QUEUED and job.media_type == "video"
    assert [t for t, _ in producer.sent] == ["video-processing-requests"]


@pytest.mark.asyncio
async def test_submit_job_for_a_missing_asset_is_not_found_and_sends_nothing(submit_stack):
    stub, producer = submit_stack
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.SubmitJob(
            raptor_pb2.SubmitJobRequest(input_asset_ids=[raptor_pb2.AssetRef(asset_path="video/mp4/nope", version_id=VERSION)]),
            metadata=AUTH,
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    assert producer.sent == []
