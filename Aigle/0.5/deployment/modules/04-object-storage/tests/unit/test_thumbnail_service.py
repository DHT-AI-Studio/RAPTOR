"""
MTR-7 / VIE01-306: Thumbnail API -- unit coverage.

Pure unit tests, no live infra, no real ffmpeg/ffprobe binary and no GPU --
asyncio.create_subprocess_exec is monkeypatched. Covers the guard clauses
(404/409/400) and the cache-hit-skips-ffmpeg requirement, which are fully
provable without real video content.

NOT covered here (needs the real thing, see test_thumbnail_cuda_node.py):
- "correct frame at several timestamps" against real video content
- actual NVDEC decode on the CUDA node

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_thumbnail_service.py -v
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
from asset_management.models import AssetMetadata
from asset_management.thumbnail_service import ThumbnailService
from fastapi import HTTPException


def _base_metadata_kwargs(**overrides):
    kwargs = {
        "asset_path": "video/mp4/foo",
        "version_id": "v1",
        "primary_filename": "foo.mp4",
        "associated_filenames": [],
        "upload_date": datetime.now(tz=ZoneInfo("Asia/Taipei")),
        "user": "tester",
        "branch": "test_branch",
        "status": "active",
        "checksum": "deadbeef",
    }
    kwargs.update(overrides)
    return kwargs


class _FakeDB:
    def __init__(self, initial_status: str = "active", metadata=None):
        self.metadata = metadata if metadata is not None else AssetMetadata(**_base_metadata_kwargs(status=initial_status))

    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return self.metadata


class _EmptyDB:
    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return None


class _FakeObjectStore:
    def __init__(self, url="https://fake-presigned/video.mp4"):
        self.url = url
        self.calls = []

    async def generate_presigned_url(self, key, version_id=None):
        self.calls.append((key, version_id))
        return self.url


class _FakeRedis:
    """In-memory stand-in -- no real Redis connection."""

    def __init__(self):
        self.store = {}
        self.get_calls = []
        self.setex_calls = []

    async def get(self, key):
        self.get_calls.append(key)
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl))
        self.store[key] = value


def _make_service(db, object_store=None, redis=None):
    svc = ThumbnailService(object_store=object_store or _FakeObjectStore(), db=db)
    svc._redis = redis if redis is not None else _FakeRedis()
    return svc


# ── 404 / 409 / 400 guard clauses ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_not_found_raises_404():
    svc = _make_service(db=_EmptyDB())
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=5.0)
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("status", ["uploaded", "processing", "failed", "unregistered", "archived", "destroyed"])
@pytest.mark.asyncio
async def test_non_active_status_raises_409(status):
    svc = _make_service(db=_FakeDB(initial_status=status))
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=5.0)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_negative_second_raises_400():
    svc = _make_service(db=_FakeDB())
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=-1.0)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_second_beyond_duration_raises_400(monkeypatch):
    """The worker decides this and returns 400 -- this service just passes
    it through unchanged."""
    svc = _make_service(db=_FakeDB())

    async def _fake_extract(self, url, second):
        raise HTTPException(status_code=400, detail="second=99.0 is beyond video duration (10.00s)")

    monkeypatch.setattr(ThumbnailService, "_extract_via_worker", _fake_extract)

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=99.0)
    assert exc_info.value.status_code == 400
    assert "beyond video duration" in exc_info.value.detail


# ── cache hit skips the worker entirely ─────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_returns_cached_bytes_and_never_calls_worker(monkeypatch):
    object_store = _FakeObjectStore()
    redis = _FakeRedis()
    svc = _make_service(db=_FakeDB(), object_store=object_store, redis=redis)
    cache_key = svc._cache_key("video/mp4/foo", "v1", 5.0)
    redis.store[cache_key] = b"cached-jpeg-bytes"

    def _poison(*a, **kw):
        raise AssertionError("thumbnail-worker must not be called on a cache hit")

    monkeypatch.setattr(ThumbnailService, "_extract_via_worker", _poison)

    result = await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=5.0)

    assert result == b"cached-jpeg-bytes"
    assert object_store.calls == [], "cache hit must not even fetch a presigned URL"


@pytest.mark.asyncio
async def test_cache_miss_calls_worker_and_populates_cache(monkeypatch):
    object_store = _FakeObjectStore(url="https://fake-presigned/video.mp4")
    redis = _FakeRedis()
    svc = _make_service(db=_FakeDB(), object_store=object_store, redis=redis)

    extract_calls = []

    async def _fake_extract(self, url, second):
        extract_calls.append((url, second))
        return b"fresh-jpeg-bytes"

    monkeypatch.setattr(ThumbnailService, "_extract_via_worker", _fake_extract)

    result = await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=12.5)

    assert result == b"fresh-jpeg-bytes"
    assert object_store.calls == [("video/mp4/foo/foo.mp4", "v1")], "presigned URL must use asset_path/primary_filename as the key"
    assert extract_calls == [("https://fake-presigned/video.mp4", 12.5)]

    cache_key = svc._cache_key("video/mp4/foo", "v1", 12.5)
    assert redis.setex_calls == [(cache_key, 7 * 24 * 3600)]
    assert redis.store[cache_key] == b"fresh-jpeg-bytes"


@pytest.mark.asyncio
async def test_redis_unavailable_still_serves_thumbnail(monkeypatch):
    """Cache is a performance optimization, not a hard dependency -- a Redis
    outage must not break thumbnail generation."""
    svc = _make_service(db=_FakeDB(), redis=None)
    svc._redis = None

    async def _fake_extract(self, url, second):
        return b"fresh-jpeg-bytes"

    monkeypatch.setattr(ThumbnailService, "_extract_via_worker", _fake_extract)

    result = await svc.get_thumbnail("video/mp4/foo", "v1", "branch", second=1.0)
    assert result == b"fresh-jpeg-bytes"


# ── _extract_via_worker itself: real HTTP call, fake worker response ───────

class _FakeWorkerResponse:
    def __init__(self, status_code, content=b"", json_body=None, text=""):
        self.status_code = status_code
        self.content = content
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class _FakeWorkerClient:
    response = _FakeWorkerResponse(200, content=b"jpeg-bytes")

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        assert url.endswith("/extract")
        return type(self).response


@pytest.mark.asyncio
async def test_extract_via_worker_returns_bytes_on_200(monkeypatch):
    monkeypatch.setattr("asset_management.thumbnail_service.httpx.AsyncClient", _FakeWorkerClient)
    _FakeWorkerClient.response = _FakeWorkerResponse(200, content=b"jpeg-bytes")

    svc = _make_service(db=_FakeDB())
    result = await svc._extract_via_worker("https://fake/video.mp4", 1.0)
    assert result == b"jpeg-bytes"


@pytest.mark.asyncio
async def test_extract_via_worker_passes_through_worker_error_status(monkeypatch):
    monkeypatch.setattr("asset_management.thumbnail_service.httpx.AsyncClient", _FakeWorkerClient)
    _FakeWorkerClient.response = _FakeWorkerResponse(502, json_body={"detail": "Thumbnail extraction failed"})

    svc = _make_service(db=_FakeDB())
    with pytest.raises(HTTPException) as exc_info:
        await svc._extract_via_worker("https://fake/video.mp4", 1.0)
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Thumbnail extraction failed"


@pytest.mark.asyncio
async def test_extract_via_worker_unreachable_raises_502(monkeypatch):
    class _UnreachableClient(_FakeWorkerClient):
        async def post(self, url, json=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("asset_management.thumbnail_service.httpx.AsyncClient", _UnreachableClient)

    svc = _make_service(db=_FakeDB())
    with pytest.raises(HTTPException) as exc_info:
        await svc._extract_via_worker("https://fake/video.mp4", 1.0)
    assert exc_info.value.status_code == 502
    assert "unreachable" in exc_info.value.detail.lower()
