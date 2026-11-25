"""Reproduces, and verifies the fix for, GET /filedownload silently dropping
associated files when an asset has enough of them (keyframes, face thumbnails,
etc.) that presigning them all needs more than redis-py's default 100-
connection pool at once.

Root cause (confirmed live, redis-py 8.1.0): retrieve_asset() used to fan out
one generate_presigned_url() per associated file via asyncio.gather() with no
concurrency limit; each one first read the presign Redis cache, and that read
wasn't wrapped in its own try/except, so once the pool's 100 connections were
all in use, the next read raised ConnectionError("Too many connections") and
the whole per-file lookup failed -- silently, since fetch_file()'s own
except swallows it and returns None. The fix is two parts: a Redis read
failure is now treated the same as a cache miss (falls through to LakeFS)
instead of failing the request, and retrieve_asset() resolves most files'
presigned URLs via one batch_generate_presigned_urls() LakeFS list_objects()
call per distinct version_id instead of one LakeFS/Redis round trip per file,
so the 100-connection burst this test reproduces no longer happens in the
first place for the common case (all associated files in one commit).

Before each case, clears this asset's own presign:* Redis keys so every
presign genuinely goes through Redis + LakeFS, not a cache hit from a
previous run.

Skips (not fails) if Redis isn't reachable, so the suite stays green on a
host that doesn't run it. Postgres/LakeFS/SeaweedFS cleanup for
PRESIGN_BATCH_BRANCH is handled by conftest.py's global_teardown.

Run from the 04-object-storage/ directory:
    python -m pytest tests/integration/test_presign_batch_many_files.py -v
"""
import asyncio
import time

import httpx
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from conftest import MODULE04_BASE_URL, REDIS_URL

# Must match conftest.py's PRESIGN_BATCH_BRANCH literal -- not imported, same
# convention test_commits_status_filter.py's TEST_BRANCH/COMMITS_FILTER_BRANCH
# pair already uses.
TEST_BRANCH = "test_presign_batch_many_files_branch"
TEST_USER = "test_presign_batch_many_files_user"

# Smallest file libmagic reports as video/mp4 (an ftyp box); unique suffix
# via run_id keeps each run's upload from deduplicating against a prior one.
MP4_HEADER = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 256 + b"\xff\xd9"


@pytest_asyncio.fixture
async def redis_client():
    r = aioredis.Redis.from_url(REDIS_URL)
    try:
        await r.ping()
    except Exception:  # noqa: BLE001 -- reachability probe; any failure means skip, reason given to pytest.skip
        await r.aclose()
        pytest.skip(f"Redis not reachable at {REDIS_URL}")
    yield r
    await r.aclose()


async def _clear_presign_cache(r: aioredis.Redis, asset_path: str) -> int:
    n = 0
    async for key in r.scan_iter(match="presign:*", count=500):
        if asset_path in (key.decode() if isinstance(key, bytes) else key):
            await r.delete(key)
            n += 1
    return n


async def _download(http: httpx.AsyncClient, asset_path: str, version_id: str) -> int:
    resp = await http.get(f"/filedownload/{asset_path}/{version_id}")
    resp.raise_for_status()
    body = resp.json()
    return sum(1 for k, v in body.items() if k.startswith("associated_file_") and v)


@pytest.mark.asyncio
async def test_filedownload_returns_all_associated_files_single_request(redis_client):
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    n_files = 150  # comfortably above redis-py 8.x's 100-connection default pool
    async with httpx.AsyncClient(base_url=MODULE04_BASE_URL, headers=headers, timeout=60.0) as http:
        tag = str(time.time_ns())
        resp = await http.post("/fileupload", files={
            "primary_file": (f"presign_batch_{tag}.mp4", MP4_HEADER + tag.encode(), "video/mp4")})
        resp.raise_for_status()
        asset_path, version_id = resp.json()["asset_path"], resp.json()["version_id"]

        files = [("associated_files", (f"repro_{i:04d}.jpg", JPEG + str(i).encode(), "image/jpeg"))
                 for i in range(n_files)]
        resp = await http.post(f"/add-associated-files/{asset_path}", files=files,
                                data={"primary_version_id": version_id})
        resp.raise_for_status()

        await _clear_presign_cache(redis_client, asset_path)
        returned = await _download(http, asset_path, version_id)
        assert returned == n_files, f"expected {n_files} associated files, got {returned}"


@pytest.mark.asyncio
async def test_filedownload_returns_all_associated_files_concurrent_requests(redis_client):
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    n_files = 60
    n_concurrent = 10
    async with httpx.AsyncClient(base_url=MODULE04_BASE_URL, headers=headers, timeout=60.0) as http:
        tag = str(time.time_ns())
        resp = await http.post("/fileupload", files={
            "primary_file": (f"presign_batch_concurrent_{tag}.mp4", MP4_HEADER + tag.encode(), "video/mp4")})
        resp.raise_for_status()
        asset_path, version_id = resp.json()["asset_path"], resp.json()["version_id"]

        files = [("associated_files", (f"repro_{i:04d}.jpg", JPEG + str(i).encode(), "image/jpeg"))
                 for i in range(n_files)]
        resp = await http.post(f"/add-associated-files/{asset_path}", files=files,
                                data={"primary_version_id": version_id})
        resp.raise_for_status()

        await _clear_presign_cache(redis_client, asset_path)
        counts = await asyncio.gather(*[_download(http, asset_path, version_id) for _ in range(n_concurrent)])
        assert all(c == n_files for c in counts), f"expected {n_files} files on all {n_concurrent} requests, got {counts}"
