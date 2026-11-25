"""
MTR-7 / VIE01-306: Thumbnail API -- live validation against a real NVIDIA
CUDA node.

Covers the two subtasks test_thumbnail_service.py's mocked unit tests
can't: "correct frame at several timestamps" against real video content,
and actual NVDEC decode on the real GPU. Generates its own tiny synthetic
test clip via `docker exec <thumbnail-worker container> ffmpeg -f lavfi
testsrc=...` so no external sample video or host-side ffmpeg install is
required -- ffmpeg lives only in the GPU-isolated thumbnail-worker
sidecar, not in Module 04's main app container.

Run from the 04-object-storage/ directory:
    python -m pytest test_thumbnail_cuda_node.py -v
"""
from __future__ import annotations

import os
import subprocess
import uuid

import httpx
import pytest
from conftest import MODULE04_BASE_URL

API_BASE_URL = MODULE04_BASE_URL
WORKER_CONTAINER_NAME = os.environ.get("TEST_THUMBNAIL_WORKER_CONTAINER", "raptor-thumbnail-worker")
TEST_USER = "test_thumbnail_mtr7_user"
TEST_BRANCH = "test_thumbnail_mtr7_branch"
CLIP_DURATION_SECONDS = 10


def _gateway_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE_URL}/docs", timeout=3.0)
        return True
    except Exception:  # noqa: BLE001 -- reachability probe for skipif; any failure means not-reachable
        return False


def _worker_has_nvdec_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["docker", "exec", WORKER_CONTAINER_NAME, "ffmpeg", "-hide_banner", "-hwaccels"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "cuda" in result.stdout.lower()
    except Exception:  # noqa: BLE001 -- capability probe for skipif; any failure means not-available
        return False


pytestmark = pytest.mark.skipif(
    not _gateway_reachable() or not _worker_has_nvdec_ffmpeg(),
    reason=f"requires Module 04 reachable at {API_BASE_URL} AND the thumbnail-worker "
           f"container ({WORKER_CONTAINER_NAME}) to have an NVDEC-capable ffmpeg "
           "(ffmpeg -hwaccels must list 'cuda')",
)


def _generate_synthetic_clip(run_id: str) -> bytes:
    """Builds a tiny test-pattern clip with a visible, changing frame
    counter (drawtext) inside Module 04's own container, so frames at
    different timestamps are visually distinguishable -- proving a real
    seek happened, not a fixed/cached dummy frame returned every time.

    run_id is burned into the overlay text so each run's actual encoded
    bytes differ -- plain testsrc+drawtext(%{pts}) output is otherwise
    bit-for-bit deterministic across runs, which made register_asset()'s
    content-dedup fast-path silently match this test's own previous run
    and skip straight to "active", making the premature-409 assertion
    meaningless (it was hitting an already-active asset, never testing
    the uploaded-state guard at all)."""
    remote_path = f"/tmp/mtr7_fixture_{uuid.uuid4().hex[:8]}.mp4"
    gen = subprocess.run(
        [
            "docker", "exec", WORKER_CONTAINER_NAME, "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc=duration={CLIP_DURATION_SECONDS}:size=320x240:rate=5",
            "-vf", f"drawtext=text='{run_id} %{{pts\\:hms}}':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black",
            "-pix_fmt", "yuv420p", "-y", remote_path,
        ],
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert gen.returncode == 0, f"failed to generate test clip: {gen.stderr}"
    try:
        cat = subprocess.run(
            ["docker", "exec", WORKER_CONTAINER_NAME, "cat", remote_path],
            check=False, capture_output=True, timeout=15,
        )
        assert cat.returncode == 0 and cat.stdout, "failed to read back generated clip bytes"
        return cat.stdout
    finally:
        subprocess.run(["docker", "exec", WORKER_CONTAINER_NAME, "rm", "-f", remote_path], check=False, capture_output=True, timeout=10)


def _generate_10bit_clip(run_id: str) -> bytes:
    """A 10-bit HEVC clip -- NVDEC decodes this to p010le/p016le, not nv12.
    Regression test for the real failure this reproduced: `hwdownload`
    rejects a requested format that doesn't match the source's actual
    native one ("Invalid output format nv12 for hwframe download"),
    confirmed live against a real asset in production."""
    remote_path = f"/tmp/mtr7_10bit_fixture_{uuid.uuid4().hex[:8]}.mp4"
    gen = subprocess.run(
        [
            "docker", "exec", WORKER_CONTAINER_NAME, "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc=duration={CLIP_DURATION_SECONDS}:size=320x240:rate=5",
            "-vf", f"drawtext=text='{run_id} %{{pts\\:hms}}':fontcolor=white:fontsize=24:x=10:y=10:box=1:boxcolor=black",
            "-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-tag:v", "hvc1", "-y", remote_path,
        ],
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert gen.returncode == 0, f"failed to generate 10-bit test clip: {gen.stderr}"
    try:
        cat = subprocess.run(
            ["docker", "exec", WORKER_CONTAINER_NAME, "cat", remote_path],
            check=False, capture_output=True, timeout=15,
        )
        assert cat.returncode == 0 and cat.stdout, "failed to read back generated 10-bit clip bytes"
        return cat.stdout
    finally:
        subprocess.run(["docker", "exec", WORKER_CONTAINER_NAME, "rm", "-f", remote_path], check=False, capture_output=True, timeout=10)


def _assert_is_jpeg(data: bytes):
    assert data[:2] == b"\xff\xd8", f"response does not start with the JPEG magic bytes: {data[:16]!r}"
    assert data[-2:] == b"\xff\xd9", f"response does not end with the JPEG EOI marker: {data[-16:]!r}"


@pytest.mark.asyncio
async def test_thumbnail_round_trip_real_gpu_decode():
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    run_id = uuid.uuid4().hex[:8]
    clip_bytes = _generate_synthetic_clip(run_id)

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0) as client:
        # ── 1. Upload -> claim -> active (same real chain as MTR-5's test) ──
        upload_resp = await client.post(
            "/fileupload", headers=headers,
            files={"primary_file": (f"mtr7_{run_id}.mp4", clip_bytes, "video/mp4")},
        )
        assert upload_resp.status_code == 200, upload_resp.text
        asset_info = upload_resp.json()
        asset_path, version_id = asset_info["asset_path"], asset_info["version_id"]

        # ── 2. Negative: 404 for a non-existent asset ─────────────────────
        missing_resp = await client.get(
            f"/thumbnail/video/mp4/does-not-exist-{run_id}/no-such-version",
            headers=headers, params={"second": 1.0},
        )
        assert missing_resp.status_code == 404, missing_resp.text

        # ── 3. Negative: 409 before the asset is active ───────────────────
        premature_resp = await client.get(
            f"/thumbnail/{asset_path}/{version_id}", headers=headers, params={"second": 1.0},
        )
        assert premature_resp.status_code == 409, premature_resp.text

        claim_resp = await client.post(
            f"/claim-processing/{asset_path}/{version_id}",
            headers=headers, params={"correlation_id": uuid.uuid4().hex},
        )
        assert claim_resp.status_code == 200, claim_resp.text

        update_resp = await client.post(
            f"/update-status/{asset_path}/{version_id}", headers=headers, params={"status": "active"},
        )
        assert update_resp.status_code == 200, update_resp.text

        # ── 4. Correct frame at several timestamps -- real NVDEC decode ───
        t1_resp = await client.get(f"/thumbnail/{asset_path}/{version_id}", headers=headers, params={"second": 1.0})
        assert t1_resp.status_code == 200, t1_resp.text
        assert t1_resp.headers["content-type"] == "image/jpeg"
        _assert_is_jpeg(t1_resp.content)

        t8_resp = await client.get(f"/thumbnail/{asset_path}/{version_id}", headers=headers, params={"second": 8.0})
        assert t8_resp.status_code == 200, t8_resp.text
        _assert_is_jpeg(t8_resp.content)

        # Different timestamps -> visibly different frames (the drawtext
        # counter differs), not the same cached/dummy image every time.
        assert t1_resp.content != t8_resp.content, "thumbnails at second=1 and second=8 must differ"

        # ── 5. Cache hit: identical bytes, and far faster than a cold decode ──
        import time
        start = time.monotonic()
        t1_again_resp = await client.get(f"/thumbnail/{asset_path}/{version_id}", headers=headers, params={"second": 1.0})
        cache_hit_elapsed = time.monotonic() - start
        assert t1_again_resp.status_code == 200
        assert t1_again_resp.content == t1_resp.content, "cache hit must return byte-identical content"
        assert cache_hit_elapsed < 2.0, f"cache hit took {cache_hit_elapsed:.2f}s -- too slow to plausibly have skipped ffmpeg"

        # ── 6. Negative: 400 for a timestamp beyond the clip's duration ───
        beyond_resp = await client.get(
            f"/thumbnail/{asset_path}/{version_id}", headers=headers, params={"second": 999.0},
        )
        assert beyond_resp.status_code == 400, beyond_resp.text


@pytest.mark.asyncio
async def test_thumbnail_10bit_source_real_gpu_decode():
    """Regression test: a 10-bit source used to fail extraction outright
    (NVDEC decodes it to p010le/p016le, not nv12 -- confirmed live against
    a real production asset, and reproduced here against a synthetic one)."""
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    run_id = uuid.uuid4().hex[:8]
    clip_bytes = _generate_10bit_clip(run_id)

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0) as client:
        upload_resp = await client.post(
            "/fileupload", headers=headers,
            files={"primary_file": (f"mtr7_10bit_{run_id}.mp4", clip_bytes, "video/mp4")},
        )
        assert upload_resp.status_code == 200, upload_resp.text
        asset_info = upload_resp.json()
        asset_path, version_id = asset_info["asset_path"], asset_info["version_id"]

        claim_resp = await client.post(
            f"/claim-processing/{asset_path}/{version_id}",
            headers=headers, params={"correlation_id": uuid.uuid4().hex},
        )
        assert claim_resp.status_code == 200, claim_resp.text

        update_resp = await client.post(
            f"/update-status/{asset_path}/{version_id}", headers=headers, params={"status": "active"},
        )
        assert update_resp.status_code == 200, update_resp.text

        thumb_resp = await client.get(f"/thumbnail/{asset_path}/{version_id}", headers=headers, params={"second": 1.0})
        assert thumb_resp.status_code == 200, thumb_resp.text
        assert thumb_resp.headers["content-type"] == "image/jpeg"
        _assert_is_jpeg(thumb_resp.content)
