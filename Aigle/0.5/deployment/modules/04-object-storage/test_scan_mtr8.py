"""
MTR-8 / VIE01-307: Scan API -- live Postgres + live HTTP coverage.

Seeds real commit_history rows with real checksums at several statuses
(including "processing", since the ticket explicitly calls out that state
needing correct reporting), then proves the batch scan finds them --
first at the Database layer directly, then through the real running
Module 04 HTTP API, then through Module 13's gateway.

Run from the 04-object-storage/ directory:
    python -m pytest test_scan_mtr8.py -v
"""
import os
import uuid
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio
from asset_management.config import settings

settings.postgres_host = os.environ.get("TEST_POSTGRES_HOST", "localhost")
settings.postgres_port = int(os.environ.get("TEST_POSTGRES_PORT", "5433"))
settings.postgres_user = os.environ.get("TEST_POSTGRES_USER", "raptor")
settings.postgres_password = os.environ.get("TEST_POSTGRES_PASSWORD", "raptor")
# "asset_management", not "raptor" (other live tests' default) -- that's
# what the real deployed container actually connects to
# (docker-compose.yml overrides POSTGRES_DB for the app service), and this
# file needs to agree with it since it also talks to that live container
# over HTTP, not just the DB directly.
settings.postgres_db = os.environ.get("TEST_POSTGRES_DB", "asset_management")

from asset_management.database import Database

API_BASE_URL = os.environ.get("TEST_MODULE_04_URL", "http://localhost:8000")
TEST_USER = "test_scan_mtr8_user"
TEST_BRANCH = "test_scan_mtr8_branch"


def _gateway_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE_URL}/docs", timeout=3.0)
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _gateway_reachable(), reason=f"Module 04 API not reachable at {API_BASE_URL}")


# Real 32-char MD5-shaped hex, one per seeded row.
CHECKSUM_ACTIVE = "a" * 32
CHECKSUM_PROCESSING = "b" * 32
CHECKSUM_ARCHIVED = "c" * 32
CHECKSUM_UNKNOWN = "d" * 32  # never inserted -- must come back not_found


@pytest_asyncio.fixture
async def seeded_db():
    database = Database()
    await database.init_db()
    rows = [
        ("active", CHECKSUM_ACTIVE),
        ("processing", CHECKSUM_PROCESSING),
        ("archived", CHECKSUM_ARCHIVED),
    ]
    async with database.pool.acquire() as conn:
        for status, checksum in rows:
            await conn.execute(
                """
                INSERT INTO commit_history (
                    asset_path, version_id, branch, updated_by, primary_filename,
                    asset_key, associated_filenames, upload_date, status, checksum
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                f"doc/scan-{status}", f"v-{uuid.uuid4().hex[:8]}", TEST_BRANCH,
                TEST_USER, f"{status}.txt", f"doc/scan-{status}/{status}.txt",
                "[]", datetime.now(timezone.utc), status, checksum,
            )
    yield database
    async with database.pool.acquire() as conn:
        await conn.execute("DELETE FROM commit_history WHERE updated_by = $1", TEST_USER)
    await database.close()


# ── Database.scan_assets() directly ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_with_known_duplicates_and_unknowns(seeded_db: Database):
    results = await seeded_db.scan_assets(
        files=[
            ("a.txt", CHECKSUM_ACTIVE),
            ("b.txt", CHECKSUM_UNKNOWN),
            ("c.txt", CHECKSUM_ARCHIVED),
        ],
        branch=TEST_BRANCH,
    )
    by_path = {r["path"]: r for r in results}
    assert by_path["a.txt"]["status"] == "active"
    assert by_path["a.txt"]["existing_asset_path"] == "doc/scan-active"
    assert by_path["b.txt"]["status"] == "not_found"
    assert "existing_asset_path" not in by_path["b.txt"]
    assert by_path["c.txt"]["status"] == "archived"


@pytest.mark.asyncio
async def test_processing_asset_reported_correctly(seeded_db: Database):
    """The ticket's own explicit callout: an asset mid-"processing" must
    be reported as exactly that, not folded into active/uploaded/etc."""
    results = await seeded_db.scan_assets(files=[("x.txt", CHECKSUM_PROCESSING)], branch=TEST_BRANCH)
    assert results[0]["status"] == "processing"
    assert results[0]["existing_asset_path"] == "doc/scan-processing"


@pytest.mark.asyncio
async def test_scan_is_scoped_to_branch(seeded_db: Database):
    """Same checksum, different branch -- must not leak across users."""
    results = await seeded_db.scan_assets(files=[("x.txt", CHECKSUM_ACTIVE)], branch="some-other-branch")
    assert results[0]["status"] == "not_found"


# ── live HTTP, through the real running Module 04 service ─────────────────

@pytest.mark.asyncio
async def test_scan_http_round_trip(seeded_db: Database):
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=20.0) as client:
        resp = await client.post(
            "/scan",
            headers=headers,
            json={"files": [
                {"path": "a.txt", "checksum": CHECKSUM_ACTIVE},
                {"path": "b.txt", "checksum": f"sha256:{CHECKSUM_UNKNOWN}"},
            ]},
        )
        assert resp.status_code == 200, resp.text
        results = {r["path"]: r for r in resp.json()["results"]}
        assert results["a.txt"]["status"] == "active"
        assert results["b.txt"]["status"] == "not_found"

        # Empty batch -> 200 with an empty list, not an error.
        empty_resp = await client.post("/scan", headers=headers, json={"files": []})
        assert empty_resp.status_code == 200, empty_resp.text
        assert empty_resp.json() == {"results": []}

        # Malformed checksum -> 400.
        bad_resp = await client.post(
            "/scan", headers=headers, json={"files": [{"path": "a.txt", "checksum": "not-a-checksum"}]},
        )
        assert bad_resp.status_code == 400, bad_resp.text


@pytest.mark.asyncio
async def test_scan_through_gateway(seeded_db: Database):
    """Same round trip, through Module 13's public-facing route, if the
    gateway happens to be reachable too -- skips cleanly if not (this
    file's own gateway-reachability check only covers Module 04)."""
    gateway_url = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
    try:
        async with httpx.AsyncClient(timeout=3.0) as probe:
            await probe.get(f"{gateway_url}/docs")
    except httpx.HTTPError:
        pytest.skip(f"gateway not reachable at {gateway_url}")

    jwt = os.environ.get("TEST_SCAN_JWT", "")
    if not jwt:
        pytest.skip("set TEST_SCAN_JWT to a real user's JWT to exercise the gateway path")

    api_version = os.environ.get("TEST_API_VERSION", "0.4")
    async with httpx.AsyncClient(base_url=gateway_url, timeout=20.0) as client:
        resp = await client.post(
            f"/api/{api_version}/asset/scan",
            headers={"Authorization": f"Bearer {jwt}"},
            json={"files": [{"path": "a.txt", "checksum": CHECKSUM_ACTIVE}]},
        )
        assert resp.status_code == 200, resp.text
