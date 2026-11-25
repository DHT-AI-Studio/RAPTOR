"""
MTR-4 / VIE01-304: GET /users/commits status filtering.

One asset per state (all 6), verifying:
- no status filter -> all 6 returned
- a single status -> exactly that one
- repeated status -> exactly that subset

Uses a real Postgres (same instance/credentials as test_claim_processing.py,
via conftest.py) since the whole point is proving status = ANY($n) actually
filters correctly, not something worth mocking.

The ticket also asks for a "failed" row produced by MTR-3's write-back
rather than inserted directly -- MTR-3 (VIE01-255) hadn't landed when this
file was first written, so that row was faked (see git history). Now that
it has, test_status_filter_reflects_real_mtr3_writeback below drives the
actual upload -> claim -> update-status(failed) chain against a live API
and confirms GET /users/commits?status=failed picks it up -- closing that
gap without touching the four DB-layer tests above, which still earn their
keep as a fast, isolated check of the SQL filter itself.

Run from the 04-object-storage/ directory (API must be running for the
live-chain test; the DB-layer tests below only need Postgres):
    python -m pytest tests/integration/test_commits_status_filter.py -v
"""
import os
import uuid
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio
from asset_management.database import Database
from conftest import MODULE04_BASE_URL

TEST_USER = "test_commits_status_filter_user"
TEST_BRANCH = "test_commits_status_filter_branch"
ALL_STATUSES = ["uploaded", "processing", "active", "failed", "unregistered", "archived"]

# TEST_MODULE_04_URL still wins if set (e.g. pointing at an ad-hoc server),
# otherwise follow the same HOST_IP-derived host as every other client in
# this suite -- was a bare "localhost:8000" default, which silently hit
# whatever's running locally instead of the host HOST_IP actually points
# at, same mismatch test_asset_lifecycle.py had.
API_BASE_URL = os.environ.get("TEST_MODULE_04_URL", MODULE04_BASE_URL)
WRITEBACK_USER = "test_commits_status_filter_writeback_user"


@pytest_asyncio.fixture
async def db():
    database = Database()
    await database.init_db()
    async with database.pool.acquire() as conn:
        for status in ALL_STATUSES:
            await conn.execute(
                """
                INSERT INTO commit_history (
                    asset_path, version_id, branch, updated_by, primary_filename,
                    asset_key, associated_filenames, upload_date, status, checksum
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                f"doc/status-filter-{status}", f"v-{uuid.uuid4().hex[:8]}", TEST_BRANCH,
                TEST_USER, f"{status}.txt", f"doc/status-filter-{status}/{status}.txt",
                "[]", datetime.now(timezone.utc), status, "fakechecksum",
            )
    yield database
    async with database.pool.acquire() as conn:
        await conn.execute("DELETE FROM commit_history WHERE updated_by = $1", TEST_USER)
    await database.close()


@pytest.mark.asyncio
async def test_no_status_filter_returns_all_six(db: Database):
    result = await db.get_user_commits(username=TEST_USER, branch=TEST_BRANCH, page_size=20)
    statuses = {c.status for c in result["commits"]}
    assert statuses == set(ALL_STATUSES)
    assert result["total_count"] == 6


@pytest.mark.asyncio
async def test_single_status_filter(db: Database):
    result = await db.get_user_commits(username=TEST_USER, branch=TEST_BRANCH, page_size=20, statuses=["failed"])
    assert result["total_count"] == 1
    assert result["commits"][0].status == "failed"


@pytest.mark.asyncio
async def test_repeated_status_filter_returns_exact_subset(db: Database):
    result = await db.get_user_commits(
        username=TEST_USER, branch=TEST_BRANCH, page_size=20,
        statuses=["failed", "processing", "archived"],
    )
    statuses = {c.status for c in result["commits"]}
    assert statuses == {"failed", "processing", "archived"}
    assert result["total_count"] == 3


@pytest.mark.asyncio
async def test_status_filter_combines_with_keyword(db: Database):
    """Pagination/keyword/date-range filters continue working in combination
    with status (VIE01-304 acceptance criterion)."""
    result = await db.get_user_commits(
        username=TEST_USER, branch=TEST_BRANCH, page_size=20,
        statuses=["failed"], keyword="failed",
    )
    assert result["total_count"] == 1
    assert result["commits"][0].status == "failed"

    result_no_match = await db.get_user_commits(
        username=TEST_USER, branch=TEST_BRANCH, page_size=20,
        statuses=["failed"], keyword="archived",
    )
    assert result_no_match["total_count"] == 0


@pytest.mark.asyncio
async def test_status_filter_reflects_real_mtr3_writeback():
    """The ticket's own acceptance criterion: a "failed" row produced by
    MTR-3's real write-back chain (upload -> claim-processing -> update-
    status), not inserted into Postgres directly, still shows up correctly
    under GET /users/commits?status=failed and disappears under any other
    single-status filter.

    Skips (rather than failing) if the API isn't reachable, matching this
    module's other live-API test (tests/integration/test_asset_lifecycle.py).
    """
    headers = {"X-User-ID": WRITEBACK_USER, "X-Branch-ID": TEST_BRANCH}
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=20.0) as client:
        try:
            await client.get("/users/commits", headers=headers, params={"page_size": 1})
        except httpx.ConnectError:
            pytest.skip(f"Module 04 API not reachable at {API_BASE_URL}")

        # ── 1. Upload -- real status starts "uploaded" ──────────────────
        run_id = uuid.uuid4().hex[:8]
        upload_resp = await client.post(
            "/fileupload", headers=headers,
            files={"primary_file": (f"writeback_{run_id}.txt", f"mtr-3 writeback test {run_id}".encode(), "text/plain")},
        )
        assert upload_resp.status_code == 200, upload_resp.text
        asset_info = upload_resp.json()
        asset_path, version_id = asset_info["asset_path"], asset_info["version_id"]

        try:
            # ── 2. Claim -- real status "uploaded" -> "processing" ──────
            claim_resp = await client.post(
                f"/claim-processing/{asset_path}/{version_id}",
                headers=headers, params={"correlation_id": uuid.uuid4().hex},
            )
            assert claim_resp.status_code == 200, claim_resp.text

            # ── 3. MTR-3's write-back -- real status "processing" -> "failed" ──
            update_resp = await client.post(
                f"/update-status/{asset_path}/{version_id}",
                headers=headers, params={"status": "failed"},
            )
            assert update_resp.status_code == 200, update_resp.text
            assert update_resp.json()["status"] == "failed"

            # ── 4. The List API sees the real failed row ─────────────────
            failed_resp = await client.get("/users/commits", headers=headers, params={"status": "failed", "page_size": 20})
            assert failed_resp.status_code == 200, failed_resp.text
            failed_commits = failed_resp.json()["commits"]
            assert any(c["asset_path"] == asset_path and c["version_id"] == version_id for c in failed_commits), failed_commits

            # ...and a filter for any other single status excludes it.
            active_resp = await client.get("/users/commits", headers=headers, params={"status": "active", "page_size": 20})
            assert active_resp.status_code == 200, active_resp.text
            active_commits = active_resp.json()["commits"]
            assert not any(c["asset_path"] == asset_path and c["version_id"] == version_id for c in active_commits), active_commits
        finally:
            database = Database()
            await database.init_db()
            async with database.pool.acquire() as conn:
                await conn.execute("DELETE FROM commit_history WHERE updated_by = $1", WRITEBACK_USER)
            await database.close()
