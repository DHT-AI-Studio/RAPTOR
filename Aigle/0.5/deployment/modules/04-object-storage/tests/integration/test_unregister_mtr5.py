"""
MTR-5 / VIE01-305: Unregister API -- end-to-end test against a real deployment.

Real chain: upload ("uploaded") -> claim-processing ("processing") ->
update-status active (MTR-3's real write-back, same mechanism
test_commits_status_filter.py's own live-chain test uses) -> Unregister
("active" -> "unregistered") -> Register fast path ("unregistered" ->
"active"). Verifies the full round trip plus both of the ticket's negative
requirements: 409 when Unregister is attempted on a non-active asset
(archived), and that GET /users/commits?status=unregistered surfaces the
asset mid-round-trip.

Note on the ticket's "default no-filter List does not show it" line: this
contradicts MTR-4 (VIE01-304), which was built -- and is tested -- so that
the default (no filter) returns all 6 statuses, including "unregistered",
per an earlier ticket (VIE01-214) that explicitly required this endpoint
never default to active-only. That earlier contract is kept; this test
does not assert the no-filter case excludes "unregistered".

Run from the 04-object-storage/ directory:
    python -m pytest tests/integration/test_unregister_mtr5.py -v
"""
from __future__ import annotations

import uuid

import httpx
import pytest
from conftest import MODULE04_BASE_URL

API_BASE_URL = MODULE04_BASE_URL
TEST_USER = "test_unregister_mtr5_user"
TEST_BRANCH = "test_unregister_mtr5_branch"


def _gateway_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE_URL}/docs", timeout=3.0)
        return True
    except Exception:  # noqa: BLE001 -- reachability probe for skipif; any failure means not-reachable
        return False


pytestmark = pytest.mark.skipif(not _gateway_reachable(), reason=f"Module 04 API not reachable at {API_BASE_URL}")


@pytest.mark.asyncio
async def test_unregister_register_round_trip_and_negative_cases():
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=20.0) as client:
        run_id = uuid.uuid4().hex[:8]

        # ── 1. Upload -- real status starts "uploaded" ──────────────────
        upload_resp = await client.post(
            "/fileupload", headers=headers,
            files={"primary_file": (f"mtr5_{run_id}.txt", f"mtr-5 unregister test {run_id}".encode(), "text/plain")},
        )
        assert upload_resp.status_code == 200, upload_resp.text
        asset_info = upload_resp.json()
        asset_path, version_id = asset_info["asset_path"], asset_info["version_id"]

        try:
            # ── 2. Negative: Unregister rejected before the asset is active ──
            premature_resp = await client.post(f"/unregister/{asset_path}/{version_id}", headers=headers)
            assert premature_resp.status_code == 409, premature_resp.text

            # ── 3. Claim -- "uploaded" -> "processing" ───────────────────
            claim_resp = await client.post(
                f"/claim-processing/{asset_path}/{version_id}",
                headers=headers, params={"correlation_id": uuid.uuid4().hex},
            )
            assert claim_resp.status_code == 200, claim_resp.text

            # ── 4. MTR-3's real write-back -- "processing" -> "active" ───
            update_resp = await client.post(
                f"/update-status/{asset_path}/{version_id}",
                headers=headers, params={"status": "active"},
            )
            assert update_resp.status_code == 200, update_resp.text
            assert update_resp.json()["status"] == "active"

            # ── 5. Unregister -- "active" -> "unregistered" ──────────────
            unreg_resp = await client.post(f"/unregister/{asset_path}/{version_id}", headers=headers)
            assert unreg_resp.status_code == 200, unreg_resp.text
            assert unreg_resp.json()["status"] == "unregistered"

            # ── 6. Explicit status filter surfaces it ─────────────────────
            filtered_resp = await client.get("/users/commits", headers=headers, params={"status": "unregistered", "page_size": 20})
            assert filtered_resp.status_code == 200, filtered_resp.text
            filtered_commits = filtered_resp.json()["commits"]
            assert any(c["asset_path"] == asset_path and c["version_id"] == version_id for c in filtered_commits), filtered_commits

            # A different single-status filter correctly excludes it.
            active_only_resp = await client.get("/users/commits", headers=headers, params={"status": "active", "page_size": 20})
            assert active_only_resp.status_code == 200, active_only_resp.text
            active_commits = active_only_resp.json()["commits"]
            assert not any(c["asset_path"] == asset_path and c["version_id"] == version_id for c in active_commits), active_commits

            # ── 7. Negative: Unregister rejected again now it's unregistered (not idempotent) ──
            repeat_resp = await client.post(f"/unregister/{asset_path}/{version_id}", headers=headers)
            assert repeat_resp.status_code == 409, repeat_resp.text

            # ── 8. Register fast path -- "unregistered" -> "active", no re-analysis ──
            register_resp = await client.post(f"/register/{asset_path}/{version_id}", headers=headers)
            assert register_resp.status_code == 200, register_resp.text
            register_body = register_resp.json()
            assert register_body["outcome"] == "registered"
            assert register_body["metadata"]["status"] == "active"

            # ── 9. Final state confirmed active via an independent read ──
            final_resp = await client.get("/users/commits", headers=headers, params={"status": "active", "page_size": 20})
            final_commits = final_resp.json()["commits"]
            assert any(c["asset_path"] == asset_path and c["version_id"] == version_id for c in final_commits), final_commits
        finally:
            from asset_management.database import Database
            database = Database()
            await database.init_db()
            async with database.pool.acquire() as conn:
                await conn.execute("DELETE FROM commit_history WHERE updated_by = $1", TEST_USER)
            await database.close()


@pytest.mark.asyncio
async def test_unregister_rejected_on_archived_real_chain():
    """Ticket's explicit negative-test subtask, against real infra: an
    archived asset (not just a non-active one in the abstract) must 409."""
    headers = {"X-User-ID": TEST_USER, "X-Branch-ID": TEST_BRANCH}
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=20.0) as client:
        run_id = uuid.uuid4().hex[:8]
        upload_resp = await client.post(
            "/fileupload", headers=headers,
            files={"primary_file": (f"mtr5_archived_{run_id}.txt", f"mtr-5 archived test {run_id}".encode(), "text/plain")},
        )
        assert upload_resp.status_code == 200, upload_resp.text
        asset_info = upload_resp.json()
        asset_path, version_id = asset_info["asset_path"], asset_info["version_id"]

        try:
            archive_resp = await client.post(f"/filearchive/{asset_path}/{version_id}", headers=headers)
            assert archive_resp.status_code == 200, archive_resp.text
            assert archive_resp.json()["status"] == "archived"

            unreg_resp = await client.post(f"/unregister/{asset_path}/{version_id}", headers=headers)
            assert unreg_resp.status_code == 409, unreg_resp.text
        finally:
            from asset_management.database import Database
            database = Database()
            await database.init_db()
            async with database.pool.acquire() as conn:
                await conn.execute("DELETE FROM commit_history WHERE updated_by = $1", TEST_USER)
            await database.close()
