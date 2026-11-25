"""VIE01-214/VIE01-215 follow-up promised in e4b0e92's test_register_clone_from.py
docstring: "the 163-only follow-up for the full real-ArcadeDB path once
Module 25 is redeployed there."

Drives the actual clone_point() -> register_asset() round trip through real
Database(), ObjectStore() and SearchSync() (no fakes) against a live Module 25
+ ArcadeDB, and verifies the Chunk/Source status string directly via ArcadeDB's
own HTTP API at each step -- "cloned" right after upload, "active" right after
Register -- the same way this was verified manually against 163 before this
file existed. test_register_clone_from.py already proves register_asset()'s
own Postgres-side branching with Module 25 stubbed; this file is the other
half, proving clone_version()/the status flip actually happen for real.

Also covers update_status()'s real Module 25 round trip for a genuinely-
indexed asset. (The check_indexed() downgrade-on-not-indexed behavior this
file used to also cover is currently disabled in client.py -- it raced
against Module 25's async PA-8 consumer and downgraded real successes to
"failed"; see update_status()'s docstring.)

Skips (not fails) if Module 25/ArcadeDB isn't reachable, so the suite stays
green on a host that doesn't run them (matches test_commits_status_filter.py's
live-chain test). Postgres/LakeFS/SeaweedFS cleanup for TEST_BRANCH is handled
by conftest.py's global_teardown (added there alongside COMMITS_FILTER_BRANCH);
this file only tears down the Module 25/ArcadeDB side, which that fixture
doesn't reach.

Run from the 04-object-storage/ directory, with HOST_IP pointed at a host that
runs Module 25 (this module's own default, host 123, does not):
    python -m pytest tests/integration/test_clone_point_real_arcadedb.py -v
"""
import logging
import re
import uuid

import httpx
import pytest
import pytest_asyncio
from asset_management.client import AssetManager
from asset_management.config import settings
from asset_management.database import Database
from asset_management.object_store import ObjectStore
from asset_management.search_sync import SearchSync
from conftest import ARCADEDB_PASSWORD, ARCADEDB_URL

logger = logging.getLogger(__name__)

# Must match conftest.py's CLONE_ARCADEDB_BRANCH literal -- not imported,
# same convention test_commits_status_filter.py's TEST_BRANCH/
# COMMITS_FILTER_BRANCH pair already uses.
TEST_BRANCH = "test_clone_point_real_arcadedb_branch"
TEST_USER = "test_clone_point_real_arcadedb_user"


def _db_name_for(branch_id: str) -> str:
    """Mirrors Module 25's own arcadedb_client.db_name_for()."""
    return "user_" + re.sub(r"[^A-Za-z0-9_]", "_", branch_id)


async def _entity_status(arcadedb: httpx.AsyncClient, db_name: str, version_id: str, entity: str) -> str | None:
    resp = await arcadedb.post(
        f"/api/v1/query/{db_name}",
        json={"language": "sql", "command": f"SELECT status FROM {entity} WHERE version_id = :v",
              "params": {"v": version_id}},
    )
    resp.raise_for_status()
    rows = resp.json().get("result", [])
    return rows[0]["status"] if rows else None


@pytest_asyncio.fixture
async def real_clients():
    arcadedb = httpx.AsyncClient(base_url=ARCADEDB_URL, auth=("root", ARCADEDB_PASSWORD), timeout=20.0)
    try:
        resp = await arcadedb.get("/api/v1/databases")
        resp.raise_for_status()
    except httpx.ConnectError:
        await arcadedb.aclose()
        pytest.skip(f"ArcadeDB not reachable at {ARCADEDB_URL}")

    m25 = httpx.AsyncClient(base_url=settings.personal_db_service_url, timeout=20.0)
    try:
        resp = await m25.post("/internal/db/init", headers={"X-Branch-ID": TEST_BRANCH})
        resp.raise_for_status()
    except httpx.ConnectError:
        await arcadedb.aclose()
        await m25.aclose()
        pytest.skip(f"Module 25 not reachable at {settings.personal_db_service_url}")

    db = Database()
    obj = ObjectStore()
    sync = SearchSync()
    await db.init_db()
    await obj.initialize()
    await sync.initialize()

    yield db, obj, sync, arcadedb

    await db.close()
    await obj.close()
    await sync.close()
    await arcadedb.aclose()
    await m25.aclose()


@pytest.mark.asyncio
async def test_clone_point_round_trip_against_real_arcadedb(real_clients):
    db, obj, sync, arcadedb = real_clients
    mgr = AssetManager(object_store=obj, search_sync=sync, db=db)
    db_name = _db_name_for(TEST_BRANCH)

    run_id = uuid.uuid4().hex[:8]
    content = f"clone-point real-arcadedb round trip {run_id}".encode()
    original = None
    dup = None

    try:
        # ── 1. Upload the original ───────────────────────────────────
        original = await mgr.upload_files(TEST_USER, TEST_BRANCH, (content, f"orig_{run_id}.txt"))
        assert original.status == "uploaded"
        assert original.cloned_from is None

        # ── 2. Simulate the orchestrator's completed analysis: index a
        # real Chunk in Module 25, then flip Postgres to "active" the
        # same way kafka_consumer.py does after a real pipeline run. ──
        async with httpx.AsyncClient(base_url=settings.personal_db_service_url, timeout=20.0) as m25:
            index_resp = await m25.post(
                "/personal/index/chunk",
                headers={"X-Branch-ID": TEST_BRANCH},
                json={
                    "chunk_id": original.version_id,
                    "type": "documents",
                    "embedding_type": "text",
                    "embedding": [0.01] * 1024,
                    "text": content.decode(),
                    "asset_path": original.asset_path,
                    "version_id": original.version_id,
                    "status": "active",
                },
            )
            index_resp.raise_for_status()
        # Through AssetManager, not db.update_status() directly.
        written = await mgr.update_status(TEST_USER, TEST_BRANCH, original.asset_path, original.version_id, "active")
        assert written == "active"
        confirmed = await db.get_asset_by_path_and_version(original.asset_path, original.version_id, TEST_BRANCH)
        assert confirmed.status == "active"

        # ── 3. Archive the original, so the dedup upload below hits the
        # "global archived, clone its index" branch. ──────────────────
        archived = await mgr.archive(TEST_USER, TEST_BRANCH, original.asset_path, original.version_id)
        assert archived.status == "archived"

        # ── 4. Upload identical content at a different path -- real
        # clone_point() -> Module 25 /personal/index/clone against the
        # real ArcadeDB, not a fake. ────────────────────────────────────
        dup = await mgr.upload_files(TEST_USER, TEST_BRANCH, (content, f"dup_{run_id}.txt"))
        assert dup.status == "uploaded", "VIE01-214: upload always leaves status=uploaded, even when cloned"
        assert dup.cloned_from == original.version_id
        assert dup.asset_path != original.asset_path

        # ── 5. Real ArcadeDB: the clone landed with status="cloned", not
        # "active" -- verified directly, not inferred from a search hit. ──
        for entity in ("Chunk", "Source"):
            status = await _entity_status(arcadedb, db_name, dup.version_id, entity)
            assert status == "cloned", f"{entity} for cloned version should be 'cloned', got {status!r}"

        # ── 6. Register the duplicate -- real register_asset() fast
        # path, real Module 25 status-flip call. ─────────────────────────
        result = await mgr.register_asset(TEST_BRANCH, dup.asset_path, dup.version_id)
        assert result["outcome"] == "registered"
        assert result["metadata"].status == "active"

        refreshed = await db.get_asset_by_path_and_version(dup.asset_path, dup.version_id, TEST_BRANCH)
        assert refreshed.status == "active"

        # ── 7. Real ArcadeDB again: the flip actually happened server-side. ──
        for entity in ("Chunk", "Source"):
            status = await _entity_status(arcadedb, db_name, dup.version_id, entity)
            assert status == "active", f"{entity} for registered version should be 'active', got {status!r}"
    finally:
        for asset in (original, dup):
            if asset is not None:
                try:
                    await sync.delete_metadata(asset.asset_path, asset.version_id, TEST_BRANCH)
                except Exception as e:  # noqa: BLE001 -- test cleanup; best-effort, doesn't affect the test's own assertions, logged
                    logger.debug(f"Teardown delete_metadata failed for {asset.asset_path}: {e}")


@pytest.mark.asyncio
async def test_update_status_active_trusts_orchestrator_against_real_module25(real_clients):
    """check_indexed()'s downgrade-on-not-indexed behavior is currently
    disabled (raced against Module 25's async PA-8 consumer -- see
    update_status()'s docstring in client.py), so an orchestrator's "active"
    claim is trusted even when this asset genuinely has nothing indexed yet."""
    db, obj, sync, _arcadedb = real_clients
    mgr = AssetManager(object_store=obj, search_sync=sync, db=db)

    run_id = uuid.uuid4().hex[:8]
    content = f"never actually indexed {run_id}".encode()
    asset = await mgr.upload_files(TEST_USER, TEST_BRANCH, (content, f"unindexed_{run_id}.txt"))
    try:
        # No /personal/index/chunk call here -- deliberately un-indexed.
        written = await mgr.update_status(TEST_USER, TEST_BRANCH, asset.asset_path, asset.version_id, "active")
        assert written == "active"
        refreshed = await db.get_asset_by_path_and_version(asset.asset_path, asset.version_id, TEST_BRANCH)
        assert refreshed.status == "active"
    finally:
        try:
            await sync.delete_metadata(asset.asset_path, asset.version_id, TEST_BRANCH)
        except Exception as e:  # noqa: BLE001 -- test cleanup; best-effort, doesn't affect the test's own assertions, logged
            logger.debug(f"Teardown delete_metadata failed for {asset.asset_path}: {e}")
