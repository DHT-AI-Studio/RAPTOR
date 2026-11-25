"""VIE01-214: register_asset()'s cloned_from fast path, against a real
Postgres -- the cloned_from column round-trip and register_asset()'s
status branching both come from real asyncpg behaviour (column exists,
ON CONFLICT upsert, read-back), not something a fake DB could prove.

The Module 25 (ArcadeDB) side of the fast path -- clone_version() writing
status="cloned", _personal_db_set_status() flipping it to "active" -- is
stubbed here, not exercised for real: this host doesn't run Module 25.
See the 163-only follow-up for the full real-ArcadeDB path once Module 25
is redeployed there.

Run from the 04-object-storage/ directory:
    python -m pytest tests/integration/test_register_clone_from.py -v
"""
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from asset_management.client import AssetManager
from asset_management.database import Database

TEST_BRANCH = "test_register_clone_from_branch"


class _FakeSearchSyncForClone:
    """Stubs the Module 25-touching call -- this host doesn't run Module 25,
    so this isolates register_asset()'s own Postgres-side branching logic."""

    def __init__(self):
        self.personal_db_calls = []

    async def _personal_db_set_status(self, asset_path, version_id, status, branch_id=None):
        self.personal_db_calls.append((asset_path, version_id, status, branch_id))


@pytest_asyncio.fixture
async def db():
    database = Database()
    await database.init_db()
    yield database
    async with database.pool.acquire() as conn:
        await conn.execute("DELETE FROM commit_history WHERE branch = $1", TEST_BRANCH)
    await database.close()


async def _insert_row(db: Database, asset_path: str, version_id: str, status: str, cloned_from: str | None = None):
    async with db.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO commit_history (
                asset_path, version_id, branch, updated_by, primary_filename,
                asset_key, associated_filenames, upload_date, status, checksum, cloned_from
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            asset_path, version_id, TEST_BRANCH, "tester", "f.txt",
            f"{asset_path}/f.txt", "[]", datetime.now(timezone.utc), status,
            "fakechecksum", cloned_from,
        )


@pytest.mark.asyncio
async def test_cloned_from_column_round_trips(db: Database):
    """Schema check: cloned_from survives a real write+read against Postgres."""
    await _insert_row(db, "doc/cloned-schema-test", "v1", "uploaded", cloned_from="source-v-abc")
    metadata = await db.get_asset_by_path_and_version("doc/cloned-schema-test", "v1", TEST_BRANCH)
    assert metadata.cloned_from == "source-v-abc"


@pytest.mark.asyncio
async def test_cloned_from_defaults_to_none(db: Database):
    """A normal upload (no clone_point() involved) reads back NULL, not an
    empty string or some other falsy-but-not-None sentinel -- register_asset()'s
    `if metadata.cloned_from:` check relies on this being a clean None."""
    await _insert_row(db, "doc/cloned-null-test", "v1", "uploaded")
    metadata = await db.get_asset_by_path_and_version("doc/cloned-null-test", "v1", TEST_BRANCH)
    assert metadata.cloned_from is None


@pytest.mark.asyncio
async def test_register_uploaded_with_cloned_from_flips_to_active_no_kafka(db: Database):
    """The new fast path: uploaded + cloned_from set -> registered, active,
    no Kafka -- real Postgres status flip, Module 25 call stubbed."""
    await _insert_row(db, "doc/cloned-register-test", "v1", "uploaded", cloned_from="source-v-xyz")
    search_sync = _FakeSearchSyncForClone()
    mgr = AssetManager(object_store=None, search_sync=search_sync, db=db)

    result = await mgr.register_asset(TEST_BRANCH, "doc/cloned-register-test", "v1")

    assert result["outcome"] == "registered"
    assert result["metadata"].status == "active"
    assert search_sync.personal_db_calls == [
        ("doc/cloned-register-test", "v1", "active", TEST_BRANCH)
    ]

    # Really landed in Postgres, not just the in-memory object this call
    # happened to mutate and return.
    refreshed = await db.get_asset_by_path_and_version("doc/cloned-register-test", "v1", TEST_BRANCH)
    assert refreshed.status == "active"


@pytest.mark.asyncio
async def test_register_uploaded_without_cloned_from_needs_processing(db: Database):
    """Regression guard: a normal (non-cloned) uploaded asset still falls
    through to needs_processing untouched, unaffected by the new branch."""
    await _insert_row(db, "doc/normal-upload-test", "v1", "uploaded")
    search_sync = _FakeSearchSyncForClone()
    mgr = AssetManager(object_store=None, search_sync=search_sync, db=db)

    result = await mgr.register_asset(TEST_BRANCH, "doc/normal-upload-test", "v1")

    assert result["outcome"] == "needs_processing"
    assert not search_sync.personal_db_calls

    refreshed = await db.get_asset_by_path_and_version("doc/normal-upload-test", "v1", TEST_BRANCH)
    assert refreshed.status == "uploaded"


@pytest.mark.asyncio
async def test_register_failed_with_cloned_from_flips_to_active(db: Database):
    """cloned_from's fast path doesn't care whether status was "uploaded"
    or "failed" -- a failed asset that happens to have cloned_from set
    still gets flipped straight to active instead of re-queued."""
    await _insert_row(db, "doc/failed-cloned-test", "v1", "failed", cloned_from="source-v-999")
    search_sync = _FakeSearchSyncForClone()
    mgr = AssetManager(object_store=None, search_sync=search_sync, db=db)

    result = await mgr.register_asset(TEST_BRANCH, "doc/failed-cloned-test", "v1")

    assert result["outcome"] == "registered"
    assert result["metadata"].status == "active"


@pytest.mark.asyncio
async def test_register_active_is_conflict_even_with_cloned_from(db: Database):
    """The conflict branch (processing/active/archived/destroyed) is
    checked before the cloned_from branch -- an already-active asset is
    rejected regardless of whatever cloned_from happens to still hold."""
    await _insert_row(db, "doc/already-active-test", "v1", "active", cloned_from="source-v-111")
    search_sync = _FakeSearchSyncForClone()
    mgr = AssetManager(object_store=None, search_sync=search_sync, db=db)

    result = await mgr.register_asset(TEST_BRANCH, "doc/already-active-test", "v1")

    assert result["outcome"] == "conflict"
    assert result["status"] == "active"
    assert not search_sync.personal_db_calls
