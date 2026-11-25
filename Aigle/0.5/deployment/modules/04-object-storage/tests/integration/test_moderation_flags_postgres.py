"""CM-2: commit_history.moderation_flags on a real Postgres -- the column added to
an already-deployed table at start-up, Database.update_moderation(), reading it
back, and save_metadata() keeping it when the row is saved again.

Run from the 04-object-storage/ directory (see conftest.py for which Postgres):
    python -m pytest tests/integration/test_moderation_flags_postgres.py -v
"""
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from asset_management.database import Database
from asset_management.models import AssetMetadata

TEST_BRANCH = "test_moderation_flags_branch"
FLAGS = {"verdict": "unsafe", "model": "shieldgemma-2", "policies": {"sexual": 0.97},
         "latency_ms": 812.4, "checked_at": "2026-11-03T04:05:06.123456+00:00"}


@pytest_asyncio.fixture
async def db():
    database = Database()
    await database.init_db()
    yield database
    async with database.pool.acquire() as conn:
        await conn.execute("DELETE FROM commit_history WHERE branch = $1", TEST_BRANCH)
    await database.close()


def _metadata(version_id="v1", **overrides):
    fields = {"asset_path": "image/jpg/harbour", "version_id": version_id, "primary_filename": "harbour.jpg",
              "associated_filenames": [], "upload_date": datetime.now(timezone.utc), "user": "tester",
              "branch": TEST_BRANCH, "status": "uploaded", "checksum": "abc"}
    return AssetMetadata(**{**fields, **overrides})


@pytest.mark.asyncio
async def test_init_db_adds_the_column_to_an_existing_table(db):
    async with db.pool.acquire() as conn:
        await conn.execute("ALTER TABLE commit_history DROP COLUMN IF EXISTS moderation_flags")
    await db.close()

    await db.init_db()

    async with db.pool.acquire() as conn:
        data_type = await conn.fetchval(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'commit_history' AND column_name = 'moderation_flags'")
    assert data_type == "jsonb"


@pytest.mark.asyncio
async def test_update_moderation_round_trips(db):
    await db.save_metadata(_metadata())

    assert await db.update_moderation("image/jpg/harbour", "v1", TEST_BRANCH, FLAGS) is True

    stored = await db.get_asset_by_path_and_version("image/jpg/harbour", "v1", TEST_BRANCH)
    assert stored.moderation_flags == FLAGS
    async with db.pool.acquire() as conn:
        verdict = await conn.fetchval(
            "SELECT moderation_flags->>'verdict' FROM commit_history WHERE branch = $1", TEST_BRANCH)
    assert verdict == "unsafe"       # a real JSONB value, queryable, not a quoted string


@pytest.mark.asyncio
async def test_update_moderation_of_an_unknown_asset_is_false(db):
    assert await db.update_moderation("image/jpg/nope", "v1", TEST_BRANCH, FLAGS) is False


@pytest.mark.asyncio
async def test_update_moderation_touches_only_its_own_branch(db):
    await db.save_metadata(_metadata())

    assert await db.update_moderation("image/jpg/harbour", "v1", "some_other_branch", FLAGS) is False
    stored = await db.get_asset_by_path_and_version("image/jpg/harbour", "v1", TEST_BRANCH)
    assert stored.moderation_flags is None


@pytest.mark.asyncio
async def test_saving_the_row_again_keeps_the_verdict(db):
    await db.save_metadata(_metadata())
    await db.update_moderation("image/jpg/harbour", "v1", TEST_BRANCH, FLAGS)

    await db.save_metadata(_metadata(status="active"))          # no verdict of its own

    stored = await db.get_asset_by_path_and_version("image/jpg/harbour", "v1", TEST_BRANCH)
    assert (stored.status, stored.moderation_flags) == ("active", FLAGS)


@pytest.mark.asyncio
async def test_a_clone_is_saved_with_its_verdict(db):
    await db.save_metadata(_metadata(version_id="v-new", cloned_from="v-old", moderation_flags=FLAGS))

    stored = await db.get_asset_by_path_and_version("image/jpg/harbour", "v-new", TEST_BRANCH)
    assert stored.moderation_flags == FLAGS
