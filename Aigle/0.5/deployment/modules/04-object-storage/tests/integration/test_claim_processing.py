"""
VIE01-215: Register API (Module 04 + 13) -- POST /claim-processing atomicity.

Database.claim_processing()'s guarantee (exactly one winner under
concurrent claims, idempotent re-claim by the same correlation_id) comes
from Postgres's own row-level locking on the conditional UPDATE, not from
application logic -- so this has to run against a real Postgres, the same
one test_asset_lifecycle.py uses (see conftest.py for how it points at the local
docker-compose stack).

Run from the 04-object-storage/ directory:
    python -m pytest tests/integration/test_claim_processing.py -v
"""
import asyncio
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from asset_management.database import Database

TEST_BRANCH = "test_claim_processing_branch"


@pytest_asyncio.fixture
async def db():
    database = Database()
    await database.init_db()
    yield database
    async with database.pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM commit_history WHERE branch = $1", TEST_BRANCH
        )
    await database.close()


async def _insert_row(db: Database, asset_path: str, version_id: str, status: str):
    async with db.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO commit_history (
                asset_path, version_id, branch, updated_by, primary_filename,
                asset_key, associated_filenames, upload_date, status, checksum
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            asset_path, version_id, TEST_BRANCH, "tester", "f.txt",
            f"{asset_path}/f.txt", "[]", datetime.now(timezone.utc), status,
            "fakechecksum",
        )


@pytest.mark.asyncio
async def test_two_concurrent_claims_exactly_one_wins(db: Database):
    """Acceptance criterion: two concurrent /claim-processing calls on the
    same asset -> exactly one succeeds (200-equivalent), the other is
    rejected (409-equivalent) -- proven here at the DB layer, which is
    where the atomicity guarantee actually lives."""
    asset_path, version_id = "doc/concurrency-test", "v1"
    await _insert_row(db, asset_path, version_id, "uploaded")

    cid_a, cid_b = str(uuid.uuid4()), str(uuid.uuid4())
    results = await asyncio.gather(
        db.claim_processing(asset_path, version_id, TEST_BRANCH, cid_a),
        db.claim_processing(asset_path, version_id, TEST_BRANCH, cid_b),
    )

    winners = [r for r in results if r is not None]
    losers = [r for r in results if r is None]
    assert len(winners) == 1, f"expected exactly one winner, got {results}"
    assert len(losers) == 1
    assert winners[0]["status"] == "processing"
    assert winners[0]["correlation_id"] in (cid_a, cid_b)


@pytest.mark.asyncio
async def test_same_correlation_id_reclaims_idempotently(db: Database):
    """Acceptance criterion: a retry with the *same* correlation_id (e.g.
    the orchestrator redelivering after a network blip) re-claims
    successfully instead of being rejected as a conflict."""
    asset_path, version_id = "doc/idempotency-test", "v1"
    await _insert_row(db, asset_path, version_id, "uploaded")
    cid = str(uuid.uuid4())

    first = await db.claim_processing(asset_path, version_id, TEST_BRANCH, cid)
    assert first is not None
    assert first["status"] == "processing"

    second = await db.claim_processing(asset_path, version_id, TEST_BRANCH, cid)
    assert second is not None, "same correlation_id must re-claim, not conflict"
    assert second["correlation_id"] == cid


@pytest.mark.asyncio
async def test_different_correlation_id_after_claim_is_rejected(db: Database):
    """A *different* correlation_id trying to claim an asset already
    claimed by someone else must be rejected (maps to 409 in the router)."""
    asset_path, version_id = "doc/conflict-test", "v1"
    await _insert_row(db, asset_path, version_id, "uploaded")
    cid_first, cid_second = str(uuid.uuid4()), str(uuid.uuid4())

    first = await db.claim_processing(asset_path, version_id, TEST_BRANCH, cid_first)
    assert first is not None

    second = await db.claim_processing(asset_path, version_id, TEST_BRANCH, cid_second)
    assert second is None


@pytest.mark.asyncio
async def test_reregistering_failed_asset_is_still_claimable(db: Database):
    """Acceptance criterion: re-registering a failed asset re-publishes
    Kafka (module 13's concern) and the asset stays "failed" until claimed
    -- here, proving the DB layer still allows a fresh claim on "failed"."""
    asset_path, version_id = "doc/failed-retry-test", "v1"
    await _insert_row(db, asset_path, version_id, "failed")

    claimed = await db.claim_processing(asset_path, version_id, TEST_BRANCH, str(uuid.uuid4()))
    assert claimed is not None
    assert claimed["status"] == "processing"


@pytest.mark.asyncio
async def test_active_asset_cannot_be_claimed(db: Database):
    """Status disallows claiming (e.g. already "active") -> rejected,
    regardless of correlation_id."""
    asset_path, version_id = "doc/active-test", "v1"
    await _insert_row(db, asset_path, version_id, "active")

    claimed = await db.claim_processing(asset_path, version_id, TEST_BRANCH, str(uuid.uuid4()))
    assert claimed is None
