"""Unit tests for the CM-5 (VIE01-408) flag store (app/services/moderation_flags.py)."""
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from app.models.moderation_flag import ModerationFlagCreate
from app.services import moderation_flags

pytestmark = pytest.mark.asyncio

_ROW = {
    "id": uuid4(), "branch_id": "branch-1", "version_id": "v1", "asset_path": "asset-1", "moment_id": None, "chunk_id": None,
    "category": "nsfw", "score": 0.91, "model": "shieldgemma-2", "policy_version": "1.0",
    "status": "flagged", "event_id": uuid4(),
    "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
}


class FakeConn:
    def __init__(self, fetchrow_return=None, fetchval_return=0, fetch_return=None):
        self.executed: list[tuple] = []
        self.fetchrow_calls: list[tuple] = []
        self.fetch_calls: list[tuple] = []
        self.fetchval_calls: list[tuple] = []
        self.fetchrow_return = fetchrow_return if fetchrow_return is not None else dict(_ROW)
        self.fetchval_return = fetchval_return
        self.fetch_return = fetch_return or []
        self.transactions = 0

    def transaction(self):
        self.transactions += 1
        return FakeAcquire(self)

    async def execute(self, query, *params):
        self.executed.append((query, params))

    async def fetchrow(self, query, *params):
        self.fetchrow_calls.append((query, params))
        return self.fetchrow_return

    async def fetchval(self, query, *params):
        self.fetchval_calls.append((query, params))
        return self.fetchval_return

    async def fetch(self, query, *params):
        self.fetch_calls.append((query, params))
        return self.fetch_return


class FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return FakeAcquire(self._conn)


@pytest.fixture()
def fake_conn():
    return FakeConn()


@pytest.fixture(autouse=True)
def patch_pool(monkeypatch, fake_conn):
    async def _get_pool():
        return FakePool(fake_conn)

    monkeypatch.setattr(moderation_flags, "get_pool", _get_pool)
    return fake_conn


@pytest.fixture(autouse=True)
def patch_kafka(monkeypatch):
    published: list[dict] = []

    async def _publish(flag):
        published.append(flag)

    monkeypatch.setattr(moderation_flags, "publish_flagged", _publish)
    return published


def _flag(**overrides) -> ModerationFlagCreate:
    return ModerationFlagCreate(**{
        "branch_id": "branch-1", "version_id": "v1", "asset_path": "asset-1", "category": "nsfw", "score": 0.91,
        "model": "shieldgemma-2", "policy_version": "1.0", **overrides,
    })


# ---------------------------------------------------------------- create_flag


async def test_create_flag_inserts_event_and_flag_row_in_one_transaction(fake_conn):
    await moderation_flags.create_flag(_flag())
    assert fake_conn.transactions == 1
    assert len(fake_conn.executed) == 1 and len(fake_conn.fetchrow_calls) == 1
    event_query, event_params = fake_conn.executed[0]
    assert "INSERT INTO audit_events" in event_query
    flag_query, flag_params = fake_conn.fetchrow_calls[0]
    assert "INSERT INTO moderation_flags" in flag_query
    assert flag_params[:9] == ("branch-1", "v1", "asset-1", None, None, "nsfw", 0.91, "shieldgemma-2", "1.0")
    assert flag_params[9] == event_params[0]        # moderation_flags.event_id == audit_events.event_id


async def test_create_flag_builds_the_cf4_moderation_event(fake_conn):
    await moderation_flags.create_flag(_flag(chunk_id="chunk-9"))
    _event_query, event_params = fake_conn.executed[0]
    (_event_id, _ts, source_module, source_service, event_type, _actor, resource_id, outcome,
     policy_version, _content_hash, metadata, _retention_days) = event_params
    assert (source_module, source_service, event_type) == (23, "guardrail-service", "moderation.flag")
    assert resource_id == "chunk-9" and outcome == "allow" and policy_version == "1.0"
    assert json.loads(metadata)["status"] == "flagged"


async def test_create_flag_returns_the_created_record(fake_conn):
    record = await moderation_flags.create_flag(_flag())
    assert record.id == _ROW["id"] and record.status == "flagged" and record.asset_path == "asset-1"


async def test_create_flag_publishes_to_kafka(fake_conn, patch_kafka):
    record = await moderation_flags.create_flag(_flag())
    assert len(patch_kafka) == 1
    assert patch_kafka[0]["id"] == str(record.id)


# ---------------------------------------------------------------- update_status


async def test_update_status_writes_its_own_audit_event(fake_conn):
    reviewed_row = dict(_ROW, status="cleared")
    fake_conn.fetchrow_return = reviewed_row
    record = await moderation_flags.update_status(_ROW["id"], "cleared", "false positive")
    assert record.status == "cleared"
    update_query, update_params = fake_conn.fetchrow_calls[0]
    assert "UPDATE moderation_flags" in update_query and update_params == (_ROW["id"], "cleared")
    event_query, event_params = fake_conn.executed[0]
    assert "INSERT INTO audit_events" in event_query
    metadata = json.loads(event_params[10])
    assert metadata == {"status": "cleared", "flag_id": str(_ROW["id"]), "note": "false positive"}
    # the transition event is NOT linked back onto moderation_flags.event_id --
    # that column stays the creation event's id (reviewed_row['event_id'] above).
    assert event_params[0] != reviewed_row["event_id"]


async def test_update_status_returns_none_for_unknown_flag(fake_conn):
    fake_conn.fetchrow_return = None
    assert await moderation_flags.update_status(uuid4(), "cleared", None) is None
    assert fake_conn.executed == []    # no audit event written for a no-op


async def test_update_status_publishes_to_kafka(fake_conn, patch_kafka):
    """CM-5: Module 25's search exclusion needs the transition too, not just the
    creation -- an admin clearing a flag must be able to reach it."""
    fake_conn.fetchrow_return = dict(_ROW, status="cleared")
    record = await moderation_flags.update_status(_ROW["id"], "cleared", "false positive")
    assert len(patch_kafka) == 1
    assert patch_kafka[0]["id"] == str(record.id) and patch_kafka[0]["status"] == "cleared"


async def test_update_status_does_not_publish_for_an_unknown_flag(fake_conn, patch_kafka):
    fake_conn.fetchrow_return = None
    await moderation_flags.update_status(uuid4(), "cleared", None)
    assert patch_kafka == []


# ---------------------------------------------------------------- list_flags


async def test_list_flags_no_filters(fake_conn):
    fake_conn.fetchval_return = 2
    rows, total = await moderation_flags.list_flags()
    assert total == 2 and rows == []
    count_query, _ = fake_conn.fetchval_calls[0]
    assert "WHERE" not in count_query
    _query, params = fake_conn.fetch_calls[0]
    assert params == (50, 0)


async def test_list_flags_applies_filters_and_pagination(fake_conn):
    await moderation_flags.list_flags(page=2, page_size=10, status="flagged", category="nsfw",
                                       version_id="v1", asset_path="a-1")
    query, params = fake_conn.fetch_calls[0]
    assert "status = $1" in query and "category = $2" in query
    assert "version_id = $3" in query and "asset_path = $4" in query
    assert params == ("flagged", "nsfw", "v1", "a-1", 10, 10)   # limit=10, offset=(2-1)*10
