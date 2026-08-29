"""Unit tests for the guardrail_violations audit log (app/services/audit_log.py)."""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.services import audit_log

pytestmark = pytest.mark.asyncio


class FakeConn:
    def __init__(self, fetchval_return=0, fetch_return=None, raise_on_execute=False):
        self.executed: list[tuple] = []
        self.fetch_calls: list[tuple] = []
        self.fetchval_calls: list[tuple] = []
        self.fetchval_return = fetchval_return
        self.fetch_return = fetch_return or []
        self.raise_on_execute = raise_on_execute

    async def execute(self, query, *params):
        if self.raise_on_execute:
            raise RuntimeError("boom")
        self.executed.append((query, params))

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

    monkeypatch.setattr(audit_log, "get_pool", _get_pool)
    return fake_conn


async def test_write_inserts_expected_params(fake_conn):
    policy_id = uuid4()
    await audit_log._write(policy_id, "07", "input", "S1", "block", "req-1", None)
    assert len(fake_conn.executed) == 1
    query, params = fake_conn.executed[0]
    assert "INSERT INTO guardrail_violations" in query
    assert params == (policy_id, "07", "input", "S1", "block", "req-1", None)


async def test_write_includes_content_when_given(fake_conn):
    policy_id = uuid4()
    await audit_log._write(policy_id, "07", "output", "pii", "redact", None, "raw content")
    _, params = fake_conn.executed[0]
    assert params[-1] == "raw content"


async def test_write_swallows_exceptions(monkeypatch):
    async def _get_pool():
        return FakePool(FakeConn(raise_on_execute=True))

    monkeypatch.setattr(audit_log, "get_pool", _get_pool)
    await audit_log._write(uuid4(), "07", "input", "S1", "block", "req-1", None)  # must not raise


async def test_record_violation_is_non_blocking_and_eventually_writes(fake_conn):
    policy_id = uuid4()
    audit_log.record_violation(policy_id, "07", "input", "S1", "block", "req-1")
    assert fake_conn.executed == []                    # nothing written synchronously
    assert len(audit_log._background_tasks) == 1

    pending = next(iter(audit_log._background_tasks))
    await pending

    assert len(fake_conn.executed) == 1
    assert audit_log._background_tasks == set()         # done_callback cleaned it up


async def test_list_violations_no_filters(fake_conn):
    fake_conn.fetchval_return = 3
    rows, total = await audit_log.list_violations()
    assert total == 3
    assert rows == []
    count_query, count_params = fake_conn.fetchval_calls[0]
    assert "WHERE" not in count_query
    fetch_query, fetch_params = fake_conn.fetch_calls[0]
    assert fetch_params == (50, 0)                       # page_size, offset


async def test_list_violations_applies_filters_and_pagination(fake_conn):
    from_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    to_ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await audit_log.list_violations(
        page=2, page_size=10, module="07", direction="input", category="S1",
        from_ts=from_ts, to_ts=to_ts,
    )
    fetch_query, fetch_params = fake_conn.fetch_calls[0]
    assert "module = $1" in fetch_query
    assert "direction = $2" in fetch_query
    assert "category = $3" in fetch_query
    assert "created_at >= $4" in fetch_query
    assert "created_at <= $5" in fetch_query
    assert fetch_params == ("07", "input", "S1", from_ts, to_ts, 10, 10)  # limit=10, offset=(2-1)*10


async def test_summary_last_24h_window_and_shape(fake_conn):
    fake_conn.fetch_return = [{"category": "S1", "action_taken": "block", "count": 5}]
    since, until, rows = await audit_log.summary_last_24h()
    assert (until - since).total_seconds() == pytest.approx(24 * 3600, abs=1)
    assert rows == fake_conn.fetch_return
    query, params = fake_conn.fetch_calls[0]
    assert "GROUP BY category, action_taken" in query
    assert params == (since, until)
