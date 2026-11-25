"""Unit tests for the guardrail_violations audit log (app/services/audit_log.py)."""
import hashlib
import json
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
        self.transactions = 0

    def transaction(self):
        self.transactions += 1
        return FakeAcquire(self)

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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event_and_violation(fake_conn):
    """The two INSERTs of one _write: (audit_events params, guardrail_violations params)."""
    assert len(fake_conn.executed) == 2
    (event_query, event_params), (violation_query, violation_params) = fake_conn.executed
    assert "INSERT INTO audit_events" in event_query
    assert "INSERT INTO guardrail_violations" in violation_query
    return event_params, violation_params


async def test_write_inserts_expected_params(fake_conn):
    policy_id = uuid4()
    await audit_log._write(policy_id, "07", "input", "S1", "block", "req-1", None)
    _event, params = _event_and_violation(fake_conn)
    assert params[:6] == (policy_id, "07", "input", "S1", "block", "req-1")
    assert len(params) == 7                                      # + event_id; no content column (GRD-X4)


async def test_write_links_both_rows_in_one_transaction(fake_conn):
    await audit_log._write(uuid4(), "07", "input", "S1", "block", "req-1", None)
    event_params, violation_params = _event_and_violation(fake_conn)
    assert fake_conn.transactions == 1
    assert violation_params[6] == event_params[0]               # guardrail_violations.event_id == audit_events.event_id


async def test_write_stores_only_the_content_hash(fake_conn):
    policy_id = uuid4()
    await audit_log._write(policy_id, "07", "output", "pii", "redact", None, "raw content",
                           user_id="kc-sub-123")
    event_params, _ = _event_and_violation(fake_conn)
    assert event_params[9] == _sha256("raw content")             # audit_events.content_hash
    assert "content" not in fake_conn.executed[1][0]             # guardrail_violations stores no content (GRD-X4)
    assert event_params[5] == _sha256("kc-sub-123")              # audit_events.actor_id_hash
    stored = repr(fake_conn.executed)
    assert "raw content" not in stored and "kc-sub-123" not in stored


async def test_write_builds_the_cf4_guardrail_event(fake_conn):
    policy_id = uuid4()
    await audit_log._write(policy_id, "13", "input", "S10", "warn", "req-7", "x",
                           session_id="s-1", policy_version="1.2.0")
    event_params, _ = _event_and_violation(fake_conn)
    (_event_id, _ts, source_module, source_service, event_type, _actor, resource_id, outcome,
     policy_version, _content_hash, metadata, retention_days) = event_params
    assert (source_module, source_service, event_type) == (23, "guardrail-service", "guardrail.decision")
    assert resource_id == "req-7"
    assert outcome == "allow"                                    # warn has no CF-4 outcome of its own
    assert policy_version == "1.2.0"
    assert retention_days == 90                                  # GRD-X4 default
    assert json.loads(metadata) == {
        "decision": "warn", "direction": "input", "category": "S10", "caller_module": "13",
        "policy_id": str(policy_id), "session_id": "s-1",
    }


@pytest.mark.parametrize("action_taken, outcome", [
    ("block", "block"), ("redact", "redact"), ("warn", "allow"), ("flag", "allow"),
])
async def test_action_taken_maps_to_a_cf4_outcome(action_taken, outcome):
    event = audit_log.build_guardrail_event(None, "07", "input", None, action_taken, "r", None)
    assert event.outcome == outcome
    assert event.metadata["decision"] == action_taken


async def test_session_id_is_the_resource_id_without_a_request_id():
    event = audit_log.build_guardrail_event(None, "07", "input", None, "block", None, None, session_id="s-1")
    assert event.resource_id == "s-1"


async def test_write_swallows_exceptions(monkeypatch):
    async def _get_pool():
        return FakePool(FakeConn(raise_on_execute=True))

    monkeypatch.setattr(audit_log, "get_pool", _get_pool)
    await audit_log._write(uuid4(), "07", "input", "S1", "block", "req-1", None)  # must not raise


async def test_record_violation_passes_hash_only_fields_through(fake_conn):
    audit_log.record_violation(None, "07", "input", "S1", "block", "req-1", "secret",
                               user_id="u-1", policy_version="2.0")
    await next(iter(audit_log._background_tasks))
    event_params, violation_params = _event_and_violation(fake_conn)
    assert event_params[5] == _sha256("u-1")
    assert event_params[8] == "2.0"
    assert event_params[9] == _sha256("secret")
    assert "secret" not in repr(violation_params)


async def test_record_violation_is_non_blocking_and_eventually_writes(fake_conn):
    policy_id = uuid4()
    audit_log.record_violation(policy_id, "07", "input", "S1", "block", "req-1")
    assert fake_conn.executed == []                    # nothing written synchronously
    assert len(audit_log._background_tasks) == 1

    pending = next(iter(audit_log._background_tasks))
    await pending

    assert len(fake_conn.executed) == 2                 # audit_events + guardrail_violations
    assert audit_log._background_tasks == set()         # done_callback cleaned it up


async def test_list_violations_no_filters(fake_conn):
    fake_conn.fetchval_return = 3
    rows, total = await audit_log.list_violations()
    assert total == 3
    assert rows == []
    count_query, _count_params = fake_conn.fetchval_calls[0]
    assert "WHERE" not in count_query
    _fetch_query, fetch_params = fake_conn.fetch_calls[0]
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


async def test_retention_days_follows_guardrail_audit_retention_days(monkeypatch):
    """GRD-X4: retention defaults to 90 days and GUARDRAIL_AUDIT_RETENTION_DAYS overrides it."""
    from app.core import config

    assert config.Settings().guardrail_audit_retention_days == 90
    monkeypatch.setenv("GUARDRAIL_AUDIT_RETENTION_DAYS", "30")
    config.get_settings.cache_clear()
    try:
        event = audit_log.build_guardrail_event(None, "07", "input", "S1", "block", "r1", "x")
        assert event.retention_days == 30
    finally:
        config.get_settings.cache_clear()
    monkeypatch.setenv("GUARDRAIL_AUDIT_RETENTION_DAYS", "0")
    with pytest.raises(ValueError):
        config.Settings()
