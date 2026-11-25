"""
V05-10 / VIE01-364: Module 33's CF-4 audit events, and the daily purge of
every audit_events table by each row's own retention_days.

Mocked -- no live Postgres/Keycloak/stores. Run from the
33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_audit_events.py -v
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import asyncpg
import pytest
from app.core.config import settings
from app.models.job import ErasureJob, JobStatus, StoreStatus, initial_store_statuses
from app.services import retention_engine as retention_mod
from app.services.audit_events import erasure_event, privacy_event
from app.services.erasure_orchestrator import ErasureOrchestrator
from app.services.retention_engine import RetentionEngine, _purge_audit_events_on
from raptor_audit import AuditWriteError

USER = "3f1c2a9e-0000-4000-8000-00000000a11c"
ADMIN = "9d0e7b1f-0000-4000-8000-0000000ad417"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# ── The event itself ─────────────────────────────────────────────────────────

def test_erasure_event_holds_only_hashes_of_the_subject_and_requester():
    event = erasure_event(
        job_id="job-1", user_id=USER, requested_by=ADMIN, job_type="erasure", outcome="erase",
        store_statuses={"module_04": StoreStatus.DELETED, "module_34": StoreStatus.SKIPPED},
    )
    dumped = event.model_dump_json()
    assert USER not in dumped and ADMIN not in dumped
    assert event.actor_id_hash == _sha(USER)
    assert event.metadata["requested_by_hash"] == _sha(ADMIN)
    assert event.metadata["requested_by_subject"] is False
    assert event.metadata["store_statuses"] == {"module_04": "deleted", "module_34": "skipped"}
    assert (event.source_module, event.event_type, event.outcome, event.resource_id) == (33, "gdpr.erasure", "erase", "job-1")
    assert event.content_hash is None
    assert event.retention_days == settings.audit_retention_days == 1095


def test_self_requested_event_has_no_requester_hash():
    event = erasure_event(job_id="j", user_id=USER, requested_by=USER, job_type="consent_withdrawal",
                          outcome="error", store_statuses={"module_26": StoreStatus.FAILED}, purpose="memory")
    assert "requested_by_hash" not in event.metadata
    assert event.metadata["requested_by_subject"] is True
    assert event.metadata["purpose"] == "memory"


PRIVACY_TYPES = ("gdpr.export", "gdpr.access", "gdpr.rectification", "gdpr.restriction",
                 "gdpr.objection", "gdpr.consent", "gdpr.legal_hold")


@pytest.mark.parametrize("event_type", PRIVACY_TYPES)
def test_privacy_event_types_are_accepted_by_raptor_audit(event_type):
    event = privacy_event(event_type, user_id=USER, action="x", ok=True, requested_by=ADMIN, resource_id="r-1")
    dumped = event.model_dump_json()
    assert USER not in dumped and ADMIN not in dumped
    assert (event.source_module, event.event_type, event.resource_id) == (33, event_type, "r-1")
    assert event.actor_id_hash == _sha(USER) and event.metadata["requested_by_hash"] == _sha(ADMIN)
    assert event.content_hash is None and event.retention_days == settings.audit_retention_days


def test_privacy_event_outcome_is_allow_or_error_and_action_says_what_was_done():
    ok = privacy_event("gdpr.consent", user_id=USER, action="withdraw", ok=True, requested_by=USER,
                       metadata={"purpose": "memory"})
    failed = privacy_event("gdpr.consent", user_id=USER, action="withdraw", ok=False, requested_by=USER)
    assert (ok.outcome, failed.outcome) == ("allow", "error")
    assert ok.metadata == {"action": "withdraw", "purpose": "memory", "requested_by_subject": True}


def test_privacy_event_by_the_system_names_no_requester():
    event = privacy_event("gdpr.legal_hold", user_id=USER, action="release", ok=True, requested_by=None)
    assert event.metadata == {"action": "release", "requested_by_system": True}


def test_raptor_audit_still_rejects_unknown_event_types():
    from pydantic import ValidationError
    from raptor_audit import build_event
    with pytest.raises(ValidationError):
        build_event(source_module=33, source_service="s", event_type="gdpr.unknown", outcome="allow")


def test_audit_retention_days_is_configurable(monkeypatch):
    monkeypatch.setenv("GD_AUDIT_RETENTION_DAYS", "30")
    from app.core.config import Settings
    assert Settings().audit_retention_days == 30
    monkeypatch.setenv("GD_AUDIT_RETENTION_DAYS", "0")
    with pytest.raises(ValueError):
        Settings()


# ── Erasure orchestrator ─────────────────────────────────────────────────────

class _FakeRedis:
    async def sadd(self, *args):
        pass

    async def expire(self, *args):
        pass


class _FakeAudit:
    def __init__(self, fail=False):
        self.fail = fail
        self.events: list = []

    async def emit(self, event, *, fail_closed=False):
        if self.fail and fail_closed:
            raise AuditWriteError("sink down")
        self.events.append((event, fail_closed))


@pytest.fixture
def make_orchestrator(monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "")
    monkeypatch.setattr(settings, "module_35_url", "")
    monkeypatch.setattr(settings, "erasure_retry_delay_seconds", 0)
    monkeypatch.setattr(settings, "erasure_max_attempts", 1)

    def _make(job: ErasureJob, *, audit_fails=False, failing_store=None):
        orch = ErasureOrchestrator(pool=None, redis=_FakeRedis(), kafka_producer=None)
        orch.audit = _FakeAudit(fail=audit_fails)
        calls: dict = {"updates": [], "dpo": None}

        async def get_job(job_id):
            return job

        async def update_job(job_id, **fields):
            calls["updates"].append(fields)
            if "store_statuses" in fields:
                job.store_statuses = fields["store_statuses"]

        async def noop(*args, **kwargs):
            pass

        async def fail(*args, **kwargs):
            raise RuntimeError("store down")

        async def notify_dpo(user_id, job_id, failed):
            calls["dpo"] = failed

        async def start(job_id, user_id):  # no legal hold: the run starts
            await update_job(job_id, status=JobStatus.IN_PROGRESS)
            return True

        monkeypatch.setattr(orch, "get_job", get_job)
        monkeypatch.setattr(orch, "_update_job", update_job)
        monkeypatch.setattr(orch, "_start_unless_held", start)
        for name in ("_disable_keycloak_account", "_hard_delete_keycloak_account", "_publish_tombstone"):
            monkeypatch.setattr(orch, name, noop)
        monkeypatch.setattr(orch, "_notify_dpo", notify_dpo)
        for key in ("04", "25", "26", "03", "23", "02"):
            monkeypatch.setattr(orch, f"_delete_module_{key}", fail if key == failing_store else noop)
        return orch, calls

    return _make


def _job(statuses=None) -> ErasureJob:
    return ErasureJob(job_id="job-1", user_id=USER, requested_at=datetime.now(timezone.utc),
                      requested_by=ADMIN, status=JobStatus.PENDING,
                      store_statuses=statuses or initial_store_statuses())


def _final_status(calls):
    return [u["status"] for u in calls["updates"] if "status" in u][-1]


@pytest.mark.asyncio
async def test_completed_erasure_writes_a_fail_closed_erase_event(make_orchestrator):
    orch, calls = make_orchestrator(_job())
    await orch.run("job-1")
    assert _final_status(calls) == JobStatus.COMPLETED
    [(event, fail_closed)] = orch.audit.events
    assert fail_closed is True
    assert event.outcome == "erase" and event.resource_id == "job-1"
    assert event.metadata["job_type"] == "erasure"


@pytest.mark.asyncio
async def test_unwritten_audit_event_keeps_the_job_from_completing(make_orchestrator):
    """CF-4: erasure confirmation is fail-closed -- no audit event, no COMPLETED."""
    orch, calls = make_orchestrator(_job(), audit_fails=True)
    await orch.run("job-1")
    assert _final_status(calls) == JobStatus.PARTIAL_FAILURE
    assert all(u.get("status") != JobStatus.COMPLETED for u in calls["updates"])
    assert "audit event" in calls["updates"][-1]["notes"]
    assert calls["dpo"] == ["audit_event"]


@pytest.mark.asyncio
async def test_retry_writes_the_missing_audit_event_and_completes(make_orchestrator):
    statuses = {k: StoreStatus.DELETED for k in initial_store_statuses()}
    job = _job(statuses)
    job.status = JobStatus.PARTIAL_FAILURE
    orch, calls = make_orchestrator(job)
    await orch.retry("job-1")
    assert _final_status(calls) == JobStatus.COMPLETED
    assert [e.outcome for e, _ in orch.audit.events] == ["erase"]


@pytest.mark.asyncio
async def test_partial_failure_writes_a_fail_open_error_event(make_orchestrator):
    orch, calls = make_orchestrator(_job(), failing_store="04")
    await orch.run("job-1")
    assert _final_status(calls) == JobStatus.PARTIAL_FAILURE
    [(event, fail_closed)] = orch.audit.events
    assert fail_closed is False
    assert event.outcome == "error"
    assert event.metadata["store_statuses"]["module_04"] == "failed"
    assert calls["dpo"] == ["module_04"]


# ── Daily audit_events purge ─────────────────────────────────────────────────

class _Conn:
    def __init__(self, table_exists=True, deleted=5, fail=False):
        self.table_exists, self.deleted, self.fail = table_exists, deleted, fail
        self.executed: list = []

    async def fetchval(self, sql, *args):
        self.executed.append(sql)
        return self.table_exists

    async def execute(self, sql, *args):
        self.executed.append(sql)
        if sql.startswith("DELETE") and self.fail:
            raise asyncpg.PostgresError("boom")
        return f"DELETE {self.deleted}" if sql.startswith("DELETE") else "INSERT 0 1"

    async def close(self):
        pass


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        conn = self.conn

        class _CM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False
        return _CM()


@pytest.mark.asyncio
async def test_purge_uses_each_rows_own_retention_days():
    conn = _Conn(deleted=3)
    assert await _purge_audit_events_on(conn) == 3
    delete = next(s for s in conn.executed if s.startswith("DELETE"))
    assert delete == "DELETE FROM audit_events WHERE timestamp < NOW() - make_interval(days => retention_days)"


@pytest.mark.asyncio
async def test_purge_skips_a_database_without_the_table():
    conn = _Conn(table_exists=False)
    assert await _purge_audit_events_on(conn) == 0
    assert not any(s.startswith("DELETE") for s in conn.executed)


@pytest.mark.asyncio
async def test_purge_covers_both_databases_and_logs_each(monkeypatch):
    own = _Conn(deleted=4)
    guardrails = _Conn(deleted=9)
    connected: list = []

    async def connect(**kwargs):
        connected.append(kwargs["database"])
        return guardrails

    monkeypatch.setattr(retention_mod.asyncpg, "connect", connect)
    engine = RetentionEngine(pool=_Pool(own))
    results = await engine.purge_audit_events()
    assert connected == ["guardrails"]
    assert {k: v["rows_deleted"] for k, v in results.items()} == {"audit_events": 4, "audit_events:guardrails": 9}
    logged = [s for s in own.executed if s.startswith("INSERT INTO gdpr_retention_log")]
    assert len(logged) == 2


@pytest.mark.asyncio
async def test_purge_failure_is_logged_as_minus_one_and_the_run_continues(monkeypatch):
    own = _Conn(fail=True)

    async def connect(**kwargs):
        return _Conn(deleted=2)

    monkeypatch.setattr(retention_mod.asyncpg, "connect", connect)
    results = await RetentionEngine(pool=_Pool(own)).purge_audit_events()
    assert results["audit_events"]["rows_deleted"] == -1
    assert results["audit_events:guardrails"]["rows_deleted"] == 2
