"""
V05-10 / VIE01-364: operator job view, CF-4 audit events and the
audit_events purge against a real PostgreSQL.

Checks what operators see for failure, partial completion, retry and
escalation: a real erasure job whose store fails, or whose audit event can't
be written, is listed as partial_failure, the DPO is emailed once, and a
retry completes it; the job list / summary filter real rows -- never
returning an export's download URLs or a DSAR's stored report. Keycloak,
Kafka, the data stores and the SMTP server are stubbed; the job table,
audit_events, every query and the DPO email itself are real. (Legal hold
has no job state yet -- PRIV-X4, VIE01-358.)

Needs an empty, throwaway database (tables are created and dropped). Skips
unless TEST_PG_DSN is set. Run from the 33-privacy-compliance-service/
directory, e.g.:
    docker run -d --rm --name pg-test -e POSTGRES_PASSWORD=pw -p 127.0.0.1:55432:5432 postgres:15
    TEST_PG_DSN=postgresql://postgres:pw@127.0.0.1:55432/postgres python -m pytest tests/test_jobs_postgres.py -v
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import httpx
import pytest
import pytest_asyncio
from app.core.config import settings
from app.models.job import JobStatus, StoreStatus, initial_store_statuses
from app.routers.jobs import router as jobs_router
from app.services import erasure_orchestrator
from app.services.erasure_orchestrator import ErasureOrchestrator
from app.services.legal_hold_state import LEGAL_HOLDS_DDL
from fastapi import FastAPI
from raptor_audit import AUDIT_EVENTS_DDL

# tests/conftest.py sets the placeholder GD_* settings before this import.
DSN = os.environ.get("TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="requires TEST_PG_DSN (a throwaway PostgreSQL)")

# gdpr_erasure_jobs as 002_gdpr.sql + 003_gdpr_export.sql + 005_gdpr_dsar.sql
# leave it (03-database/init/postgresql/).
_JOBS_DDL = """
CREATE TABLE gdpr_erasure_jobs (
    job_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        VARCHAR(255) NOT NULL,
    requested_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    requested_by   VARCHAR(255) NOT NULL,
    status         VARCHAR(50)  NOT NULL DEFAULT 'pending',
    store_statuses JSONB        NOT NULL DEFAULT '{}',
    completed_at   TIMESTAMPTZ,
    notes          TEXT,
    type           VARCHAR(20)  NOT NULL DEFAULT 'erasure',
    download_urls  JSONB,
    size_bytes     BIGINT,
    expires_at     TIMESTAMPTZ,
    result         JSONB
);
"""
USER = "e2e-user-" + uuid.uuid4().hex[:8]


@pytest_asyncio.fixture
async def pool():
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=3)
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS gdpr_erasure_jobs, audit_events, gdpr_legal_holds")
        await conn.execute(_JOBS_DDL)
        await conn.execute(AUDIT_EVENTS_DDL)
        await conn.execute(LEGAL_HOLDS_DDL)  # PRIV-X4: run() checks for a legal hold first
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS gdpr_erasure_jobs, audit_events, gdpr_legal_holds")
    await pool.close()


async def _insert(pool, *, type_="erasure", status="completed", user=USER, hours_ago=1.0, statuses=None, **extra):
    job_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO gdpr_erasure_jobs (job_id, user_id, requested_by, requested_at, status, store_statuses, type, "
            "download_urls, result) VALUES ($1, $2, $2, $3, $4, $5, $6, $7, $8)",
            job_id, user, datetime.now(timezone.utc) - timedelta(hours=hours_ago), status,
            json.dumps(statuses or {}), type_, extra.get("download_urls"), extra.get("result"),
        )
    return job_id


def _orchestrator(pool, monkeypatch) -> ErasureOrchestrator:
    """A real orchestrator on the real pool, with every external call stubbed."""
    monkeypatch.setattr(settings, "module_34_url", "")
    monkeypatch.setattr(settings, "module_35_url", "")
    orch = ErasureOrchestrator(pool=pool, redis=None, kafka_producer=None)

    async def noop(*args, **kwargs):
        pass

    class _Redis:
        sadd = expire = noop

    orch.redis = _Redis()
    for name in ("_disable_keycloak_account", "_hard_delete_keycloak_account", "_publish_tombstone"):
        monkeypatch.setattr(orch, name, noop)
    for key in ("04", "25", "26", "03", "23", "02"):
        monkeypatch.setattr(orch, f"_delete_module_{key}", noop)
    return orch


def _capture_dpo_mail(monkeypatch) -> list:
    """Configure a DPO address and replace only the SMTP connection, so
    _notify_dpo builds and sends its real message into the returned list."""
    sent: list = []

    class _SMTP:
        def __init__(self, host, port):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent.append(msg)

    monkeypatch.setattr(settings, "dpo_email", "dpo@example.test")
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.test")
    monkeypatch.setattr(erasure_orchestrator.smtplib, "SMTP", _SMTP)
    return sent


async def _events(pool) -> list:
    async with pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM audit_events ORDER BY timestamp")


def _client(pool) -> httpx.AsyncClient:
    app = FastAPI()
    app.state.pool = pool
    app.include_router(jobs_router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m33")


@pytest.mark.asyncio
async def test_job_list_filters_real_rows_and_never_returns_urls_or_reports(pool):
    failed = await _insert(pool, status="partial_failure", statuses={"module_04": "deleted", "module_25": "failed"})
    stuck = await _insert(pool, type_="export", status="in_progress", hours_ago=30,
                          download_urls=json.dumps(["https://s3/secret-link"]))
    await _insert(pool, type_="dsar", status="completed", result=json.dumps({"email": "a@x.io"}))
    await _insert(pool, type_="consent_withdrawal", status="pending", user="someone-else")

    async with _client(pool) as c:
        everything = (await c.get("/jobs")).json()
        assert everything["total"] == 4
        dumped = json.dumps(everything)
        assert "secret-link" not in dumped and "a@x.io" not in dumped

        partial = (await c.get("/jobs", params={"status": "partial_failure"})).json()
        assert [i["job_id"] for i in partial["items"]] == [failed]
        assert partial["items"][0]["failed_stores"] == ["module_25"]

        assert [i["job_id"] for i in (await c.get("/jobs", params={"stuck_hours": 24})).json()["items"]] == [stuck]
        both = {"status": "in_progress", "stuck_hours": 24}
        assert [i["job_id"] for i in (await c.get("/jobs", params=both)).json()["items"]] == [stuck]
        never_stuck = await c.get("/jobs", params={"status": "partial_failure", "stuck_hours": 24})
        assert never_stuck.status_code == 422 and "partial_failure" in never_stuck.json()["detail"]
        assert (await c.get("/jobs", params={"user_id": USER, "type": "consent_withdrawal"})).json()["total"] == 0
        assert len((await c.get("/jobs", params={"page_size": 2, "page": 2})).json()["items"]) == 2
        assert (await c.get("/jobs", params={"status": "nope"})).status_code == 422

        summary = (await c.get("/jobs/summary")).json()
    assert summary["by_type"] == {
        "erasure": {"partial_failure": 1}, "export": {"in_progress": 1},
        "dsar": {"completed": 1}, "consent_withdrawal": {"pending": 1},
    }
    assert summary["partial_failure"] == 1
    assert summary["stuck"] == {"export": 1}


@pytest.mark.asyncio
async def test_erasure_without_its_audit_event_is_visible_and_a_retry_completes_it(pool, monkeypatch):
    orch = _orchestrator(pool, monkeypatch)
    sent = _capture_dpo_mail(monkeypatch)

    job = await orch.create_job(USER, "dpo-admin")
    # The audit sink is down: the erasure itself succeeds, its confirmation must not.
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE audit_events RENAME TO audit_events_down")
    await orch.run(job.job_id)

    async with _client(pool) as c:
        listed = (await c.get("/jobs", params={"status": "partial_failure"})).json()["items"]
        assert [i["job_id"] for i in listed] == [job.job_id]
        assert listed[0]["failed_stores"] == [] and "audit event" in listed[0]["notes"]

        async with pool.acquire() as conn:
            await conn.execute("ALTER TABLE audit_events_down RENAME TO audit_events")
        retried = await orch.retry(job.job_id)
        assert retried.status == JobStatus.COMPLETED
        assert (await c.get("/jobs/summary")).json()["partial_failure"] == 0

    # Escalated once, on the first run, naming the missing audit event.
    assert len(sent) == 1 and "audit_event" in sent[0].get_content()
    [event] = await _events(pool)
    assert (event["event_type"], event["outcome"], event["resource_id"]) == ("gdpr.erasure", "erase", job.job_id)
    assert USER not in str(dict(event)) and "dpo-admin" not in str(dict(event))
    assert all(v in ("deleted", "skipped") for v in json.loads(event["metadata"])["store_statuses"].values())
    assert set(json.loads(event["metadata"])["store_statuses"]) == set(initial_store_statuses())
    assert StoreStatus.SKIPPED.value in json.loads(event["metadata"])["store_statuses"].values()


def _flaky_store(orch, monkeypatch, *, failures: int) -> None:
    """Module 25's deleter raises for its first `failures` calls."""
    calls = {"n": 0}

    async def delete_module_25(user_id):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise httpx.ConnectError("module 25 unreachable")

    monkeypatch.setattr(orch, "_delete_module_25", delete_module_25)
    monkeypatch.setattr(settings, "erasure_max_attempts", 1)


@pytest.mark.asyncio
async def test_failed_store_is_visible_escalated_once_and_a_retry_completes_it(pool, monkeypatch):
    orch = _orchestrator(pool, monkeypatch)
    sent = _capture_dpo_mail(monkeypatch)
    _flaky_store(orch, monkeypatch, failures=1)

    job = await orch.create_job(USER, "dpo-admin")
    await orch.run(job.job_id)

    async with _client(pool) as c:
        [listed] = (await c.get("/jobs", params={"status": "partial_failure"})).json()["items"]
        assert listed["job_id"] == job.job_id
        assert listed["failed_stores"] == ["module_25"] and "module_25" in listed["notes"]
        assert listed["completed_at"] is None
        assert (await c.get("/jobs/summary")).json()["partial_failure"] == 1

        # Escalation: one real DPO email naming the job, the failed store and the retry route.
        [mail] = sent
        assert mail["To"] == "dpo@example.test" and job.job_id in mail["Subject"]
        body = mail.get_content()
        assert "module_25" in body and f"/gdpr/erasure/jobs/{job.job_id}/retry" in body

        retried = await orch.retry(job.job_id)
        assert retried.status == JobStatus.COMPLETED
        [done] = (await c.get("/jobs", params={"status": "completed"})).json()["items"]
        assert done["job_id"] == job.job_id and done["failed_stores"] == [] and done["completed_at"]
        assert (await c.get("/jobs/summary")).json()["partial_failure"] == 0

    assert len(sent) == 1  # a retry never re-escalates
    events = await _events(pool)
    assert [e["outcome"] for e in events] == ["error", "erase"]
    assert all(USER not in str(dict(e)) and "dpo-admin" not in str(dict(e)) for e in events)
    assert json.loads(events[0]["metadata"])["store_statuses"]["module_25"] == StoreStatus.FAILED.value


@pytest.mark.asyncio
async def test_a_retry_that_fails_again_stays_visible_as_partial_failure(pool, monkeypatch):
    orch = _orchestrator(pool, monkeypatch)
    sent = _capture_dpo_mail(monkeypatch)
    _flaky_store(orch, monkeypatch, failures=2)

    job = await orch.create_job(USER, "dpo-admin")
    await orch.run(job.job_id)
    retried = await orch.retry(job.job_id)
    assert retried.status == JobStatus.PARTIAL_FAILURE

    async with _client(pool) as c:
        [listed] = (await c.get("/jobs", params={"status": "partial_failure"})).json()["items"]
        assert listed["job_id"] == job.job_id and listed["failed_stores"] == ["module_25"]
        assert (await c.get("/jobs", params={"status": "completed"})).json()["total"] == 0

    assert len(sent) == 1
    assert [e["outcome"] for e in await _events(pool)] == ["error", "error"]
