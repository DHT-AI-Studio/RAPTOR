"""
PRIV-X4 / VIE01-358: legal holds and erasure exceptions against a real
PostgreSQL.

One test per acceptance criterion, plus the edges the flow relies on: a hold
placed after an erasure job was created still stops it, a hold commits only
once Keycloak keeps its attributes, and an expiry run interrupted half way is
finished by the next. The tables come from the real init SQL
(010_gdpr_legal_hold.sql, 004_gdpr_retention.sql with its seed rows); the
routers, the orchestrator and every query are real. Keycloak (ConsentStore),
the data stores, Kafka and Redis are stubbed.

Needs an empty, throwaway database (tables are created and dropped). Skips
unless TEST_PG_DSN is set. Run from the 33-privacy-compliance-service/
directory, e.g.:
    docker run -d --rm --name pg-test -e POSTGRES_PASSWORD=pw -p 127.0.0.1:55432:5432 postgres:15
    TEST_PG_DSN=postgresql://postgres:pw@127.0.0.1:55432/postgres python -m pytest tests/test_legal_hold_postgres.py -v
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import asyncpg
import httpx
import pytest
import pytest_asyncio
from app.core.config import settings
from app.routers.admin import router as admin_router
from app.routers.erasure import router as erasure_router
from app.routers.erasure import start_erasure
from app.services.consent_store import UserNotFound
from app.services.erasure_orchestrator import ErasureOrchestrator
from app.services.legal_hold import LegalHoldService
from app.services.legal_hold_state import lock_user, today_utc
from fastapi import FastAPI
from raptor_audit import AUDIT_EVENTS_DDL

# tests/conftest.py sets the placeholder GD_* settings before this import.
DSN = os.environ.get("TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="requires TEST_PG_DSN (a throwaway PostgreSQL)")

INIT_SQL = Path(__file__).resolve().parents[2] / "03-database" / "init" / "postgresql"
# gdpr_erasure_jobs as 002_gdpr.sql + 003_gdpr_export.sql + 005_gdpr_dsar.sql leave it.
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
_TABLES = "gdpr_erasure_jobs, gdpr_legal_holds, gdpr_retention_policies, gdpr_retention_log, audit_events"
USER = "held-user-" + uuid.uuid4().hex[:8]
ADMIN = "dpo-1"
FUTURE = (today_utc() + timedelta(days=60)).isoformat()


def _init_sql(name: str) -> str:
    """An init script without its psql-only \\connect line."""
    text = (INIT_SQL / name).read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.startswith("\\connect"))


@pytest_asyncio.fixture
async def pool():
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_TABLES}")
        await conn.execute(_JOBS_DDL)
        await conn.execute(_init_sql("010_gdpr_legal_hold.sql"))
        await conn.execute(_init_sql("004_gdpr_retention.sql"))
        await conn.execute(AUDIT_EVENTS_DDL)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_TABLES}")
    await pool.close()


class _FakeConsentStore:
    """ConsentStore.update_user as Keycloak behaves: read-modify-write of
    the whole user, unknown users 404, unmanaged attributes optionally
    dropped silently."""

    def __init__(self):
        self.users: dict = {USER: {"id": USER, "attributes": {"locale": ["en"]}}}
        self.drop_unmanaged = False

    async def update_user(self, user_id, mutate):
        if user_id not in self.users:
            raise UserNotFound(user_id)
        user = copy.deepcopy(self.users[user_id])
        mutate(user)
        if self.drop_unmanaged:
            user["attributes"] = self.users[user_id]["attributes"]
        self.users[user_id] = user
        return copy.deepcopy(user)


class _Env:
    def __init__(self, pool, monkeypatch):
        monkeypatch.setattr(settings, "module_34_url", "")
        monkeypatch.setattr(settings, "module_35_url", "")
        self.pool = pool
        self.deleted: list = []  # (store, user_id) for every store deletion
        orch = ErasureOrchestrator(pool=pool, redis=None, kafka_producer=None)

        async def noop(*args, **kwargs):
            pass

        class _Redis:
            sadd = expire = noop

        orch.redis = _Redis()
        for name in ("_disable_keycloak_account", "_hard_delete_keycloak_account", "_publish_tombstone"):
            monkeypatch.setattr(orch, name, noop)
        for key in ("04", "25", "26", "03", "23", "02"):
            async def delete(user_id, key=key):
                self.deleted.append((key, user_id))
            monkeypatch.setattr(orch, f"_delete_module_{key}", delete)
        self.orch = orch
        self.keycloak = _FakeConsentStore()

        app = FastAPI()
        app.state.orchestrator = orch
        app.state.background_tasks = set()
        app.state.legal_hold_service = LegalHoldService(
            pool=pool, consent_store=self.keycloak, orchestrator=orch,
            start_erasure=lambda job_id: start_erasure(app, job_id),
        )
        app.include_router(erasure_router)
        app.include_router(admin_router)
        self.app = app
        self.holds: LegalHoldService = app.state.legal_hold_service
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m33")

    async def place(self, user=USER, **body):
        payload = {"user_id": user, "reason": "regulatory_audit", "hold_until": FUTURE, "authority": "DPO", **body}
        return await self.client.post("/admin/legal-hold", json=payload, headers={"X-Requested-By": ADMIN})

    async def erase(self, user=USER):
        return await self.client.post("/erasure", headers={"X-User-ID": user})

    async def settle(self):
        """Wait for every background erasure to finish."""
        for _ in range(200):
            if not self.app.state.background_tasks:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("background erasure did not finish")

    async def job_status(self, job_id):
        return await self.pool.fetchval("SELECT status FROM gdpr_erasure_jobs WHERE job_id = $1", uuid.UUID(job_id))

    async def holds_rows(self, user=USER):
        return await self.pool.fetch("SELECT * FROM gdpr_legal_holds WHERE user_id = $1 ORDER BY id", user)


@pytest_asyncio.fixture
async def env(pool, monkeypatch):
    env = _Env(pool, monkeypatch)
    yield env
    await env.client.aclose()


# ── AC: POST /gdpr/admin/legal-hold ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_placing_a_hold_writes_the_row_and_the_keycloak_attributes(env):
    r = await env.place()
    assert r.status_code == 200, r.text
    hold = r.json()
    assert (hold["user_id"], hold["reason"], hold["authority"], hold["hold_until"]) == (USER, "regulatory_audit", "DPO", FUTURE)
    assert hold["released_at"] is None and hold["placed_at"]
    [row] = await env.holds_rows()
    assert (row["reason"], row["authority"], row["hold_until"], row["released_at"]) == ("regulatory_audit", "DPO", date.fromisoformat(FUTURE), None)
    attributes = env.keycloak.users[USER]["attributes"]
    assert attributes == {"locale": ["en"], "gdpr_legal_hold": ["true"], "gdpr_hold_until": [FUTURE]}


@pytest.mark.asyncio
async def test_a_hold_is_refused_for_a_past_date_a_second_open_hold_and_an_unknown_user(env):
    assert (await env.place(hold_until=today_utc().isoformat())).status_code == 422
    assert (await env.place()).status_code == 200
    second = await env.place(hold_until=(today_utc() + timedelta(days=90)).isoformat())
    assert second.status_code == 409 and FUTURE in second.json()["detail"]
    assert (await env.place(user="nobody")).status_code == 404
    assert await env.holds_rows("nobody") == []
    assert len(await env.holds_rows()) == 1
    missing = await env.client.post("/admin/legal-hold", json={"user_id": USER, "reason": "x", "hold_until": FUTURE, "authority": "DPO"})
    assert missing.status_code == 400  # no X-Requested-By


@pytest.mark.asyncio
async def test_a_hold_keycloak_did_not_keep_is_not_recorded(env):
    env.keycloak.drop_unmanaged = True
    r = await env.place()
    assert r.status_code == 500 and "Unmanaged attributes" in r.json()["detail"]
    assert await env.holds_rows() == []  # rolled back with the attribute write


@pytest.mark.asyncio
async def test_a_hold_is_refused_while_an_erasure_runs_or_after_it_completed(env):
    job = await env.orch.create_job(USER, USER)
    await env.pool.execute("UPDATE gdpr_erasure_jobs SET status = 'in_progress' WHERE job_id = $1", uuid.UUID(job.job_id))
    r = await env.place()
    assert r.status_code == 409 and "running" in r.json()["detail"]
    await env.pool.execute("UPDATE gdpr_erasure_jobs SET status = 'completed' WHERE job_id = $1", uuid.UUID(job.job_id))
    r = await env.place()
    assert r.status_code == 409 and "already been erased" in r.json()["detail"]
    assert await env.holds_rows() == []


# ── AC: POST /gdpr/erasure checks for an active hold ────────────────────────

@pytest.mark.asyncio
async def test_erasure_under_a_hold_is_deferred_recorded_and_deletes_nothing(env):
    await env.place()
    r = await env.erase()
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["status"], body["reason"], body["hold_until"]) == ("DEFERRED", "legal_hold", FUTURE)
    assert body["estimated_completion_at"] is None
    await env.settle()
    assert await env.job_status(body["job_id"]) == "deferred"
    assert env.deleted == []
    # Asking again returns the same deferred request, not a second one.
    again = (await env.erase()).json()
    assert (again["job_id"], again["status"]) == (body["job_id"], "DEFERRED")
    assert await env.pool.fetchval("SELECT count(*) FROM gdpr_erasure_jobs WHERE user_id = $1", USER) == 1


@pytest.mark.asyncio
async def test_erasure_without_a_hold_runs_as_before(env):
    r = (await env.erase()).json()
    assert r["status"] == "pending" and r["reason"] is None and r["hold_until"] is None
    await env.settle()
    assert await env.job_status(r["job_id"]) == "completed"
    assert {store for store, _ in env.deleted} == {"04", "25", "26", "03", "23", "02"}


@pytest.mark.asyncio
async def test_a_hold_placed_after_the_job_was_created_still_stops_it(env):
    job = await env.orch.create_job(USER, USER)  # submitted, not started yet
    assert (await env.place()).status_code == 200
    await env.orch.run(job.job_id)
    assert await env.job_status(job.job_id) == "deferred"
    assert env.deleted == []


@pytest.mark.asyncio
async def test_an_erasure_starting_while_a_hold_is_being_placed_waits_for_it_and_defers(env):
    job = await env.orch.create_job(USER, USER)
    async with env.pool.acquire() as conn:
        async with conn.transaction():  # place(), mid-way: locked, row written, not committed
            await lock_user(conn, USER)
            await conn.execute(
                "INSERT INTO gdpr_legal_holds (user_id, reason, authority, hold_until) VALUES ($1, 'audit', 'DPO', $2)",
                USER, date.fromisoformat(FUTURE),
            )
            run = asyncio.create_task(env.orch.run(job.job_id))
            await asyncio.sleep(0.3)
            assert not run.done() and await env.job_status(job.job_id) == "pending"
    await run
    assert await env.job_status(job.job_id) == "deferred"
    assert env.deleted == []


@pytest.mark.asyncio
async def test_a_hold_placed_while_an_erasure_is_starting_waits_for_it_and_is_refused(env):
    job = await env.orch.create_job(USER, USER)
    async with env.pool.acquire() as conn:
        async with conn.transaction():  # run(), mid-way: locked, IN_PROGRESS written, not committed
            await lock_user(conn, USER)
            await conn.execute("UPDATE gdpr_erasure_jobs SET status = 'in_progress' WHERE job_id = $1", uuid.UUID(job.job_id))
            place = asyncio.create_task(env.place())
            await asyncio.sleep(0.3)
            assert not place.done()
    r = await place
    assert r.status_code == 409 and "running" in r.json()["detail"]
    assert await env.holds_rows() == []


@pytest.mark.asyncio
async def test_another_users_hold_does_not_defer_this_erasure(env):
    env.keycloak.users["other"] = {"id": "other", "attributes": {}}
    await env.place(user="other")
    r = (await env.erase()).json()
    assert r["status"] == "pending"
    await env.settle()
    assert await env.job_status(r["job_id"]) == "completed"


# ── AC: daily job releases expired holds and runs the deferred erasures ──────

async def _expire(env, user=USER):
    """End the user's open hold: hold_until today, the first day it no longer
    applies (the API refuses dates that aren't in the future)."""
    await env.pool.execute(
        "UPDATE gdpr_legal_holds SET hold_until = $2 WHERE user_id = $1 AND released_at IS NULL",
        user, today_utc(),
    )


@pytest.mark.asyncio
async def test_expiry_releases_the_hold_clears_keycloak_and_runs_the_deferred_erasure(env):
    await env.place()
    deferred = (await env.erase()).json()["job_id"]
    await _expire(env)
    await env.holds.release_expired()
    await env.settle()
    [row] = await env.holds_rows()
    assert row["released_at"] is not None
    assert env.keycloak.users[USER]["attributes"] == {"locale": ["en"]}
    assert await env.job_status(deferred) == "completed"
    assert {store for store, _ in env.deleted} == {"04", "25", "26", "03", "23", "02"}


# ── V05-10 / VIE01-364: CF-4 gdpr.legal_hold audit events ───────────────────

async def _hold_events(env):
    rows = await env.pool.fetch(
        "SELECT * FROM audit_events WHERE event_type = 'gdpr.legal_hold' ORDER BY timestamp")
    return [dict(r, metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"])
            for r in rows]


@pytest.mark.asyncio
async def test_placing_and_expiring_a_hold_write_audit_events_without_its_free_text(env):
    hold_id = (await env.place()).json()["id"]
    assert (await env.place()).status_code == 409          # refused: no event
    await _expire(env)
    await env.holds.release_expired()
    await env.settle()
    place, release = await _hold_events(env)
    for event in (place, release):
        assert (event["source_module"], event["outcome"], event["resource_id"]) == (33, "allow", str(hold_id))
        dumped = json.dumps(event, default=str)
        assert USER not in dumped and "regulatory_audit" not in dumped and '"DPO"' not in dumped
    assert place["metadata"]["action"] == "place" and place["metadata"]["requested_by_subject"] is False
    assert place["metadata"]["hold_until"] == FUTURE
    assert release["metadata"] == {"action": "release", "hold_until": release["metadata"]["hold_until"],
                                   "reason": "expired", "requested_by_system": True}


@pytest.mark.asyncio
async def test_a_hold_keycloak_did_not_keep_writes_an_error_event(env):
    env.keycloak.drop_unmanaged = True
    assert (await env.place()).status_code == 500
    [event] = await _hold_events(env)
    assert (event["outcome"], event["resource_id"], event["metadata"]["action"]) == ("error", None, "place")


@pytest.mark.asyncio
async def test_expiry_leaves_holds_still_in_force_alone(env):
    await env.place()
    deferred = (await env.erase()).json()["job_id"]
    await env.holds.release_expired()
    await env.settle()
    [row] = await env.holds_rows()
    assert row["released_at"] is None
    assert await env.job_status(deferred) == "deferred" and env.deleted == []


@pytest.mark.asyncio
async def test_an_expiry_run_interrupted_after_the_release_is_finished_by_the_next(env):
    await env.place()
    deferred = (await env.erase()).json()["job_id"]
    # The previous run released the hold, then stopped before queueing.
    await env.pool.execute("UPDATE gdpr_legal_holds SET released_at = now() WHERE user_id = $1", USER)
    await env.holds.release_expired()
    await env.settle()
    assert await env.job_status(deferred) == "completed"


@pytest.mark.asyncio
async def test_expiry_still_runs_the_erasure_when_the_account_is_already_gone(env):
    await env.place()
    deferred = (await env.erase()).json()["job_id"]
    await _expire(env)
    del env.keycloak.users[USER]
    await env.holds.release_expired()
    await env.settle()
    assert await env.job_status(deferred) == "completed"


# ── AC: GET /gdpr/admin/legal-holds ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_listing_shows_only_holds_in_force_soonest_first(env):
    later = (today_utc() + timedelta(days=120)).isoformat()
    for user, until in (("a-later", later), ("b-sooner", FUTURE), ("c-expired", FUTURE)):
        env.keycloak.users[user] = {"id": user, "attributes": {}}
        assert (await env.place(user=user, hold_until=until)).status_code == 200
    await _expire(env, "c-expired")
    holds = (await env.client.get("/admin/legal-holds")).json()["holds"]
    assert [h["user_id"] for h in holds] == ["b-sooner", "a-later"]


# ── AC: exception_fields in the erasure response ────────────────────────────

def _tokens(fields) -> list[str]:
    """The "<data>_<retention>" part of each "<data>_<retention>: <basis>" entry."""
    return [f.split(": ", 1)[0] for f in fields]


NOT_PURGED = " -- policy period, not deleted automatically"


@pytest.mark.asyncio
async def test_every_erasure_response_lists_everything_kept_with_retention_and_basis(env):
    expected = [
        # GD-3 seed rows: 1095 and 2555 days with auto_delete=false (never purged), 365 days purged.
        "gdpr_erasure_jobs_3yr", "gdpr_consent_history_7yr", "gdpr_access_log_1yr",
        "gdpr_restriction_log_indefinite", "gdpr_objection_log_indefinite", "gdpr_legal_holds_indefinite",
        "audit_events_3yr",  # GD_AUDIT_RETENTION_DAYS, 1095 by default
    ]
    first = (await env.erase()).json()["exception_fields"]
    assert all(isinstance(f, str) for f in first)  # the ticket's array of strings
    assert _tokens(first) == expected
    assert all(len(f.split(": ", 1)[1]) > 10 for f in first)  # each cites its basis
    assert [f.endswith(NOT_PURGED) for f in first] == [True, True, False, False, False, False, False]
    await env.settle()
    assert (await env.erase()).json()["exception_fields"] == first  # already_erased
    env.keycloak.users["held"] = {"id": "held", "attributes": {}}
    await env.place(user="held")
    assert (await env.erase(user="held")).json()["exception_fields"] == first  # DEFERRED
    # Periods follow the GD-3 table as an admin changes it; a period not in whole years is in days.
    await env.pool.execute(
        "UPDATE gdpr_retention_policies SET retention_days = 400, auto_delete = TRUE WHERE data_category = 'gdpr_consent_history'"
    )
    changed = (await env.erase(user="held")).json()["exception_fields"][1]
    assert changed.startswith("gdpr_consent_history_400d: ") and not changed.endswith(NOT_PURGED)


# ── GD-1 cannot erase held data: the retry of a partial failure ─────────────

@pytest.mark.asyncio
async def test_a_partial_failure_is_not_retried_while_the_user_is_held(env):
    job = await env.orch.create_job(USER, USER)
    # What a first run leaves when one store kept failing: the others deleted.
    statuses = {key: "deleted" for key in job.store_statuses} | {"module_04": "failed"}
    await env.pool.execute(
        "UPDATE gdpr_erasure_jobs SET status = 'partial_failure', store_statuses = $2 WHERE job_id = $1",
        uuid.UUID(job.job_id), json.dumps(statuses),
    )
    assert (await env.place()).status_code == 200  # a hold can still be placed on what remains
    r = await env.client.post(f"/erasure/jobs/{job.job_id}/retry")
    assert r.status_code == 409 and FUTURE in r.json()["detail"]
    assert env.deleted == []
    assert await env.job_status(job.job_id) == "partial_failure"
    # Once the hold has ended the retry goes ahead.
    await _expire(env)
    r = await env.client.post(f"/erasure/jobs/{job.job_id}/retry")
    assert r.status_code == 200 and r.json()["status"] == "completed"
    assert env.deleted == [("04", USER)]


async def _partial_failure(env) -> str:
    job = await env.orch.create_job(USER, USER)
    statuses = {key: "deleted" for key in job.store_statuses} | {"module_04": "failed"}
    await env.pool.execute(
        "UPDATE gdpr_erasure_jobs SET status = 'partial_failure', store_statuses = $2 WHERE job_id = $1",
        uuid.UUID(job.job_id), json.dumps(statuses),
    )
    return job.job_id


@pytest.mark.asyncio
async def test_a_retry_requested_while_a_hold_is_being_placed_waits_for_it_and_is_refused(env):
    job_id = await _partial_failure(env)
    async with env.pool.acquire() as conn:
        async with conn.transaction():  # place(), mid-way: locked, row written, not committed
            await lock_user(conn, USER)
            await conn.execute(
                "INSERT INTO gdpr_legal_holds (user_id, reason, authority, hold_until) VALUES ($1, 'audit', 'DPO', $2)",
                USER, date.fromisoformat(FUTURE),
            )
            retry = asyncio.create_task(env.client.post(f"/erasure/jobs/{job_id}/retry"))
            await asyncio.sleep(0.3)
            assert not retry.done()
    r = await retry
    assert r.status_code == 409 and "legal hold" in r.json()["detail"]
    assert env.deleted == [] and await env.job_status(job_id) == "partial_failure"


@pytest.mark.asyncio
async def test_a_hold_is_refused_while_a_retry_runs_and_a_job_is_retried_once(env):
    job_id = await _partial_failure(env)
    assert await env.orch.claim_retry(job_id, USER) == (None, True)  # a retry has started
    assert await env.job_status(job_id) == "in_progress"
    r = await env.place()
    assert r.status_code == 409 and "running" in r.json()["detail"]
    second = await env.client.post(f"/erasure/jobs/{job_id}/retry")
    assert second.status_code == 409  # not partial_failure any more: no second, overlapping retry
    # Two retries that both read partial_failure before either claimed: one wins.
    await env.pool.execute("UPDATE gdpr_erasure_jobs SET status = 'partial_failure' WHERE job_id = $1", uuid.UUID(job_id))
    claims = await asyncio.gather(env.orch.claim_retry(job_id, USER), env.orch.claim_retry(job_id, USER))
    assert sorted(claimed for _, claimed in claims) == [False, True]
