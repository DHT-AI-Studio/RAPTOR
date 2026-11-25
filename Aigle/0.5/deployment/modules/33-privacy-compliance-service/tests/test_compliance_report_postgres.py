"""
CM-7 / VIE01-410: the compliance report's SQL against a real PostgreSQL.

Two databases, like a deployment: the throwaway one TEST_PG_DSN names stands
in for privacy_compliance (gdpr_erasure_jobs, audit_events, gdpr_access_log),
and a `guardrails` database is created next to it with Module 23's own DDL
(app/db/schema.py, loaded from that module's file). Rows are written inside
and outside the period; the report must count and list only the ones inside,
hash every person id, and never carry a raw id, file name or free text.

Needs an empty, throwaway PostgreSQL (tables and a database are created and
dropped). Skips unless TEST_PG_DSN is set. Run from the
33-privacy-compliance-service/ directory, e.g.:
    docker run -d --rm --name pg-cm7 -e POSTGRES_PASSWORD=pw -p 127.0.0.1:55473:5432 postgres:15
    TEST_PG_DSN=postgresql://postgres:pw@127.0.0.1:55473/postgres python -m pytest tests/test_compliance_report_postgres.py -v
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import httpx
import pytest
import pytest_asyncio
from app.routers import reports
from app.services import compliance_report as cr
from fastapi import FastAPI
from raptor_audit import AUDIT_EVENTS_DDL, hash_value

# tests/conftest.py sets the placeholder GD_* settings before these imports.
DSN = os.environ.get("TEST_PG_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="requires TEST_PG_DSN (a throwaway PostgreSQL)")
GUARDRAILS_DB = "cm7_guardrails_test"

_M23_SCHEMA = Path(__file__).resolve().parents[2] / "23-guardrail-service" / "app" / "db" / "schema.py"
_spec = importlib.util.spec_from_file_location("m23_schema", _M23_SCHEMA)
_m23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m23)
GUARDRAILS_DDL = _m23.DDL

# As tests/test_jobs_postgres.py (002/003/005_gdpr*.sql), and 007_gdpr_access_log.sql.
_PRIVACY_DDL = """
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
CREATE TABLE gdpr_access_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(255),
    accessor_id     VARCHAR(255) NOT NULL,
    endpoint        VARCHAR(512) NOT NULL,
    method          VARCHAR(16)  NOT NULL,
    data_categories VARCHAR(64)[] NOT NULL DEFAULT '{}',
    purpose         VARCHAR(64),
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    ip_hash         VARCHAR(64)
);
"""

ALICE, BOB, ADMIN = (str(uuid.uuid4()) for _ in range(3))
START = datetime(2026, 11, 1, tzinfo=timezone.utc)
END = datetime(2026, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
IN1, IN2, IN3 = START + timedelta(days=2), START + timedelta(days=2, hours=5), START + timedelta(days=9)
BEFORE, AFTER = START - timedelta(seconds=1), END + timedelta(seconds=1)
RAW = (ALICE, BOB, ADMIN, "alice-birthday", "Restricted because of a dispute", "free text note")


def _dsn_for(db: str) -> str:
    base, _, _ = DSN.rpartition("/")
    return f"{base}/{db}"


@pytest_asyncio.fixture
async def dbs():
    admin = await asyncpg.connect(DSN)
    await admin.execute(f"DROP DATABASE IF EXISTS {GUARDRAILS_DB}")
    await admin.execute(f"CREATE DATABASE {GUARDRAILS_DB}")
    await admin.execute("DROP TABLE IF EXISTS gdpr_erasure_jobs, gdpr_access_log, audit_events")
    await admin.execute(_PRIVACY_DDL)
    await admin.execute(AUDIT_EVENTS_DDL)
    guard = await asyncpg.connect(_dsn_for(GUARDRAILS_DB))
    await guard.execute(GUARDRAILS_DDL)
    await _seed(admin, guard)
    yield admin, guard
    await guard.close()
    await admin.execute("DROP TABLE IF EXISTS gdpr_erasure_jobs, gdpr_access_log, audit_events")
    await admin.execute(f"DROP DATABASE IF EXISTS {GUARDRAILS_DB}")
    await admin.close()


async def _event(conn, ts, event_type, actor, metadata, outcome="allow") -> uuid.UUID:
    event_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO audit_events (event_id, timestamp, source_module, source_service, event_type, actor_id_hash, "
        "outcome, metadata) VALUES ($1, $2, 33, 'test', $3, $4, $5, $6::text::jsonb)",
        event_id, ts, event_type, hash_value(actor), outcome, json.dumps(metadata))
    return event_id


async def _seed(privacy, guard) -> None:
    # Module 23: violations (one linked to its audit event) and flags.
    linked = uuid.uuid4()
    await guard.execute(
        "INSERT INTO audit_events (event_id, timestamp, source_module, source_service, event_type, actor_id_hash, "
        "outcome, policy_version) VALUES ($1, $2, 23, 'guardrail-service', 'guardrail.decision', $3, 'block', '2.0')",
        linked, IN1, hash_value(ALICE))
    for ts, category, action, module, event_id, request_id in [
        (IN1, "S1", "block", "13", linked, "caller-chosen request text"),
        (IN2, "S1", "block", "07", None, None),
        (IN3, "S7", "redact", "13", None, None),
        (BEFORE, "S1", "block", "13", None, None),
        (AFTER, "S1", "block", "13", None, None),
    ]:
        await guard.execute(
            "INSERT INTO guardrail_violations (module, direction, category, action_taken, request_id, created_at, "
            "event_id) VALUES ($1, 'input', $2, $3, $4, $5, $6)", module, category, action, request_id, ts, event_id)
    for ts, owner, category, status, moment, chunk in [
        (IN1, ALICE, "violence", "flagged", "m" * 64, None),
        (IN3, BOB, "nsfw", "cleared", None, str(uuid.uuid4())),
        (AFTER, ALICE, "violence", "flagged", None, None),
    ]:
        await guard.execute(
            "INSERT INTO moderation_flags (branch_id, version_id, asset_path, moment_id, chunk_id, category, score, "
            "model, status, created_at, updated_at) VALUES ($1, 'v1', 'video/mp4/alice-birthday', $2, $3, $4, 0.9, "
            "'shieldgemma-2', $5, $6, $6)", owner, moment, chunk, category, status, ts)

    # Module 33: jobs, rights events (+ events that must stay out), access log.
    for ts, user, by, type_, status in [
        (IN1, ALICE, ADMIN, "erasure", "partial_failure"),
        (IN2, BOB, BOB, "export", "completed"),
        (BEFORE, ALICE, ALICE, "dsar", "completed"),
    ]:
        await privacy.execute(
            "INSERT INTO gdpr_erasure_jobs (user_id, requested_by, type, status, requested_at, store_statuses, notes) "
            "VALUES ($1, $2, $3, $4, $5, $6, 'free text note')",
            user, by, type_, status, ts, json.dumps({"module_04": "deleted", "module_25": "failed"}))
    await _event(privacy, IN1, "gdpr.restriction", ALICE, {"action": "restrict", "requested_by_subject": True})
    await _event(privacy, IN2, "gdpr.legal_hold", BOB,
                 {"action": "place", "requested_by_subject": False, "requested_by_hash": hash_value(ADMIN)})
    await _event(privacy, IN3, "gdpr.consent", ALICE, {"action": "withdraw", "purpose": "memory",
                                                       "requested_by_subject": True})
    await _event(privacy, IN3, "gdpr.erasure", ALICE, {"job_type": "erasure"}, outcome="erase")   # a job: not here
    await _event(privacy, AFTER, "gdpr.objection", ALICE, {"action": "object", "requested_by_subject": True})
    for ts, user, accessor, endpoint, method, purpose in [
        (IN1, ALICE, ADMIN, f"/api/0.5/gdpr/erasure/{ALICE}", "POST", "right_to_erasure"),
        (IN2, ALICE, ALICE, "/api/0.5/search/bm25", "POST", "service_provision"),
        (IN3, BOB, BOB, "/api/0.5/search/bm25", "POST", "service_provision"),
        (AFTER, BOB, BOB, "/api/0.5/memory/x", "GET", "service_provision"),
    ]:
        await privacy.execute(
            "INSERT INTO gdpr_access_log (user_id, accessor_id, endpoint, method, data_categories, purpose, "
            "timestamp, ip_hash) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            user, accessor, endpoint, method, ["indexed_content"], purpose, ts, "f" * 64)


async def _report(dbs, max_rows=1000):
    privacy, guard = dbs
    return await cr.build_report(privacy, guard, START, END, max_rows)


def _no_raw(text: str) -> None:
    for raw in RAW:
        assert raw not in text, raw


@pytest.mark.asyncio
async def test_counts_only_the_period(dbs):
    r = await _report(dbs)
    assert r["totals"] == {"guardrail_violations": 3, "moderation_flags": 2, "gdpr_jobs": 2,
                           "gdpr_rights_requests": 3, "access_log": 3}
    assert r["unavailable_sections"] == []
    assert all(s["available"] and not s["truncated"] for s in r["sections"].values())


@pytest.mark.asyncio
async def test_violations_summary_and_actor_from_the_audit_event(dbs):
    s = (await _report(dbs))["sections"]["guardrail_violations"]
    assert {(g["category"], g["action_taken"]): g["count"] for g in s["summary"]["by_category_and_action"]} == \
        {("S1", "block"): 2, ("S7", "redact"): 1}
    assert {g["module"]: g["count"] for g in s["summary"]["by_module"]} == {"13": 2, "07": 1}
    assert s["summary"]["by_day"] == [{"day": "2026-11-03", "count": 2}, {"day": "2026-11-10", "count": 1}]
    first = s["rows"][0]
    assert (first["actor_hash"], first["policy_version"]) == (hash_value(ALICE), "2.0")
    assert s["rows"][1]["actor_hash"] is None                       # no linked audit event


@pytest.mark.asyncio
async def test_flags_are_hashed_and_summarised(dbs):
    s = (await _report(dbs))["sections"]["moderation_flags"]
    assert {g["status"]: g["count"] for g in s["summary"]["by_status"]} == {"flagged": 1, "cleared": 1}
    assert [r["scope"] for r in s["rows"]] == ["moment", "chunk"]
    assert [r["owner_hash"] for r in s["rows"]] == [hash_value(ALICE), hash_value(BOB)]


@pytest.mark.asyncio
async def test_gdpr_jobs_and_rights_requests(dbs):
    sections = (await _report(dbs))["sections"]
    jobs = sections["gdpr_jobs"]
    assert {(g["type"], g["status"]): g["count"] for g in jobs["summary"]["by_type_and_status"]} == \
        {("erasure", "partial_failure"): 1, ("export", "completed"): 1}
    assert jobs["rows"][0]["failed_stores"] == ["module_25"]
    assert jobs["rows"][0]["requester_hash"] == hash_value(ADMIN)
    rights = sections["gdpr_rights_requests"]
    assert [r["type"] for r in rights["rows"]] == ["gdpr.restriction", "gdpr.legal_hold", "gdpr.consent"]
    hold = rights["rows"][1]
    assert (hold["requested_by"], hold["requester_hash"], hold["subject_hash"]) == \
        ("other", hash_value(ADMIN), hash_value(BOB))
    assert rights["rows"][2]["purpose"] == "memory"
    assert {(g["event_type"], g["action"]) for g in rights["summary"]["by_type_action_and_outcome"]} == \
        {("gdpr.restriction", "restrict"), ("gdpr.legal_hold", "place"), ("gdpr.consent", "withdraw")}


@pytest.mark.asyncio
async def test_access_log_scrubs_ids_and_counts_people(dbs):
    s = (await _report(dbs))["sections"]["access_log"]
    assert (s["summary"]["distinct_accessors"], s["summary"]["distinct_subjects"]) == (3, 2)
    assert s["summary"]["by_endpoint"] == [
        {"method": "POST", "endpoint": "/api/0.5/search/bm25", "count": 2},
        {"method": "POST", "endpoint": "/api/0.5/gdpr/erasure/{user_id}", "count": 1},
    ]
    assert s["rows"][0]["accessor_hash"] == hash_value(ADMIN)


@pytest.mark.asyncio
async def test_truncation_keeps_exact_counts(dbs):
    r = await _report(dbs, max_rows=1)
    for s in r["sections"].values():
        assert s["rows_included"] == 1 and s["truncated"] == (s["total"] > 1)
    assert r["sections"]["guardrail_violations"]["summary"]["by_module"][0]["count"] == 2   # still the full period


@pytest.mark.asyncio
async def test_a_missing_table_is_reported_as_unavailable(dbs):
    _, guard = dbs
    await guard.execute("DROP TABLE moderation_flags")
    r = await _report(dbs)
    assert r["unavailable_sections"] == ["moderation_flags"]
    assert r["sections"]["moderation_flags"]["available"] is False
    assert "moderation_flags" in r["sections"]["moderation_flags"]["error"]
    assert r["totals"]["guardrail_violations"] == 3


@pytest.mark.asyncio
async def test_json_and_csv_never_carry_raw_ids_or_text_and_agree(dbs):
    r = await _report(dbs)
    _no_raw(json.dumps(r, default=str))
    text = cr.to_csv(r)
    _no_raw(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    for name, s in r["sections"].items():
        assert sum(row["section"] == name for row in rows) == s["rows_included"]


@pytest.mark.asyncio
async def test_the_route_end_to_end(dbs, monkeypatch):  # dbs seeds both databases
    """GET /reports/compliance with a real pool and the real guardrails database."""
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2)

    async def guardrails():
        return await asyncpg.connect(_dsn_for(GUARDRAILS_DB))

    monkeypatch.setattr(reports, "_guardrails_connection", guardrails)
    app = FastAPI()
    app.state.pool = pool
    app.include_router(reports.router)
    try:
        params = {"from": START.isoformat(), "to": END.isoformat()}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m33") as client:
            as_json = await client.get("/reports/compliance", params=params)
            as_csv = await client.get("/reports/compliance", params={**params, "format": "csv"})
    finally:
        await pool.close()
    assert as_json.status_code == 200 and as_json.json()["totals"]["access_log"] == 3
    assert as_csv.status_code == 200 and as_csv.headers["content-type"].startswith("text/csv")
    assert len(list(csv.DictReader(io.StringIO(as_csv.text)))) == sum(as_json.json()["totals"].values())
    _no_raw(as_json.text + as_csv.text)
