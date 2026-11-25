"""
CM-7 / VIE01-410: compliance report -- id hashing, endpoint scrubbing, row
mapping, the CSV export and the route's checks. The SQL runs against a real
PostgreSQL in tests/test_compliance_report_postgres.py.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_compliance_report.py -v
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone

import pytest
from app.routers import reports
from app.services import compliance_report as cr
from fastapi import FastAPI
from fastapi.testclient import TestClient
from raptor_audit import hash_value

ALICE = "6f1c2a7e-0d4b-4c1e-9a55-1b2c3d4e5f60"
ADMIN = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
T = datetime(2026, 11, 3, 8, 30, tzinfo=timezone.utc)


# ── raw rows as the database returns them ─────────────────────────────────────


def violation(**kw):
    return {"id": uuid.uuid4(), "created_at": T, "module": "13", "direction": "input", "category": "S1",
            "action_taken": "block", "policy_id": None, "event_id": uuid.uuid4(), "actor_id_hash": hash_value(ALICE),
            "policy_version": "1.0", **kw}


def flag(**kw):
    return {"id": uuid.uuid4(), "created_at": T, "updated_at": T, "status": "flagged", "category": "violence",
            "score": 0.97, "model": "shieldgemma-2", "policy_version": None, "branch_id": ALICE,
            "asset_path": "video/mp4/alice-birthday", "version_id": "ab" * 32, "moment_id": "m" * 64,
            "chunk_id": None, "event_id": uuid.uuid4(), **kw}


def job(**kw):
    return {"job_id": uuid.uuid4(), "requested_at": T, "completed_at": None, "type": "erasure",
            "status": "partial_failure", "user_id": ALICE, "requested_by": ADMIN,
            "store_statuses": '{"module_04": "deleted", "module_25": "failed"}', **kw}


def rights(meta, **kw):
    return {"event_id": uuid.uuid4(), "timestamp": T, "event_type": "gdpr.restriction", "outcome": "allow",
            "actor_id_hash": hash_value(ALICE), "metadata": json.dumps(meta), **kw}


def access(**kw):
    return {"id": uuid.uuid4(), "timestamp": T, "user_id": ALICE, "accessor_id": ADMIN,
            "endpoint": f"/api/0.5/gdpr/erasure/{ALICE}", "method": "POST",
            "data_categories": ["identity", "memory"], "purpose": "right_to_erasure", "ip_hash": "f" * 64, **kw}


def section(rows):
    return {"available": True, "total": len(rows), "rows_included": len(rows), "truncated": False,
            "summary": {}, "rows": rows}


def report_with(**rows):
    return {"sections": {name: section(rows.get(name, [])) for name in cr.SECTIONS}}


# ── hashing and scrubbing ─────────────────────────────────────────────────────


def test_scrub_endpoint_replaces_the_rows_own_ids_and_any_other_id():
    other = str(uuid.uuid4())
    assert cr.scrub_endpoint(f"/api/0.5/gdpr/erasure/{ALICE}", ALICE, ADMIN) == "/api/0.5/gdpr/erasure/{user_id}"
    assert cr.scrub_endpoint(f"/api/0.5/gdpr/erasure/jobs/{other}", ALICE, ADMIN) == "/api/0.5/gdpr/erasure/jobs/{id}"
    assert cr.scrub_endpoint("/api/0.5/memory/x/" + "a" * 40, None) == "/api/0.5/memory/x/{id}"
    assert cr.scrub_endpoint("/api/0.5/search/bm25?q=secret", None) == "/api/0.5/search/bm25"
    assert cr.scrub_endpoint("/api/0.5/gdpr/consent", None) == "/api/0.5/gdpr/consent"


def test_violation_row_keeps_the_audit_events_hash_and_drops_the_request_id():
    row = cr.violation_row(violation(request_id="caller text"))
    assert row["actor_hash"] == hash_value(ALICE)
    assert "request_id" not in row
    assert (row["category"], row["action_taken"], row["module"]) == ("S1", "block", "13")


def test_flag_row_hashes_the_owner_and_the_file_name():
    row = cr.flag_row(flag())
    assert row["owner_hash"] == hash_value(ALICE) and row["asset_hash"] == hash_value("video/mp4/alice-birthday")
    assert "branch_id" not in row and "asset_path" not in row
    assert row["scope"] == "moment"
    assert cr.flag_row(flag(moment_id=None, chunk_id="c1"))["scope"] == "chunk"
    assert cr.flag_row(flag(moment_id=None))["scope"] == "asset"


def test_job_row_hashes_subject_and_requester_and_lists_failed_stores():
    row = cr.job_row(job(notes="free text"))
    assert row["subject_hash"] == hash_value(ALICE) and row["requester_hash"] == hash_value(ADMIN)
    assert row["requested_by_subject"] is False
    assert row["failed_stores"] == ["module_25"]
    assert "notes" not in row and "user_id" not in row
    assert cr.job_row(job(requested_by=ALICE))["requested_by_subject"] is True


@pytest.mark.parametrize("meta, who, requester", [
    ({"action": "restrict", "requested_by_subject": True}, "subject", hash_value(ALICE)),
    ({"action": "lift", "requested_by_subject": False, "requested_by_hash": hash_value(ADMIN)}, "other",
     hash_value(ADMIN)),
    ({"action": "release", "requested_by_system": True}, "system", None),
])
def test_rights_row_says_who_asked(meta, who, requester):
    row = cr.rights_row(rights(meta))
    assert (row["requested_by"], row["requester_hash"], row["subject_hash"]) == (who, requester, hash_value(ALICE))
    assert row["action"] == meta["action"]


def test_rights_row_carries_only_whitelisted_metadata():
    row = cr.rights_row(rights({"action": "grant", "purpose": "memory", "reason": "free text", "value": "x@y.z"}))
    assert row["purpose"] == "memory"
    assert "free text" not in json.dumps(row) and "x@y.z" not in json.dumps(row)


def test_access_row_hashes_both_people_and_scrubs_the_endpoint():
    row = cr.access_row(access())
    assert (row["accessor_hash"], row["subject_hash"]) == (hash_value(ADMIN), hash_value(ALICE))
    assert row["endpoint"] == "/api/0.5/gdpr/erasure/{user_id}"
    assert ALICE not in json.dumps(row) and ADMIN not in json.dumps(row)


def test_access_row_without_a_subject():
    assert cr.access_row(access(user_id=None, endpoint="/api/0.5/gdpr/jobs"))["subject_hash"] is None


# ── CSV export ────────────────────────────────────────────────────────────────


def _full_report():
    return report_with(
        guardrail_violations=[cr.violation_row(violation())],
        moderation_flags=[cr.flag_row(flag())],
        gdpr_jobs=[cr.job_row(job())],
        gdpr_rights_requests=[cr.rights_row(rights({"action": "restrict", "requested_by_subject": True}))],
        access_log=[cr.access_row(access())],
    )


def _parse(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_csv_has_the_fixed_header_and_one_row_per_event():
    text = cr.to_csv(_full_report())
    assert text.splitlines()[0] == ",".join(cr.CSV_COLUMNS)
    rows = _parse(text)
    assert [r["section"] for r in rows] == list(cr.SECTIONS)


def test_csv_and_json_agree_on_every_section_count():
    report = _full_report()
    report["sections"]["access_log"] = section([cr.access_row(access()) for _ in range(3)])
    rows = _parse(cr.to_csv(report))
    for name in cr.SECTIONS:
        assert sum(r["section"] == name for r in rows) == len(report["sections"][name]["rows"])


def test_csv_maps_actor_and_subject_per_section():
    rows = {r["section"]: r for r in _parse(cr.to_csv(_full_report()))}
    assert rows["gdpr_jobs"]["actor_hash"] == hash_value(ADMIN)
    assert rows["gdpr_jobs"]["subject_hash"] == hash_value(ALICE)
    assert rows["access_log"]["actor_hash"] == hash_value(ADMIN)
    assert rows["moderation_flags"]["subject_hash"] == hash_value(ALICE)
    assert json.loads(rows["access_log"]["details"])["endpoint"] == "/api/0.5/gdpr/erasure/{user_id}"


def test_csv_never_contains_a_raw_id_or_a_file_name():
    text = cr.to_csv(_full_report())
    for raw in (ALICE, ADMIN, "alice-birthday"):
        assert raw not in text


def test_csv_neutralises_formula_cells():
    report = report_with(guardrail_violations=[cr.violation_row(violation(category="=HYPERLINK(1)",
                                                                          module="+13"))])
    row = _parse(cr.to_csv(report))[0]
    assert row["category"] == "'=HYPERLINK(1)" and row["module"] == "'+13"


# ── the route ─────────────────────────────────────────────────────────────────


class _Pool:
    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool

            async def __aexit__(self, *exc):
                return False
        return _Ctx()


class _Conn:
    closed = False

    async def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    seen: dict = {}
    conn = _Conn()

    async def fake_connection():
        return conn

    async def fake_build(privacy_conn, guardrails_conn, start, end, max_rows):
        seen.update(start=start, end=end, max_rows=max_rows, guardrails=guardrails_conn)
        report = _full_report()
        report.update(period={"from": cr.utc(start).isoformat(), "to": cr.utc(end).isoformat()},
                      unavailable_sections=["moderation_flags"])
        report["sections"]["access_log"]["truncated"] = True
        return report

    monkeypatch.setattr(reports, "_guardrails_connection", fake_connection)
    monkeypatch.setattr(reports.compliance_report, "build_report", fake_build)
    app = FastAPI()
    app.state.pool = _Pool()
    app.include_router(reports.router)
    c = TestClient(app)
    c.seen, c.conn = seen, conn
    return c


def test_period_is_required(client):
    assert client.get("/reports/compliance", params={"from": "2026-11-01"}).status_code == 422
    assert client.get("/reports/compliance", params={"to": "2026-11-01"}).status_code == 422


def test_from_after_to_is_rejected(client):
    r = client.get("/reports/compliance", params={"from": "2026-11-05T00:00:00", "to": "2026-11-01T00:00:00"})
    assert r.status_code == 422


def test_max_rows_cannot_exceed_the_setting(client, monkeypatch):
    monkeypatch.setattr(reports.settings, "compliance_report_max_rows", 100)
    params = {"from": "2026-11-01T00:00:00", "to": "2026-11-30T00:00:00"}
    assert client.get("/reports/compliance", params={**params, "max_rows": 101}).status_code == 422
    assert client.get("/reports/compliance", params={**params, "max_rows": 10}).status_code == 200
    assert client.seen["max_rows"] == 10
    assert client.get("/reports/compliance", params=params).status_code == 200
    assert client.seen["max_rows"] == 100


def test_json_report_and_its_headers(client):
    r = client.get("/reports/compliance", params={"from": "2026-11-01T00:00:00", "to": "2026-11-30T23:59:59"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.headers["x-report-unavailable-sections"] == "moderation_flags"
    assert r.headers["x-report-truncated-sections"] == "access_log"
    assert set(r.json()["sections"]) == set(cr.SECTIONS)
    assert client.conn.closed


def test_csv_report_is_a_download(client):
    r = client.get("/reports/compliance",
                   params={"from": "2026-11-01T00:00:00Z", "to": "2026-11-30T00:00:00Z", "format": "csv"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="compliance-report_2026-11-01_2026-11-30.csv"'
    assert len(_parse(r.text)) == len(cr.SECTIONS)


def test_unknown_format_is_rejected(client):
    r = client.get("/reports/compliance", params={"from": "2026-11-01", "to": "2026-11-02", "format": "xlsx"})
    assert r.status_code == 422


def test_unreachable_guardrails_database_is_a_503(client, monkeypatch):
    async def down():
        raise OSError("connection refused")
    monkeypatch.setattr(reports, "_guardrails_connection", down)
    r = client.get("/reports/compliance", params={"from": "2026-11-01", "to": "2026-11-02"})
    assert r.status_code == 503 and "guardrails" in r.json()["detail"]
