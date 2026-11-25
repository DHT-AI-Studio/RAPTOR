"""
GD-7 / VIE01-252: unit tests for the data access audit log -- AuditMiddleware
(app/middlewares/audit_middleware.py) and the /gdpr/audit-log gateway routes.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_audit_middleware.py -v

Module 33 is never called: AuditLogClient.write is replaced by a recorder,
and the gateway routes' HTTP client by a fake. The last group runs the real
gateway app with only the JWT signature/UMA checks stubbed, so the real
get_current_user -> request.state.user -> middleware chain is exercised.
"""
import asyncio
import hashlib
import os
import sys
import time

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
# Same Kafka values as test_register_processing.py / test_fileupload_decoupling.py:
# settings are read once per test session, and this file sorts (and so
# imports) first -- a different value here would break those tests.
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.middlewares import audit_middleware as audit
from app.middlewares.audit_middleware import ALL_CATEGORIES, AuditMiddleware, classify
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

SUB = "jwt-verified-user-42"


@pytest.fixture
def records(monkeypatch):
    """Every record the middleware hands to Module 33, in order."""
    written: list = []

    async def _write(record):
        written.append(record)

    monkeypatch.setattr(audit.audit_log, "write", _write)
    return written


# ── A minimal app: the middleware + routes shaped like the gateway's ────────

def _login(request: Request):
    """Stand-in for get_current_user: verified identity on request.state."""
    request.state.user = {"sub": request.headers.get("x-test-sub", SUB)}
    return request.state.user


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuditMiddleware)

    @app.get("/api/0.4/memory/stats", dependencies=[Depends(_login)])
    async def memory_stats():
        return {"ok": True}

    @app.post("/api/0.4/search/hybrid", dependencies=[Depends(_login)])
    async def search():
        return {"results": []}

    @app.get("/api/0.4/gdpr/dsar/{user_id}", dependencies=[Depends(_login)])
    async def admin_dsar(user_id: str):
        return {"user_id": user_id}

    @app.get("/api/0.4/gdpr/dsar/jobs/{job_id}", dependencies=[Depends(_login)])
    async def dsar_job(job_id: str):
        return {"job_id": job_id}

    @app.get("/api/0.4/gdpr/audit-log", dependencies=[Depends(_login)])
    async def admin_log():
        return {"items": []}

    @app.get("/api/0.4/gdpr/jobs", dependencies=[Depends(_login)])
    async def jobs():
        return {}

    @app.get("/api/0.4/gdpr/jobs/summary", dependencies=[Depends(_login)])
    async def jobs_summary():
        return {}

    @app.get("/api/0.4/gdpr/retention/policy", dependencies=[Depends(_login)])
    async def retention():
        return []

    @app.post("/api/0.4/gdpr/restrict/lift", dependencies=[Depends(_login)])
    async def lift():
        return {"restricted": False}

    @app.post("/api/0.4/gdpr/admin/legal-hold", dependencies=[Depends(_login)])
    async def legal_hold():
        return {"id": 1}

    @app.get("/api/0.4/gdpr/admin/legal-holds", dependencies=[Depends(_login)])
    async def legal_holds():
        return {"holds": []}

    @app.post("/api/0.4/gdpr/rectify", dependencies=[Depends(_login)])
    async def rectify():
        return {"fields_updated": ["keycloak.firstName"]}

    @app.get("/api/0.4/benchmark/forbidden", dependencies=[Depends(_login)])
    async def forbidden():
        raise HTTPException(status_code=403, detail="no")

    @app.get("/api/0.4/memory/anonymous")
    async def anonymous():
        return {"ok": True}

    @app.get("/api/0.4/asset/files", dependencies=[Depends(_login)])
    async def assets():
        return []

    @app.get("/api/0.4/memory/slow", dependencies=[Depends(_login)])
    async def slow():
        return {"ok": True}

    @app.get("/api/0.4/moderation/flags", dependencies=[Depends(_login)])
    async def moderation_list():
        return {"items": []}

    @app.get("/api/0.4/moderation/flags/{flag_id}", dependencies=[Depends(_login)])
    async def moderation_get(flag_id: str):
        return {"id": flag_id}

    @app.patch("/api/0.4/moderation/flags/{flag_id}", dependencies=[Depends(_login)])
    async def moderation_patch(flag_id: str):
        return {"id": flag_id}

    @app.get("/api/0.4/compliance/report", dependencies=[Depends(_login)])
    async def compliance_report():
        return {"totals": {}}

    return app


@pytest.fixture
def client():
    return TestClient(_app())


# ── The record itself ───────────────────────────────────────────────────────

def test_self_access_writes_correct_record(client, records):
    r = client.get("/api/0.4/memory/stats", headers={"X-Forwarded-For": "203.0.113.7"})
    assert r.status_code == 200
    assert len(records) == 1
    rec = records[0]
    assert rec["user_id"] == SUB and rec["accessor_id"] == SUB
    assert rec["endpoint"] == "/api/0.4/memory/stats" and rec["method"] == "GET"
    assert rec["data_categories"] == ["memory"] and rec["purpose"] == "service_provision"
    assert rec["ip_hash"] == hashlib.sha256(b"203.0.113.7").hexdigest()
    assert rec["timestamp"]


def test_admin_access_takes_user_id_from_path(client, records):
    client.get("/api/0.4/gdpr/dsar/alice", headers={"x-test-sub": "dpo-1"})
    rec = records[0]
    assert (rec["user_id"], rec["accessor_id"]) == ("alice", "dpo-1")
    assert rec["data_categories"] == ALL_CATEGORIES and rec["purpose"] == "right_of_access"


def test_job_poll_path_is_not_mistaken_for_a_user_id(client, records):
    client.get("/api/0.4/gdpr/dsar/jobs/job-123")
    assert records[0]["user_id"] == SUB


def test_admin_log_filter_is_the_subject(client, records):
    client.get("/api/0.4/gdpr/audit-log", params={"user_id": "bob"}, headers={"x-test-sub": "dpo-1"})
    client.get("/api/0.4/gdpr/audit-log", headers={"x-test-sub": "dpo-1"})
    assert [r["user_id"] for r in records] == ["bob", None]
    assert records[0]["data_categories"] == ["audit_log"]


def test_lifting_a_restriction_names_no_subject_not_the_admin(client, records):
    """PRIV-X3: the user being lifted is in the body, which the middleware never reads."""
    client.post("/api/0.4/gdpr/restrict/lift", json={"user_id": "bob"}, headers={"x-test-sub": "dpo-1"})
    assert (records[0]["user_id"], records[0]["accessor_id"], records[0]["purpose"]) == (None, "dpo-1", "restriction_of_processing")


def test_legal_hold_routes_name_no_subject_not_the_admin(client, records):
    """PRIV-X4: the held user is in the body; the listing covers many users."""
    client.post("/api/0.4/gdpr/admin/legal-hold", json={"user_id": "bob"}, headers={"x-test-sub": "dpo-1"})
    client.get("/api/0.4/gdpr/admin/legal-holds", headers={"x-test-sub": "dpo-1"})
    assert [(r["user_id"], r["accessor_id"]) for r in records] == [(None, "dpo-1"), (None, "dpo-1")]


def test_rectification_is_the_callers_own_identity_data(client, records):
    client.post("/api/0.4/gdpr/rectify", json={"field": "firstName", "value": "x"})
    assert (records[0]["user_id"], records[0]["data_categories"], records[0]["purpose"]) == (SUB, ["identity"], "rectification")


def test_job_list_subject_is_the_filtered_user_and_summary_has_none(client, records):
    """V05-10: an admin's job list is about the user it's filtered on, if any."""
    client.get("/api/0.4/gdpr/jobs", params={"user_id": "bob"}, headers={"x-test-sub": "dpo-1"})
    client.get("/api/0.4/gdpr/jobs", headers={"x-test-sub": "dpo-1"})
    client.get("/api/0.4/gdpr/jobs/summary", headers={"x-test-sub": "dpo-1"})
    assert [r["user_id"] for r in records] == ["bob", None, None]
    assert records[0]["data_categories"] == ["gdpr_records"] and records[0]["purpose"] == "gdpr_administration"


def test_retention_policy_has_no_data_subject(client, records):
    client.get("/api/0.4/gdpr/retention/policy")
    assert records[0]["user_id"] is None and records[0]["data_categories"] == []


def test_moderation_review_is_recorded(client, records):
    """CM-5 / VIE01-408: an admin reviewing a flag reaches the actual uploaded asset
    (asset_url) -- previously invisible to this audit trail entirely."""
    client.get("/api/0.4/moderation/flags/flag-1", headers={"x-test-sub": "mod-admin-1"})
    rec = records[0]
    assert (rec["accessor_id"], rec["data_categories"], rec["purpose"]) == (
        "mod-admin-1", ["uploaded_files"], "content_moderation")
    assert rec["user_id"] is None  # branch_id isn't in the path -- see the filtered case below


def test_moderation_list_filtered_by_branch_id_names_that_subject(client, records):
    client.get("/api/0.4/moderation/flags", params={"branch_id": "branch-9"}, headers={"x-test-sub": "mod-admin-1"})
    client.get("/api/0.4/moderation/flags", headers={"x-test-sub": "mod-admin-1"})
    assert [r["user_id"] for r in records] == ["branch-9", None]


def test_moderation_status_update_is_recorded(client, records):
    client.patch("/api/0.4/moderation/flags/flag-1", headers={"x-test-sub": "mod-admin-1"})
    assert records[0]["method"] == "PATCH" and records[0]["purpose"] == "content_moderation"


def test_compliance_report_is_recorded_with_no_single_subject(client, records):
    """CM-7 / VIE01-410: the report covers many people's (hashed) records."""
    client.get("/api/0.4/compliance/report", params={"from": "2026-11-01", "to": "2026-11-30"},
               headers={"x-test-sub": "auditor-1"})
    rec = records[0]
    assert (rec["accessor_id"], rec["user_id"], rec["purpose"]) == ("auditor-1", None, "compliance_reporting")
    assert rec["data_categories"] == ["audit_log", "gdpr_records"]


@pytest.mark.parametrize("path,categories,purpose", [
    ("/api/0.4/personal-db/search/hybrid", ["indexed_content"], "service_provision"),
    ("/api/0.4/search/hybrid", ["indexed_content"], "service_provision"),
    ("/api/0.4/memory/stats", ["memory"], "service_provision"),
    ("/api/0.4/benchmark/runs", ["benchmark"], "service_provision"),
    ("/api/0.4/gdpr/dsar", ALL_CATEGORIES, "right_of_access"),
    ("/api/0.4/gdpr/export/alice", ALL_CATEGORIES, "data_portability"),
    ("/api/0.4/gdpr/rectify", ["identity"], "rectification"),
    ("/api/0.4/gdpr/restrict", [], "restriction_of_processing"),
    ("/api/0.4/gdpr/restrict/lift", [], "restriction_of_processing"),
    ("/api/0.4/gdpr/object", ["consent"], "objection"),
    ("/api/0.4/gdpr/erasure", ALL_CATEGORIES, "right_to_erasure"),
    ("/api/0.4/gdpr/consent/memory", ["consent"], "consent_management"),
    ("/api/0.4/gdpr/audit-log/my", ["audit_log"], "right_of_access"),
    ("/api/0.4/gdpr/jobs", ["gdpr_records"], "gdpr_administration"),
    ("/api/0.4/moderation/flags", ["uploaded_files"], "content_moderation"),
    ("/api/0.4/compliance/report", ["audit_log", "gdpr_records"], "compliance_reporting"),
])
def test_categories_and_purpose_inferred_from_path(path, categories, purpose):
    assert classify(path) == (categories, purpose)


# ── What is NOT recorded ────────────────────────────────────────────────────

def test_paths_outside_the_ticket_list_are_not_recorded(client, records):
    client.get("/api/0.4/asset/files")
    assert records == []


def test_unauthenticated_request_is_not_recorded(client, records):
    client.get("/api/0.4/memory/anonymous")
    assert records == []


def test_failed_request_is_not_recorded(client, records):
    assert client.get("/api/0.4/benchmark/forbidden").status_code == 403
    assert records == []


# ── Non-blocking ────────────────────────────────────────────────────────────

def test_write_never_delays_the_response(client, monkeypatch):
    finished: list = []

    async def _slow_write(record):
        await asyncio.sleep(1.0)
        finished.append(record)

    monkeypatch.setattr(audit.audit_log, "write", _slow_write)
    start = time.monotonic()
    assert client.get("/api/0.4/memory/slow").status_code == 200
    assert time.monotonic() - start < 0.5
    assert finished == []  # still running in the background


@pytest.mark.asyncio
async def test_write_failure_is_swallowed(monkeypatch):
    """Module 33 down: logged, never raised into the request."""
    monkeypatch.setattr(audit, "get_settings", lambda: type("S", (), {
        "gdpr_internal_token": "t", "privacy_compliance_url": "http://127.0.0.1:9"})())
    await audit.AuditLogClient().write({"method": "GET", "endpoint": "/x", "accessor_id": "a"})


# ── The real gateway: get_current_user -> request.state.user -> record ───────

from app.api import dependencies as deps
from app.api.dependencies import get_http_client
from app.core.config import get_settings
from app.main import app as gateway

GDPR = f"/api/{get_settings().api_version}/gdpr"


class _FakeClient:
    def __init__(self):
        self.calls: list = []

    async def request(self, method, url, headers=None, timeout=None, params=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "params": params})
        return httpx.Response(200, json={"total": 0, "items": []}, request=httpx.Request(method, url))


@pytest.fixture
def real_gateway(monkeypatch, records):
    """The real gateway app; only the token's signature/UMA checks are stubbed,
    so get_current_user itself runs. The test sets the caller's claims."""
    claims: dict = {"sub": SUB}

    async def _no_uma(token, path):
        return None

    monkeypatch.setattr(deps, "verify_jwt", lambda request, settings: dict(claims))
    monkeypatch.setattr(deps, "check_uma_permission", _no_uma)
    fake = _FakeClient()
    gateway.dependency_overrides[get_http_client] = lambda: fake
    yield TestClient(gateway), fake, claims
    gateway.dependency_overrides.pop(get_http_client, None)


def test_real_gateway_records_access_via_get_current_user(real_gateway, records):
    client, _fake, _ = real_gateway
    r = client.get(f"{GDPR}/audit-log/my", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert records and records[0]["accessor_id"] == SUB and records[0]["user_id"] == SUB
    assert records[0]["endpoint"] == f"{GDPR}/audit-log/my"


def test_my_audit_log_is_scoped_to_the_jwt_sub(real_gateway):
    client, fake, _ = real_gateway
    client.get(f"{GDPR}/audit-log/my", params={"user_id": "someone-else", "page": "2"},
               headers={"Authorization": "Bearer x", "X-User-ID": "someone-else"})
    call = fake.calls[0]
    assert call["url"].endswith("/audit-log/my")
    assert call["headers"]["X-User-ID"] == SUB
    assert call["params"] == {"page": "2"}  # a caller can't pick whose log to read


def test_admin_audit_log_requires_gdpr_admin(real_gateway):
    client, fake, _ = real_gateway
    r = client.get(f"{GDPR}/audit-log", headers={"Authorization": "Bearer x"})
    assert r.status_code == 403
    assert fake.calls == []


def test_admin_audit_log_forwards_filters(real_gateway):
    client, fake, claims = real_gateway
    claims["realm_access"] = {"roles": ["gdpr-admin"]}
    params = {"user_id": "alice", "from": "2026-01-01T00:00:00Z", "to": "2026-02-01T00:00:00Z",
              "page": "3", "page_size": "1000"}
    r = client.get(f"{GDPR}/audit-log", params=params, headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert fake.calls[0]["url"].endswith("/audit-log") and fake.calls[0]["params"] == params
