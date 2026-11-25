"""
GD-4 / VIE01-249: unit tests for the DSAR assembler.

Mocked -- no live Postgres/Keycloak/Modules 04/25/26 needed. Every store
fetcher is replaced per test, so these exercise the assembler's own logic:
the report's exact shape, the >10k-items async rule, the 10-second bound's
overrun handoff, partial-failure reporting, and job persistence.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_dsar_assembler.py -v
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from app.core.config import settings
from app.models.dsar import DSARPendingResponse, DSARReport
from app.services import dsar_assembler as mod
from app.services.dsar_assembler import DSARAssembler, indexed_items, retention_label
from raptor_audit import hash_value

USER = "user-123"


# ── Pure helpers ────────────────────────────────────────────────────────────

def test_indexed_items_uses_internal_field_not_document_count():
    """20 documents split into 50k chunks is a large account -- the
    threshold must not be fooled by the small document count."""
    assert indexed_items({"documents": 20, "moments": 0, "entities": 0, "indexed_items": 50000}) == 50000
    assert indexed_items({"documents": 20, "moments": 5, "entities": 5}) == 0  # field absent
    assert indexed_items({}) == 0


def test_retention_label():
    assert retention_label({"retention_days": 365, "auto_delete": True}) == "365 days"
    # Legal-hold / operator-disabled row: not actually enforced.
    assert retention_label({"retention_days": 90, "auto_delete": False}) == "indefinite"
    assert retention_label(None) == "indefinite"


# ── Fakes ───────────────────────────────────────────────────────────────────

_AUDIT_COLUMNS = ("event_id", "timestamp", "source_module", "source_service", "event_type", "actor_id_hash",
                  "resource_id", "outcome", "policy_version", "content_hash", "metadata", "retention_days")


def _audit_events(pool) -> list[dict]:
    """Every executed statement must be an audit_events INSERT; returns them as dicts."""
    events = []
    for sql, args in pool.executed:
        assert "INSERT INTO audit_events" in sql, sql
        event = dict(zip(_AUDIT_COLUMNS, args))
        event["metadata"] = json.loads(event["metadata"])
        events.append(event)
    return events


class _FakeConn:
    def __init__(self, db):
        self.db = db

    async def execute(self, sql, *args):
        self.db.executed.append((sql, args))
        if sql.lstrip().startswith("UPDATE gdpr_erasure_jobs SET result = NULL"):
            return f"UPDATE {self.db.cleanup_count}"
        return "OK"

    async def fetchrow(self, sql, *args):
        return self.db.row


class _FakePool:
    def __init__(self):
        self.executed: list = []
        self.row = None
        self.cleanup_count = 0

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _FakeConn(pool)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _patch_stores(monkeypatch, assembler, *, m25=None, m25_delay=0.0, slow=0.0, fail=()):
    m25 = m25 if m25 is not None else {"documents": 3, "moments": 2, "entities": 5, "pii_flagged": 1, "indexed_items": 30}

    def _maybe_fail(name):
        if name in fail:
            raise RuntimeError(f"{name} down")

    async def m25_fetch(user_id):
        await asyncio.sleep(m25_delay)
        _maybe_fail("module_25")
        return m25

    async def identity(user_id):
        await asyncio.sleep(slow)
        _maybe_fail("module_06")
        return {"email": "a@b.c", "username": "alice", "created_at": "2026-01-01T00:00:00+00:00",
                "consent": {"memory": True, "analytics": False, "benchmark": True, "voice_recording": False,
                            "face_detection": False, "voice_clone": False}}

    async def files(user_id):
        _maybe_fail("module_04")
        return {"count": 7, "file_types": ["mp4", "pdf"]}

    async def m26(user_id):
        _maybe_fail("module_26")
        return {"sessions": 4, "long_term_facts": 9, "oldest_session": "2026-02-01T00:00:00+00:00"}

    async def benchmark(user_id):
        _maybe_fail("module_03")
        return {"runs": 2, "schemas_owned": 1}

    async def moderation_flags(user_id):
        _maybe_fail("module_23")
        return {"count": 6}

    async def policies():
        _maybe_fail("retention")
        return {
            "memvid_sessions": {"data_category": "memvid_sessions", "retention_days": 365, "auto_delete": True},
            "moderation_flags": {"data_category": "moderation_flags", "retention_days": 90, "auto_delete": True},
            "moderation_flags_actionable": {
                "data_category": "moderation_flags_actionable", "retention_days": 1095, "auto_delete": True,
            },
        }

    monkeypatch.setattr(assembler, "_fetch_module_25", m25_fetch)
    monkeypatch.setattr(assembler, "_fetch_identity", identity)
    monkeypatch.setattr(assembler, "_fetch_uploaded_files", files)
    monkeypatch.setattr(assembler, "_fetch_module_26", m26)
    monkeypatch.setattr(assembler, "_fetch_benchmark", benchmark)
    monkeypatch.setattr(assembler, "_fetch_moderation_flags", moderation_flags)
    monkeypatch.setattr(assembler, "_fetch_retention_policies", policies)


@pytest.fixture
def pool():
    return _FakePool()


@pytest.fixture
def assembler(pool):
    return DSARAssembler(pool=pool)


async def _drain(assembler):
    while assembler._tasks:
        await asyncio.gather(*list(assembler._tasks), return_exceptions=True)


# ── Synchronous path ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_report_matches_ticket_structure(monkeypatch, assembler, pool):
    _patch_stores(monkeypatch, assembler)
    report = await assembler.request(USER, USER)
    assert isinstance(report, DSARReport)

    body = report.model_dump(mode="json")
    assert set(body) >= {
        "user_id", "generated_at", "data_categories", "consent_status", "retention_summary",
        "ccpa_category", "ccpa_source", "ccpa_sells_data", "rights",
    }
    cats = body["data_categories"]
    assert cats["identity"] == {"email": "a@b.c", "username": "alice", "created_at": "2026-01-01T00:00:00+00:00"}
    assert cats["uploaded_files"] == {"count": 7, "total_bytes": None, "file_types": ["mp4", "pdf"]}
    assert cats["indexed_content"] == {"documents": 3, "entities": 5, "moments": 2, "pii_flagged": 1}
    assert cats["memory"] == {"sessions": 4, "long_term_facts": 9, "oldest_session": "2026-02-01T00:00:00+00:00"}
    assert cats["benchmark"] == {"schemas_owned": 1, "runs": 2}
    assert cats["safety_violations"] == {"count": 6}
    assert body["consent_status"] == {"memory": True, "analytics": False, "benchmark": True, "voice_recording": False,
                                      "face_detection": False, "voice_clone": False}
    assert body["retention_summary"] == {
        "files": "indefinite", "memory": "365 days",
        "violations": "90 days (cleared) / 1095 days (flagged or removed)",
    }
    assert body["ccpa_category"] == "Identifiers, Commercial information, Internet activity"
    assert body["ccpa_source"] == "Directly from consumer"
    assert body["ccpa_sells_data"] is False
    v = settings.api_version
    assert body["rights"] == {
        "erasure_endpoint": f"/api/{v}/gdpr/erasure",
        "export_endpoint": f"/api/{v}/gdpr/export",
        "consent_endpoint": f"/api/{v}/gdpr/consent",
    }
    assert body["stores_failed"] == []
    # Every null the report carries is explained, not silently zeroed.
    assert set(body["unavailable_fields"]) == {"data_categories.uploaded_files.total_bytes"}
    # A synchronous DSAR persists no job -- only its V05-10 `gdpr.access`
    # audit event, which carries the user only as a hash.
    [event] = _audit_events(pool)
    assert event["event_type"] == "gdpr.access" and event["outcome"] == "allow"
    assert event["actor_id_hash"] == hash_value(USER) and USER not in json.dumps(event, default=str)
    assert event["resource_id"] is None and event["metadata"]["async"] is False


@pytest.mark.asyncio
async def test_failed_store_leaves_nulls_and_is_named(monkeypatch, assembler):
    _patch_stores(monkeypatch, assembler, fail=("module_26", "module_06"))
    report = await assembler.request(USER, USER)
    assert isinstance(report, DSARReport)
    assert sorted(report.stores_failed) == ["module_06", "module_26"]
    assert report.data_categories.memory.sessions is None
    assert report.data_categories.identity.email is None
    # No Keycloak profile read, so no consent to report -- null, not defaults.
    assert report.consent_status.model_dump() == {
        "memory": None, "analytics": None, "benchmark": None, "voice_recording": None, "face_detection": None,
        "voice_clone": None,
    }
    # Unaffected stores still report.
    assert report.data_categories.indexed_content.documents == 3


@pytest.mark.asyncio
async def test_module_25_failure_still_answers_synchronously(monkeypatch, assembler, pool):
    """Without Module 25's count there's no >10k signal -- the DSAR is still
    answered (within budget) rather than refused or forced async."""
    _patch_stores(monkeypatch, assembler, fail=("module_25",))
    report = await assembler.request(USER, USER)
    assert isinstance(report, DSARReport)
    assert report.stores_failed == ["module_25"]
    assert report.data_categories.indexed_content.documents is None
    [event] = _audit_events(pool)   # no job row; a partial answer is an `error` event
    assert event["outcome"] == "error" and event["metadata"]["stores_failed"] == ["module_25"]


@pytest.mark.asyncio
async def test_moderation_flags_failure_leaves_safety_violations_null(monkeypatch, assembler):
    """CM-5 / VIE01-408: module_23 (moderation_flags) is fetched and scored
    like every other store -- a failure there nulls just that field, named
    in stores_failed, same as any other per-store failure."""
    _patch_stores(monkeypatch, assembler, fail=("module_23",))
    report = await assembler.request(USER, USER)
    assert "module_23" in report.stores_failed
    assert report.data_categories.safety_violations.count is None
    # Unaffected stores still report.
    assert report.data_categories.uploaded_files.count == 7


@pytest.mark.asyncio
async def test_retention_read_failure_reports_unknown(monkeypatch, assembler):
    _patch_stores(monkeypatch, assembler, fail=("retention",))
    report = await assembler.request(USER, USER)
    assert report.retention_summary.model_dump() == {"files": "unknown", "memory": "unknown", "violations": "unknown"}
    assert "retention" in report.stores_failed


# ── Async path ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exactly_threshold_items_stays_synchronous(monkeypatch, assembler):
    _patch_stores(monkeypatch, assembler, m25={"documents": 1, "moments": 0, "entities": 0, "indexed_items": settings.dsar_async_threshold})
    assert isinstance(await assembler.request(USER, USER), DSARReport)


@pytest.mark.asyncio
async def test_large_account_returns_pending_job_and_completes(monkeypatch, assembler, pool):
    # Few documents, many chunks: still the async path.
    _patch_stores(monkeypatch, assembler, m25={"documents": 12, "moments": 4000, "entities": 1, "indexed_items": 10001})
    resp = await assembler.request(USER, "admin-1")
    assert isinstance(resp, DSARPendingResponse)
    assert resp.status == "pending"

    insert_sql, insert_args = pool.executed[0]
    assert "INSERT INTO gdpr_erasure_jobs" in insert_sql and "'dsar'" in insert_sql
    assert insert_args[0] == resp.job_id
    assert insert_args[1] == USER and insert_args[3] == "admin-1" and insert_args[4] == "pending"

    await _drain(assembler)
    updates = [(sql, args) for sql, args in pool.executed if sql.startswith("UPDATE gdpr_erasure_jobs SET")]
    assert "in_progress" in updates[0][1]
    final_sql, final_args = updates[-1]
    assert "result" in final_sql and "expires_at" in final_sql
    assert "completed" in final_args
    stored = json.loads(next(a for a in final_args if isinstance(a, str) and a.startswith("{")))
    assert stored["user_id"] == USER
    assert stored["data_categories"]["indexed_content"]["documents"] == 12
    # Internal-only: never leaks into the report.
    assert "indexed_items" not in stored["data_categories"]["indexed_content"]
    # V05-10 / VIE01-364: the job, not the request, writes the one gdpr.access event.
    [event] = [dict(zip(_AUDIT_COLUMNS, args)) for sql, args in pool.executed if "INSERT INTO audit_events" in sql]
    assert (event["event_type"], event["outcome"], event["resource_id"]) == ("gdpr.access", "allow", resp.job_id)
    meta = json.loads(event["metadata"])
    assert meta["async"] is True and meta["requested_by_hash"] == hash_value("admin-1")


@pytest.mark.asyncio
async def test_large_account_with_store_failure_is_partial_failure(monkeypatch, assembler, pool):
    _patch_stores(monkeypatch, assembler, m25={"documents": 3, "moments": 0, "entities": 0, "indexed_items": 20000}, fail=("module_04",))
    resp = await assembler.request(USER, USER)
    assert isinstance(resp, DSARPendingResponse)
    await _drain(assembler)
    _, final_args = [e for e in pool.executed if e[0].startswith("UPDATE")][-1]
    assert "partial_failure" in final_args
    assert "stores_failed: module_04" in final_args


@pytest.mark.asyncio
async def test_sync_overrun_is_handed_off_not_abandoned(monkeypatch, assembler, pool):
    """A small account whose assembly overruns the sync budget still gets an
    answer inside it -- a pending job carrying on the SAME in-flight work."""
    monkeypatch.setattr(settings, "dsar_sync_timeout_seconds", 0.1)
    _patch_stores(monkeypatch, assembler, slow=0.3)
    loop = asyncio.get_running_loop()
    start = loop.time()
    resp = await assembler.request(USER, USER)
    assert loop.time() - start < 0.25
    assert isinstance(resp, DSARPendingResponse)

    await _drain(assembler)
    _, final_args = [e for e in pool.executed if e[0].startswith("UPDATE")][-1]
    assert "completed" in final_args


@pytest.mark.asyncio
async def test_module_25_overrun_is_handed_off(monkeypatch, assembler):
    monkeypatch.setattr(settings, "dsar_sync_timeout_seconds", 0.1)
    _patch_stores(monkeypatch, assembler, m25_delay=0.3)
    assert isinstance(await assembler.request(USER, USER), DSARPendingResponse)
    await _drain(assembler)


# ── Job lookup + cleanup ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_parses_stored_result(assembler, pool):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    pool.row = {
        "job_id": "11111111-1111-1111-1111-111111111111", "user_id": USER, "requested_at": now,
        "requested_by": USER, "status": "completed", "completed_at": now, "notes": None,
        "result": json.dumps({"user_id": USER}), "expires_at": now,
    }
    job = await assembler.get_job("11111111-1111-1111-1111-111111111111")
    assert job.result == {"user_id": USER}
    assert job.status.value == "completed"


@pytest.mark.asyncio
async def test_cleanup_clears_only_expired_dsar_results(assembler, pool):
    pool.cleanup_count = 3
    assert await assembler.cleanup_expired() == 3
    sql, _ = pool.executed[0]
    assert "SET result = NULL" in sql and "type = 'dsar'" in sql and "expires_at < now()" in sql
    assert "DELETE" not in sql  # the job row itself is a legal-hold record (GD-3)


def test_unavailable_fields_cover_every_structural_null():
    # consent_status, rights.consent_endpoint (GD-5), benchmark.schemas_owned
    # (GD-5's owner_id), and safety_violations.count (CM-5's moderation_flags,
    # VIE01-408) are backed now.
    assert set(mod.UNAVAILABLE_FIELDS) == {"data_categories.uploaded_files.total_bytes"}


# ── GD-5 consent, read-only ─────────────────────────────────────────────────

def _profile(**attrs):
    return {"attributes": {f"gdpr_consent_{k}": [v] for k, v in attrs.items()}}


def test_consent_from_profile_reads_gd5_attributes():
    profile = _profile(memory="false", analytics="true", benchmark="false", voice_recording="true",
                       face_detection="true", voice_clone="true")
    assert mod.consent_from_profile(profile) == {
        "memory": False, "analytics": True, "benchmark": False, "voice_recording": True, "face_detection": True,
        "voice_clone": True,
    }


def test_consent_never_set_reports_gd5_defaults():
    """An account from before GD-5: same values GET /gdpr/consent shows."""
    assert mod.consent_from_profile({}) == {
        "memory": True, "analytics": False, "benchmark": True, "voice_recording": False, "face_detection": False,
        "voice_clone": False,
    }
    assert mod.consent_from_profile(_profile(analytics="true")) == {
        "memory": True, "analytics": True, "benchmark": True, "voice_recording": False, "face_detection": False,
        "voice_clone": False,
    }


@pytest.mark.asyncio
async def test_fetch_identity_returns_consent_without_writing(monkeypatch, assembler):
    """The DSAR reads Keycloak only -- a GET on the user, never a PUT."""
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(200, json={
            "email": "a@b.c", "username": "alice", "createdTimestamp": 1767225600000,
            "attributes": {"gdpr_consent_memory": ["false"]},
        })

    assembler.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    identity = await assembler._fetch_identity(USER)
    assert identity["consent"] == {"memory": False, "analytics": False, "benchmark": True, "voice_recording": False,
                                   "face_detection": False, "voice_clone": False}
    assert [m for m, _ in calls] == ["POST", "GET"]  # admin token, then the user


@pytest.mark.asyncio
async def test_erased_user_reports_no_consent(monkeypatch, assembler):
    def handler(request):
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(404)

    assembler.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await assembler._fetch_identity(USER) == {}
