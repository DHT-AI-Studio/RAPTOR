"""
GD-5 / VIE01-250: unit tests for consent management (consent_store.py and
the routes in routers/consent.py).

Mocked -- no live Keycloak/Postgres/Redis/Modules 22/26 needed. The fakes
keep just enough real behaviour to test the store's own guarantees:
Keycloak keeps or (like Keycloak 24+ with unmanaged attributes disabled)
silently drops attributes, and the fake Postgres transaction only
"commits" history rows if the block exits cleanly.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_consent_store.py -v
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from app.core.config import settings
from app.models.consent import ConsentPurposeId
from app.services import consent_store as mod
from app.services.consent_store import (
    ConsentStore,
    ConsentStoreError,
    UserNotFound,
    hash_ip,
)
from fastapi.testclient import TestClient
from keycloak.exceptions import KeycloakGetError

USER = "user-123"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
ALL_SET = {
    "gdpr_consent_memory": ["true"], "gdpr_consent_analytics": ["false"],
    "gdpr_consent_benchmark": ["true"], "gdpr_consent_voice_recording": ["false"],
    "gdpr_consent_face_detection": ["false"], "gdpr_consent_voice_clone": ["false"],
    "gdpr_consent_version": ["1.0"],
}


# ── Fakes ───────────────────────────────────────────────────────────────────

class _FakeKeycloakAdmin:
    def __init__(self, connection=None):
        self.users: dict = {}
        self.updates: list = []
        self.drop_unmanaged = False

    async def a_get_user(self, user_id):
        if user_id not in self.users:
            raise KeycloakGetError(error_message="User not found", response_code=404)
        return copy.deepcopy(self.users[user_id])

    async def a_update_user(self, user_id, payload):
        self.updates.append(copy.deepcopy(payload))
        stored = copy.deepcopy(payload)
        if self.drop_unmanaged:
            stored["attributes"] = self.users[user_id].get("attributes", {})
        self.users[user_id] = stored


class _FakeConn:
    def __init__(self, db):
        self.db = db
        self.pending: list | None = None

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self):
                conn.pending = []

            async def __aexit__(self, exc_type, *exc):
                if exc_type is None:
                    conn.db.history.extend(conn.pending)
                conn.pending = None
                return False

        return _Tx()

    async def executemany(self, sql, rows):
        assert "gdpr_consent_history" in sql
        target = self.pending if self.pending is not None else self.db.history
        for user_id, purpose, action, ip_hash, version, source in rows:
            target.append({"user_id": user_id, "purpose": purpose, "action": action,
                           "ip_hash": ip_hash, "policy_version": version, "source": source})

    async def execute(self, sql, *args):
        if sql.lstrip().startswith("INSERT INTO gdpr_erasure_jobs"):
            job_id, user_id, _, status, store_statuses, job_type, notes = args
            self.db.jobs[job_id] = {"user_id": user_id, "status": status, "store_statuses": json.loads(store_statuses),
                                    "type": job_type, "notes": notes}
        elif sql.lstrip().startswith("UPDATE gdpr_erasure_jobs"):
            status, store_statuses, completed_at, job_id = args
            self.db.jobs[job_id].update(status=status, store_statuses=json.loads(store_statuses), completed_at=completed_at)
        return "OK"

    async def fetch(self, sql, *args):
        user_id, action = args
        latest: dict = {}
        for i, row in enumerate(self.db.history):
            if row["user_id"] == user_id and row["action"] == action:
                latest[row["purpose"]] = T0 + timedelta(seconds=i)
        return [{"purpose": p, "granted_at": t} for p, t in latest.items()]


class _FakePool:
    def __init__(self):
        self.history: list = []
        self.jobs: dict = {}

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _FakeConn(pool)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class _FakeRedis:
    def __init__(self):
        self.data: dict = {}

    async def set(self, key, value, ex=None):
        self.data[key] = (value, ex)

    async def delete(self, key):
        self.data.pop(key, None)


class _FakeAudit:
    def __init__(self):
        self.events: list = []

    async def emit(self, event, *, fail_closed=False):
        self.events.append((event, fail_closed))


class _FakeOrchestrator:
    def __init__(self):
        self.audit = _FakeAudit()

    async def _with_retry(self, fn, user_id):
        await fn(user_id)


class _FakeHttp:
    def __init__(self):
        self.calls: list = []
        self.status = 200

    async def delete(self, url, headers=None):
        self.calls.append(url)
        return httpx.Response(self.status, json={}, request=httpx.Request("DELETE", url))

    async def aclose(self):
        pass


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(mod, "KeycloakAdmin", _FakeKeycloakAdmin)
    monkeypatch.setattr(mod, "KeycloakOpenIDConnection", lambda **kw: None)
    s = ConsentStore(pool=_FakePool(), redis=_FakeRedis(), orchestrator=_FakeOrchestrator())
    s.http = _FakeHttp()
    s.audit = _FakeAudit()
    s.admin.users[USER] = {"id": USER, "username": "alice", "email": "a@x.io", "firstName": "Alice",
                           "attributes": dict(ALL_SET)}
    return s


async def _drain(store):
    await asyncio.gather(*list(store._tasks))


def _attr(store, name):
    return store.admin.users[USER]["attributes"].get(name)


# ── Pure helpers ────────────────────────────────────────────────────────────

def test_hash_ip_is_sha256_and_never_the_raw_ip():
    assert hash_ip("203.0.113.7") == hashlib.sha256(b"203.0.113.7").hexdigest()
    assert hash_ip(" 203.0.113.7, 10.0.0.1 ") == hashlib.sha256(b"203.0.113.7, 10.0.0.1").hexdigest()
    assert hash_ip(None) is None
    assert hash_ip("  ") is None


# ── GET ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_returns_every_purpose_in_ticket_order(store):
    store.pool.history.append({"user_id": USER, "purpose": "memory", "action": "grant"})
    resp = await store.get(USER)
    assert [p.id.value for p in resp.purposes] == ["memory", "analytics", "benchmark", "voice_recording",
                                                     "face_detection", "voice_clone"]
    by_id = {p.id.value: p for p in resp.purposes}
    assert by_id["memory"].granted is True and by_id["memory"].granted_at == T0
    assert by_id["analytics"].granted is False and by_id["analytics"].granted_at is None
    assert by_id["benchmark"].granted_at is None  # no grant on record
    assert all(p.description for p in resp.purposes)
    assert store.admin.updates == []  # a complete user is never written on read


@pytest.mark.asyncio
async def test_get_for_pre_gd5_user_writes_defaults_without_version(store):
    store.admin.users[USER]["attributes"] = {"locale": ["en"]}
    resp = await store.get(USER)
    assert {p.id.value: p.granted for p in resp.purposes} == {
        "memory": True, "analytics": False, "benchmark": True, "voice_recording": False, "face_detection": False,
        "voice_clone": False,
    }
    assert _attr(store, "gdpr_consent_memory") == ["true"]
    assert _attr(store, "locale") == ["en"]  # unrelated attributes survive
    # No version: the REVIEW_CONSENT action must still treat them as not
    # having seen the notice.
    assert _attr(store, "gdpr_consent_version") is None
    assert {r["source"] for r in store.pool.history} == {"first_read_default"}
    assert len(store.pool.history) == 6


@pytest.mark.asyncio
async def test_get_unknown_user(store):
    with pytest.raises(UserNotFound):
        await store.get("nobody")


# ── PUT ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_withdraw_memory_sets_attribute_records_history_and_erases_module_26_only(store):
    resp = await store.set(USER, ConsentPurposeId.MEMORY, False, ip_hash="h" * 64)
    await _drain(store)
    assert resp.changed is True and resp.granted is False
    assert _attr(store, "gdpr_consent_memory") == ["false"]
    assert store.pool.history == [{"user_id": USER, "purpose": "memory", "action": "withdraw", "ip_hash": "h" * 64,
                                   "policy_version": settings.consent_policy_version, "source": "user"}]
    job = store.pool.jobs[resp.erasure_job_id]
    assert job["type"] == "consent_withdrawal"
    assert job["status"] == "completed" and job["store_statuses"] == {"module_26": "deleted"}
    assert store.http.calls == [f"{settings.module_26_url}/internal/gdpr/memory/user/{USER}"]


@pytest.mark.asyncio
async def test_withdraw_benchmark_erases_module_22(store):
    resp = await store.set(USER, ConsentPurposeId.BENCHMARK, False, ip_hash=None)
    await _drain(store)
    assert store.http.calls == [f"{settings.module_22_url}/internal/gdpr/benchmark/user/{USER}"]
    assert store.pool.jobs[resp.erasure_job_id]["store_statuses"] == {"module_22": "deleted"}


@pytest.mark.asyncio
async def test_unchanged_value_records_nothing_and_erases_nothing(store):
    resp = await store.set(USER, ConsentPurposeId.ANALYTICS, False, ip_hash=None)  # already false
    assert resp.changed is False and resp.erasure_job_id is None
    assert store.pool.history == [] and store.admin.updates == []


@pytest.mark.asyncio
async def test_withdraw_analytics_has_no_cascade(store):
    await store.set(USER, ConsentPurposeId.ANALYTICS, True, ip_hash=None)
    resp = await store.set(USER, ConsentPurposeId.ANALYTICS, False, ip_hash=None)
    assert resp.changed is True and resp.erasure_job_id is None
    assert store.pool.jobs == {}


@pytest.mark.asyncio
async def test_grant_records_history_and_never_erases(store):
    resp = await store.set(USER, ConsentPurposeId.ANALYTICS, True, ip_hash=None)
    assert resp.changed is True and resp.erasure_job_id is None
    assert store.pool.history[0]["action"] == "grant"


@pytest.mark.asyncio
async def test_withdraw_voice_sets_redis_flag_and_skips_module_34_until_it_exists(store, monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "")
    await store.set(USER, ConsentPurposeId.VOICE_RECORDING, True, ip_hash=None)
    resp = await store.set(USER, ConsentPurposeId.VOICE_RECORDING, False, ip_hash=None)
    await _drain(store)
    assert _attr(store, "gdpr_consent_voice_recording") == ["false"]
    assert store.redis.data[f"gdpr:voice_recording_disabled:{USER}"] == ("1", 90 * 86400)
    job = store.pool.jobs[resp.erasure_job_id]
    assert job["status"] == "completed" and job["store_statuses"] == {"module_34": "skipped"}
    assert store.http.calls == []


@pytest.mark.asyncio
async def test_withdraw_voice_calls_module_34_once_configured(store, monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "http://raptor-realtime-audio:8034")
    await store.set(USER, ConsentPurposeId.VOICE_RECORDING, True, ip_hash=None)
    await store.set(USER, ConsentPurposeId.VOICE_RECORDING, False, ip_hash=None)
    await _drain(store)
    assert store.http.calls == [f"http://raptor-realtime-audio:8034/internal/gdpr/user/{USER}"]


@pytest.mark.asyncio
async def test_withdraw_face_detection_erases_module_25_face_layer_only(store, monkeypatch):
    """GD-11: the face-only purge, never GD-1's whole-database drop."""
    monkeypatch.setattr(settings, "module_25_url", "http://raptor-personal-db-service:8000")
    await store.set(USER, ConsentPurposeId.FACE_DETECTION, True, ip_hash=None)
    resp = await store.set(USER, ConsentPurposeId.FACE_DETECTION, False, ip_hash=None)
    await _drain(store)
    assert store.http.calls == [f"http://raptor-personal-db-service:8000/internal/gdpr/faces/user/{USER}"]
    job = store.pool.jobs[resp.erasure_job_id]
    assert job["status"] == "completed" and job["store_statuses"] == {"module_25": "deleted"}
    assert job["notes"] == "consent withdrawn: face_detection"


@pytest.mark.asyncio
async def test_withdraw_voice_clone_erases_module_35_speaker_embeddings(store, monkeypatch):
    monkeypatch.setattr(settings, "module_35_url", "http://raptor-inference-service:8035")
    await store.set(USER, ConsentPurposeId.VOICE_CLONE, True, ip_hash=None)
    resp = await store.set(USER, ConsentPurposeId.VOICE_CLONE, False, ip_hash=None)
    await _drain(store)
    assert store.http.calls == [f"http://raptor-inference-service:8035/internal/gdpr/user/{USER}"]
    job = store.pool.jobs[resp.erasure_job_id]
    assert job["status"] == "completed" and job["store_statuses"] == {"module_35": "deleted"}
    # voice_clone has no Redis flag of its own -- Module 35 asks Module 33 per request.
    assert f"gdpr:voice_recording_disabled:{USER}" not in store.redis.data


@pytest.mark.asyncio
async def test_withdraw_voice_clone_skips_module_35_when_not_deployed(store, monkeypatch):
    monkeypatch.setattr(settings, "module_35_url", "")
    await store.set(USER, ConsentPurposeId.VOICE_CLONE, True, ip_hash=None)
    resp = await store.set(USER, ConsentPurposeId.VOICE_CLONE, False, ip_hash=None)
    await _drain(store)
    assert store.http.calls == []
    job = store.pool.jobs[resp.erasure_job_id]
    assert job["status"] == "completed" and job["store_statuses"] == {"module_35": "skipped"}


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", [ConsentPurposeId.FACE_DETECTION, ConsentPurposeId.VOICE_CLONE])
async def test_biometric_purposes_default_to_withdrawn(store, purpose):
    """GDPR Art. 9: explicit opt-in -- registration never grants them."""
    store.admin.users[USER]["attributes"] = {}
    await store.set_registration_defaults(USER)
    assert _attr(store, f"gdpr_consent_{purpose.value}") == ["false"]
    assert store.pool.jobs == {}  # a default is not a withdrawal: nothing erased


@pytest.mark.asyncio
async def test_grant_voice_deletes_redis_flag(store):
    store.redis.data[f"gdpr:voice_recording_disabled:{USER}"] = ("1", 1)
    await store.set(USER, ConsentPurposeId.VOICE_RECORDING, True, ip_hash=None)
    assert f"gdpr:voice_recording_disabled:{USER}" not in store.redis.data
    assert _attr(store, "gdpr_consent_voice_recording") == ["true"]


@pytest.mark.asyncio
async def test_failed_cascade_marks_job_partial_failure(store):
    store.http.status = 503
    resp = await store.set(USER, ConsentPurposeId.MEMORY, False, ip_hash=None)
    await _drain(store)
    job = store.pool.jobs[resp.erasure_job_id]
    assert job["status"] == "partial_failure" and job["store_statuses"] == {"module_26": "failed"}
    # The withdrawal itself still stands -- only the deletion needs a retry.
    assert _attr(store, "gdpr_consent_memory") == ["false"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,outcome", [(200, "erase"), (503, "error")])
async def test_withdrawal_job_writes_a_gdpr_erasure_audit_event(store, status, outcome):
    """V05-10 / VIE01-364: one CF-4 event per finished withdrawal job."""
    store.http.status = status
    resp = await store.set(USER, ConsentPurposeId.MEMORY, False, ip_hash=None)
    await _drain(store)
    [(event, fail_closed)] = store.orchestrator.audit.events
    assert fail_closed is False
    assert event.event_type == "gdpr.erasure" and event.outcome == outcome
    assert event.resource_id == resp.erasure_job_id
    assert event.metadata["job_type"] == "consent_withdrawal" and event.metadata["purpose"] == "memory"
    assert USER not in event.model_dump_json()


@pytest.mark.asyncio
async def test_consent_change_writes_a_gdpr_consent_audit_event(store):
    """V05-10 / VIE01-364: one `gdpr.consent` event per request, its job (if any) as resource."""
    withdrawn = await store.set(USER, ConsentPurposeId.MEMORY, False, ip_hash="h")
    await store.set(USER, ConsentPurposeId.MEMORY, False, ip_hash="h")      # unchanged repeat
    await store.set(USER, ConsentPurposeId.ANALYTICS, True, ip_hash="h")
    await _drain(store)
    [(w, fc), (repeat, _), (grant, _)] = store.audit.events
    assert fc is False
    assert (w.event_type, w.outcome, w.metadata["action"], w.metadata["purpose"]) == (
        "gdpr.consent", "allow", "withdraw", "memory")
    assert w.resource_id == withdrawn.erasure_job_id and w.metadata["changed"] is True
    assert repeat.metadata["changed"] is False and repeat.resource_id is None
    assert (grant.metadata["action"], grant.metadata["purpose"]) == ("grant", "analytics")
    assert USER not in w.model_dump_json() and w.metadata["requested_by_subject"] is True


@pytest.mark.asyncio
async def test_failed_consent_change_is_an_error_event(store):
    store.admin.drop_unmanaged = True
    with pytest.raises(ConsentStoreError):
        await store.set(USER, ConsentPurposeId.ANALYTICS, True, ip_hash=None)
    [(event, _)] = store.audit.events
    assert (event.event_type, event.outcome, event.metadata["action"]) == ("gdpr.consent", "error", "grant")


@pytest.mark.asyncio
async def test_registration_defaults_write_no_consent_event(store):
    store.admin.users[USER]["attributes"] = {}
    await store.set_registration_defaults(USER)
    assert store.audit.events == []


@pytest.mark.asyncio
async def test_update_puts_full_user_back(store):
    """Keycloak 24+ clears profile fields a partial PUT omits."""
    await store.set(USER, ConsentPurposeId.ANALYTICS, True, ip_hash=None)
    payload = store.admin.updates[0]
    assert payload["email"] == "a@x.io" and payload["firstName"] == "Alice" and payload["username"] == "alice"


@pytest.mark.asyncio
async def test_dropped_attribute_raises_and_history_rolls_back(store):
    store.admin.drop_unmanaged = True
    with pytest.raises(ConsentStoreError, match="Unmanaged"):
        await store.set(USER, ConsentPurposeId.ANALYTICS, True, ip_hash=None)
    assert store.pool.history == []
    assert store.pool.jobs == {}


# ── Registration webhook ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registration_sets_ticket_defaults_and_version(store):
    store.admin.users[USER]["attributes"] = {}
    await store.set_registration_defaults(USER)
    attrs = store.admin.users[USER]["attributes"]
    assert attrs == {
        "gdpr_consent_memory": ["true"], "gdpr_consent_analytics": ["false"],
        "gdpr_consent_benchmark": ["true"], "gdpr_consent_voice_recording": ["false"],
        "gdpr_consent_face_detection": ["false"], "gdpr_consent_voice_clone": ["false"],
        "gdpr_consent_version": [settings.consent_policy_version],
    }
    assert [(r["purpose"], r["action"], r["source"]) for r in store.pool.history] == [
        ("memory", "grant", "registration_default"), ("analytics", "withdraw", "registration_default"),
        ("benchmark", "grant", "registration_default"), ("voice_recording", "withdraw", "registration_default"),
        ("face_detection", "withdraw", "registration_default"), ("voice_clone", "withdraw", "registration_default"),
    ]


@pytest.mark.asyncio
async def test_registration_is_idempotent_and_never_overrides_a_choice(store):
    store.admin.users[USER]["attributes"] = {"gdpr_consent_analytics": ["true"]}
    await store.set_registration_defaults(USER)
    await store.set_registration_defaults(USER)
    assert _attr(store, "gdpr_consent_analytics") == ["true"]
    assert len(store.pool.history) == 5  # analytics untouched; second call a no-op


# ── Routes ──────────────────────────────────────────────────────────────────

@pytest.fixture
def api(store):
    from app.main import app
    app.state.consent_store = store
    return TestClient(app, headers={"X-Internal-Token": settings.internal_token})


def test_route_requires_internal_token(store):
    from app.main import app
    app.state.consent_store = store
    assert TestClient(app).get("/consent", headers={"X-User-ID": USER}).status_code == 401


def test_route_get_and_missing_user_header(api):
    assert api.get("/consent", headers={"X-User-ID": USER}).status_code == 200
    assert api.get("/consent").status_code == 400
    assert api.get("/consent", headers={"X-User-ID": "nobody"}).status_code == 404


def test_route_put_hashes_forwarded_for_and_rejects_unknown_purpose(api, store):
    r = api.put("/consent/analytics", json={"granted": True},
                headers={"X-User-ID": USER, "X-Forwarded-For": "203.0.113.7"})
    assert r.status_code == 200 and r.json()["changed"] is True
    assert store.pool.history[0]["ip_hash"] == hashlib.sha256(b"203.0.113.7").hexdigest()
    assert api.put("/consent/nope", json={"granted": True}, headers={"X-User-ID": USER}).status_code == 422


def test_route_new_user(api, store):
    store.admin.users["new-1"] = {"id": "new-1", "attributes": {}}
    assert api.post("/internal/gdpr/new-user", json={"user_id": "new-1"}).status_code == 204
    assert store.admin.users["new-1"]["attributes"]["gdpr_consent_version"] == [settings.consent_policy_version]
