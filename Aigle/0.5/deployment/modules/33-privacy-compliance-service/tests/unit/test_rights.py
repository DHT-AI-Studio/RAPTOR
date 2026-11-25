"""
PRIV-X3 / VIE01-357: unit tests for rectification, restriction of processing
and objection (services/rights_service.py, routers/rights.py, and
ConsentStore.stop_processing / update_user).

Mocked -- no live Keycloak/Postgres/Redis. The fakes keep the behaviour the
code relies on: Keycloak can silently drop unmanaged attributes, a Postgres
transaction only keeps its writes if the block exits cleanly, and at most one
open restriction per user exists (the real table's partial unique index).

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_rights.py -v
"""
from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone

import asyncpg
import pytest
from app.core.config import settings
from app.models.consent import ConsentPurposeId
from app.models.rights import RectifiableField
from app.services import consent_store as consent_mod
from app.services.consent_store import ConsentStore, ConsentStoreError
from app.services.rights_service import RightsError, RightsService
from fastapi.testclient import TestClient
from keycloak.exceptions import KeycloakGetError, KeycloakPutError

USER = "user-123"
ADMIN = "dpo-1"


# ── Fakes ───────────────────────────────────────────────────────────────────

class _FakeKeycloakAdmin:
    def __init__(self, connection=None):
        self.users: dict = {}
        self.drop_unmanaged = False
        self.conflict = False
        # Email verification required, and a mail server to send the link.
        self.realm: dict = {"verifyEmail": True, "smtpServer": {"host": "smtp.example.com", "from": "no-reply@example.com"}}
        self.realm_error = None

    async def a_get_realm(self, realm_name):
        if self.realm_error:
            raise self.realm_error
        return copy.deepcopy(self.realm)

    async def a_get_user(self, user_id):
        if user_id not in self.users:
            raise KeycloakGetError(error_message="User not found", response_code=404)
        return copy.deepcopy(self.users[user_id])

    async def a_update_user(self, user_id, payload):
        if self.conflict:
            raise KeycloakPutError(error_message="User exists with same email", response_code=409)
        stored = copy.deepcopy(payload)
        if self.drop_unmanaged:
            stored["attributes"] = self.users[user_id].get("attributes", {})
        self.users[user_id] = stored


class _ConsentConn:
    """Just what ConsentStore._write needs: the history insert, in a transaction."""

    def __init__(self, db):
        self.db = db

    def transaction(self):
        class _Tx:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return False

        return _Tx()

    async def executemany(self, sql, rows):
        for user_id, purpose, action, ip_hash, version, source in rows:
            self.db.history.append({"user_id": user_id, "purpose": purpose, "action": action,
                                    "ip_hash": ip_hash, "source": source})

    async def execute(self, sql, *args):
        self.db.jobs.append(sql)  # any job insert would be a cascade -- objection must create none
        return "OK"


class _RightsConn:
    """gdpr_restriction_log / gdpr_objection_log, understanding exactly the
    statements rights_service.py issues."""

    def __init__(self, db):
        self.db = db

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self):
                conn.snapshot = copy.deepcopy((conn.db.restrictions, conn.db.objections))

            async def __aexit__(self, exc_type, *exc):
                if exc_type is not None:  # roll back
                    conn.db.restrictions, conn.db.objections = conn.snapshot
                return False

        return _Tx()

    async def fetchrow(self, sql, *args):
        if sql.lstrip().startswith("SELECT"):
            rows = [r for r in self.db.restrictions if r["user_id"] == args[0]]
            return rows[-1] if rows else None
        if sql.lstrip().startswith("INSERT INTO gdpr_restriction_log"):
            user_id, reason = args
            if any(r["user_id"] == user_id and r["lifted_at"] is None for r in self.db.restrictions):
                raise asyncpg.UniqueViolationError("idx_gdpr_restriction_log_open")
            row = {"user_id": user_id, "reason": reason,
                   "restricted_at": datetime.now(timezone.utc), "lifted_at": None}
            self.db.restrictions.append(row)
            return row
        if sql.lstrip().startswith("UPDATE gdpr_restriction_log"):
            (user_id,) = args
            for row in self.db.restrictions:
                if row["user_id"] == user_id and row["lifted_at"] is None:
                    row.update(lifted_at=datetime.now(timezone.utc))
                    return row
            return None
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        assert "INSERT INTO gdpr_objection_log" in sql
        user_id, processing_purpose = args
        self.db.objections.append({"user_id": user_id, "processing_purpose": processing_purpose})
        return "OK"


class _Pool:
    def __init__(self, conn_cls):
        self.conn_cls = conn_cls
        self.history: list = []
        self.jobs: list = []
        self.restrictions: list = []
        self.objections: list = []

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn_cls(pool)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class _FakeRedis:
    def __init__(self):
        self.data: dict = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def delete(self, key):
        self.data.pop(key, None)


class _Audit:
    """Collects the CF-4 events instead of writing them."""

    def __init__(self):
        self.events: list = []

    async def emit(self, event, *, fail_closed=False):
        self.events.append(event)


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.setattr(consent_mod, "KeycloakAdmin", _FakeKeycloakAdmin)
    monkeypatch.setattr(consent_mod, "KeycloakOpenIDConnection", lambda **kw: None)
    store = ConsentStore(pool=_Pool(_ConsentConn), redis=_FakeRedis(), orchestrator=None)
    store.admin.users[USER] = {
        "id": USER, "username": "alice", "email": "alice@example.com", "firstName": "Alice", "lastName": "Liu",
        "attributes": {"gdpr_consent_memory": ["true"], "gdpr_consent_analytics": ["false"],
                       "gdpr_consent_benchmark": ["true"], "gdpr_consent_voice_recording": ["true"],
                       "gdpr_consent_face_detection": ["true"]},
    }
    store.audit = _Audit()
    rights = RightsService(pool=_Pool(_RightsConn), consent_store=store)
    rights.audit = _Audit()
    return rights


def _user(svc):
    return svc.consent_store.admin.users[USER]


# ── Rectification (Art. 16) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rectify_updates_one_field_and_keeps_the_rest(svc):
    resp = await svc.rectify(USER, RectifiableField.FIRST_NAME, "Alicia")
    assert resp.model_dump() == {"fields_updated": ["keycloak.firstName"], "propagation_status": "complete"}
    user = _user(svc)
    assert user["firstName"] == "Alicia"
    # The whole user was PUT back -- nothing else lost.
    assert (user["lastName"], user["email"], user["attributes"]["gdpr_consent_memory"]) == ("Liu", "alice@example.com", ["true"])


@pytest.mark.asyncio
async def test_rectify_email_marks_it_unverified(svc):
    resp = await svc.rectify(USER, RectifiableField.EMAIL, "  Alice.New@Example.com ")
    assert resp.fields_updated == ["keycloak.email"]
    assert _user(svc)["email"] == "Alice.New@Example.com"
    assert _user(svc)["emailVerified"] is False


@pytest.mark.asyncio
async def test_rectify_rejects_an_invalid_email_without_touching_keycloak(svc):
    with pytest.raises(RightsError) as err:
        await svc.rectify(USER, RectifiableField.EMAIL, "not-an-email")
    assert err.value.status_code == 422
    assert _user(svc)["email"] == "alice@example.com"


@pytest.mark.asyncio
@pytest.mark.parametrize("smtp", [{}, None, {"host": ""}, {"host": "  ", "from": "x@example.com"}])
async def test_rectify_email_refused_when_the_realm_cannot_send_the_verification_link(svc, smtp):
    # The live realm: verifyEmail on, no mail server -- an unverified email
    # would lock the user out at their next login.
    svc.consent_store.admin.realm = {"verifyEmail": True, "smtpServer": smtp}
    with pytest.raises(RightsError) as err:
        await svc.rectify(USER, RectifiableField.EMAIL, "alice.new@example.com")
    assert err.value.status_code == 409 and "mail server" in err.value.detail
    assert _user(svc)["email"] == "alice@example.com" and "emailVerified" not in _user(svc)
    assert svc.audit.events == []                     # rejected by the rules: nothing changed


@pytest.mark.asyncio
async def test_rectify_email_allowed_when_the_realm_does_not_require_verification(svc):
    svc.consent_store.admin.realm = {"verifyEmail": False, "smtpServer": {}}
    await svc.rectify(USER, RectifiableField.EMAIL, "alice.new@example.com")
    assert _user(svc)["email"] == "alice.new@example.com"


@pytest.mark.asyncio
async def test_rectify_email_refused_when_the_realm_cannot_be_read(svc):
    svc.consent_store.admin.realm_error = KeycloakGetError(error_message="Forbidden", response_code=403)
    with pytest.raises(RightsError) as err:
        await svc.rectify(USER, RectifiableField.EMAIL, "alice.new@example.com")
    assert err.value.status_code == 503
    assert _user(svc)["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_rectify_names_never_needs_the_mail_server(svc):
    svc.consent_store.admin.realm = {"verifyEmail": True, "smtpServer": {}}
    svc.consent_store.admin.realm_error = KeycloakGetError(error_message="down", response_code=503)
    await svc.rectify(USER, RectifiableField.FIRST_NAME, "Alicia")
    await svc.rectify(USER, RectifiableField.LAST_NAME, "Lin")
    assert (_user(svc)["firstName"], _user(svc)["lastName"]) == ("Alicia", "Lin")


@pytest.mark.asyncio
async def test_rectify_fails_loudly_if_keycloak_keeps_the_old_value(svc, monkeypatch):
    async def ignore(user_id, payload):
        return None
    monkeypatch.setattr(svc.consent_store.admin, "a_update_user", ignore)
    with pytest.raises(ConsentStoreError):
        await svc.rectify(USER, RectifiableField.LAST_NAME, "Lin")


# ── Restriction (Art. 18) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_restrict_sets_the_keycloak_attribute_and_log(svc):
    assert await svc.is_restricted(USER) is False
    status = await svc.restrict(USER, "disputed accuracy")
    assert status.restricted is True and status.reason == "disputed accuracy"
    assert _user(svc)["attributes"]["gdpr_processing_restricted"] == ["true"]
    assert await svc.is_restricted(USER) is True
    assert [(r["user_id"], r["reason"], r["lifted_at"]) for r in svc.pool.restrictions] == [
        (USER, "disputed accuracy", None)]


@pytest.mark.asyncio
async def test_restrict_is_idempotent_and_repairs_a_removed_attribute(svc):
    first = await svc.restrict(USER, None)
    del _user(svc)["attributes"]["gdpr_processing_restricted"]  # e.g. removed by hand in Keycloak
    second = await svc.restrict(USER, "again")
    assert second.restricted_at == first.restricted_at
    assert len(svc.pool.restrictions) == 1
    assert await svc.is_restricted(USER) is True


@pytest.mark.asyncio
async def test_restrict_rolls_back_when_keycloak_drops_the_attribute(svc):
    svc.consent_store.admin.drop_unmanaged = True
    with pytest.raises(ConsentStoreError):
        await svc.restrict(USER, None)
    assert svc.pool.restrictions == []  # no legal record of a restriction that didn't happen


@pytest.mark.asyncio
async def test_lift_removes_the_attribute_and_keeps_the_record(svc):
    await svc.restrict(USER, "disputed accuracy")
    status = await svc.lift(USER, ADMIN)
    assert status.restricted is False and status.lifted_at is not None
    assert "gdpr_processing_restricted" not in _user(svc)["attributes"]
    assert await svc.is_restricted(USER) is False
    assert set(svc.pool.restrictions[0]) == {"user_id", "reason", "restricted_at", "lifted_at"}
    # Restricting again opens a new row; the old one stays as history.
    await svc.restrict(USER, None)
    assert len(svc.pool.restrictions) == 2


@pytest.mark.asyncio
async def test_lift_when_not_restricted_is_404(svc):
    with pytest.raises(RightsError) as err:
        await svc.lift(USER, ADMIN)
    assert err.value.status_code == 404


# ── Objection (Art. 21) ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_object_withdraws_consent_without_deleting_anything(svc):
    resp = await svc.object_to(USER, "memory", "a" * 64)
    assert resp.model_dump() == {"purposes_stopped": [ConsentPurposeId.MEMORY], "data_deleted": False}
    assert _user(svc)["attributes"]["gdpr_consent_memory"] == ["false"]
    history = svc.consent_store.pool.history
    assert [(h["purpose"], h["action"], h["source"], h["ip_hash"]) for h in history] == [
        ("memory", "withdraw", "objection", "a" * 64)]
    assert svc.consent_store.pool.jobs == []  # no GD-5 partial-erasure cascade
    assert svc.pool.objections == [{"user_id": USER, "processing_purpose": "memory"}]


@pytest.mark.asyncio
async def test_object_to_voice_also_sets_the_module_34_flag(svc):
    await svc.object_to(USER, "voice_recording", None)
    assert f"gdpr:voice_recording_disabled:{USER}" in svc.consent_store.redis.data


@pytest.mark.asyncio
async def test_object_when_already_withdrawn_is_still_logged(svc):
    await svc.object_to(USER, "analytics", None)
    assert svc.pool.objections == [{"user_id": USER, "processing_purpose": "analytics"}]
    assert svc.consent_store.pool.history == []  # nothing changed in Keycloak


@pytest.mark.asyncio
async def test_object_to_face_detection_withdraws_it(svc):
    """MTR-24 made face_detection a consent purpose, so it can be objected to (People Search stops)."""
    resp = await svc.object_to(USER, "face_detection", None)
    assert resp.model_dump() == {"purposes_stopped": [ConsentPurposeId.FACE_DETECTION], "data_deleted": False}
    assert _user(svc)["attributes"]["gdpr_consent_face_detection"] == ["false"]
    assert svc.consent_store.pool.jobs == []


@pytest.mark.asyncio
async def test_object_to_unsupported_purpose_explains_why(svc):
    with pytest.raises(RightsError) as err:
        await svc.object_to(USER, "speaker_cloning", None)
    assert err.value.status_code == 422 and "isn't supported yet" in err.value.detail
    assert svc.pool.objections == []


@pytest.mark.asyncio
async def test_object_to_unknown_purpose_lists_the_valid_ones(svc):
    with pytest.raises(RightsError) as err:
        await svc.object_to(USER, "marketing", None)
    assert err.value.status_code == 422 and "memory, analytics, benchmark, voice_recording" in err.value.detail


# ── V05-10 / VIE01-364: CF-4 audit events ────────────────────────────────────

def _events(svc):
    return [(e.event_type, e.metadata["action"], e.outcome) for e in svc.audit.events]


@pytest.mark.asyncio
async def test_rectify_event_names_the_field_never_the_value(svc):
    await svc.rectify(USER, RectifiableField.LAST_NAME, "Lin")
    [event] = svc.audit.events
    assert (event.event_type, event.outcome, event.metadata["field"]) == ("gdpr.rectification", "allow", "lastName")
    assert "Lin" not in event.model_dump_json() and USER not in event.model_dump_json()


@pytest.mark.asyncio
async def test_rectify_failure_is_an_error_event_and_a_rejected_value_none(svc):
    with pytest.raises(RightsError):
        await svc.rectify(USER, RectifiableField.EMAIL, "not-an-email")
    assert svc.audit.events == []                     # rejected by the rules: nothing changed
    svc.consent_store.admin.conflict = True
    with pytest.raises(KeycloakPutError):
        await svc.rectify(USER, RectifiableField.EMAIL, "taken@example.com")
    assert _events(svc) == [("gdpr.rectification", "rectify", "error")]


@pytest.mark.asyncio
async def test_restrict_and_lift_events_name_who_lifted(svc):
    await svc.restrict(USER, "disputed accuracy")
    await svc.restrict(USER, None)                    # repeat: still carried out
    await svc.lift(USER, ADMIN)
    with pytest.raises(RightsError):
        await svc.lift(USER, ADMIN)                   # not restricted: 404, no event
    assert _events(svc) == [("gdpr.restriction", "restrict", "allow"), ("gdpr.restriction", "restrict", "allow"),
                            ("gdpr.restriction", "lift", "allow")]
    first, repeat, lift = svc.audit.events
    assert "disputed accuracy" not in first.model_dump_json()   # free-text reason never stored
    assert repeat.metadata["already_restricted"] is True
    assert lift.metadata["requested_by_hash"] == hashlib.sha256(ADMIN.encode()).hexdigest()


@pytest.mark.asyncio
async def test_restrict_failure_is_an_error_event(svc):
    svc.consent_store.admin.drop_unmanaged = True
    with pytest.raises(ConsentStoreError):
        await svc.restrict(USER, None)
    assert _events(svc) == [("gdpr.restriction", "restrict", "error")]


@pytest.mark.asyncio
async def test_object_event_names_the_purpose(svc):
    await svc.object_to(USER, "analytics", None)
    [event] = svc.audit.events
    assert (event.event_type, event.outcome, event.metadata["purpose"]) == ("gdpr.objection", "allow", "analytics")
    assert svc.consent_store.audit.events == []       # an objection is not a consent change event


# ── Routes ──────────────────────────────────────────────────────────────────

@pytest.fixture
def api(svc):
    from app.main import app
    app.state.rights_service = svc
    return TestClient(app, headers={"X-Internal-Token": settings.internal_token})


def test_routes_require_internal_token(svc):
    from app.main import app
    app.state.rights_service = svc
    bare = TestClient(app)
    for path in ("/rectify", "/restrict", "/restrict/lift", "/object"):
        assert bare.post(path, json={}, headers={"X-User-ID": USER}).status_code == 401
    assert bare.get("/restrict/status", headers={"X-User-ID": USER}).status_code == 401


def test_rectify_route(api, svc):
    r = api.post("/rectify", headers={"X-User-ID": USER}, json={"field": "lastName", "value": "Lin"})
    assert r.status_code == 200 and r.json()["fields_updated"] == ["keycloak.lastName"]
    assert api.post("/rectify", headers={"X-User-ID": USER},
                    json={"field": "display_name", "value": "x"}).status_code == 422
    assert api.post("/rectify", json={"field": "lastName", "value": "Lin"}).status_code == 400


def test_rectify_route_refuses_an_email_change_without_a_mail_server(api, svc):
    svc.consent_store.admin.realm = {"verifyEmail": True, "smtpServer": {}}
    r = api.post("/rectify", headers={"X-User-ID": USER}, json={"field": "email", "value": "alice.new@example.com"})
    assert r.status_code == 409 and "mail server" in r.json()["detail"]
    svc.consent_store.admin.realm_error = KeycloakGetError(error_message="down", response_code=503)
    r = api.post("/rectify", headers={"X-User-ID": USER}, json={"field": "email", "value": "alice.new@example.com"})
    assert r.status_code == 503


def test_rectify_route_maps_an_email_taken_by_another_account_to_409(api, svc):
    svc.consent_store.admin.conflict = True
    r = api.post("/rectify", headers={"X-User-ID": USER}, json={"field": "email", "value": "bob@example.com"})
    assert r.status_code == 409


def test_restrict_and_lift_routes(api, svc):
    assert api.get("/restrict/status", headers={"X-User-ID": USER}).json() == {"restricted": False}
    r = api.post("/restrict", headers={"X-User-ID": USER})  # no body: reason optional
    assert r.status_code == 200 and r.json()["restricted"] is True
    assert api.get("/restrict/status", headers={"X-User-ID": USER}).json() == {"restricted": True}
    assert api.get("/restrict/status").status_code == 400
    assert api.post("/restrict/lift", json={"user_id": USER}).status_code == 400  # who lifted it is required
    r = api.post("/restrict/lift", headers={"X-Requested-By": ADMIN}, json={"user_id": USER})
    assert r.status_code == 200 and r.json()["restricted"] is False
    assert api.get("/restrict/status", headers={"X-User-ID": USER}).json() == {"restricted": False}
    assert api.post("/restrict/lift", headers={"X-Requested-By": ADMIN}, json={"user_id": USER}).status_code == 404


def test_object_route_hashes_the_forwarded_ip(api, svc):
    r = api.post("/object", headers={"X-User-ID": USER, "X-Forwarded-For": "203.0.113.7"},
                 json={"processing_purpose": "benchmark"})
    assert r.status_code == 200 and r.json() == {"purposes_stopped": ["benchmark"], "data_deleted": False}
    assert svc.consent_store.pool.history[0]["ip_hash"] == hashlib.sha256(b"203.0.113.7").hexdigest()
    assert api.post("/object", headers={"X-User-ID": USER},
                    json={"processing_purpose": "speaker_cloning"}).status_code == 422
