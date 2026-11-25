"""
GD-5 / VIE01-250: end-to-end consent management test against a real
deployment -- the ticket's acceptance criteria, each verified through a
DIFFERENT path than the one that made the change:

- consent attributes: read straight from Keycloak's Admin API
- consent history: read straight from privacy_compliance.gdpr_consent_history
- memory withdrawal: Module 26's own GET /memory/stats (header trust), and
  the consent_withdrawal job row in gdpr_erasure_jobs
- benchmark withdrawal: benchmark_schemas/benchmark_runs queried directly
- voice_recording: the Redis key and its TTL read straight from Redis
- registration defaults: a user registered through Module 06's own
  POST /SSO/register, then read back from Keycloak

Deliberately live-infrastructure, not mocked (tests/unit/test_consent_store.py
covers the logic with fakes). Requires the realm's "Unmanaged attributes"
policy to allow admin writes (06-authentication/README.md). Skips cleanly if
the gateway isn't reachable, matching the other *_e2e.py tests here.

Run from the 33-privacy-compliance-service/ directory:
    TEST_KEYCLOAK_ADMIN_PASSWORD=... TEST_POSTGRES_PASSWORD=... [TEST_REDIS_PASSWORD=...] \
        python -m pytest tests/test_consent_e2e.py -v
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid

import asyncpg
import httpx
import pytest
import pytest_asyncio
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

GATEWAY_URL = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
KEYCLOAK_API_URL = os.environ.get("TEST_KEYCLOAK_API_URL", "http://localhost:8800")
MODULE_26_URL = os.environ.get("TEST_MODULE_26_URL", "http://localhost:8026")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
POSTGRES_HOST = os.environ.get("TEST_POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("TEST_POSTGRES_PORT", "5433"))
POSTGRES_USER = os.environ.get("TEST_POSTGRES_USER", "raptor")
POSTGRES_PASSWORD = os.environ.get("TEST_POSTGRES_PASSWORD", "")
REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("TEST_REDIS_PASSWORD", "")
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")
# Must match the deployed Module 33's GD_CONSENT_POLICY_VERSION.
POLICY_VERSION = os.environ.get("TEST_CONSENT_POLICY_VERSION", "1.0")

GDPR = f"{GATEWAY_URL}/api/{API_VERSION}/gdpr"
POLL_TIMEOUT_SECONDS = 60
PURPOSES = ("memory", "analytics", "benchmark", "voice_recording", "face_detection")
DEFAULTS = {"memory": True, "analytics": False, "benchmark": True, "voice_recording": False,
            "face_detection": False}


def _gateway_reachable() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/docs", timeout=3.0)
        return True
    except Exception:  # noqa: BLE001 -- reachability probe feeding pytest.skip -- skip reason itself communicates it
        return False


pytestmark = pytest.mark.skipif(
    not (KEYCLOAK_ADMIN_PASSWORD and POSTGRES_PASSWORD) or not _gateway_reachable(),
    reason="requires a live deployment (gateway reachable; TEST_KEYCLOAK_ADMIN_PASSWORD, "
           "TEST_POSTGRES_PASSWORD set)",
)


# ── Keycloak helpers (same shape as the other *_e2e.py tests) ───────────────

def _keycloak_admin_token() -> str:
    resp = httpx.post(
        f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "admin-cli",
              "username": KEYCLOAK_ADMIN_USER, "password": KEYCLOAK_ADMIN_PASSWORD},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _admin_headers() -> dict:
    return {"Authorization": f"Bearer {_keycloak_admin_token()}"}


def _user_id(username: str) -> str:
    return httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=_admin_headers(), params={"username": username, "exact": "true"}, timeout=10.0,
    ).json()[0]["id"]


def _create_test_user(username: str, password: str) -> str:
    """Straight through the Admin API -- like a user that existed before
    GD-5, it never passes Module 06's registration webhook."""
    headers = _admin_headers()
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users", headers=headers,
        json={"username": username, "firstName": "GD5", "lastName": "E2ETest",
              "email": f"{username}@test.invalid", "emailVerified": True, "enabled": True, "requiredActions": [],
              "credentials": [{"type": "password", "value": password, "temporary": False}]},
        timeout=10.0,
    ).raise_for_status()
    user_id = _user_id(username)
    client_uuid = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients",
        headers=headers, params={"clientId": KEYCLOAK_CLIENT_ID}, timeout=10.0,
    ).json()[0]["id"]
    role = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients/{client_uuid}/roles/user_standard",
        headers=headers, timeout=10.0,
    ).json()
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
        headers=headers, json=[role], timeout=10.0,
    ).raise_for_status()
    return user_id


def _keycloak_user(user_id: str) -> dict:
    return httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}", headers=_admin_headers(), timeout=10.0,
    ).json()


def _consent_attributes(user_id: str) -> dict:
    attrs = _keycloak_user(user_id).get("attributes") or {}
    return {k: v[0] for k, v in attrs.items() if k.startswith("gdpr_consent_")}


def _user_jwt(username: str, password: str) -> str:
    resp = httpx.post(
        f"{GATEWAY_URL}/api/{API_VERSION}/sso/login",
        data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM, "client_id": KEYCLOAK_CLIENT_ID},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Independent ground-truth readers ────────────────────────────────────────

async def _pg(database: str) -> asyncpg.Connection:
    return await asyncpg.connect(host=POSTGRES_HOST, port=POSTGRES_PORT, user=POSTGRES_USER,
                                 password=POSTGRES_PASSWORD, database=database)


async def _history(user_id: str) -> list[dict]:
    conn = await _pg("privacy_compliance")
    try:
        rows = await conn.fetch("SELECT * FROM gdpr_consent_history WHERE user_id = $1 ORDER BY id", user_id)
    finally:
        await conn.close()
    return [dict(r) for r in rows]


async def _wait_for_job(job_id: str) -> dict:
    conn = await _pg("privacy_compliance")
    try:
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while True:
            row = await conn.fetchrow("SELECT * FROM gdpr_erasure_jobs WHERE job_id = $1", uuid.UUID(job_id))
            if row["status"] in ("completed", "partial_failure") or time.monotonic() > deadline:
                job = dict(row)
                job["store_statuses"] = json.loads(job["store_statuses"])  # asyncpg returns JSONB as text
                return job
            await asyncio.sleep(1)
    finally:
        await conn.close()


def _redis() -> aioredis.Redis:
    kwargs = {"host": REDIS_HOST, "port": REDIS_PORT, "decode_responses": True}
    if REDIS_PASSWORD:
        kwargs["password"] = REDIS_PASSWORD
    return aioredis.Redis(**kwargs)


def _put(auth: dict, purpose: str, granted: bool, forwarded_for: str = "198.51.100.23") -> dict:
    resp = httpx.put(f"{GDPR}/consent/{purpose}", headers={**auth, "X-Forwarded-For": forwarded_for},
                     json={"granted": granted}, timeout=30.0)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def cleanup():
    """Keycloak test users, seeded benchmark rows and voice Redis keys are
    always removed. gdpr_consent_history rows are not -- the table is
    append-only by design (the trigger refuses DELETE), same as any real
    user's consent record outliving their account."""
    state = {"usernames": [], "user_ids": [], "schema_ids": []}
    yield state
    headers = _admin_headers()
    for username in state["usernames"]:
        try:
            users = httpx.get(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users", headers=headers,  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                              params={"username": username, "exact": "true"}, timeout=10.0).json()
            if users:
                httpx.delete(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{users[0]['id']}",  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                             headers=headers, timeout=10.0)
        except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
            logger.debug(f"Teardown Keycloak user delete failed for {username}: {e}")
    conn = await _pg("benchmark")
    try:
        await conn.execute("DELETE FROM benchmark_runs WHERE branch_id = ANY($1)", state["user_ids"])
        await conn.execute("DELETE FROM benchmark_schemas WHERE id = ANY($1::uuid[])", state["schema_ids"])
    finally:
        await conn.close()
    redis = _redis()
    try:
        for user_id in state["user_ids"]:
            await redis.delete(f"gdpr:voice_recording_disabled:{user_id}")
    finally:
        await redis.aclose()


def _new_user(cleanup: dict) -> tuple[str, dict]:
    username = f"gd5-e2e-{uuid.uuid4().hex[:12]}"
    password = "TempPassw0rd!1"
    user_id = _create_test_user(username, password)
    cleanup["usernames"].append(username)
    cleanup["user_ids"].append(user_id)
    return user_id, {"Authorization": f"Bearer {_user_jwt(username, password)}"}


# ── Tests ───────────────────────────────────────────────────────────────────

def test_consent_routes_require_login():
    assert httpx.get(f"{GDPR}/consent", timeout=10.0).status_code == 401
    assert httpx.put(f"{GDPR}/consent/memory", json={"granted": False}, timeout=10.0).status_code == 401


@pytest.mark.asyncio
async def test_registration_through_module_06_sets_default_attributes(cleanup):
    username = f"gd5-e2e-reg-{uuid.uuid4().hex[:10]}"
    cleanup["usernames"].append(username)
    # example.com (RFC 2606), not .invalid: Module 06's email validator
    # rejects special-use domains.
    resp = httpx.post(f"{KEYCLOAK_API_URL}/SSO/register",  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                      data={"username": username, "email": f"{username}@example.com"}, timeout=30.0)
    assert resp.status_code == 200, resp.text
    user_id = _user_id(username)
    cleanup["user_ids"].append(user_id)

    assert _consent_attributes(user_id) == {
        "gdpr_consent_memory": "true", "gdpr_consent_analytics": "false",
        "gdpr_consent_benchmark": "true", "gdpr_consent_voice_recording": "false",
        "gdpr_consent_face_detection": "false", "gdpr_consent_version": POLICY_VERSION,
    }
    history = await _history(user_id)
    assert {(r["purpose"], r["action"], r["source"]) for r in history} == {
        ("memory", "grant", "registration_default"), ("analytics", "withdraw", "registration_default"),
        ("benchmark", "grant", "registration_default"), ("voice_recording", "withdraw", "registration_default"),
        ("face_detection", "withdraw", "registration_default"),
    }


@pytest.mark.asyncio
async def test_get_consent_for_user_without_attributes_writes_defaults(cleanup):
    user_id, auth = _new_user(cleanup)
    assert _consent_attributes(user_id) == {}

    resp = httpx.get(f"{GDPR}/consent", headers=auth, timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 200, resp.text
    purposes = resp.json()["purposes"]
    assert [p["id"] for p in purposes] == list(PURPOSES)
    assert {p["id"]: p["granted"] for p in purposes} == DEFAULTS
    for p in purposes:
        assert p["description"]
        assert (p["granted_at"] is not None) == p["granted"]

    attrs = _consent_attributes(user_id)
    assert attrs == {f"gdpr_consent_{k}": str(v).lower() for k, v in DEFAULTS.items()}  # no version
    assert {r["source"] for r in await _history(user_id)} == {"first_read_default"}


@pytest.mark.asyncio
async def test_withdraw_memory_erases_module_26_memory_only(cleanup):
    user_id, auth = _new_user(cleanup)
    httpx.get(f"{GDPR}/consent", headers=auth, timeout=30.0).raise_for_status()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    resp = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/memory/archive", headers=auth, params={"session_id": "gd5"},  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                      json={"user_message": "gd-5 consent e2e", "assistant_response": "ok"}, timeout=30.0)
    assert resp.status_code in (200, 201), resp.text
    before = httpx.get(f"{MODULE_26_URL}/memory/stats", headers={"X-User-ID": user_id}, timeout=10.0).json()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert before["session_count"] >= 1

    body = _put(auth, "memory", False, forwarded_for="198.51.100.23")
    assert body["changed"] is True and body["erasure_job_id"]

    assert _consent_attributes(user_id)["gdpr_consent_memory"] == "false"
    last = (await _history(user_id))[-1]
    assert (last["purpose"], last["action"], last["source"]) == ("memory", "withdraw", "user")
    assert last["ip_hash"] == hashlib.sha256(b"198.51.100.23").hexdigest()

    job = await _wait_for_job(body["erasure_job_id"])
    assert job["type"] == "consent_withdrawal" and job["status"] == "completed", job
    after = httpx.get(f"{MODULE_26_URL}/memory/stats", headers={"X-User-ID": user_id}, timeout=10.0).json()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert after["session_count"] == 0 and after["storage_bytes_used"] == 0
    # A partial erasure: the account itself is untouched.
    assert _keycloak_user(user_id)["enabled"] is True

    # Same value again: nothing new recorded, no second job.
    again = _put(auth, "memory", False)
    assert again["changed"] is False and again["erasure_job_id"] is None


@pytest.mark.asyncio
async def test_withdraw_benchmark_deletes_users_schemas_and_runs(cleanup):
    user_id, auth = _new_user(cleanup)
    other_id = f"gd5-e2e-other-{uuid.uuid4().hex[:8]}"
    cleanup["user_ids"].append(other_id)
    schema = {
        "name": "gd5 consent e2e", "target_pipeline": "chat",
        "test_cases": [{"id": "c1", "input": {"q": "x"}, "expected_keywords": ["x"]}],
        "scoring_schema": {"dimensions": [{"name": "kw", "weight": 1.0, "method": "keyword_match"}]},
    }
    schema_ids = []
    for _ in range(2):
        # X-User-ID is a spoof attempt -- the gateway must use the JWT sub.
        resp = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/benchmark/schemas", json=schema,  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                          headers={**auth, "X-User-ID": other_id}, timeout=30.0)
        assert resp.status_code == 201, resp.text
        schema_ids.append(resp.json()["id"])
    cleanup["schema_ids"].extend(schema_ids)
    own_only, shared = schema_ids

    conn = await _pg("benchmark")
    try:
        owners = await conn.fetch("SELECT owner_id FROM benchmark_schemas WHERE id = ANY($1::uuid[])", schema_ids)
        assert {r["owner_id"] for r in owners} == {user_id}
        for schema_id, branch in ((own_only, user_id), (shared, user_id), (shared, other_id)):
            await conn.execute("INSERT INTO benchmark_runs (status, branch_id, schema_id) VALUES ('completed', $1, $2)",
                               branch, uuid.UUID(schema_id))
    finally:
        await conn.close()

    body = _put(auth, "benchmark", False)
    job = await _wait_for_job(body["erasure_job_id"])
    assert job["status"] == "completed", job

    conn = await _pg("benchmark")
    try:
        assert await conn.fetchval("SELECT COUNT(*) FROM benchmark_runs WHERE branch_id = $1", user_id) == 0
        assert await conn.fetchval("SELECT COUNT(*) FROM benchmark_schemas WHERE owner_id = $1", user_id) == 0
        assert await conn.fetchval("SELECT COUNT(*) FROM benchmark_schemas WHERE id = $1", uuid.UUID(own_only)) == 0
        # Another user's run depends on `shared`: kept, but detached from the user.
        kept = await conn.fetchrow("SELECT owner_id FROM benchmark_schemas WHERE id = $1", uuid.UUID(shared))
        assert kept is not None and kept["owner_id"] is None
        assert await conn.fetchval("SELECT COUNT(*) FROM benchmark_runs WHERE branch_id = $1", other_id) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_voice_recording_toggles_redis_flag(cleanup):
    user_id, auth = _new_user(cleanup)
    key = f"gdpr:voice_recording_disabled:{user_id}"
    redis = _redis()
    try:
        granted = _put(auth, "voice_recording", True)
        assert granted["changed"] is True and granted["erasure_job_id"] is None
        assert _consent_attributes(user_id)["gdpr_consent_voice_recording"] == "true"
        assert await redis.exists(key) == 0

        withdrawn = _put(auth, "voice_recording", False)
        assert _consent_attributes(user_id)["gdpr_consent_voice_recording"] == "false"
        assert await redis.get(key) == "1"
        assert 90 * 86400 - 60 <= await redis.ttl(key) <= 90 * 86400
        job = await _wait_for_job(withdrawn["erasure_job_id"])
        # `deleted` when GD_MODULE_34_URL points at Module 34, `skipped` when it is empty.
        assert job["status"] == "completed"
        assert job["store_statuses"]["module_34"] in ("deleted", "skipped")
        assert set(job["store_statuses"]) == {"module_34"}

        _put(auth, "voice_recording", True)
        assert await redis.exists(key) == 0
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_unknown_purpose_rejected(cleanup):
    _, auth = _new_user(cleanup)
    resp = httpx.put(f"{GDPR}/consent/marketing", headers=auth, json={"granted": True}, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_consent_history_is_append_only(cleanup):
    user_id, auth = _new_user(cleanup)
    _put(auth, "analytics", True)
    conn = await _pg("privacy_compliance")
    try:
        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            await conn.execute("UPDATE gdpr_consent_history SET action = 'withdraw' WHERE user_id = $1", user_id)
        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            await conn.execute("DELETE FROM gdpr_consent_history WHERE user_id = $1", user_id)
    finally:
        await conn.close()
    assert [r["action"] for r in await _history(user_id) if r["purpose"] == "analytics"][-1] == "grant"
