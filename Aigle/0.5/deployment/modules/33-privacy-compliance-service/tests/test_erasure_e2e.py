"""
GD-1 / VIE01-246: end-to-end erasure test against a real deployment.

create Keycloak test user -> upload file -> chat -> submit erasure ->
poll until COMPLETED -> assert every store is actually empty, exactly
the ticket's own acceptance-criteria workflow.

Deliberately a live-infrastructure test, not a mocked unit test -- the
whole point is proving cross-store deletion actually happened, which a
mock can't demonstrate. Skips cleanly (rather than failing) if the
gateway isn't reachable, so a plain `pytest` run with no deployment up
doesn't break.

Post-erasure assertions independently verify each store through a
*different* code path than the deletion itself used, not by re-calling
the same DELETE endpoint and trusting its own answer:
- Module 04 (LakeFS-backed commit_history): GET /users/commits, called
  directly against the module (X-User-ID/X-Branch-ID header trust --
  the same convention Module 13 itself uses, see app/core/dependencies.py
  in modules 04/25/26), not through Module 13's JWT-gated proxy -- the
  erased account has no working JWT anymore by this point anyway.
- Module 25 (ArcadeDB): GET /internal/db/status, same header-trust.
- Module 26 (MemVID): GET /memory/stats, same header-trust.
- Module 03 (Postgres, `benchmark` database): direct asyncpg query.
- Module 23 (Postgres, `guardrails` database, moderation_flags table --
  CM-5 / VIE01-408): direct asyncpg query, same connect-directly pattern.
- Module 02 (Redis): direct SCAN, same patterns the orchestrator itself
  deletes.
- Module 06 (Keycloak): Admin API user lookup.

Scope note: only the 7 real stores this environment has (02, 03, 04, 06,
23, 25, 26). Modules 34/35 don't exist in this codebase (see GD-1's
implementation notes) and are not asserted on here.

Run against a live deployment from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/test_erasure_e2e.py -v
"""
from __future__ import annotations

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
MODULE_04_URL = os.environ.get("TEST_MODULE_04_URL", "http://localhost:8000")
MODULE_25_URL = os.environ.get("TEST_MODULE_25_URL", "http://localhost:8025")
MODULE_26_URL = os.environ.get("TEST_MODULE_26_URL", "http://localhost:8026")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")

POSTGRES_HOST = os.environ.get("TEST_POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("TEST_POSTGRES_PORT", "5433"))
POSTGRES_USER = os.environ.get("TEST_POSTGRES_USER", "raptor")
POSTGRES_PASSWORD = os.environ.get("TEST_POSTGRES_PASSWORD", "raptor")

REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("TEST_REDIS_PASSWORD", "")

POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 600  # generous -- store fan-out has its own 300s per-store timeout


def _gateway_reachable() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/docs", timeout=3.0)
        return True
    except Exception:  # noqa: BLE001 -- reachability probe feeding pytest.skip -- skip reason itself communicates it
        return False


pytestmark = pytest.mark.skipif(
    not KEYCLOAK_ADMIN_PASSWORD or not _gateway_reachable(),
    reason="requires a live deployment (gateway + Keycloak reachable, TEST_KEYCLOAK_ADMIN_PASSWORD set)",
)


# ── Keycloak admin helpers ──────────────────────────────────────────────────

def _keycloak_admin_token() -> str:
    resp = httpx.post(
        f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
        data={
            "grant_type": "password", "client_id": "admin-cli",
            "username": KEYCLOAK_ADMIN_USER, "password": KEYCLOAK_ADMIN_PASSWORD,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _raptor_client_uuid(admin_token: str) -> str:
    resp = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={"clientId": KEYCLOAK_CLIENT_ID}, timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()[0]["id"]


def _create_test_user(admin_token: str, username: str, password: str) -> str:
    """Full profile + the client role Keycloak's own authorization policy
    actually checks (user_standard on the `raptor` client) -- a user created
    with only realm roles passes login but fails every permission check,
    discovered the hard way testing this exact ticket."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    resp = httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers,
        json={
            "username": username, "firstName": "GD1", "lastName": "E2ETest",
            "email": f"{username}@test.invalid", "emailVerified": True,
            "enabled": True, "requiredActions": [],
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    user_id = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers, params={"username": username, "exact": "true"}, timeout=10.0,
    ).json()[0]["id"]

    client_uuid = _raptor_client_uuid(admin_token)
    role = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients/{client_uuid}/roles/user_standard",
        headers=headers, timeout=10.0,
    ).json()
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
        headers=headers, json=[role], timeout=10.0,
    ).raise_for_status()
    return user_id


def _user_jwt(username: str, password: str) -> str:
    # `raptor` is a confidential Keycloak client -- direct ROPC against
    # Keycloak's own token endpoint 401s without its client secret. Module
    # 13's /sso/login exists precisely to proxy this: it attaches that
    # secret server-side (see app/routers/sso.py / app/keycloak/user.py's
    # user_login_secret_client), so real callers only ever need their own
    # username/password, never the client secret.
    resp = httpx.post(
        f"{GATEWAY_URL}/api/{API_VERSION}/sso/login",
        data={
            "username": username, "password": password,
            "realm_name": KEYCLOAK_REALM, "client_id": KEYCLOAK_CLIENT_ID,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _keycloak_user_exists(admin_token: str, username: str) -> bool:
    resp = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={"username": username, "exact": "true"}, timeout=10.0,
    )
    resp.raise_for_status()
    return len(resp.json()) > 0


@pytest_asyncio.fixture
async def cleanup_registry():
    """Best-effort cleanup of the Keycloak test user if the test fails
    partway through (erasure normally removes it itself)."""
    created_usernames: list[str] = []
    yield created_usernames
    if not created_usernames:
        return
    admin_token = _keycloak_admin_token()
    for username in created_usernames:
        try:
            headers = {"Authorization": f"Bearer {admin_token}"}
            users = httpx.get(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
                headers=headers, params={"username": username, "exact": "true"}, timeout=10.0,
            ).json()
            if users:
                httpx.delete(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                    f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{users[0]['id']}",
                    headers=headers, timeout=10.0,
                )
        except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
            logger.debug(f"Teardown Keycloak user delete failed for {username}: {e}")


@pytest.mark.asyncio
async def test_erasure_deletes_across_every_real_store(cleanup_registry):
    admin_token = _keycloak_admin_token()
    username = f"gd1-e2e-{uuid.uuid4().hex[:12]}"
    password = "TempPassw0rd!1"
    user_id = _create_test_user(admin_token, username, password)
    cleanup_registry.append(username)

    user_jwt = _user_jwt(username, password)
    auth_headers = {"Authorization": f"Bearer {user_jwt}"}

    # ── Upload a file (Module 04) ───────────────────────────────────────
    upload_resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{GATEWAY_URL}/api/{API_VERSION}/asset/fileupload",
        headers=auth_headers,
        files={"primary_file": ("gd1_e2e.txt", b"hello gd-1 e2e test", "text/plain")},
        timeout=30.0,
    )
    assert upload_resp.status_code == 200, upload_resp.text

    # ── Chat, to populate Module 15/26's memory (Redis short-term cache +
    # MemVID long-term storage) -- POST /chat/chat (RAG, saves memory),
    # not /chat/completions (a stateless OpenAI-compatible proxy with no
    # memory persistence at all). Body is {"message": str}, not an OpenAI
    # messages array -- see chat.py's ChatRequest. ─────────────────────
    chat_resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{GATEWAY_URL}/api/{API_VERSION}/chat/chat",
        headers=auth_headers,
        json={"message": "Hello, this is a GD-1 e2e test message."},
        timeout=60.0,
    )
    assert chat_resp.status_code == 200, chat_resp.text

    # ── Sanity: data actually exists before erasure ─────────────────────
    # (independent, header-trust reads -- same code path used for the
    # post-erasure assertions below, so a false-negative there can't be
    # explained away as "there was never anything there to begin with")
    branch_headers = {"X-User-ID": user_id, "X-Branch-ID": user_id}
    pre_commits = httpx.get(f"{MODULE_04_URL}/users/commits", headers=branch_headers, timeout=10.0).json()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert pre_commits["total_count"] >= 1, "upload should have produced at least one commit_history row"

    # ── Submit erasure ───────────────────────────────────────────────
    submit_resp = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/erasure", headers=auth_headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert submit_resp.status_code == 200, submit_resp.text
    job_id = submit_resp.json()["job_id"]
    assert submit_resp.json()["status"] == "pending"

    # ── Poll until COMPLETED -- no auth needed (see gdpr.py's public_router:
    # the submitter's own session is dead within moments of submission, so
    # requiring a login here would make it impossible to ever confirm
    # completion) ────────────────────────────────────────────────────────
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    job = None
    while time.time() < deadline:
        resp = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/erasure/jobs/{job_id}", timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["status"] in ("completed", "partial_failure"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)  # noqa: ASYNC251 -- e2e polling loop, no concurrent event loop pressure
    assert job is not None and job["status"] == "completed", f"job ended as {job and job['status']}: {job and job.get('notes')}"

    # ── Assert every real store is actually empty ───────────────────────
    # Module 04 -- LakeFS-backed commit_history, via the same header-trust
    # GET /users/commits used for the pre-erasure sanity check above.
    post_commits = httpx.get(f"{MODULE_04_URL}/users/commits", headers=branch_headers, timeout=10.0).json()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert post_commits["total_count"] == 0, "commit_history rows still present after erasure"

    # Module 25 -- ArcadeDB per-user database.
    db_status = httpx.get(f"{MODULE_25_URL}/internal/db/status", headers=branch_headers, timeout=10.0).json()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert db_status["db_exists"] is False, "ArcadeDB per-user database directory still exists"

    # Module 26 -- MemVID .mv2/.meta.json files.
    mem_stats = httpx.get(f"{MODULE_26_URL}/memory/stats", headers={"X-User-ID": user_id}, timeout=10.0).json()  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert mem_stats["session_count"] == 0, "memory session(s) still present after erasure"
    assert mem_stats["storage_bytes_used"] == 0, "MemVID files still present after erasure"

    # Module 03 -- Postgres (benchmark_runs.branch_id, the one real,
    # per-user-scoped table in the ticket's Module 03 data inventory).
    conn = await asyncpg.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, user=POSTGRES_USER,
        password=POSTGRES_PASSWORD, database="benchmark",
    )
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM benchmark_runs WHERE branch_id = $1", user_id)
    finally:
        await conn.close()
    assert count == 0, "benchmark_runs rows still present after erasure"

    # Module 23 -- Postgres (guardrails.moderation_flags.branch_id, CM-5 /
    # VIE01-408) -- distinct from guardrail_violations in the same database,
    # which has no owner column and is never erased.
    conn = await asyncpg.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, user=POSTGRES_USER,
        password=POSTGRES_PASSWORD, database="guardrails",
    )
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM moderation_flags WHERE branch_id = $1", user_id)
    finally:
        await conn.close()
    assert count == 0, "moderation_flags rows still present after erasure"

    # Module 02 -- Redis, same patterns the orchestrator itself deletes.
    redis_kwargs = {"host": REDIS_HOST, "port": REDIS_PORT, "decode_responses": True}
    if REDIS_PASSWORD:
        redis_kwargs["password"] = REDIS_PASSWORD
    redis = aioredis.Redis(**redis_kwargs)
    try:
        leftover_keys = []
        for pattern in (f"*:{user_id}:*", f"chat_memory:{user_id}:*", f"chat_memory:{user_id}"):
            async for key in redis.scan_iter(match=pattern):
                leftover_keys.append(key)
        assert leftover_keys == [], f"Redis keys still present after erasure: {leftover_keys}"
    finally:
        await redis.aclose()

    # Module 06 -- Keycloak account itself.
    assert not _keycloak_user_exists(admin_token, username), "Keycloak account still exists after erasure"
