"""
GD-4 / VIE01-249: end-to-end DSAR test against a real deployment.

create Keycloak test user -> upload 2 files -> archive 3 memory turns across
2 sessions with known timestamps -> seed known Chunk/Entity vertices into
the user's ArcadeDB database -> seed 2 benchmark_runs rows -> GET
/gdpr/dsar -> verify every count against a DIFFERENT code path than the
DSAR itself used (ArcadeDB queried directly, not through Module 25's new
summary endpoint; the seeded ground truth for memory/benchmark; the
gdpr_retention_policies table read directly), plus the admin route,
role-gating, and the 10-second response bound.

The async path (>10k indexed items) can't be reached with a test account
in reasonable time, so test_dsar_async_job_* only runs when the operator
has deployed Module 33 with GD_DSAR_ASYNC_THRESHOLD=-1 (every DSAR goes
async) and set TEST_DSAR_FORCE_ASYNC=1; it skips otherwise.

Deliberately live-infrastructure, not mocked (tests/unit/test_dsar_assembler.py
covers the pure logic with fakes). Skips cleanly if the gateway isn't
reachable, matching the other *_e2e.py tests here.

Run from the 33-privacy-compliance-service/ directory:
    TEST_KEYCLOAK_ADMIN_PASSWORD=... TEST_ARCADEDB_PASSWORD=... TEST_POSTGRES_PASSWORD=... \
        python -m pytest tests/test_dsar_e2e.py -v
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone

import asyncpg
import httpx
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

GATEWAY_URL = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
ARCADEDB_URL = os.environ.get("TEST_ARCADEDB_URL", "http://localhost:2480")
ARCADEDB_PASSWORD = os.environ.get("TEST_ARCADEDB_PASSWORD", "")
POSTGRES_HOST = os.environ.get("TEST_POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("TEST_POSTGRES_PORT", "5433"))
POSTGRES_USER = os.environ.get("TEST_POSTGRES_USER", "raptor")
POSTGRES_PASSWORD = os.environ.get("TEST_POSTGRES_PASSWORD", "")
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")
FORCE_ASYNC = os.environ.get("TEST_DSAR_FORCE_ASYNC") == "1"

GDPR = f"{GATEWAY_URL}/api/{API_VERSION}/gdpr"
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120


def _gateway_reachable() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/docs", timeout=3.0)
        return True
    except Exception:  # noqa: BLE001 -- reachability probe feeding pytest.skip -- skip reason itself communicates it
        return False


pytestmark = pytest.mark.skipif(
    not (KEYCLOAK_ADMIN_PASSWORD and ARCADEDB_PASSWORD and POSTGRES_PASSWORD) or not _gateway_reachable(),
    reason="requires a live deployment (gateway reachable; TEST_KEYCLOAK_ADMIN_PASSWORD, "
           "TEST_ARCADEDB_PASSWORD, TEST_POSTGRES_PASSWORD set)",
)


# ── Keycloak helpers (same shape as test_export_e2e.py / test_retention_e2e.py) ──

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


def _create_test_user(admin_token: str, username: str, password: str, gdpr_admin: bool = False) -> str:
    headers = {"Authorization": f"Bearer {admin_token}"}
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers,
        json={
            "username": username, "firstName": "GD4", "lastName": "E2ETest",
            "email": f"{username}@test.invalid", "emailVerified": True,
            "enabled": True, "requiredActions": [],
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        },
        timeout=10.0,
    ).raise_for_status()
    user_id = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers, params={"username": username, "exact": "true"}, timeout=10.0,
    ).json()[0]["id"]

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
    if gdpr_admin:
        realm_role = httpx.get(
            f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/roles/gdpr-admin", headers=headers, timeout=10.0,
        ).json()
        httpx.post(
            f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/realm",
            headers=headers, json=[realm_role], timeout=10.0,
        ).raise_for_status()
    return user_id


def _keycloak_profile(admin_token: str, user_id: str) -> dict:
    return httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}"}, timeout=10.0,
    ).json()


def _user_jwt(username: str, password: str) -> str:
    resp = httpx.post(
        f"{GATEWAY_URL}/api/{API_VERSION}/sso/login",
        data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM, "client_id": KEYCLOAK_CLIENT_ID},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Independent ground-truth readers ────────────────────────────────────────

_ARCADE_AUTH = ("root", ARCADEDB_PASSWORD)


def _arcadedb_name(user_id: str) -> str:
    """Same mapping as Module 25's db_name_for, re-derived here rather than
    imported so the test doesn't share code with what it's checking."""
    return "user_" + (re.sub(r"[^A-Za-z0-9_]", "_", user_id).strip("_") or "anon")


def _arcadedb_exists(db: str) -> bool:
    return db in httpx.get(f"{ARCADEDB_URL}/api/v1/databases", auth=_ARCADE_AUTH, timeout=10.0).json()["result"]


def _arcadedb_seed(user_id: str, chunks: list[tuple[str, str]], doc_sources: list[str], entities: int) -> None:
    """Writes known Chunk/Entity vertices straight into the user's database.
    The upload pipeline can't be relied on for this: a deployment without
    the processing workers running (as this one is at time of writing)
    leaves uploads at status=uploaded and never indexes anything, which
    would reduce the indexed_content check to 0 == 0."""
    db = _arcadedb_name(user_id)
    if not _arcadedb_exists(db):
        httpx.post(f"{ARCADEDB_URL}/api/v1/server", auth=_ARCADE_AUTH,
                   json={"command": f"create database {db}"}, timeout=30.0).raise_for_status()
    commands = [f"CREATE VERTEX TYPE {t} IF NOT EXISTS" for t in ("Chunk", "Source", "Entity")]
    commands += [
        f"INSERT INTO Chunk SET chunk_id = 'gd4-{i}', type = '{t}', version_id = '{v}'"
        for i, (t, v) in enumerate(chunks)
    ]
    commands += [f"INSERT INTO Source SET version_id = '{v}', media_type = 'documents'" for v in doc_sources]
    commands += [f"INSERT INTO Entity SET entity_id = 'gd4-e{i}', name = 'gd4 entity {i}'" for i in range(entities)]
    for command in commands:
        httpx.post(f"{ARCADEDB_URL}/api/v1/command/{db}", auth=_ARCADE_AUTH,
                   json={"language": "sql", "command": command}, timeout=10.0).raise_for_status()


def _arcadedb_drop(user_id: str) -> None:
    db = _arcadedb_name(user_id)
    if _arcadedb_exists(db):
        httpx.post(f"{ARCADEDB_URL}/api/v1/server", auth=_ARCADE_AUTH,
                   json={"command": f"drop database {db}"}, timeout=30.0)


def _arcadedb_counts(user_id: str) -> dict:
    """Straight to ArcadeDB's own HTTP API -- not through Module 25 at all."""
    db = _arcadedb_name(user_id)
    auth = _ARCADE_AUTH
    if not _arcadedb_exists(db):
        return {"documents": 0, "moments": 0, "entities": 0}

    def _sql(command: str) -> list:
        resp = httpx.post(
            f"{ARCADEDB_URL}/api/v1/query/{db}", auth=auth,
            json={"language": "sql", "command": command}, timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    by_type = {r["type"]: r["c"] for r in _sql("SELECT type, count(*) AS c FROM Chunk GROUP BY type") if r.get("type")}
    # documents = distinct document versions across tagged Sources and
    # document Chunks, re-derived here in Python rather than with Module
    # 25's own queries.
    doc_versions = {r["version_id"] for r in _sql("SELECT version_id, type FROM Chunk") if r.get("type") == "documents"}
    doc_versions |= {r["version_id"] for r in _sql("SELECT version_id, media_type FROM Source") if r.get("media_type") == "documents"}
    entities = _sql("SELECT count(*) AS c FROM Entity")
    return {
        "documents": len({v for v in doc_versions if v}),
        "moments": sum(by_type.get(t, 0) for t in ("videos", "images", "audios")),
        "entities": entities[0]["c"] if entities else 0,
    }


async def _pg(database: str) -> asyncpg.Connection:
    return await asyncpg.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, user=POSTGRES_USER,
        password=POSTGRES_PASSWORD, database=database,
    )


async def _retention_policies() -> dict:
    conn = await _pg("privacy_compliance")
    try:
        rows = await conn.fetch("SELECT data_category, retention_days, auto_delete FROM gdpr_retention_policies")
    finally:
        await conn.close()
    return {r["data_category"]: dict(r) for r in rows}


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def cleanup():
    """Keycloak test users, seeded benchmark_runs rows and seeded ArcadeDB
    databases -- always removed. Uploaded files / memory are left for the
    same reason test_export_e2e.py leaves them: the account itself is
    deleted, and those stores are covered by GD-1's erasure test."""
    state = {"usernames": [], "benchmark_branch_ids": [], "arcadedb_user_ids": []}
    yield state
    for user_id in state["arcadedb_user_ids"]:
        try:
            _arcadedb_drop(user_id)
        except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
            logger.debug(f"Teardown ArcadeDB drop failed for {user_id}: {e}")
    admin_token = _keycloak_admin_token()
    headers = {"Authorization": f"Bearer {admin_token}"}
    for username in state["usernames"]:
        try:
            users = httpx.get(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
                f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
                headers=headers, params={"username": username, "exact": "true"}, timeout=10.0,
            ).json()
            if users:
                httpx.delete(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{users[0]['id']}", headers=headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
            logger.debug(f"Teardown Keycloak user delete failed for {username}: {e}")
    if state["benchmark_branch_ids"]:
        conn = await _pg("benchmark")
        try:
            await conn.execute("DELETE FROM benchmark_runs WHERE branch_id = ANY($1)", state["benchmark_branch_ids"])
        finally:
            await conn.close()


def _new_user(cleanup: dict, admin_token: str, gdpr_admin: bool = False) -> tuple[str, str, dict]:
    username = f"gd4-e2e-{uuid.uuid4().hex[:12]}"
    password = "TempPassw0rd!1"
    user_id = _create_test_user(admin_token, username, password, gdpr_admin=gdpr_admin)
    cleanup["usernames"].append(username)
    return username, user_id, {"Authorization": f"Bearer {_user_jwt(username, password)}"}


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.skipif(FORCE_ASYNC, reason="sync-path test; deployment is configured to force every DSAR async")
async def test_dsar_reports_real_cross_store_counts(cleanup):
    admin_token = _keycloak_admin_token()
    username, user_id, auth = _new_user(cleanup, admin_token)

    # ── Seed ground truth ────────────────────────────────────────────────
    for name, content, ctype in (("gd4_e2e.txt", b"hello gd-4 dsar e2e", "text/plain"),
                                 ("gd4_e2e.md", b"# gd-4 dsar e2e", "text/markdown")):
        resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
            f"{GATEWAY_URL}/api/{API_VERSION}/asset/fileupload", headers=auth,
            files={"primary_file": (name, content, ctype)}, timeout=30.0,
        )
        assert resp.status_code == 200, resp.text

    t0 = 1_760_000_000.0  # oldest turn, deliberately in the past
    for session_id, ts in (("s1", t0), ("s1", t0 + 60), ("s2", t0 + 3600)):
        resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
            f"{GATEWAY_URL}/api/{API_VERSION}/memory/archive", headers=auth, params={"session_id": session_id},
            json={"user_message": f"gd-4 turn at {ts}", "assistant_response": "ok", "timestamp": ts},
            timeout=60.0,
        )
        assert resp.status_code in (200, 201), resp.text

    conn = await _pg("benchmark")
    try:
        cleanup["benchmark_branch_ids"].append(user_id)
        for _ in range(2):
            await conn.execute("INSERT INTO benchmark_runs (status, branch_id) VALUES ('completed', $1)", user_id)
    finally:
        await conn.close()

    cleanup["arcadedb_user_ids"].append(user_id)
    # 3 documents in 4 document chunks (so a chunk count would read 4):
    # doc-a (3 chunks), doc-b (1 chunk, no Source -- as live data has),
    # doc-c (Source only, chunks not indexed yet). 3 moments.
    _arcadedb_seed(
        user_id,
        chunks=[("documents", "doc-a"), ("documents", "doc-a"), ("documents", "doc-a"), ("documents", "doc-b"),
                ("videos", "vid-1"), ("videos", "vid-1"), ("audios", "aud-1")],
        doc_sources=["doc-a", "doc-c"],
        entities=4,
    )

    # ── DSAR (self) -- within the ticket's 10-second bound ───────────────
    start = time.monotonic()
    resp = httpx.get(f"{GDPR}/dsar", headers=auth, timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    elapsed = time.monotonic() - start
    assert resp.status_code == 200, resp.text
    assert elapsed < 10.0, f"DSAR took {elapsed:.1f}s"
    report = resp.json()

    assert report["user_id"] == user_id
    assert report["stores_failed"] == [], report["stores_failed"]
    # Pydantic serializes UTC as a trailing "Z", which 3.10's fromisoformat rejects.
    datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))

    cats = report["data_categories"]
    # Re-issued: Keycloak admin tokens live 60s by default.
    admin_token = _keycloak_admin_token()
    profile = _keycloak_profile(admin_token, user_id)
    assert cats["identity"] == {
        "email": f"{username}@test.invalid",
        "username": username,
        "created_at": datetime.fromtimestamp(profile["createdTimestamp"] / 1000, tz=timezone.utc).isoformat(),
    }

    assert cats["uploaded_files"]["count"] == 2
    assert cats["uploaded_files"]["file_types"] == ["md", "txt"]
    assert cats["uploaded_files"]["total_bytes"] is None

    indexed = {k: cats["indexed_content"][k] for k in ("documents", "moments", "entities")}
    assert indexed == {"documents": 3, "moments": 3, "entities": 4}
    assert indexed == _arcadedb_counts(user_id)
    # GD-6: a real count now -- the seeded chunks carry no pii_flag.
    assert cats["indexed_content"]["pii_flagged"] == 0

    assert cats["memory"]["sessions"] == 2
    assert cats["memory"]["oldest_session"] == datetime.fromtimestamp(t0, tz=timezone.utc).isoformat()
    assert isinstance(cats["memory"]["long_term_facts"], int)

    assert cats["benchmark"] == {"schemas_owned": None, "runs": 2}
    # CM-5 / VIE01-408: a real count now (this test user has no moderation
    # flags) -- distinct from guardrail_violations, which stays unscopable.
    assert cats["safety_violations"] == {"count": 0}

    policies = await _retention_policies()

    def _label(cat: str) -> str:
        p = policies.get(cat)
        return f"{p['retention_days']} days" if p and p["auto_delete"] else "indefinite"

    # CM-5 / VIE01-408: moderation_flags is split by status into two policies
    # ('cleared' vs. 'flagged'/'removed') -- violations reports both, same
    # policy-level-not-per-user pattern as every other retention_summary field.
    assert report["retention_summary"] == {
        "files": "indefinite", "memory": _label("memvid_sessions"),
        "violations": f"{_label('moderation_flags')} (cleared) / {_label('moderation_flags_actionable')} (flagged or removed)",
    }
    assert report["consent_status"] == {"memory": None, "analytics": None}
    assert report["ccpa_category"] == "Identifiers, Commercial information, Internet activity"
    assert report["ccpa_source"] == "Directly from consumer"
    assert report["ccpa_sells_data"] is False
    assert report["rights"] == {
        "erasure_endpoint": f"/api/{API_VERSION}/gdpr/erasure",
        "export_endpoint": f"/api/{API_VERSION}/gdpr/export",
        "consent_endpoint": None,
    }
    assert set(report["unavailable_fields"]) == {
        "data_categories.uploaded_files.total_bytes",
        "data_categories.benchmark.schemas_owned",
        "consent_status", "rights.consent_endpoint",
    }

    # ── Role gating: a normal user can't DSAR someone else ──────────────
    _, _, other_auth = _new_user(cleanup, admin_token)
    resp = httpx.get(f"{GDPR}/dsar/{user_id}", headers=other_auth, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 403, resp.text

    # ── Admin route: same report for the target user ────────────────────
    _, _, admin_auth = _new_user(cleanup, admin_token, gdpr_admin=True)
    resp = httpx.get(f"{GDPR}/dsar/{user_id}", headers=admin_auth, timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 200, resp.text
    admin_report = resp.json()
    assert admin_report["user_id"] == user_id
    for key in ("identity", "uploaded_files", "indexed_content", "memory", "benchmark"):
        assert admin_report["data_categories"][key] == cats[key]

    # ── A fresh account with nothing stored reports zeros, not errors ────
    resp = httpx.get(f"{GDPR}/dsar", headers=other_auth, timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 200, resp.text
    empty = resp.json()["data_categories"]
    assert empty["uploaded_files"]["count"] == 0
    assert empty["indexed_content"]["documents"] == 0
    assert empty["memory"] == {"sessions": 0, "long_term_facts": 0, "oldest_session": None}
    assert empty["benchmark"]["runs"] == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not FORCE_ASYNC, reason="needs Module 33 deployed with GD_DSAR_ASYNC_THRESHOLD=-1 and TEST_DSAR_FORCE_ASYNC=1")
async def test_dsar_async_job_is_polled_to_completion(cleanup):
    admin_token = _keycloak_admin_token()
    _, user_id, auth = _new_user(cleanup, admin_token)

    start = time.monotonic()
    resp = httpx.get(f"{GDPR}/dsar", headers=auth, timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert time.monotonic() - start < 10.0
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "pending" and set(body) == {"job_id", "status"}
    job_id = body["job_id"]

    # Another user can't read this job's report.
    _, _, other_auth = _new_user(cleanup, admin_token)
    resp = httpx.get(f"{GDPR}/dsar/jobs/{job_id}", headers=other_auth, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 403, resp.text

    deadline = time.time() + POLL_TIMEOUT_SECONDS
    job = None
    while time.time() < deadline:
        resp = httpx.get(f"{GDPR}/dsar/jobs/{job_id}", headers=auth, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["status"] in ("completed", "partial_failure"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)  # noqa: ASYNC251 -- e2e polling loop, no concurrent event loop pressure
    assert job is not None and job["status"] == "completed", job
    assert job["user_id"] == user_id
    assert job["result"]["user_id"] == user_id
    assert job["result"]["data_categories"]["uploaded_files"]["count"] == 0
    assert job["expires_at"] is not None

    # The stored row really is a type='dsar' job carrying the report.
    conn = await _pg("privacy_compliance")
    try:
        row = await conn.fetchrow("SELECT type, status, result FROM gdpr_erasure_jobs WHERE job_id = $1", uuid.UUID(job_id))
    finally:
        await conn.close()
    assert row["type"] == "dsar" and row["status"] == "completed" and row["result"] is not None
