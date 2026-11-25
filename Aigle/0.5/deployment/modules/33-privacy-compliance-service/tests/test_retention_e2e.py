"""
GD-3 / VIE01-248: end-to-end retention test against a real deployment.

Seeds one deliberately-old and one fresh row/file per enforceable category
(benchmark_runs, guardrail_violations, moderation_flags, gdpr_access_log,
memvid_sessions) -- moderation_flags gets two old+fresh pairs instead of one,
since CM-5 / VIE01-408 split it into two categories by status (moderation_flags:
'cleared', 90 days; moderation_flags_actionable: 'flagged'/'removed', 1095 days) --
confirms GET /retention/policy/dry-run counts exactly the old ones, runs
the engine's actual enforce() (constructed directly, not over HTTP -- the
ticket's endpoint set has no manual-trigger route, deliberately: enforcement
only ever runs on APScheduler's daily schedule) against real Postgres/disk,
and confirms only the old ones are gone -- fresh data survives, and
gdpr_retention_log picks up real entries. Also exercises all three gateway
routes (GET policy, PUT policy, GET dry-run) through Module 13 with a real
gdpr-admin JWT.

Deliberately live-infrastructure, not mocked (tests/unit/test_retention_engine.py
already covers the pure logic with fakes) -- the point here is proving the
SQL/file-age filtering is actually correct against real data, and that role
enforcement on the gateway really works. Skips cleanly if the gateway isn't
reachable, matching test_erasure_e2e.py / test_export_e2e.py.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/test_retention_e2e.py -v
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import asyncpg
import httpx
import pytest
import pytest_asyncio
from app.core.config import settings

logger = logging.getLogger(__name__)

GATEWAY_URL = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")

MEMVID_STORAGE_ROOT = os.environ.get("TEST_MEMVID_STORAGE_ROOT", "/mnt/memvid-storage")

# Live-DB overrides -- same convention test_commits_status_filter.py uses:
# set real connection info on the already-constructed settings singleton
# before anything in this test module uses it. conftest.py already supplied
# dummy placeholders so the import itself doesn't fail.
settings.postgres_host = os.environ.get("TEST_POSTGRES_HOST", "localhost")
settings.postgres_port = int(os.environ.get("TEST_POSTGRES_PORT", "5433"))
settings.postgres_user = os.environ.get("TEST_POSTGRES_USER", "raptor")
settings.postgres_password = os.environ.get("TEST_POSTGRES_PASSWORD", "raptor")
settings.postgres_db = os.environ.get("TEST_POSTGRES_DB", "privacy_compliance")
settings.module_26_url = os.environ.get("TEST_MODULE_26_URL", "http://localhost:8026")
settings.internal_token = os.environ.get("TEST_GD_INTERNAL_TOKEN", settings.internal_token)

from app.services.retention_engine import (
    RetentionEngine,
)

TEST_TAG = f"gd3-e2e-{uuid.uuid4().hex[:10]}"


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

# Only the seeded-data test touches Module 26's on-disk session files
# directly (to backdate mtime) -- the two gateway-role tests below don't,
# and shouldn't be skipped just because this host process can't see a
# Docker-managed volume's data directly (e.g. no root, or the test runner
# is entirely outside the Docker host).
_memvid_fs_reachable = pytest.mark.skipif(
    not os.path.isdir(MEMVID_STORAGE_ROOT) or not os.access(MEMVID_STORAGE_ROOT, os.W_OK),
    reason=f"this host process can't write to TEST_MEMVID_STORAGE_ROOT ({MEMVID_STORAGE_ROOT}) -- "
           "set it to a path this process can actually read/write (e.g. the Docker volume's host "
           "mount point, if running with sufficient permissions), or run from inside the container",
)


# ── Keycloak admin helpers (same pattern as test_erasure_e2e.py) ───────────

def _keycloak_admin_token() -> str:
    resp = httpx.post(
        f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "admin-cli", "username": KEYCLOAK_ADMIN_USER, "password": KEYCLOAK_ADMIN_PASSWORD},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _create_gdpr_admin_user(admin_token: str, username: str, password: str) -> str:
    headers = {"Authorization": f"Bearer {admin_token}"}
    resp = httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers,
        json={
            "username": username, "firstName": "GD3", "lastName": "E2ETest",
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

    client_uuid = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients",
        headers=headers, params={"clientId": KEYCLOAK_CLIENT_ID}, timeout=10.0,
    ).json()[0]["id"]
    user_standard_role = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients/{client_uuid}/roles/user_standard",
        headers=headers, timeout=10.0,
    ).json()
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
        headers=headers, json=[user_standard_role], timeout=10.0,
    ).raise_for_status()

    gdpr_admin_role = httpx.get(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/roles/gdpr-admin", headers=headers, timeout=10.0,
    ).json()
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/realm",
        headers=headers, json=[gdpr_admin_role], timeout=10.0,
    ).raise_for_status()
    return user_id


def _user_jwt(username: str, password: str) -> str:
    resp = httpx.post(
        f"{GATEWAY_URL}/api/{API_VERSION}/sso/login",
        data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM, "client_id": KEYCLOAK_CLIENT_ID},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


@pytest_asyncio.fixture
async def cleanup_registry():
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
                httpx.delete(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{users[0]['id']}", headers=headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
            logger.debug(f"Teardown Keycloak user delete failed for {username}: {e}")


async def _pg_execute(database: str, sql: str, *args) -> None:
    conn = await asyncpg.connect(host=settings.postgres_host, port=settings.postgres_port, user=settings.postgres_user, password=settings.postgres_password, database=database)
    try:
        await conn.execute(sql, *args)
    finally:
        await conn.close()


async def _pg_fetchval(database: str, sql: str, *args):
    conn = await asyncpg.connect(host=settings.postgres_host, port=settings.postgres_port, user=settings.postgres_user, password=settings.postgres_password, database=database)
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


@_memvid_fs_reachable
@pytest.mark.asyncio
async def test_dry_run_and_enforce_against_real_seeded_data():
    old_tag, fresh_tag = f"{TEST_TAG}-old", f"{TEST_TAG}-fresh"
    # CM-5 / VIE01-408: moderation_flags is split by status into two
    # categories -- 'cleared' purges at 90 days (moderation_flags),
    # 'flagged'/'removed' at 1095 days (moderation_flags_actionable). Four
    # tags instead of two, so each category's own age boundary is exercised
    # independently: old_cleared is 100 days old (past 90, cleared) but
    # old_actionable is 1200 days old (past 1095, flagged) -- 100 days is
    # nowhere near old enough for the actionable category.
    old_cleared_tag, fresh_cleared_tag = f"{TEST_TAG}-mf-old-cleared", f"{TEST_TAG}-mf-fresh-cleared"
    old_actionable_tag, fresh_actionable_tag = f"{TEST_TAG}-mf-old-actionable", f"{TEST_TAG}-mf-fresh-actionable"

    # ── Seed one old + one fresh row per SQL category ────────────────────
    await _pg_execute(
        "benchmark", "INSERT INTO benchmark_runs (id, status, created_at, branch_id) VALUES (gen_random_uuid(), 'completed', NOW() - INTERVAL '100 days', $1)", old_tag,
    )
    await _pg_execute(
        "benchmark", "INSERT INTO benchmark_runs (id, status, created_at, branch_id) VALUES (gen_random_uuid(), 'completed', NOW() - INTERVAL '10 days', $1)", fresh_tag,
    )
    await _pg_execute(
        "guardrails", "INSERT INTO guardrail_violations (id, direction, action_taken, request_id, created_at) VALUES (gen_random_uuid(), 'inbound', 'blocked', $1, NOW() - INTERVAL '100 days')", old_tag,
    )
    await _pg_execute(
        "guardrails", "INSERT INTO guardrail_violations (id, direction, action_taken, request_id, created_at) VALUES (gen_random_uuid(), 'inbound', 'blocked', $1, NOW() - INTERVAL '10 days')", fresh_tag,
    )
    await _pg_execute(
        "guardrails",
        "INSERT INTO moderation_flags (id, branch_id, version_id, asset_path, category, model, status, created_at) "
        "VALUES (gen_random_uuid(), $1, 'v1', 'path/old-cleared', 'nudity', 'test-model', 'cleared', NOW() - INTERVAL '100 days')",
        old_cleared_tag,
    )
    await _pg_execute(
        "guardrails",
        "INSERT INTO moderation_flags (id, branch_id, version_id, asset_path, category, model, status, created_at) "
        "VALUES (gen_random_uuid(), $1, 'v1', 'path/fresh-cleared', 'nudity', 'test-model', 'cleared', NOW() - INTERVAL '10 days')",
        fresh_cleared_tag,
    )
    await _pg_execute(
        "guardrails",
        "INSERT INTO moderation_flags (id, branch_id, version_id, asset_path, category, model, status, created_at) "
        "VALUES (gen_random_uuid(), $1, 'v1', 'path/old-actionable', 'nudity', 'test-model', 'flagged', NOW() - INTERVAL '1200 days')",
        old_actionable_tag,
    )
    await _pg_execute(
        "guardrails",
        "INSERT INTO moderation_flags (id, branch_id, version_id, asset_path, category, model, status, created_at) "
        "VALUES (gen_random_uuid(), $1, 'v1', 'path/fresh-actionable', 'nudity', 'test-model', 'flagged', NOW() - INTERVAL '10 days')",
        fresh_actionable_tag,
    )
    await _pg_execute(
        "privacy_compliance", "INSERT INTO gdpr_access_log (user_id, action, timestamp) VALUES ($1, 'retrieve', NOW() - INTERVAL '400 days')", old_tag,
    )
    await _pg_execute(
        "privacy_compliance", "INSERT INTO gdpr_access_log (user_id, action, timestamp) VALUES ($1, 'retrieve', NOW() - INTERVAL '10 days')", fresh_tag,
    )

    # ── Seed one old + one fresh memvid session (real files, backdated mtime) ──
    user_dir = Path(MEMVID_STORAGE_ROOT) / f"user_{TEST_TAG}"
    user_dir.mkdir(parents=True, exist_ok=True)
    old_shard = user_dir / f"session_{old_tag}_0.mv2"
    fresh_shard = user_dir / f"session_{fresh_tag}_0.mv2"
    old_shard.touch()
    fresh_shard.touch()
    old_mtime = time.time() - 400 * 86400
    os.utime(old_shard, (old_mtime, old_mtime))

    pool = await asyncpg.create_pool(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db, min_size=1, max_size=3,
    )
    engine = RetentionEngine(pool=pool)

    try:
        # ── Dry-run counts at least the 5 old records we just seeded ────────
        counts = await engine.dry_run()
        assert counts["benchmark_runs"] >= 1
        assert counts["guardrail_violations"] >= 1
        assert counts["moderation_flags"] >= 1
        assert counts["moderation_flags_actionable"] >= 1
        assert counts["gdpr_access_log"] >= 1
        assert counts["memvid_sessions"] >= 1

        # ── Enforce for real ──────────────────────────────────────────────
        results = await engine.enforce()
        for category in (
            "benchmark_runs", "guardrail_violations", "moderation_flags", "moderation_flags_actionable",
            "gdpr_access_log", "memvid_sessions",
        ):
            assert results[category]["rows_deleted"] >= 1, results

        # ── Old rows/files gone, fresh ones survive ─────────────────────────
        assert await _pg_fetchval("benchmark", "SELECT COUNT(*) FROM benchmark_runs WHERE branch_id = $1", old_tag) == 0
        assert await _pg_fetchval("benchmark", "SELECT COUNT(*) FROM benchmark_runs WHERE branch_id = $1", fresh_tag) == 1
        assert await _pg_fetchval("guardrails", "SELECT COUNT(*) FROM guardrail_violations WHERE request_id = $1", old_tag) == 0
        assert await _pg_fetchval("guardrails", "SELECT COUNT(*) FROM guardrail_violations WHERE request_id = $1", fresh_tag) == 1
        # moderation_flags: old+cleared (90-day policy) and old+actionable
        # (1095-day policy, flagged/removed) each gone via their own category;
        # the fresh row of each status survives regardless of status.
        assert await _pg_fetchval("guardrails", "SELECT COUNT(*) FROM moderation_flags WHERE branch_id = $1", old_cleared_tag) == 0
        assert await _pg_fetchval("guardrails", "SELECT COUNT(*) FROM moderation_flags WHERE branch_id = $1", fresh_cleared_tag) == 1
        assert await _pg_fetchval("guardrails", "SELECT COUNT(*) FROM moderation_flags WHERE branch_id = $1", old_actionable_tag) == 0
        assert await _pg_fetchval("guardrails", "SELECT COUNT(*) FROM moderation_flags WHERE branch_id = $1", fresh_actionable_tag) == 1
        assert await _pg_fetchval("privacy_compliance", "SELECT COUNT(*) FROM gdpr_access_log WHERE user_id = $1", old_tag) == 0
        assert await _pg_fetchval("privacy_compliance", "SELECT COUNT(*) FROM gdpr_access_log WHERE user_id = $1", fresh_tag) == 1
        assert not old_shard.exists()
        assert not old_shard.with_suffix(".meta.json").exists()
        assert fresh_shard.exists()

        # ── gdpr_retention_log recorded real entries for this run ───────────
        log_count = await _pg_fetchval(
            "privacy_compliance",
            "SELECT COUNT(*) FROM gdpr_retention_log WHERE data_category = ANY($1) AND run_at > NOW() - INTERVAL '1 minute'",
            [
                "benchmark_runs", "guardrail_violations", "moderation_flags", "moderation_flags_actionable",
                "gdpr_access_log", "memvid_sessions",
            ],
        )
        assert log_count == 6
    finally:
        await engine.close()
        await pool.close()
        await _pg_execute("benchmark", "DELETE FROM benchmark_runs WHERE branch_id = ANY($1)", [old_tag, fresh_tag])
        await _pg_execute("guardrails", "DELETE FROM guardrail_violations WHERE request_id = ANY($1)", [old_tag, fresh_tag])
        await _pg_execute(
            "guardrails", "DELETE FROM moderation_flags WHERE branch_id = ANY($1)",
            [old_cleared_tag, fresh_cleared_tag, old_actionable_tag, fresh_actionable_tag],
        )
        await _pg_execute("privacy_compliance", "DELETE FROM gdpr_access_log WHERE user_id = ANY($1)", [old_tag, fresh_tag])
        shutil.rmtree(user_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_gateway_retention_routes_require_gdpr_admin_role(cleanup_registry):
    admin_token = _keycloak_admin_token()
    username = f"gd3-e2e-{uuid.uuid4().hex[:12]}"
    password = "TempPassw0rd!1"
    user_id = _create_gdpr_admin_user(admin_token, username, password)
    cleanup_registry.append(username)

    user_jwt = _user_jwt(username, password)
    auth_headers = {"Authorization": f"Bearer {user_jwt}"}

    get_resp = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/retention/policy", headers=auth_headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert get_resp.status_code == 200, get_resp.text
    policies = {p["data_category"]: p for p in get_resp.json()["policies"]}
    assert set(policies) == {
        "benchmark_runs", "guardrail_violations", "moderation_flags", "moderation_flags_actionable", "memvid_sessions",
        "gdpr_consent_history", "gdpr_access_log", "gdpr_erasure_jobs",
        "voice_recordings", "loki_logs",
    }

    dry_run_resp = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/retention/policy/dry-run", headers=auth_headers, timeout=15.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert dry_run_resp.status_code == 200, dry_run_resp.text
    assert set(dry_run_resp.json()["counts"]) == {
        "benchmark_runs", "guardrail_violations", "moderation_flags", "moderation_flags_actionable",
        "gdpr_access_log", "memvid_sessions",
    }

    original_days = policies["guardrail_violations"]["retention_days"]
    put_resp = httpx.put(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/retention/policy", headers=auth_headers, timeout=10.0,
        json=[{"data_category": "guardrail_violations", "retention_days": original_days}],  # no-op value, just proves the round trip
    )
    assert put_resp.status_code == 200, put_resp.text
    updated = {p["data_category"]: p for p in put_resp.json()["policies"]}
    assert updated["guardrail_violations"]["retention_days"] == original_days
    # X-Requested-By carries the JWT's `sub` (a stable UUID), not the
    # username -- same convention erasure/export's admin routes already use.
    assert updated["guardrail_violations"]["updated_by"] == user_id


@pytest.mark.asyncio
async def test_gateway_retention_routes_reject_non_admin(cleanup_registry):
    """The same _is_gdpr_admin() check erasure/export's admin routes already
    use -- one negative case here is enough to prove retention's 3 new
    routes are actually wired to it, not just copy-pasted decoration."""
    admin_token = _keycloak_admin_token()
    username = f"gd3-e2e-nonadmin-{uuid.uuid4().hex[:8]}"
    password = "TempPassw0rd!1"

    headers = {"Authorization": f"Bearer {admin_token}"}
    resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users", headers=headers,
        json={
            "username": username, "firstName": "GD3", "lastName": "NonAdmin",
            "email": f"{username}@test.invalid", "emailVerified": True,
            "enabled": True, "requiredActions": [],
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    cleanup_registry.append(username)
    user_id = httpx.get(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers, params={"username": username, "exact": "true"}, timeout=10.0,
    ).json()[0]["id"]
    client_uuid = httpx.get(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients",
        headers=headers, params={"clientId": KEYCLOAK_CLIENT_ID}, timeout=10.0,
    ).json()[0]["id"]
    user_standard_role = httpx.get(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients/{client_uuid}/roles/user_standard",
        headers=headers, timeout=10.0,
    ).json()
    httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
        headers=headers, json=[user_standard_role], timeout=10.0,
    ).raise_for_status()
    # Deliberately NOT granting gdpr-admin.

    user_jwt = _user_jwt(username, password)
    auth_headers = {"Authorization": f"Bearer {user_jwt}"}

    resp = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/retention/policy", headers=auth_headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert resp.status_code == 403, resp.text
