"""
GD-2 / VIE01-247: end-to-end export test against a real deployment.

create Keycloak test user -> upload 2 files -> chat 3 turns -> submit
export (GET, the ticket's method) -> poll until COMPLETED -> download the
ZIP(s) -> verify the manifest and the ticket's sections: present where this
codebase backs them, explained in manifest.omitted where it doesn't. Each
files/index.json link must download the very bytes that were uploaded,
while the ZIP itself carries none.
Mirrors test_erasure_e2e.py's structure and the ticket's own workflow.

Deliberately live-infrastructure, not mocked -- the point is proving the
ZIP actually contains real cross-store content, not that the orchestrator
calls the right functions. Skips cleanly if the gateway isn't reachable.

Run against a live deployment from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/test_export_e2e.py -v
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
import zipfile
from urllib.parse import urlparse, urlunparse

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
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")

POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 300
# S3_PUBLIC_URL is a LAN IP meant for clients on that LAN (e.g. a browser
# hitting Swagger); it isn't necessarily reachable from wherever this test
# itself runs. Rewritten to the S3 gateway's host-mapped port on localhost
# for the download step only -- the URL's signature/query string (and thus
# whether the presign+upload logic itself is correct) is untouched.
DOWNLOAD_HOST_OVERRIDE = os.environ.get("TEST_S3_DOWNLOAD_HOST", "")


def _resolve_download_url(url: str) -> str:
    if not DOWNLOAD_HOST_OVERRIDE:
        return url
    parsed = urlparse(url)
    override = urlparse(DOWNLOAD_HOST_OVERRIDE if "//" in DOWNLOAD_HOST_OVERRIDE else f"//{DOWNLOAD_HOST_OVERRIDE}")
    return urlunparse(parsed._replace(scheme=override.scheme or parsed.scheme, netloc=override.netloc))


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


# ── Keycloak admin helpers (same as test_erasure_e2e.py) ───────────────────

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
    headers = {"Authorization": f"Bearer {admin_token}"}
    resp = httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
        headers=headers,
        json={
            "username": username, "firstName": "GD2", "lastName": "E2ETest",
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
    resp = httpx.post(
        f"{GATEWAY_URL}/api/{API_VERSION}/sso/login",
        data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM, "client_id": KEYCLOAK_CLIENT_ID},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


@pytest_asyncio.fixture
async def cleanup_registry():
    """Export doesn't delete the test account itself -- always clean it up."""
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
async def test_export_produces_zip_with_real_cross_store_content(cleanup_registry):
    admin_token = _keycloak_admin_token()
    username = f"gd2-e2e-{uuid.uuid4().hex[:12]}"
    password = "TempPassw0rd!1"
    user_id = _create_test_user(admin_token, username, password)
    cleanup_registry.append(username)

    user_jwt = _user_jwt(username, password)
    auth_headers = {"Authorization": f"Bearer {user_jwt}"}

    # ── Upload 2 files (Module 04) ───────────────────────────────────────
    uploaded: dict[str, bytes] = {}
    for i in range(2):
        uploaded[f"gd2_e2e_{i}.txt"] = f"hello gd-2 e2e test file {i}".encode()
        upload_resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
            f"{GATEWAY_URL}/api/{API_VERSION}/asset/fileupload",
            headers=auth_headers,
            files={"primary_file": (f"gd2_e2e_{i}.txt", uploaded[f"gd2_e2e_{i}.txt"], "text/plain")},
            timeout=30.0,
        )
        assert upload_resp.status_code == 200, upload_resp.text

    # ── Chat, 3 turns (Module 15/26's memory) ────────────────────────────
    for i in range(3):
        chat_resp = httpx.post(  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
            f"{GATEWAY_URL}/api/{API_VERSION}/chat/chat",
            headers=auth_headers,
            json={"message": f"Hello, this is GD-2 e2e chat turn {i}."},
            timeout=60.0,
        )
        assert chat_resp.status_code == 200, chat_resp.text

    # ── Submit export ─────────────────────────────────────────────────
    submit_resp = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/export", headers=auth_headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
    assert submit_resp.status_code == 200, submit_resp.text
    job_id = submit_resp.json()["job_id"]
    assert submit_resp.json()["status"] == "pending"

    # ── Poll until COMPLETED -- JWT-gated (unlike erasure's poll, export
    # doesn't kill the submitter's session, so no self-lockout problem to
    # design around; ownership is also checked server-side in gdpr.py) ──
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    job = None
    while time.time() < deadline:
        resp = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/export/jobs/{job_id}", headers=auth_headers, timeout=10.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["status"] in ("completed", "partial_failure"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)  # noqa: ASYNC251 -- e2e polling loop, no concurrent event loop pressure
    assert job is not None, "export job never reached a terminal state"
    assert job["status"] in ("completed", "partial_failure"), f"unexpected status: {job['status']}"
    assert job["download_urls"], f"no download_urls on a {job['status']} job: {job}"
    assert job["download_url"] == job["download_urls"][0]
    assert job["size_bytes"] and job["expires_at"]

    # ── Download and inspect every ZIP part ──────────────────────────────
    all_names: set[str] = set()
    manifest = None
    files_index = None
    for url in job["download_urls"]:
        zip_resp = httpx.get(_resolve_download_url(url), timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        assert zip_resp.status_code == 200, f"failed to download {url}: {zip_resp.status_code}"
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            names = zf.namelist()
            all_names.update(names)
            if "manifest.json" in names:
                manifest = json.loads(zf.read("manifest.json"))
            if "files/index.json" in names:
                files_index = json.loads(zf.read("files/index.json"))

    assert manifest is not None, "manifest.json missing from every ZIP part"
    assert manifest["user_id"] == user_id
    assert manifest["export_schema_version"] == "1.0"
    assert manifest["gdpr_article"] == "Art. 20 Right to Data Portability"
    assert manifest["ccpa_section"] == "§ 1798.100"
    assert manifest["exported_at"]
    assert len(manifest["parts"]) == len(job["download_urls"])
    assert "speaker_embeddings_cached" in manifest
    assert "module_04" in manifest["stores_included"], f"module_04 missing/failed: {manifest['stores_failed']}"

    # Files: metadata + a working link each, never bytes in the ZIP.
    assert files_index is not None and not any(
        n.startswith("files/") and n != "files/index.json" for n in all_names), all_names
    by_name = {f["name"]: f for f in files_index}
    assert set(by_name) == set(uploaded), files_index
    for name, content in uploaded.items():
        entry = by_name[name]
        assert entry["size"] == len(content) and entry["type"] and entry["uploaded_at"]
        file_resp = httpx.get(_resolve_download_url(entry["download_url"]), timeout=30.0)  # noqa: ASYNC210 -- e2e test, real service calls run sequentially, no concurrent event loop
        assert file_resp.status_code == 200 and file_resp.content == content, (name, file_resp.status_code)

    # The ticket's 7 sections: present for every store that answered,
    # explained in manifest.omitted where this codebase has no backing.
    if "module_06" in manifest["stores_included"]:
        assert "identity/keycloak_profile.json" in all_names
    if "module_25" in manifest["stores_included"]:
        assert {"content/documents.json", "content/entities.json", "content/moments.json"} <= all_names
    if "module_26" in manifest["stores_included"]:
        assert "memory/long_term.json" in all_names
        assert any(n.startswith("memory/sessions/") for n in all_names), all_names
    if "module_03" in manifest["stores_included"]:
        assert "benchmark/runs.json" in all_names
    if "module_23" in manifest["stores_included"]:
        assert "moderation/flags.json" in all_names
    if "module_34" in manifest["stores_included"]:
        assert "voice/recordings.json" in all_names
    elif "module_34" not in manifest["stores_failed"]:   # GD_MODULE_34_URL empty here
        assert "voice/recordings.json" in manifest["omitted"]
    assert "audit/violations.json" in manifest["omitted"]
