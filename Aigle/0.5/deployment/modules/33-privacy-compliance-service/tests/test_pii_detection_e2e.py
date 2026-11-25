"""
GD-6 / VIE01-251: end-to-end PII detection at document ingestion, against a
real deployment.

Two documents go through the real pipeline (gateway upload -> process-file
-> Module 12 analysis/summary/indexer -> Module 23 guard-model check ->
Kafka -> Module 25): one with a name, ID number and phone number, one
without. Each
effect is checked through an independent path:
- pii_flag on the Chunk vertices: ArcadeDB's own HTTP API, not Module 25
- pii_warning in search results: Module 25's hybrid search
- pii_flagged count: the DSAR, through the gateway

Requires Module 12's document-indexer running with GUARDRAIL_URL set,
Module 23's guardrail switched on, and the document summary model
reachable (the pipeline stops at summary otherwise). Skips cleanly if the
gateway or Module 23's /guard/check/input isn't answering.

Run from the 33-privacy-compliance-service/ directory:
    TEST_KEYCLOAK_ADMIN_PASSWORD=... TEST_ARCADEDB_PASSWORD=... \
        python -m pytest tests/test_pii_detection_e2e.py -v
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid

import httpx
import pytest

logger = logging.getLogger(__name__)

GATEWAY_URL = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
MODULE_23_URL = os.environ.get("TEST_MODULE_23_URL", "http://localhost:8023")
MODULE_25_URL = os.environ.get("TEST_MODULE_25_URL", "http://localhost:8025")
ARCADEDB_URL = os.environ.get("TEST_ARCADEDB_URL", "http://localhost:2480")
ARCADEDB_PASSWORD = os.environ.get("TEST_ARCADEDB_PASSWORD", "")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("TEST_PIPELINE_TIMEOUT_SECONDS", "900"))

PII_DOC = ("gd6_e2e_pii.txt",
           # A sample Llama Guard reliably flags S7 -- it can miss others (e.g. an
           # English email + phone sentence did), see the GD-6 PR notes.
           ("GD-6 e2e document. Employee record: John Smith, national ID A123456789, "
           "phone +886 912 345 678. The quarterly report is attached.\n"))
CLEAN_DOC = ("gd6_e2e_clean.txt",
             ("GD-6 e2e document. The quarterly report shows revenue grew twelve percent "
             "compared to last year.\n"))


def _reachable(url: str) -> bool:
    try:
        httpx.get(url, timeout=3.0)
        return True
    except Exception:  # noqa: BLE001 -- reachability probe feeding pytest.skip -- skip reason itself communicates it
        return False


def _guard_model_answers() -> bool:
    """Module 23's /guard/check/input is up and actually classifying (not
    switched off, which answers every request "guardrail disabled")."""
    try:
        resp = httpx.post(f"{MODULE_23_URL}/guard/check/input", json={"content": "hello", "module": "e2e"},
                          timeout=60.0)
        return resp.status_code == 200 and resp.json().get("raw") != "guardrail disabled"
    except Exception:  # noqa: BLE001 -- reachability probe feeding pytest.skip -- skip reason itself communicates it
        return False


pytestmark = pytest.mark.skipif(
    not (KEYCLOAK_ADMIN_PASSWORD and ARCADEDB_PASSWORD) or not _reachable(f"{GATEWAY_URL}/docs")
    or not _guard_model_answers(),
    reason="requires a live deployment (gateway and Module 23 /guard/check/input answering; "
           "TEST_KEYCLOAK_ADMIN_PASSWORD, TEST_ARCADEDB_PASSWORD set)",
)


# ── Keycloak helpers (same shape as the other *_e2e.py tests) ───────────────

def _admin_headers() -> dict:
    resp = httpx.post(
        f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "admin-cli",
              "username": KEYCLOAK_ADMIN_USER, "password": KEYCLOAK_ADMIN_PASSWORD},
        timeout=10.0,
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_test_user(username: str, password: str) -> str:
    headers = _admin_headers()
    httpx.post(
        f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users", headers=headers,
        json={"username": username, "firstName": "GD6", "lastName": "E2ETest",
              "email": f"{username}@example.com", "emailVerified": True, "enabled": True, "requiredActions": [],
              "credentials": [{"type": "password", "value": password, "temporary": False}]},
        timeout=10.0,
    ).raise_for_status()
    user_id = httpx.get(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users", headers=headers,
                        params={"username": username, "exact": "true"}, timeout=10.0).json()[0]["id"]
    client_uuid = httpx.get(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients", headers=headers,
                            params={"clientId": KEYCLOAK_CLIENT_ID}, timeout=10.0).json()[0]["id"]
    role = httpx.get(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/clients/{client_uuid}/roles/user_standard",
                     headers=headers, timeout=10.0).json()
    httpx.post(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
               headers=headers, json=[role], timeout=10.0).raise_for_status()
    return user_id


def _user_jwt(username: str, password: str) -> str:
    resp = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/sso/login",
                      data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM,
                            "client_id": KEYCLOAK_CLIENT_ID}, timeout=10.0)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── ArcadeDB (ground truth, read directly) ──────────────────────────────────

_ARCADE_AUTH = ("root", ARCADEDB_PASSWORD)


def _arcadedb_name(user_id: str) -> str:
    """Same mapping as Module 25's db_name_for, re-derived here."""
    return "user_" + (re.sub(r"[^A-Za-z0-9_]", "_", user_id).strip("_") or "anon")


def _chunks(user_id: str) -> list[dict]:
    db = _arcadedb_name(user_id)
    if db not in httpx.get(f"{ARCADEDB_URL}/api/v1/databases", auth=_ARCADE_AUTH, timeout=10.0).json()["result"]:
        return []
    resp = httpx.post(f"{ARCADEDB_URL}/api/v1/query/{db}", auth=_ARCADE_AUTH, timeout=10.0,
                      json={"language": "sql", "command": "SELECT chunk_id, filename, pii_flag FROM Chunk"})
    return resp.json().get("result", []) if resp.status_code == 200 else []


# ── Fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def user():
    """A throwaway user; its Keycloak account and ArcadeDB database are
    always removed. Uploaded files stay in Module 04, as in the other
    *_e2e.py tests (covered by GD-1's erasure test)."""
    username, password = f"gd6-e2e-{uuid.uuid4().hex[:10]}", "TempPassw0rd!1"
    user_id = _create_test_user(username, password)
    yield user_id, {"Authorization": f"Bearer {_user_jwt(username, password)}"}
    try:
        headers = _admin_headers()
        httpx.delete(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}", headers=headers, timeout=10.0)
    except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
        logger.debug(f"Teardown Keycloak user delete failed for {user_id}: {e}")
    try:
        httpx.post(f"{ARCADEDB_URL}/api/v1/server", auth=_ARCADE_AUTH, timeout=30.0,
                   json={"command": f"drop database {_arcadedb_name(user_id)}"})
    except Exception as e:  # noqa: BLE001 -- best-effort test teardown cleanup, already logged
        logger.debug(f"Teardown ArcadeDB drop failed for {user_id}: {e}")


def _ingest(auth: dict, name: str, text: str) -> None:
    """Upload, then explicitly start processing (upload and analysis are
    decoupled since VIE01-215)."""
    up = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/asset/fileupload", headers=auth, timeout=60.0,
                    files={"primary_file": (name, text.encode(), "text/plain")})
    assert up.status_code == 200, up.text
    asset = up.json()
    proc = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/processing/process-file", headers=auth, timeout=60.0,
                      json={"asset_path": asset["asset_path"], "version_id": asset["version_id"]})
    assert proc.status_code == 200, proc.text


# ── Test ────────────────────────────────────────────────────────────────────

def test_pii_document_is_flagged_and_surfaced(user):
    user_id, auth = user
    for name, text in (PII_DOC, CLEAN_DOC):
        _ingest(auth, name, text)

    deadline = time.monotonic() + PIPELINE_TIMEOUT_SECONDS
    while True:
        chunks = _chunks(user_id)
        if {c["filename"] for c in chunks} >= {PII_DOC[0], CLEAN_DOC[0]}:
            break
        if time.monotonic() > deadline:
            pytest.fail(f"both documents not indexed within {PIPELINE_TIMEOUT_SECONDS}s (got {chunks}) -- "
                        "check Module 12's summary model is reachable")
        time.sleep(10)

    # ── ArcadeDB: the PII document's chunk is flagged, the other isn't ──────
    pii_chunks = [c for c in chunks if c["filename"] == PII_DOC[0]]
    clean_chunks = [c for c in chunks if c["filename"] == CLEAN_DOC[0]]
    assert pii_chunks and all(c.get("pii_flag") is True for c in pii_chunks), pii_chunks
    assert clean_chunks and all(not c.get("pii_flag") for c in clean_chunks), clean_chunks
    flagged = len(pii_chunks)

    # ── Module 25 hybrid search: annotated, never hidden ────────────────────
    resp = httpx.post(f"{MODULE_25_URL}/personal/search/hybrid", headers={"X-Branch-ID": user_id},
                      json={"query": "quarterly report", "top_k": 10}, timeout=60.0)
    assert resp.status_code == 200, resp.text
    by_file: dict = {}
    for result in resp.json()["results"]:
        payload = result["payload"]
        if payload.get("chunk_id"):  # Chunk rows; Source (summary) rows aren't scanned
            by_file.setdefault(payload.get("filename"), []).append(payload)
    assert by_file.get(PII_DOC[0]), f"PII chunk missing from search results: {by_file}"
    assert all(p.get("pii_warning") is True and p.get("text") for p in by_file[PII_DOC[0]])
    assert all("pii_warning" not in p for p in by_file.get(CLEAN_DOC[0], []))
    assert all("pii_flag" not in p for ps in by_file.values() for p in ps)

    # ── DSAR through the gateway: the real count, no longer "unavailable" ───
    dsar = httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/dsar", headers=auth, timeout=30.0)
    assert dsar.status_code == 200, dsar.text
    report = dsar.json()
    assert report["data_categories"]["indexed_content"]["pii_flagged"] == flagged
    assert "data_categories.indexed_content.pii_flagged" not in report["unavailable_fields"]
