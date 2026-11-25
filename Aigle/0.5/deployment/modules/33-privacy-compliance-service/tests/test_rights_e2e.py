"""
PRIV-X3 / VIE01-357: end-to-end rectification, restriction and objection
against a real deployment, through the gateway.

A throwaway user (and a throwaway gdpr-admin for the lift) is created in
Keycloak; every effect is checked through an independent path -- Keycloak's
own Admin API for the profile and attributes, the gateway's consent route
for the objection, real write/read requests for the restriction.

Skips cleanly if the gateway isn't reachable. Run from the
33-privacy-compliance-service/ directory:
    TEST_KEYCLOAK_ADMIN_PASSWORD=... python -m pytest tests/test_rights_e2e.py -v
"""
from __future__ import annotations

import os
import uuid

import httpx
import pytest

GATEWAY_URL = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
API_VERSION = os.environ.get("TEST_API_VERSION", "0.5")
GDPR = f"{GATEWAY_URL}/api/{API_VERSION}/gdpr"


def _reachable() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/docs", timeout=3.0)
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not KEYCLOAK_ADMIN_PASSWORD or not _reachable(),
    reason="requires a live deployment (gateway reachable, TEST_KEYCLOAK_ADMIN_PASSWORD set)",
)


def _admin_headers() -> dict:
    resp = httpx.post(
        f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "admin-cli",
              "username": KEYCLOAK_ADMIN_USER, "password": KEYCLOAK_ADMIN_PASSWORD},
        timeout=10.0,
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_user(username: str, password: str, gdpr_admin: bool = False) -> str:
    h = _admin_headers()
    base = f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}"
    httpx.post(f"{base}/users", headers=h, timeout=10.0, json={
        "username": username, "firstName": "PrivX3", "lastName": "E2E", "email": f"{username}@example.com",
        "emailVerified": True, "enabled": True, "requiredActions": [],
        "credentials": [{"type": "password", "value": password, "temporary": False}],
    }).raise_for_status()
    user_id = httpx.get(f"{base}/users", headers=h, params={"username": username, "exact": "true"},
                        timeout=10.0).json()[0]["id"]
    client_uuid = httpx.get(f"{base}/clients", headers=h, params={"clientId": KEYCLOAK_CLIENT_ID},
                            timeout=10.0).json()[0]["id"]
    role = httpx.get(f"{base}/clients/{client_uuid}/roles/user_standard", headers=h, timeout=10.0).json()
    httpx.post(f"{base}/users/{user_id}/role-mappings/clients/{client_uuid}", headers=h, json=[role],
               timeout=10.0).raise_for_status()
    if gdpr_admin:
        realm_role = httpx.get(f"{base}/roles/gdpr-admin", headers=h, timeout=10.0).json()
        httpx.post(f"{base}/users/{user_id}/role-mappings/realm", headers=h, json=[realm_role],
                   timeout=10.0).raise_for_status()
    return user_id


def _login(username: str, password: str) -> dict:
    resp = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/sso/login", timeout=10.0,
                      data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM,
                            "client_id": KEYCLOAK_CLIENT_ID})
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _keycloak_user(user_id: str) -> dict:
    return httpx.get(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}",
                     headers=_admin_headers(), timeout=10.0).json()


@pytest.fixture
def users():
    """A throwaway data subject and a throwaway gdpr-admin, always removed.
    Restriction/objection/consent log rows stay (append-only legal records)."""
    tag, password = uuid.uuid4().hex[:10], "TempPassw0rd!1"
    subject = _create_user(f"privx3-e2e-{tag}", password)
    admin = _create_user(f"privx3-e2e-admin-{tag}", password, gdpr_admin=True)
    yield ((subject, _login(f"privx3-e2e-{tag}", password)), (admin, _login(f"privx3-e2e-admin-{tag}", password)))
    h = _admin_headers()
    for user_id in (subject, admin):
        httpx.delete(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}", headers=h, timeout=10.0)


def test_rectification_updates_the_keycloak_profile(users):
    (user_id, auth), _ = users
    r = httpx.post(f"{GDPR}/rectify", headers=auth, json={"field": "lastName", "value": "Rectified"}, timeout=30.0)
    assert r.status_code == 200, r.text
    assert r.json() == {"fields_updated": ["keycloak.lastName"], "propagation_status": "complete"}
    assert _keycloak_user(user_id)["lastName"] == "Rectified"
    r = httpx.post(f"{GDPR}/rectify", headers=auth, json={"field": "username", "value": "x"}, timeout=30.0)
    assert r.status_code == 422


def test_objection_stops_the_purpose_without_deleting(users):
    (_user_id, auth), _ = users
    r = httpx.post(f"{GDPR}/object", headers=auth, json={"processing_purpose": "memory"}, timeout=30.0)
    assert r.status_code == 200, r.text
    assert r.json() == {"purposes_stopped": ["memory"], "data_deleted": False}
    consent = httpx.get(f"{GDPR}/consent", headers=auth, timeout=30.0).json()
    assert {p["id"]: p["granted"] for p in consent["purposes"]}["memory"] is False
    r = httpx.post(f"{GDPR}/object", headers=auth, json={"processing_purpose": "face_detection"}, timeout=30.0)
    assert r.status_code == 422


def test_restriction_blocks_writes_until_an_admin_lifts_it(users):
    (user_id, auth), (_, admin_auth) = users
    upload = {"primary_file": ("privx3_e2e.txt", f"PRIV-X3 e2e {uuid.uuid4().hex}".encode(), "text/plain")}

    r = httpx.post(f"{GDPR}/restrict", headers=auth, json={"reason": "e2e"}, timeout=30.0)
    assert r.status_code == 200 and r.json()["restricted"] is True, r.text
    assert _keycloak_user(user_id)["attributes"]["gdpr_processing_restricted"] == ["true"]

    # Writes refused with the ticket's body; reads and GDPR rights still work.
    r = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/asset/fileupload", headers=auth, files=upload, timeout=30.0)
    assert r.status_code == 451 and r.json() == {"reason": "processing_restricted"}
    assert httpx.get(f"{GATEWAY_URL}/api/{API_VERSION}/asset/users/commits", headers=auth,
                     timeout=30.0).status_code == 200
    assert httpx.get(f"{GDPR}/dsar", headers=auth, timeout=60.0).status_code == 200

    # Only a gdpr-admin lifts it.
    assert httpx.post(f"{GDPR}/restrict/lift", headers=auth, json={"user_id": user_id},
                      timeout=30.0).status_code == 403
    r = httpx.post(f"{GDPR}/restrict/lift", headers=admin_auth, json={"user_id": user_id},
                   timeout=30.0)
    assert r.status_code == 200 and r.json()["restricted"] is False, r.text
    assert "gdpr_processing_restricted" not in (_keycloak_user(user_id).get("attributes") or {})

    r = httpx.post(f"{GATEWAY_URL}/api/{API_VERSION}/asset/fileupload", headers=auth, files=upload, timeout=60.0)
    assert r.status_code == 200, r.text
