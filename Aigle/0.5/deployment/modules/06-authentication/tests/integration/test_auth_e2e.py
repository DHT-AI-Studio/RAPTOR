"""
Module 06 — Authentication API Integration Test

Exercises the live keycloak-api (Module 06) + Keycloak stack end-to-end:
admin login, group CRUD, admin-created user CRUD (role/group assignment
verified directly against Keycloak), user login + user-facing APIs, path
permission allow/deny, logout, self-registration, and cleanup.

Tests run in file order (pytest's default) and intentionally depend on
each other via the module-scoped `state` fixture, since each step builds
on resources the previous one created -- same flow as the original
standalone script this file replaces, just as real pytest tests instead
of a hand-rolled runner.

Config (host, ports, admin credentials) comes from conftest.py, which
reads the repo-root unified .env.
"""

import pytest
import requests
from conftest import (
    ADMIN_CLIENT,
    ADMIN_PASS,
    ADMIN_REALM,
    ADMIN_USER,
    KC_BASE_URL,
    MODULE06_BASE_URL,
    REG_EMAIL,
    TEST_CLIENT,
    TEST_REALM,
)

API_BASE = MODULE06_BASE_URL
KC_BASE = KC_BASE_URL

ADM_USERNAME = "pytest_admin_created"
ADM_EMAIL = "pytest_admin_created@example.com"

REG_USERNAME = "pytest_self_register"
REG_PASSWORD = "PytestNewPass99!"

TEST_GROUP = "__pytest_group__"

TIMEOUT = 10


@pytest.fixture(scope="module")
def state():
    """Threaded through this module's sequential test steps."""
    return {}


def get_admin_token() -> str:
    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADMIN_USER,
            "password": ADMIN_PASS,
            "realm_name": ADMIN_REALM,
            "client_id": ADMIN_CLIENT,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, f"Cannot get admin token: {resp.status_code} {resp.text}"
    token = resp.json()
    assert isinstance(token, str), f"Unexpected login response: {token}"
    return token


def kc_clear_required_actions(user_id: str, admin_token: str) -> bool:
    """Directly patch Keycloak to clear requiredActions + set emailVerified=true (test helper)."""
    resp = requests.put(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"},
        json={"requiredActions": [], "emailVerified": True},
        timeout=TIMEOUT,
    )
    return resp.status_code == 204


def kc_get_user_id(username: str, admin_token: str) -> str | None:
    resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={"username": username, "exact": "true"},
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        return None
    users = resp.json()
    return users[0]["id"] if users else None


def kc_get_client_roles(user_id: str, admin_token: str) -> list[str]:
    client_resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/clients",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={"clientId": TEST_CLIENT},
        timeout=TIMEOUT,
    )
    clients = client_resp.json()
    exact = next((c for c in clients if c.get("clientId") == TEST_CLIENT), None)
    if not exact:
        return []
    client_uuid = exact["id"]
    roles_resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=TIMEOUT,
    )
    return [r["name"] for r in roles_resp.json()] if roles_resp.status_code == 200 else []


def kc_get_user_groups(user_id: str, admin_token: str) -> list[str]:
    resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}/groups",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=TIMEOUT,
    )
    return [g.get("path", g.get("name", "")) for g in resp.json()] if resp.status_code == 200 else []


def kc_delete_user(username: str, admin_token: str) -> bool:
    user_id = kc_get_user_id(username, admin_token)
    if not user_id:
        return True
    resp = requests.delete(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=TIMEOUT,
    )
    return resp.status_code == 204


def kc_delete_group_by_name(group_name: str, admin_token: str) -> bool:
    resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/groups",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        return False
    groups = resp.json()
    target = next((g for g in groups if g.get("name") == group_name), None)
    if not target:
        return True
    del_resp = requests.delete(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/groups/{target['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=TIMEOUT,
    )
    return del_resp.status_code == 204


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    try:
        kc_token = get_admin_token()
        for username in [ADM_USERNAME, REG_USERNAME]:
            kc_delete_user(username, kc_token)
        kc_delete_group_by_name(TEST_GROUP, kc_token)
    except Exception as e:  # noqa: BLE001 -- test teardown; best-effort, failure printed and doesn't affect test assertions
        print(f"cleanup error: {e}")


def test_health():
    resp = requests.get(f"{API_BASE}/keycloak/health", timeout=TIMEOUT)
    assert resp.status_code == 200
    assert resp.json().get("reachable") is True


def test_admin_login(state):
    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADMIN_USER,
            "password": ADMIN_PASS,
            "realm_name": ADMIN_REALM,
            "client_id": ADMIN_CLIENT,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()
    assert isinstance(token, str) and token.startswith("eyJ")
    state["admin_token"] = token

    bad = requests.post(
        f"{API_BASE}/SSO/login",
        data={"username": "nobody", "password": "wrong", "realm_name": ADMIN_REALM, "client_id": ADMIN_CLIENT},
        timeout=TIMEOUT,
    )
    assert bad.status_code in (401, 400)


def test_group_crud(state):
    token = state["admin_token"]

    resp = requests.post(
        f"{API_BASE}/keycloak/group",
        params={"group_name": TEST_GROUP},
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 201, resp.text

    resp = requests.get(
        f"{API_BASE}/keycloak/group",
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert any(g.get("name") == TEST_GROUP for g in resp.json())

    no_auth = requests.get(f"{API_BASE}/keycloak/group", timeout=TIMEOUT)
    assert no_auth.status_code in (401, 403)


def test_admin_user_crud(state):
    token = state["admin_token"]

    resp = requests.post(
        f"{API_BASE}/admin/keycloak/user",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "username": ADM_USERNAME,
            "email": ADM_EMAIL,
            "firstName": "Pytest",
            "lastName": "AdminCreated",
            "groups": [f"/{TEST_GROUP}"],
            "clientRoles": {"raptor": ["user_basic"]},
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("message") == "User created successfully."
    state["adm_temp_pass"] = body.get("temporaryPassword")

    resp = requests.get(
        f"{API_BASE}/admin/keycloak/user",
        params={"keyword": ADM_USERNAME, "query": "username"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert any(u.get("username") == ADM_USERNAME for u in resp.json())

    admin_kc_token = get_admin_token()
    uid = kc_get_user_id(ADM_USERNAME, admin_kc_token)
    assert uid, f"user '{ADM_USERNAME}' not found via Keycloak"

    roles = kc_get_client_roles(uid, admin_kc_token)
    assert "user_basic" in roles, f"roles: {roles}"

    groups = kc_get_user_groups(uid, admin_kc_token)
    assert any(TEST_GROUP in g for g in groups), f"groups: {groups}"

    # clear requiredActions so this user can login in the next step
    assert kc_clear_required_actions(uid, admin_kc_token)
    state["adm_user_id"] = uid


def test_user_login_and_user_apis(state):
    temp_pass = state["adm_temp_pass"]

    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADM_USERNAME,
            "password": temp_pass,
            "realm_name": TEST_REALM,
            "client_id": TEST_CLIENT,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    user_token = resp.json()
    assert isinstance(user_token, str) and user_token.startswith("eyJ")
    state["user_token"] = user_token

    resp = requests.get(
        f"{API_BASE}/keycloak/user-groups",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = requests.get(
        f"{API_BASE}/keycloak/user-group-id",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    new_pass = "PytestChanged99!"
    resp = requests.post(
        f"{API_BASE}/SSO/change-password",
        headers={"Authorization": f"Bearer {user_token}"},
        data={"new_password": new_pass, "realm_name": TEST_REALM},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert resp.json().get("message") == "Password updated successfully."

    resp2 = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADM_USERNAME,
            "password": new_pass,
            "realm_name": TEST_REALM,
            "client_id": TEST_CLIENT,
        },
        timeout=TIMEOUT,
    )
    assert resp2.status_code == 200, resp2.text
    new_token = resp2.json()
    assert isinstance(new_token, str) and new_token.startswith("eyJ")
    state["user_token"] = new_token  # update to fresh token


def test_path_and_permission(state):
    user_token = state["user_token"]

    resp = requests.post(
        f"{API_BASE}/API/inference/1",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200 and "path" in resp.json()

    resp = requests.post(
        f"{API_BASE}/API/finetune/1",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 403

    resp = requests.get(
        f"{API_BASE}/auth/permission",
        params={"request_path": "/API/inference/1"},
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert resp.json().get("message") == "Permission granted"

    resp = requests.get(
        f"{API_BASE}/auth/permission",
        params={"request_path": "/API/finetune/1"},
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 403
    assert resp.json().get("detail") == "not_authorized"

    resp = requests.get(
        f"{API_BASE}/auth/permission",
        params={"request_path": "/does/not/exist"},
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 403


def test_logout(state):
    user_token = state["user_token"]

    resp = requests.post(
        f"{API_BASE}/SSO/logout",
        headers={
            "Authorization": f"Bearer {user_token}",
            "Cookie": f"realm_name={TEST_REALM}",
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert "Logout successful" in resp.json().get("message", "")


def test_self_register(state):
    resp = requests.post(
        f"{API_BASE}/SSO/register",
        data={
            "username": REG_USERNAME,
            "email": REG_EMAIL,
            "firstName": "Pytest",
            "lastName": "SelfReg",
            "realm_name": TEST_REALM,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("message") == "Registration successful."

    kc_token = get_admin_token()
    uid = kc_get_user_id(REG_USERNAME, kc_token)
    assert uid, f"'{REG_USERNAME}' not found via Keycloak"

    roles = kc_get_client_roles(uid, kc_token)
    assert "user_basic" in roles, f"roles: {roles}"

    groups = kc_get_user_groups(uid, kc_token)
    assert not groups, f"unexpectedly in: {groups}"

    # simulate user completing email verification (test helper — clear requiredActions)
    assert kc_clear_required_actions(uid, kc_token)
    state["reg_user_id"] = uid

    pw_resp = requests.put(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{uid}/reset-password",
        headers={"Authorization": f"Bearer {kc_token}", "Content-Type": "application/json"},
        json={"type": "password", "value": REG_PASSWORD, "temporary": False},
        timeout=TIMEOUT,
    )
    assert pw_resp.status_code == 204

    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": REG_USERNAME,
            "password": REG_PASSWORD,
            "realm_name": TEST_REALM,
            "client_id": TEST_CLIENT,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    reg_token = resp.json()
    assert isinstance(reg_token, str) and reg_token.startswith("eyJ")

    resp = requests.post(
        f"{API_BASE}/API/inference/1",
        headers={"Authorization": f"Bearer {reg_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200

    resp = requests.post(
        f"{API_BASE}/API/finetune/1",
        headers={"Authorization": f"Bearer {reg_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 403

    resp = requests.delete(
        f"{API_BASE}/SSO/account",
        params={"realm_name": TEST_REALM},
        headers={"Authorization": f"Bearer {reg_token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    assert "deleted" in resp.json().get("message", "").lower()
    state.pop("reg_user_id", None)  # cleaned up


def test_delete_admin_user(state):
    token = state["admin_token"]
    uid = state["adm_user_id"]

    resp = requests.delete(
        f"{API_BASE}/admin/keycloak/user",
        params={"user_id": uid},
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
