#!/usr/bin/env python3
"""
Module 06 — Authentication API Integration Test
Run against a live stack: python3 test.py

Requirements: pip install requests
"""

import sys
import requests

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_BASE = "http://localhost:8800"   # keycloak-api (FastAPI)
KC_BASE  = "http://localhost:8080"   # Keycloak directly (for test helpers)

ADMIN_USER   = "dhtsolution"
ADMIN_PASS   = "password"
ADMIN_REALM  = "master"
ADMIN_CLIENT = "admin-cli"

TEST_REALM   = "dhtsolution"
TEST_CLIENT  = "raptor"

# test user created via admin API
ADM_USERNAME  = "pytest_admin_created"
ADM_EMAIL     = "pytest_admin_created@example.com"
ADM_PASSWORD  = None   # filled from API response

# test user created via self-register
REG_USERNAME  = "pytest_self_register"
REG_EMAIL     = "cing@dhtsolution.group"   # real address so SMTP works
REG_PASSWORD  = "PytestNewPass99!"         # set after email verification

TEST_GROUP    = "__pytest_group__"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

pass_count = 0
fail_count = 0
skip_count = 0

def ok(name: str):
    global pass_count
    pass_count += 1
    print(f"  {GREEN}PASS{RESET}  {name}")

def fail(name: str, detail: str = ""):
    global fail_count
    fail_count += 1
    msg = f"  {RED}FAIL{RESET}  {name}"
    if detail:
        msg += f"\n        {RED}↳ {detail}{RESET}"
    print(msg)

def skip(name: str, reason: str = ""):
    global skip_count
    skip_count += 1
    print(f"  {YELLOW}SKIP{RESET}  {name}" + (f" ({reason})" if reason else ""))

def section(title: str):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def get_admin_token() -> str:
    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADMIN_USER,
            "password": ADMIN_PASS,
            "realm_name": ADMIN_REALM,
            "client_id": ADMIN_CLIENT,
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Cannot get admin token: {resp.status_code} {resp.text}")
    token = resp.json()
    if not isinstance(token, str):
        raise RuntimeError(f"Unexpected login response: {token}")
    return token


def kc_clear_required_actions(user_id: str, admin_token: str):
    """Directly patch Keycloak to clear requiredActions + set emailVerified=true (test helper)."""
    resp = requests.put(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"},
        json={"requiredActions": [], "emailVerified": True},
    )
    return resp.status_code == 204


def kc_get_user_id(username: str, admin_token: str) -> str | None:
    resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        params={"username": username, "exact": "true"},
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
    )
    clients = client_resp.json()
    exact = next((c for c in clients if c.get("clientId") == TEST_CLIENT), None)
    if not exact:
        return []
    client_uuid = exact["id"]
    roles_resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}/role-mappings/clients/{client_uuid}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    return [r["name"] for r in roles_resp.json()] if roles_resp.status_code == 200 else []


def kc_get_user_groups(user_id: str, admin_token: str) -> list[str]:
    resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}/groups",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    return [g.get("path", g.get("name", "")) for g in resp.json()] if resp.status_code == 200 else []


def kc_delete_user(username: str, admin_token: str) -> bool:
    user_id = kc_get_user_id(username, admin_token)
    if not user_id:
        return True
    resp = requests.delete(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    return resp.status_code == 204


def kc_delete_group_by_name(group_name: str, admin_token: str) -> bool:
    resp = requests.get(
        f"{KC_BASE}/admin/realms/{TEST_REALM}/groups",
        headers={"Authorization": f"Bearer {admin_token}"},
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
    )
    return del_resp.status_code == 204


# ──────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────

def test_health():
    section("Debug / Health")

    resp = requests.get(f"{API_BASE}/keycloak/health")
    if resp.status_code == 200 and resp.json().get("reachable") is True:
        ok("GET /keycloak/health — reachable")
    else:
        fail("GET /keycloak/health", f"{resp.status_code} {resp.text[:100]}")


def test_admin_login(state: dict):
    section("Public — Admin Login")

    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADMIN_USER,
            "password": ADMIN_PASS,
            "realm_name": ADMIN_REALM,
            "client_id": ADMIN_CLIENT,
        },
    )
    token = resp.json() if resp.status_code == 200 else None
    if isinstance(token, str) and token.startswith("eyJ"):
        ok("POST /SSO/login — admin-cli (master realm)")
        state["admin_token"] = token
    else:
        fail("POST /SSO/login — admin-cli", f"{resp.status_code} {resp.text[:100]}")

    # invalid credentials
    bad = requests.post(
        f"{API_BASE}/SSO/login",
        data={"username": "nobody", "password": "wrong", "realm_name": ADMIN_REALM, "client_id": ADMIN_CLIENT},
    )
    if bad.status_code in (401, 400):
        ok("POST /SSO/login — invalid credentials → reject")
    else:
        fail("POST /SSO/login — invalid credentials", f"expected 401/400, got {bad.status_code}")


def test_group_crud(state: dict):
    section("Admin — Group CRUD")
    token = state.get("admin_token")
    if not token:
        skip("group tests", "no admin token")
        return

    # create
    resp = requests.post(
        f"{API_BASE}/keycloak/group",
        params={"group_name": TEST_GROUP},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code == 201:
        ok(f"POST /keycloak/group — create '{TEST_GROUP}'")
    else:
        fail("POST /keycloak/group", f"{resp.status_code} {resp.text[:100]}")

    # list
    resp = requests.get(
        f"{API_BASE}/keycloak/group",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code == 200 and any(g.get("name") == TEST_GROUP for g in resp.json()):
        ok("GET /keycloak/group — list contains test group")
    else:
        fail("GET /keycloak/group", f"{resp.status_code} {resp.text[:100]}")

    # unauthorised (no token)
    no_auth = requests.get(f"{API_BASE}/keycloak/group")
    if no_auth.status_code in (401, 403):
        ok("GET /keycloak/group — no token → 401/403")
    else:
        fail("GET /keycloak/group — no token", f"expected 401/403, got {no_auth.status_code}")


def test_admin_user_crud(state: dict):
    section("Admin — User CRUD")
    token = state.get("admin_token")
    if not token:
        skip("admin user tests", "no admin token")
        return

    global ADM_PASSWORD

    # create
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
    )
    body = resp.json() if resp.status_code == 200 else {}
    if resp.status_code == 200 and body.get("message") == "User created successfully.":
        ADM_PASSWORD = body.get("temporaryPassword")
        ok(f"POST /admin/keycloak/user — created '{ADM_USERNAME}'")
        state["adm_temp_pass"] = ADM_PASSWORD
    else:
        fail("POST /admin/keycloak/user", f"{resp.status_code} {resp.text[:150]}")

    # get by username
    resp = requests.get(
        f"{API_BASE}/admin/keycloak/user",
        params={"keyword": ADM_USERNAME, "query": "username"},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code == 200 and any(u.get("username") == ADM_USERNAME for u in resp.json()):
        ok("GET /admin/keycloak/user — found by username")
    else:
        fail("GET /admin/keycloak/user", f"{resp.status_code} {resp.text[:100]}")

    # verify roles and group via Keycloak directly
    admin_kc_token = get_admin_token()
    uid = kc_get_user_id(ADM_USERNAME, admin_kc_token)
    if uid:
        roles = kc_get_client_roles(uid, admin_kc_token)
        if "user_basic" in roles:
            ok("Admin user — raptor:user_basic assigned")
        else:
            fail("Admin user — raptor:user_basic assigned", f"roles: {roles}")

        groups = kc_get_user_groups(uid, admin_kc_token)
        if any(TEST_GROUP in g for g in groups):
            ok(f"Admin user — group '/{TEST_GROUP}' assigned")
        else:
            fail(f"Admin user — group '/{TEST_GROUP}' assigned", f"groups: {groups}")

        # clear requiredActions so this user can login
        if kc_clear_required_actions(uid, admin_kc_token):
            state["adm_user_id"] = uid
        else:
            fail("Admin user — clear requiredActions", "failed to clear for login test")
    else:
        fail("Admin user — lookup via KC", f"user '{ADM_USERNAME}' not found")


def test_user_login_and_user_apis(state: dict):
    section("User — Login & User APIs")
    temp_pass = state.get("adm_temp_pass")
    if not temp_pass:
        skip("user login tests", "no temp password from admin-create")
        return

    # login
    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADM_USERNAME,
            "password": temp_pass,
            "realm_name": TEST_REALM,
            "client_id": TEST_CLIENT,
        },
    )
    user_token = resp.json() if resp.status_code == 200 else None
    if isinstance(user_token, str) and user_token.startswith("eyJ"):
        ok(f"POST /SSO/login — '{ADM_USERNAME}' (dhtsolution realm)")
        state["user_token"] = user_token
    else:
        fail("POST /SSO/login — user login", f"{resp.status_code} {resp.text[:100]}")
        return

    # user-groups
    resp = requests.get(
        f"{API_BASE}/keycloak/user-groups",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 200 and isinstance(resp.json(), list):
        ok(f"GET /keycloak/user-groups — OK ({len(resp.json())} groups)")
    else:
        fail("GET /keycloak/user-groups", f"{resp.status_code} {resp.text[:100]}")

    # user-group-id
    resp = requests.get(
        f"{API_BASE}/keycloak/user-group-id",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 200 and isinstance(resp.json(), list):
        ok(f"GET /keycloak/user-group-id — OK ({len(resp.json())} entries)")
    else:
        fail("GET /keycloak/user-group-id", f"{resp.status_code} {resp.text[:100]}")

    # change-password
    new_pass = "PytestChanged99!"
    resp = requests.post(
        f"{API_BASE}/SSO/change-password",
        headers={"Authorization": f"Bearer {user_token}"},
        data={"new_password": new_pass, "realm_name": TEST_REALM},
    )
    if resp.status_code == 200 and resp.json().get("message") == "Password updated successfully.":
        ok("POST /SSO/change-password — OK")
        state["adm_new_pass"] = new_pass
    else:
        fail("POST /SSO/change-password", f"{resp.status_code} {resp.text[:100]}")

    # re-login with new password
    resp2 = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": ADM_USERNAME,
            "password": new_pass,
            "realm_name": TEST_REALM,
            "client_id": TEST_CLIENT,
        },
    )
    new_token = resp2.json() if resp2.status_code == 200 else None
    if isinstance(new_token, str) and new_token.startswith("eyJ"):
        ok("POST /SSO/login — re-login with new password")
        state["user_token"] = new_token  # update to fresh token
    else:
        fail("POST /SSO/login — re-login with new password", f"{resp2.status_code} {resp2.text[:100]}")


def test_path_and_permission(state: dict):
    section("System — Permission & Path Access")
    user_token = state.get("user_token")
    if not user_token:
        skip("permission tests", "no user token")
        return

    # inference allowed (user_basic)
    resp = requests.post(
        f"{API_BASE}/API/inference/1",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 200 and "path" in resp.json():
        ok("POST /API/inference/1 — user_basic allowed")
    else:
        fail("POST /API/inference/1 — user_basic", f"{resp.status_code} {resp.text[:100]}")

    # finetune denied (user_basic has no permission)
    resp = requests.post(
        f"{API_BASE}/API/finetune/1",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 403:
        ok("POST /API/finetune/1 — user_basic denied (403)")
    else:
        fail("POST /API/finetune/1 — user_basic should be 403", f"got {resp.status_code}")

    # /auth/permission — granted
    resp = requests.get(
        f"{API_BASE}/auth/permission",
        params={"request_path": "/API/inference/1"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 200 and resp.json().get("message") == "Permission granted":
        ok("GET /auth/permission — inference granted")
    else:
        fail("GET /auth/permission — inference", f"{resp.status_code} {resp.text[:100]}")

    # /auth/permission — not authorized
    resp = requests.get(
        f"{API_BASE}/auth/permission",
        params={"request_path": "/API/finetune/1"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 403 and resp.json().get("detail") == "not_authorized":
        ok("GET /auth/permission — finetune denied (not_authorized)")
    else:
        fail("GET /auth/permission — finetune", f"{resp.status_code} {resp.text[:100]}")

    # /auth/permission — unknown resource
    resp = requests.get(
        f"{API_BASE}/auth/permission",
        params={"request_path": "/does/not/exist"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    if resp.status_code == 403:
        ok("GET /auth/permission — unknown resource → 403")
    else:
        fail("GET /auth/permission — unknown resource", f"{resp.status_code} {resp.text[:100]}")


def test_logout(state: dict):
    section("User — Logout")
    user_token = state.get("user_token")
    if not user_token:
        skip("logout test", "no user token")
        return

    resp = requests.post(
        f"{API_BASE}/SSO/logout",
        headers={
            "Authorization": f"Bearer {user_token}",
            "Cookie": f"realm_name={TEST_REALM}",
        },
    )
    if resp.status_code == 200 and "Logout successful" in resp.json().get("message", ""):
        ok("POST /SSO/logout — OK")
    else:
        fail("POST /SSO/logout", f"{resp.status_code} {resp.text[:100]}")


def test_self_register(state: dict):
    section("Public — Self-Register")

    # register
    resp = requests.post(
        f"{API_BASE}/SSO/register",
        data={
            "username": REG_USERNAME,
            "email": REG_EMAIL,
            "firstName": "Pytest",
            "lastName": "SelfReg",
            "realm_name": TEST_REALM,
        },
    )
    if resp.status_code == 200 and resp.json().get("message") == "Registration successful.":
        ok(f"POST /SSO/register — created '{REG_USERNAME}'")
    else:
        fail("POST /SSO/register", f"{resp.status_code} {resp.text[:150]}")
        return

    # no temporaryPassword in response when SMTP configured
    if "temporaryPassword" not in resp.json():
        ok("POST /SSO/register — no temporaryPassword (SMTP mode)")
    else:
        ok(f"POST /SSO/register — temporaryPassword returned (no-SMTP mode): {resp.json()['temporaryPassword'][:8]}…")

    # verify via Keycloak
    kc_token = get_admin_token()
    uid = kc_get_user_id(REG_USERNAME, kc_token)
    if uid:
        roles = kc_get_client_roles(uid, kc_token)
        if "user_basic" in roles:
            ok("Self-register user — raptor:user_basic assigned")
        else:
            fail("Self-register user — raptor:user_basic", f"roles: {roles}")

        groups = kc_get_user_groups(uid, kc_token)
        if groups:
            fail("Self-register user — no group", f"unexpectedly in: {groups}")
        else:
            ok("Self-register user — no group assigned")

        # simulate user completing email verification (test helper — clear requiredActions)
        if kc_clear_required_actions(uid, kc_token):
            state["reg_user_id"] = uid

            # set a known password so we can test login
            pw_resp = requests.put(
                f"{KC_BASE}/admin/realms/{TEST_REALM}/users/{uid}/reset-password",
                headers={"Authorization": f"Bearer {kc_token}", "Content-Type": "application/json"},
                json={"type": "password", "value": REG_PASSWORD, "temporary": False},
            )
            if pw_resp.status_code == 204:
                ok("Self-register user — password set (test helper)")
            else:
                fail("Self-register user — set password", f"{pw_resp.status_code}")
                return
        else:
            fail("Self-register user — clear requiredActions", "test setup failed")
            return
    else:
        fail("Self-register user — lookup via KC", f"'{REG_USERNAME}' not found")
        return

    # login as self-registered user
    resp = requests.post(
        f"{API_BASE}/SSO/login",
        data={
            "username": REG_USERNAME,
            "password": REG_PASSWORD,
            "realm_name": TEST_REALM,
            "client_id": TEST_CLIENT,
        },
    )
    reg_token = resp.json() if resp.status_code == 200 else None
    if isinstance(reg_token, str) and reg_token.startswith("eyJ"):
        ok("POST /SSO/login — self-registered user")
        state["reg_token"] = reg_token
    else:
        fail("POST /SSO/login — self-registered user", f"{resp.status_code} {resp.text[:100]}")
        return

    # self-register user can access inference
    resp = requests.post(
        f"{API_BASE}/API/inference/1",
        headers={"Authorization": f"Bearer {reg_token}"},
    )
    if resp.status_code == 200:
        ok("Self-register user — POST /API/inference/1 allowed")
    else:
        fail("Self-register user — POST /API/inference/1", f"{resp.status_code}")

    # self-register user cannot access finetune
    resp = requests.post(
        f"{API_BASE}/API/finetune/1",
        headers={"Authorization": f"Bearer {reg_token}"},
    )
    if resp.status_code == 403:
        ok("Self-register user — POST /API/finetune/1 denied (403)")
    else:
        fail("Self-register user — POST /API/finetune/1 should 403", f"{resp.status_code}")

    # delete own account
    resp = requests.delete(
        f"{API_BASE}/SSO/account",
        params={"realm_name": TEST_REALM},
        headers={"Authorization": f"Bearer {reg_token}"},
    )
    if resp.status_code == 200 and "deleted" in resp.json().get("message", "").lower():
        ok("DELETE /SSO/account — self-register user deleted own account")
        state.pop("reg_user_id", None)  # cleaned up
    else:
        fail("DELETE /SSO/account", f"{resp.status_code} {resp.text[:100]}")


def test_delete_admin_user(state: dict):
    section("Admin — Delete User")
    token = state.get("admin_token")
    uid = state.get("adm_user_id")
    if not token or not uid:
        skip("DELETE /admin/keycloak/user", "no admin token or user id")
        return

    resp = requests.delete(
        f"{API_BASE}/admin/keycloak/user",
        params={"user_id": uid},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code == 200:
        ok(f"DELETE /admin/keycloak/user — '{ADM_USERNAME}' deleted")
    else:
        fail("DELETE /admin/keycloak/user", f"{resp.status_code} {resp.text[:100]}")


def cleanup():
    section("Cleanup")
    try:
        kc_token = get_admin_token()
        for username in [ADM_USERNAME, REG_USERNAME]:
            if kc_delete_user(username, kc_token):
                print(f"  cleaned  user '{username}'")
            else:
                print(f"  skipped  user '{username}' (not found)")

        if kc_delete_group_by_name(TEST_GROUP, kc_token):
            print(f"  cleaned  group '{TEST_GROUP}'")
        else:
            print(f"  skipped  group '{TEST_GROUP}' (not found)")
    except Exception as e:
        print(f"  {YELLOW}cleanup error: {e}{RESET}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print(f"\n{BOLD}Module 06 — Authentication API Integration Test{RESET}")
    print(f"  API : {API_BASE}")
    print(f"  KC  : {KC_BASE}")

    state: dict = {}

    try:
        test_health()
        test_admin_login(state)
        test_group_crud(state)
        test_admin_user_crud(state)
        test_user_login_and_user_apis(state)
        test_path_and_permission(state)
        test_logout(state)
        test_self_register(state)
        test_delete_admin_user(state)
    finally:
        cleanup()

    section("Results")
    total = pass_count + fail_count + skip_count
    print(f"  {GREEN}PASS{RESET} {pass_count}  |  {RED}FAIL{RESET} {fail_count}  |  {YELLOW}SKIP{RESET} {skip_count}  |  TOTAL {total}")

    if fail_count > 0:
        print(f"\n  {RED}{BOLD}Some tests failed.{RESET}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}All tests passed.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
