"""
Live, end-to-end two-user isolation test for Module 21's memory integration.

Kept separate from test_memory_integration.py (which mocks Keycloak and
Module 26 entirely) so this file can be excluded by pattern from normal/CI
runs -- it needs a real running stack, not just Python deps. It uses the
same production MemoryClient class and set_auth_token() JWT forwarding, but
against a real Module 06 (Keycloak) + Module 13 (gateway) + Module 26
(memory service), with two real Keycloak-issued JWTs, to prove the
isolation holds end to end and not just in mocked proxy logic.

Skipped by default. Requires:
  - Modules 06, 13 and 26 running locally (docker compose up, see each
    module's docker-compose.yml) and reachable at the URLs below.
  - Two Keycloak accounts in the `dhtsolution` realm with known passwords
    (the repo ships test_basicuser/test_standarduser in
    06-authentication/realm-import/dhtsolution-realm.json with no password
    set -- set one via the Keycloak admin console or keycloak-api first).

Run:
    RAPTOR_LIVE_INTEGRATION=1 \\
    LIVE_TEST_USER_A_PASSWORD=... LIVE_TEST_USER_B_PASSWORD=... \\
    conda run -n CIE python -m pytest test_memory_live_isolation.py -v
"""
import os
import sys
import uuid
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))

from memory_client import MemoryClient, set_auth_token

AUTH_URL = os.environ.get("LIVE_AUTH_URL", "http://localhost:8800")
GATEWAY_URL = os.environ.get("LIVE_GATEWAY_URL", f"http://localhost:8012/api/{os.environ.get('API_VERSION', '0.5')}")
REALM = os.environ.get("LIVE_TEST_REALM", "dhtsolution")
CLIENT_ID = os.environ.get("LIVE_TEST_CLIENT_ID", "raptor")

USER_A = os.environ.get("LIVE_TEST_USER_A", "test_basicuser")
USER_B = os.environ.get("LIVE_TEST_USER_B", "test_standarduser")
PASSWORD_A = os.environ.get("LIVE_TEST_USER_A_PASSWORD")
PASSWORD_B = os.environ.get("LIVE_TEST_USER_B_PASSWORD")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("RAPTOR_LIVE_INTEGRATION") != "1" or not (PASSWORD_A and PASSWORD_B),
        reason=(
            "needs a running Module 06/13/26 stack and real credentials -- set "
            "RAPTOR_LIVE_INTEGRATION=1, LIVE_TEST_USER_A_PASSWORD and LIVE_TEST_USER_B_PASSWORD"
        ),
    ),
]


async def _login(username: str, password: str) -> str:
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"{AUTH_URL}/SSO/login",
            data={"username": username, "password": password, "realm_name": REALM, "client_id": CLIENT_ID},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


async def test_two_real_users_cannot_see_each_others_memory():
    """Same guarantee as test_search_longterm_scoped_to_callers_jwt_no_cross_contamination
    in test_memory_integration.py, but through the real MemoryClient -> Module 13 ->
    Module 26 path with two real Keycloak-issued JWTs."""
    token_a = f"Bearer {await _login(USER_A, PASSWORD_A)}"
    token_b = f"Bearer {await _login(USER_B, PASSWORD_B)}"

    async with httpx.AsyncClient() as http:
        client = MemoryClient(GATEWAY_URL, http, timeout=10.0)
        session_id = f"live_isolation_{uuid.uuid4().hex[:8]}"
        secret_a = f"ALPHA_SECRET_{uuid.uuid4().hex[:8]}"
        secret_b = f"BRAVO_SECRET_{uuid.uuid4().hex[:8]}"

        set_auth_token(token_a)
        await client.append_turn("ignored", session_id, f"my secret code is {secret_a}", "noted")
        set_auth_token(token_b)
        await client.append_turn("ignored", session_id, f"my secret code is {secret_b}", "noted")

        set_auth_token(token_a)
        hits_a = await client.search_longterm("ignored", secret_a)
        set_auth_token(token_b)
        hits_b = await client.search_longterm("ignored", secret_b)

        assert any(secret_a in h["text"] for h in hits_a), "user A must find their own secret"
        assert not any(secret_b in h["text"] for h in hits_a), "user A must never see user B's secret"
        assert any(secret_b in h["text"] for h in hits_b), "user B must find their own secret"
        assert not any(secret_a in h["text"] for h in hits_b), "user B must never see user A's secret"


async def test_shared_session_id_does_not_merge_across_users():
    """The same session_id string used by two different real users must
    resolve to two independent per-user histories in Module 26, not a
    shared/appended one -- storage is scoped by JWT sub, not session_id."""
    token_a = f"Bearer {await _login(USER_A, PASSWORD_A)}"
    token_b = f"Bearer {await _login(USER_B, PASSWORD_B)}"
    session_id = f"live_isolation_shared_{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient() as http:
        resp_a = await http.post(
            f"{GATEWAY_URL}/memory/archive",
            params={"session_id": session_id},
            headers={"Authorization": token_a},
            json={"user_message": "user A turn", "assistant_response": "ack A"},
            timeout=10.0,
        )
        resp_a.raise_for_status()

        resp_b = await http.post(
            f"{GATEWAY_URL}/memory/archive",
            params={"session_id": session_id},
            headers={"Authorization": token_b},
            json={"user_message": "user B turn", "assistant_response": "ack B"},
            timeout=10.0,
        )
        resp_b.raise_for_status()

    assert resp_a.json()["turn_index"] == resp_b.json()["turn_index"] == 1
