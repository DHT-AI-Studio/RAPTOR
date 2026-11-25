"""Fixtures shared by the E2E suite -- real HTTP against an already-running stack.
See README.md for what needs to be up first and how to point these at it.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
import pytest

BASE_URL = os.environ.get("E2E_INFERENCE_URL", "http://localhost:8036")
TIMEOUT_S = float(os.environ.get("E2E_TIMEOUT_S", "120"))  # real inference, incl. cold model loads
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# For consented_user_id below -- any ordinary logged-in user (no Keycloak admin credentials
# needed: granting one's own GDPR consent is always self-service, the same PUT
# /consent/{purpose} a real client would call). Deliberately no credential defaults here --
# whoever runs this suite supplies their own via the env vars below.
GATEWAY_URL = os.environ.get("E2E_GATEWAY_URL", "http://localhost:8012")
API_VERSION = os.environ.get("E2E_API_VERSION", "0.5")
GATEWAY_USER = os.environ.get("E2E_GATEWAY_USER", "")
GATEWAY_PASSWORD = os.environ.get("E2E_GATEWAY_PASSWORD", "")
GATEWAY_REALM = os.environ.get("E2E_GATEWAY_REALM", "dhtsolution")
GATEWAY_CLIENT_ID = os.environ.get("E2E_GATEWAY_CLIENT_ID", "raptor")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session", autouse=True)
def _require_module_35(base_url: str) -> None:
    try:
        r = httpx.get(f"{base_url}/health", timeout=5.0)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"module 35 not reachable at {base_url} -- bring up the stack first (see README.md): {exc}")


@pytest.fixture(scope="session")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT_S) as c:
        yield c


@pytest.fixture(scope="session")
def voice_enabled(client: httpx.Client) -> bool:
    return client.get("/health").json().get("voice", {}).get("diarize_loaded", False)


def _jwt_sub(token: str) -> str:
    """No signature check -- only reading our own just-issued token's `sub` claim."""
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]


@pytest.fixture(scope="session")
def consented_user_id() -> str | None:
    """A real Keycloak user id with ``voice_clone`` consent actually granted through the
    real gateway + Module 33 (GD-11) -- consent is keyed by the caller's real Keycloak
    user id, so an arbitrary made-up ``X-User-ID`` (e.g. "e2e-test-user") can never pass
    Module 33's check: it 404s there as "not a Keycloak user", surfacing to the caller as
    403 consent_required regardless of how many times consent is "granted" for that made-up
    string. Logs in as E2E_GATEWAY_USER/PASSWORD and grants consent for *that* real id.
    None (callers skip) if those aren't set, the gateway isn't reachable, or login fails."""
    if not GATEWAY_USER or not GATEWAY_PASSWORD:
        return None
    try:
        resp = httpx.post(
            f"{GATEWAY_URL}/api/{API_VERSION}/sso/login", timeout=10.0,
            data={"username": GATEWAY_USER, "password": GATEWAY_PASSWORD,
                  "realm_name": GATEWAY_REALM, "client_id": GATEWAY_CLIENT_ID},
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        user_id = _jwt_sub(token)
        httpx.put(f"{GATEWAY_URL}/api/{API_VERSION}/gdpr/consent/voice_clone", timeout=10.0,
                 headers={"Authorization": f"Bearer {token}"}, json={"granted": True}).raise_for_status()
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        return None
    return user_id


def fixture_path(name: str) -> Path:
    """An audio fixture's path, skipping the test by name if it's missing (see README.md)."""
    p = FIXTURES_DIR / name
    if not p.is_file():
        pytest.skip(f"missing audio fixture {p} -- see tests/e2e/README.md's 'Audio fixtures' section")
    return p
