"""
client_credentials is a client-only grant with no per-user identity, so
every agent_token registered against the same Keycloak client_id resolves
to that client's one shared service-account -- not the registering human.
This doesn't (can't, from Module 27 alone) fix that underlying limitation,
but it does close the practical failure mode: two DIFFERENT humans
accidentally registering the same client_id (e.g. copy-pasted from a
shared setup doc) and unknowingly sharing one identity.

Tests TokenManager.register_agent's ownership-claim logic directly, reusing
test_mc7_token_manager.py's FakeRedis/tm fixture pattern (own copy here
rather than importing it, to keep this file runnable standalone).

Run from the 27-mcp-server/ directory:
    python -m pytest tests/unit/test_agent_ownership.py -v
"""
import os
import sys
import time
import types

import pytest
from cryptography.fernet import Fernet

_MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _MODULE_ROOT)

# See test_agent_allowlist.py for why this is needed in some sandboxes.
if "app" not in sys.modules or not getattr(sys.modules["app"], "__path__", None):
    _ns = types.ModuleType("app")
    _ns.__path__ = [os.path.join(os.path.abspath(_MODULE_ROOT), "app")]
    sys.modules["app"] = _ns

from app.core.config import Settings
from app.services.token_manager import OwnershipConflictError, TokenManager


class FakeRedis:
    """Minimal in-memory async Redis stand-in — matches
    test_mc7_token_manager.py's fixture."""

    def __init__(self):
        self.store: dict = {}

    async def ping(self):
        return True

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
        return n

    async def expire(self, k, ttl):
        return k in self.store


@pytest.fixture
def tm(monkeypatch):
    manager = TokenManager(Settings(secret_encryption_key=Fernet.generate_key().decode()))
    fake = FakeRedis()

    async def _get_redis():
        return fake

    monkeypatch.setattr(manager, "_get_redis", _get_redis)

    calls = {"n": 0}

    async def _fetch(client_id, client_secret, scope):
        calls["n"] += 1
        return f"kc-token-{calls['n']}", time.time() + 3600

    monkeypatch.setattr(manager, "_fetch_client_credentials_token", _fetch)
    manager._fake = fake
    return manager


@pytest.mark.asyncio
async def test_first_registration_claims_the_client_id(tm):
    await tm.register_agent("agent-1", "secret", None, registered_by="alice")
    assert tm._fake.store["mcp:agent-owner:agent-1"] == "alice"


@pytest.mark.asyncio
async def test_same_registrant_can_reregister_same_client_id(tm):
    """Token rotation / re-registration by the SAME human must keep working
    — the ownership check should never block its own owner."""
    await tm.register_agent("agent-1", "secret", None, registered_by="alice")
    token2 = await tm.register_agent("agent-1", "secret", None, registered_by="alice")
    assert token2  # didn't raise


@pytest.mark.asyncio
async def test_different_registrant_is_rejected(tm):
    await tm.register_agent("agent-1", "secret", None, registered_by="alice")

    with pytest.raises(OwnershipConflictError) as exc_info:
        await tm.register_agent("agent-1", "secret", None, registered_by="bob")

    assert "agent-1" in str(exc_info.value)
    assert "alice" in str(exc_info.value)
    # Ownership must not have been silently overwritten by the rejected attempt.
    assert tm._fake.store["mcp:agent-owner:agent-1"] == "alice"


@pytest.mark.asyncio
async def test_conflict_check_runs_before_credential_validation(tm, monkeypatch):
    """The ownership conflict must be caught before ever calling out to
    Keycloak to validate the (possibly different) client_secret -- an
    unauthorized registrant shouldn't be able to probe whether a secret is
    valid for a client_id they don't own."""
    await tm.register_agent("agent-1", "secret", None, registered_by="alice")

    async def _poison(*a, **kw):
        raise AssertionError("Keycloak credential validation must not run for a rejected owner")

    monkeypatch.setattr(tm, "_fetch_client_credentials_token", _poison)

    with pytest.raises(OwnershipConflictError):
        await tm.register_agent("agent-1", "wrong-secret", None, registered_by="bob")


@pytest.mark.asyncio
async def test_different_client_ids_are_independent(tm):
    """Two distinct client_ids, each claimed by a different human, must not
    interfere with each other -- this is exactly the "one client_id per
    agent" pattern the allowlist (MCP_AGENT_ALLOWED_CLIENT_IDS) is meant to
    support."""
    await tm.register_agent("agent-alice", "secret", None, registered_by="alice")
    await tm.register_agent("agent-bob", "secret", None, registered_by="bob")

    assert tm._fake.store["mcp:agent-owner:agent-alice"] == "alice"
    assert tm._fake.store["mcp:agent-owner:agent-bob"] == "bob"


@pytest.mark.asyncio
async def test_unknown_registrant_can_claim_unclaimed_client_id(tm):
    """A JWT that couldn't be decoded falls back to registered_by="unknown"
    (see auth.py::_best_effort_sub) -- still allowed to claim a fresh
    client_id, since failing open here would just make legitimate first
    registration impossible, not close any real gap."""
    token = await tm.register_agent("agent-1", "secret", None, registered_by="unknown")
    assert token
    assert tm._fake.store["mcp:agent-owner:agent-1"] == "unknown"


@pytest.mark.asyncio
async def test_unknown_registrant_cannot_override_a_claimed_client_id(tm):
    """The flip side of the above: once claimed by a real identity,
    "unknown" can't steal it -- fails closed, not open."""
    await tm.register_agent("agent-1", "secret", None, registered_by="alice")

    with pytest.raises(OwnershipConflictError):
        await tm.register_agent("agent-1", "secret", None, registered_by="unknown")
