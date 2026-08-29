import json
import time

import pytest
from cryptography.fernet import Fernet

from app.core.config import Settings
from app.services.token_manager import TokenManager


class FakeRedis:
    """Minimal in-memory async Redis stand-in for token_manager tests."""

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
    manager._fake, manager._calls = fake, calls  # expose for assertions
    return manager


@pytest.mark.asyncio
async def test_register_returns_opaque_agent_token(tm):
    token = await tm.register_agent("cid", "secret", "search chat")
    assert token.startswith("mcp-agent-")
    assert tm._creds_key(token) in tm._fake.store
    assert tm._calls["n"] == 1  # validated credentials once


@pytest.mark.asyncio
async def test_get_valid_returns_cached_without_refresh(tm):
    token = await tm.register_agent("cid", "secret", None)
    got = await tm.get_valid_keycloak_token(token)
    assert got == "kc-token-1"
    assert tm._calls["n"] == 1  # no extra fetch — cached token still valid


@pytest.mark.asyncio
async def test_get_valid_refreshes_when_expired(tm):
    token = await tm.register_agent("cid", "secret", None)
    key = tm._creds_key(token)
    blob = json.loads(tm._fake.store[key])
    blob["expires_at"] = time.time() - 1  # force expiry
    tm._fake.store[key] = json.dumps(blob)

    got = await tm.get_valid_keycloak_token(token)
    assert got == "kc-token-2"      # refreshed
    assert tm._calls["n"] == 2


@pytest.mark.asyncio
async def test_unknown_token_returns_none(tm):
    assert await tm.get_valid_keycloak_token("mcp-agent-nope") is None


@pytest.mark.asyncio
async def test_revoke_removes_credentials(tm):
    token = await tm.register_agent("cid", "secret", None)
    assert await tm.revoke_agent(token) is True
    assert await tm.get_valid_keycloak_token(token) is None
    assert await tm.revoke_agent(token) is False  # already gone


# ── client_secret encryption at rest ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_client_secret_not_stored_in_plaintext(tm):
    token = await tm.register_agent("cid", "s3cr3t-value", None)
    blob = json.loads(tm._fake.store[tm._creds_key(token)])

    assert "client_secret" not in blob            # old plaintext field is gone
    assert "s3cr3t-value" not in blob["client_secret_enc"]


@pytest.mark.asyncio
async def test_client_secret_round_trips_through_encryption(tm, monkeypatch):
    seen = {}

    async def _fetch(client_id, client_secret, scope):
        seen["client_secret"] = client_secret
        tm._calls["n"] += 1
        return f"kc-token-{tm._calls['n']}", time.time() + 3600

    monkeypatch.setattr(tm, "_fetch_client_credentials_token", _fetch)

    token = await tm.register_agent("cid", "s3cr3t-value", None)
    key = tm._creds_key(token)
    blob = json.loads(tm._fake.store[key])
    blob["expires_at"] = time.time() - 1  # force the refresh path to run
    tm._fake.store[key] = json.dumps(blob)

    await tm.get_valid_keycloak_token(token)

    assert seen["client_secret"] == "s3cr3t-value"  # decrypted correctly for Keycloak


@pytest.mark.asyncio
async def test_register_fails_clearly_without_encryption_key():
    manager = TokenManager(Settings(secret_encryption_key=""))
    fake = FakeRedis()

    async def _get_redis():
        return fake

    manager._get_redis = _get_redis

    with pytest.raises(RuntimeError, match="MCP_SECRET_ENCRYPTION_KEY"):
        await manager.register_agent("cid", "secret", None)
