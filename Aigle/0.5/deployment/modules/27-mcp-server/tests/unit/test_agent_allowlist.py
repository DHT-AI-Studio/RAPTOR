"""
VIE01, MCP OAuth follow-up: POST /auth/register
previously accepted ANY client_id+secret a caller supplied, gated only by
"some human JWT" (any Bearer token) -- no restriction on who was allowed to
register an autonomous agent. MCP_AGENT_ALLOWED_CLIENT_IDS closes that gap:
default-deny, only listed client_ids may register.

Tests the router function directly (matching this suite's existing
TokenManager-direct-call convention in test_mc7_token_manager.py) rather
than through a full HTTP TestClient, since register_agent's allowlist check
runs before TokenManager is ever touched -- no real Keycloak/Redis needed.

Run from the 27-mcp-server/ directory:
    python -m pytest tests/unit/test_agent_allowlist.py -v
"""
import os
import sys
import types
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

_MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _MODULE_ROOT)

# Module 27's own app/ has no __init__.py (a PEP 420 namespace package). Some
# dev sandboxes have an unrelated pip package literally named "app" in user
# site-packages, which — per Python's import rules — always wins over a
# namespace portion regardless of sys.path order, since a regular package
# match short-circuits the namespace search entirely. Pre-registering the
# real local "app" in sys.modules forces correct resolution regardless of
# what else is installed; harmless no-op in an environment without the
# collision. Affects every test file in this suite equally, not just this
# one — confirmed test_mc7_token_manager.py hits the identical
# "No module named 'app.core'" failure here without this.
if "app" not in sys.modules or not getattr(sys.modules["app"], "__path__", None):
    _ns = types.ModuleType("app")
    _ns.__path__ = [os.path.join(os.path.abspath(_MODULE_ROOT), "app")]
    sys.modules["app"] = _ns

os.environ.setdefault("MCP_SECRET_ENCRYPTION_KEY", "gAAAAABkX0Y9dGVzdC1mZXJuZXQta2V5LWZvci10ZXN0cw==")

from app.core.config import get_settings
from app.routers.auth import RegisterRequest, register_agent


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings() is @lru_cache'd — clear it so each test's monkeypatched
    env var is actually picked up, not a stale cached Settings instance."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_token_manager():
    tm = AsyncMock()
    tm.register_agent.return_value = "mcp-agent-fake-token"
    return tm


@pytest.mark.asyncio
async def test_registration_disabled_by_default(monkeypatch):
    """Default-deny: with MCP_AGENT_ALLOWED_CLIENT_IDS unset (empty), every
    client_id is rejected -- the endpoint is disabled entirely, not
    "allow everything"."""
    monkeypatch.delenv("MCP_AGENT_ALLOWED_CLIENT_IDS", raising=False)
    tm = _fake_token_manager()

    with pytest.raises(HTTPException) as exc_info:
        await register_agent(
            body=RegisterRequest(client_id="raptor", client_secret="s3cr3t"),
            _human_jwt="some.jwt.token",
            tm=tm,
        )
    assert exc_info.value.status_code == 403
    tm.register_agent.assert_not_called()


@pytest.mark.asyncio
async def test_client_id_not_on_allowlist_rejected(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ALLOWED_CLIENT_IDS", "approved-agent-1,approved-agent-2")
    tm = _fake_token_manager()

    with pytest.raises(HTTPException) as exc_info:
        await register_agent(
            body=RegisterRequest(client_id="raptor", client_secret="s3cr3t"),
            _human_jwt="some.jwt.token",
            tm=tm,
        )
    assert exc_info.value.status_code == 403
    assert "raptor" in exc_info.value.detail
    tm.register_agent.assert_not_called()


@pytest.mark.asyncio
async def test_client_id_on_allowlist_proceeds(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ALLOWED_CLIENT_IDS", "approved-agent-1,approved-agent-2")
    tm = _fake_token_manager()

    result = await register_agent(
        body=RegisterRequest(client_id="approved-agent-2", client_secret="s3cr3t"),
        _human_jwt="some.jwt.token",  # not a real JWT -> registered_by falls back to "unknown"
        tm=tm,
    )
    tm.register_agent.assert_awaited_once_with(
        "approved-agent-2", "s3cr3t", None, registered_by="unknown",
    )
    assert result.agent_token == "mcp-agent-fake-token"


@pytest.mark.asyncio
async def test_registering_human_sub_is_extracted_and_forwarded(monkeypatch):
    """_best_effort_sub decodes the registering human's JWT (unverified --
    Module 13 already did the real validation) and register_agent forwards
    it as registered_by, for the ownership-conflict check in TokenManager."""
    monkeypatch.setenv("MCP_AGENT_ALLOWED_CLIENT_IDS", "approved-agent-1")
    tm = _fake_token_manager()

    # header.payload.signature -- unsigned/fake signature is fine, we only
    # ever call get_unverified_claims on this, never verify it.
    import base64
    import json as _json

    def _b64(d):
        return base64.urlsafe_b64encode(_json.dumps(d).encode()).rstrip(b"=").decode()

    fake_jwt = f"{_b64({'alg': 'none'})}.{_b64({'sub': 'alice-keycloak-id'})}.sig"

    await register_agent(
        body=RegisterRequest(client_id="approved-agent-1", client_secret="s3cr3t"),
        _human_jwt=fake_jwt,
        tm=tm,
    )
    tm.register_agent.assert_awaited_once_with(
        "approved-agent-1", "s3cr3t", None, registered_by="alice-keycloak-id",
    )


@pytest.mark.asyncio
async def test_allowlist_trims_whitespace_around_commas(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ALLOWED_CLIENT_IDS", " approved-agent-1 , approved-agent-2 ")
    tm = _fake_token_manager()

    result = await register_agent(
        body=RegisterRequest(client_id="approved-agent-1", client_secret="s3cr3t"),
        _human_jwt="some.jwt.token",
        tm=tm,
    )
    assert result.agent_token == "mcp-agent-fake-token"
