"""QA e2e scenarios. Scenario 2 is skipped (no live Module 07 to call).
Scenario 3's actual agent run is in test_agent_orchestration.py; the
>24h credential-refresh piece is verified here offline.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from cryptography.fernet import Fernet
from mcp.server.fastmcp import FastMCP

from app.context import current_bearer
from app.core.config import Settings
from app.services.raptor_client import MCPAuthError, RaptorClient
from app.services.token_manager import TokenManager
from app.tools import get_client
from app.tools import chat as chat_tool
from app.tools import memory as memory_tool
from app.tools import search as search_tool


# ── Scenario 1 — direct Python client, no LLM ─────────────────────────────────

@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def mock_ctx(mock_client):
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.request_context.lifespan_context = {"raptor_client": mock_client}
    return ctx


@pytest.fixture
def three_tools():
    """search_vector (raptor_search_vector), chat (raptor_chat), and
    memory_retrieve (raptor_memory_retrieve) — the exact trio named in the
    AC's example — registered on a real FastMCP instance, same as every
    other tool test file in this suite."""
    mcp = FastMCP("test")
    search_tool.register(mcp)
    chat_tool.register(mcp)
    memory_tool.register(mcp)
    return {n: mcp._tool_manager._tools[n].fn for n in mcp._tool_manager._tools}


@pytest.mark.asyncio
async def test_scenario1_search_vector_returns_correct_response(mock_client, mock_ctx, three_tools):
    mock_client.post_json.return_value = {
        "results": [{"id": "h1", "score": 0.9, "payload": {"text": "hello"}}],
    }

    result = json.loads(await three_tools["raptor_search_vector"](query="hello", ctx=mock_ctx))

    assert result[0]["id"] == "h1"
    assert mock_client.post_json.call_args[0][0] == "/search/vector"


@pytest.mark.asyncio
async def test_scenario1_chat_returns_correct_response(mock_client, mock_ctx, three_tools):
    mock_client.post_json.return_value = {"response": "Hi there!", "session_id": "s1"}

    result = json.loads(await three_tools["raptor_chat"](message="hi", ctx=mock_ctx))

    assert result["response"] == "Hi there!"


@pytest.mark.asyncio
async def test_scenario1_memory_retrieve_returns_correct_response(mock_client, mock_ctx, three_tools):
    mock_client.get_json.return_value = {"results": [{"text": "prefers Traditional Chinese"}]}

    result = json.loads(await three_tools["raptor_memory_retrieve"](query="language preference", ctx=mock_ctx))

    assert result["results"][0]["text"] == "prefers Traditional Chinese"


@pytest.mark.asyncio
async def test_scenario1_missing_auth_returns_mcp_auth_error():
    """Same assertion as app/tests/test_token_resolution.py, kept here too
    since the AC names it as part of this scenario file explicitly."""
    settings = MagicMock(transport="http")
    token_manager = AsyncMock()
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"settings": settings, "token_manager": token_manager}

    bearer_token = current_bearer.set(None)
    try:
        with pytest.raises(MCPAuthError):
            await get_client(ctx)
    finally:
        current_bearer.reset(bearer_token)


# ── Scenario 2 — LLM function call via Module 07 (Gemma 3 / Llama) ───────────

@pytest.mark.skip(
    reason="Requires a live Module 07 inference endpoint serving Gemma 3 or "
           "Llama; not reachable in this environment. See docstring at the "
           "top of this file — this scenario needs to run against a "
           "deployed/staging stack, not as an offline unit test."
)
@pytest.mark.asyncio
async def test_llm_function_call_scenario2():
    """Intended flow (documented, not executed here):

    1. Fetch the MCP tool manifest via tools/list.
    2. Call Module 07's /inference/infer with the manifest translated to
       that model's function-calling format, asking a question that
       requires a tool (e.g. "what's in my recent uploads?").
    3. Parse the model's tool-call request, invoke the matching MCP tool
       (e.g. raptor_list_assets) for real.
    4. Feed the tool result back into a second /inference/infer call as
       context and confirm a final natural-language answer is produced.
    """


# ── Scenario 3 — agent credentials survive a session > 24h ───────────────────

class _FakeRedis:
    def __init__(self):
        self.store: dict = {}
        self.expire_calls = 0
        self.last_set_ex: int | None = None

    async def ping(self):
        return True

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v
        self.last_set_ex = ex

    async def delete(self, *keys):
        return sum(1 for k in keys if self.store.pop(k, None) is not None)

    async def expire(self, k, ttl):
        self.expire_calls += 1
        return k in self.store


@pytest.mark.asyncio
async def test_scenario3_agent_credentials_refresh_past_24h_session(monkeypatch):
    """A multi-step agent (Scenario 3) can run well past the 24h
    agent_token idle window as long as it keeps making tool calls — each
    successful resolution both refreshes the underlying (short-lived)
    Keycloak access token and extends the agent_token's 24h idle TTL, so
    an active agent session never has to re-register mid-task."""
    manager = TokenManager(Settings(secret_encryption_key=Fernet.generate_key().decode()))
    fake = _FakeRedis()

    async def _get_redis():
        return fake

    monkeypatch.setattr(manager, "_get_redis", _get_redis)

    fetch_calls = {"n": 0}

    async def _fetch(client_id, client_secret, scope):
        fetch_calls["n"] += 1
        # Keycloak access tokens are short-lived regardless of the agent's
        # own 24h window — simulate one that's already stale.
        return f"kc-token-{fetch_calls['n']}", time.time() - 1

    monkeypatch.setattr(manager, "_fetch_client_credentials_token", _fetch)

    agent_token = await manager.register_agent("cid", "secret", scope=None)

    # Simulate the agent's task running long past 24 real-world hours by
    # jumping the cached credential's clock forward, then using it again —
    # this is the same "session exceeds 24 hours" case the AC describes.
    key = manager._creds_key(agent_token)
    blob = json.loads(fake.store[key])
    blob["expires_at"] = time.time() - 90000  # ~25h ago
    fake.store[key] = json.dumps(blob)

    resolved = await manager.get_valid_keycloak_token(agent_token)

    assert resolved == "kc-token-2"        # transparently refreshed, no re-registration
    assert fetch_calls["n"] == 2           # 1 at register, 1 on this refresh
    # Refreshing rewrites the credential blob via redis.set(..., ex=...),
    # not redis.expire() (that path is only for the still-valid/no-refresh
    # case) — either way the 24h idle window gets reset, so the session
    # keeps living past the AC's ">24h" mark.
    assert fake.last_set_ex == manager._s.agent_token_ttl_seconds

    # A session that goes fully idle (no calls for the whole 24h window, so
    # Redis would have actually evicted the key) must NOT be silently
    # treated as valid — the agent has to re-register.
    await fake.delete(key)
    assert await manager.get_valid_keycloak_token(agent_token) is None


# ── Downstream service down (e.g. Module 26) — bounded, non-sensitive error ──

@pytest.mark.asyncio
async def test_downstream_service_down_returns_bounded_nonsensitive_error():
    """Module 26 unreachable → the MCP client gets a short, generic error,
    never the raw exception text (which could include internal hostnames)
    or a downstream stack trace. Exercised through a real tool call
    (raptor_memory_retrieve), not the raw RaptorClient — see also
    app/tests/test_error_sanitization.py for the client-layer coverage this
    builds on."""
    client = RaptorClient("http://raptor-api-gateway:8012/api/0.4", "fake-jwt")
    boom = httpx.ConnectError("Connection refused to raptor-memory-service:8026")

    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.request_context.lifespan_context = {"raptor_client": client}

    mcp = FastMCP("test")
    memory_tool.register(mcp)
    tool_fn = mcp._tool_manager._tools["raptor_memory_retrieve"].fn

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(client._http, "request", AsyncMock(side_effect=boom))
        with pytest.raises(Exception) as exc_info:
            await tool_fn(query="anything", ctx=ctx)

    message = str(exc_info.value)
    assert "raptor-memory-service" not in message
    assert "8026" not in message
    assert len(message) < 200
