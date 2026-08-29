import pytest
from unittest.mock import AsyncMock, MagicMock

from app.context import current_bearer
from app.services.raptor_client import MCPAuthError
from app.tools import _resolve_token, get_client


@pytest.fixture
def token_manager():
    return AsyncMock()


@pytest.mark.asyncio
async def test_missing_bearer_raises_mcp_auth_error_over_http(token_manager):
    with pytest.raises(MCPAuthError):
        await _resolve_token(None, token_manager, transport="http")

    token_manager.get_server_token.assert_not_called()


@pytest.mark.asyncio
async def test_missing_bearer_uses_server_token_over_stdio(token_manager):
    """stdio has no HTTP layer to carry a caller Authorization header, so the
    locally-spawned process authenticates as its own configured Keycloak
    service account for the whole session — this is not the "silent
    fallback" the http-transport check above guards against."""
    token_manager.get_server_token.return_value = "server-service-account-token"

    token = await _resolve_token(None, token_manager, transport="stdio")

    assert token == "server-service-account-token"
    token_manager.get_server_token.assert_awaited_once()


@pytest.mark.asyncio
async def test_raw_jwt_forwarded_as_is(token_manager):
    token = await _resolve_token("a-raw-jwt", token_manager, transport="http")

    assert token == "a-raw-jwt"
    token_manager.get_valid_keycloak_token.assert_not_called()
    token_manager.get_server_token.assert_not_called()


@pytest.mark.asyncio
async def test_valid_agent_token_resolves_to_keycloak_token(token_manager):
    token_manager.get_valid_keycloak_token.return_value = "live-keycloak-token"

    token = await _resolve_token("mcp-agent-abc123", token_manager, transport="http")

    assert token == "live-keycloak-token"
    token_manager.get_valid_keycloak_token.assert_awaited_once_with("mcp-agent-abc123")


@pytest.mark.asyncio
async def test_unknown_agent_token_raises_mcp_auth_error(token_manager):
    token_manager.get_valid_keycloak_token.return_value = None

    with pytest.raises(MCPAuthError):
        await _resolve_token("mcp-agent-revoked", token_manager, transport="http")

    token_manager.get_server_token.assert_not_called()


# ── end-to-end through get_client(), the real path every tool call uses ──────

@pytest.mark.asyncio
async def test_get_client_rejects_http_call_with_no_bearer_set(token_manager):
    """Same scenario as above but exercised through get_client(), matching
    the actual code path a real (non-mocked) tool call goes through — the
    _resolve_token unit tests above never touch this integration."""
    settings = MagicMock(transport="http")
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"settings": settings, "token_manager": token_manager}

    token = current_bearer.set(None)
    try:
        with pytest.raises(MCPAuthError):
            await get_client(ctx)
    finally:
        current_bearer.reset(token)

    token_manager.get_server_token.assert_not_called()


@pytest.mark.asyncio
async def test_get_client_uses_server_token_over_stdio_with_no_bearer_set(token_manager):
    """stdio never populates current_bearer at all (only app/main.py's HTTP
    handler does), so this is the normal, expected path for every stdio tool
    call — it must keep working after the http-transport rejection above."""
    token_manager.get_server_token.return_value = "server-service-account-token"
    settings = MagicMock(transport="stdio", api_gateway_url="http://gateway/api/0.4")
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"settings": settings, "token_manager": token_manager}

    token = current_bearer.set(None)
    try:
        client = await get_client(ctx)
    finally:
        current_bearer.reset(token)

    assert client._token == "server-service-account-token"
