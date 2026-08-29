from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context

from app.context import current_bearer
from app.core.config import get_settings

if TYPE_CHECKING:
    from app.services.raptor_client import RaptorClient
    from app.services.token_manager import TokenManager
    from app.core.config import Settings


async def _resolve_token(bearer: str | None, token_manager: "TokenManager", transport: str) -> str:
    if bearer:
        # Opaque agent_token → resolve to a live Keycloak token, auto-refreshed.
        if bearer.startswith("mcp-agent-"):
            token = await token_manager.get_valid_keycloak_token(bearer)
            if token:
                return token
            # Unknown, revoked, or unresolvable (e.g. Redis unreachable) —
            # reject outright rather than silently continuing under the
            # server's own service-account identity.
            from app.services.raptor_client import MCPAuthError
            raise MCPAuthError(f"Unknown or revoked agent_token: {bearer[:20]}...")
        return bearer  # already a raw JWT — forward as-is
    if transport == "stdio":
        # stdio has no HTTP layer, so there's no per-call Authorization
        # header to forward in the first place — the locally-spawned process
        # always acts as its own configured Keycloak service account for the
        # whole session (see MCP_KEYCLOAK_USERNAME/PASSWORD).
        return await token_manager.get_server_token()
    # http transport with no Authorization header — reject rather than
    # silently falling back to the server's own service-account identity.
    # token_manager's get_server_token() is still used deliberately elsewhere
    # (e.g. public resource reads in resources/raptor_resources.py, and the
    # stdio case above); over HTTP, tool calls must always carry the
    # caller's own JWT.
    from app.services.raptor_client import MCPAuthError
    raise MCPAuthError("Missing Authorization: Bearer <token> — MCP tool calls require a caller JWT")


async def get_client(ctx: Context, timeout: float = get_settings().timeout_default) -> "RaptorClient":
    """Return an authenticated RaptorClient for this request.

    Tests inject a mock by putting ``"raptor_client"`` directly in the lifespan
    context.  Production code resolves the JWT from the ``current_bearer``
    ContextVar and creates a real client.
    """
    from app.services.raptor_client import RaptorClient

    lifespan = ctx.request_context.lifespan_context

    if "raptor_client" in lifespan:
        return lifespan["raptor_client"]

    settings: Settings = lifespan["settings"]
    token_manager: TokenManager = lifespan["token_manager"]
    bearer = current_bearer.get(None)
    token = await _resolve_token(bearer, token_manager, settings.transport)
    return RaptorClient(settings.api_gateway_url, token)
