"""MCP router — raw proxy to the Raptor MCP server (module 27).

No JWT/UMA gate at the gateway level: the MCP server extracts and forwards
the bearer token itself (see module 27's `app.context.current_bearer`), and
each Raptor tool call re-validates against Module 13 downstream. This is a
byte-for-byte reverse proxy, not a JSON-in/JSON-out proxy like the other
routers — the MCP Streamable HTTP transport returns either `application/json`
or `text/event-stream`, and both must pass through unmodified.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.dependencies import get_current_user, get_http_client
from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)

router = APIRouter()

# Hop-by-hop headers that must not be forwarded either direction — Starlette/
# httpx recompute these themselves based on the actual streamed body.
_HOP_BY_HOP = {"content-length", "transfer-encoding", "connection", "keep-alive"}

# MCP-specific timeout: some tools (raptor_chat, raptor_a2a_agent) document up
# to 120s latency, well beyond the gateway's default request_timeout.
_MCP_TIMEOUT = httpx.Timeout(150.0)


async def _stream_and_close(resp: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    finally:
        await resp.aclose()


@router.post("", include_in_schema=True, tags=["mcp"])
async def mcp_proxy(
    request: Request,
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Proxy JSON-RPC requests to the MCP server's Streamable HTTP endpoint.

    Forwards the raw body and the headers the MCP session protocol depends
    on (Authorization, Mcp-Session-Id, Accept), and streams the response
    back unmodified so both plain JSON and text/event-stream responses work.
    """
    body = await request.body()

    forward_headers = {
        "content-type": request.headers.get("content-type", "application/json"),
        "accept": request.headers.get("accept", "application/json, text/event-stream"),
    }
    if "authorization" in request.headers:
        forward_headers["authorization"] = request.headers["authorization"]
    if "mcp-session-id" in request.headers:
        forward_headers["mcp-session-id"] = request.headers["mcp-session-id"]

    downstream_request = http_client.build_request(
        "POST",
        f"{settings.mcp_server_url}/mcp",
        content=body,
        headers=forward_headers,
        timeout=_MCP_TIMEOUT,
    )

    try:
        resp = await http_client.send(downstream_request, stream=True)
    except httpx.RequestError as exc:
        _logger.error("MCP proxy request failed: %s", exc)
        raise

    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "content-type"
    }

    return StreamingResponse(
        _stream_and_close(resp),
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )


# ── Agent registration ────────────────────────────────────────────────────────
# Unlike the raw JSON-RPC proxy above, this is a plain JSON-in/JSON-out call
# to module 27's own /mcp/auth/register endpoint (agent credential storage,
# not part of the MCP protocol itself) — and unlike the proxy above, IS gated
# by the gateway's own JWT check, since registering agent credentials is a
# privileged action distinct from using already-registered tools.

# Field names match Module 27's real app/routers/auth.py::RegisterRequest/
# RegisterResponse exactly — this used to be agent_id/username/password ->
# agent_id/message, which doesn't correspond to any request Module 27's
# endpoint actually accepts (it validates the given client_id/client_secret
# as a Keycloak confidential-client service account, not a username/password
# pair). Confirmed live: neither shape ever reached Module 27 at all, because
# /api/{version}/mcp/* was never registered as a Keycloak UMA resource (see
# KEYCLOAK_INFERENCE_URIS) -- every request was rejected at the gateway's own
# authorization check before hitting this handler, which is presumably why
# the schema mismatch went unnoticed.
class RegisterRequest(BaseModel):
    client_id: str
    client_secret: str
    scope: Optional[str] = None


class RegisterResponse(BaseModel):
    agent_token: str
    expires_in: int


@router.post("/auth/register", response_model=RegisterResponse, tags=["mcp"])
async def register_agent(
    request: Request,
    body: RegisterRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> RegisterResponse:
    """Register an agent's Keycloak service-account credentials with the MCP
    server for autonomous token refresh. Returns an opaque agent_token
    (Module 27-issued) -- use it as `Authorization: Bearer <agent_token>` on
    MCP tool calls; Module 27 resolves it to a Keycloak token and
    auto-refreshes via client_credentials."""
    try:
        resp = await http_client.post(
            f"{settings.mcp_server_url}/mcp/auth/register",
            json=body.model_dump(exclude_none=True),
            # Module 27's require_human_jwt only checks that *some* Bearer
            # token is present (Module 13 already did the real validation via
            # get_current_user above) -- but it still needs to see one on the
            # forwarded request, or it 401s with "Human JWT required" even
            # though the caller was already authenticated at the gateway.
            headers={"authorization": request.headers["authorization"]},
            timeout=_MCP_TIMEOUT,
        )
        resp.raise_for_status()
        return RegisterResponse(**resp.json())
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        _logger.error("MCP agent registration proxy failed: %s", e)
        raise HTTPException(status_code=502, detail=f"MCP server unreachable: {e}")


@router.delete("/auth/register/{agent_token}", status_code=204, response_model=None, tags=["mcp"])
async def revoke_agent(
    request: Request,
    agent_token: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Revoke a previously-registered agent_token (removes its Redis credential
    blob on Module 27). Mirrors Module 27's own DELETE /mcp/auth/register/{agent_token}
    -- was missing from the gateway despite register being proxied, leaving no
    way to revoke an agent through the public API."""
    try:
        resp = await http_client.delete(
            f"{settings.mcp_server_url}/mcp/auth/register/{agent_token}",
            headers={"authorization": request.headers["authorization"]},
            timeout=_MCP_TIMEOUT,
        )
        resp.raise_for_status()
        return Response(status_code=204)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        _logger.error("MCP agent revoke proxy failed: %s", e)
        raise HTTPException(status_code=502, detail=f"MCP server unreachable: {e}")
