"""MCP router — raw proxy to the Raptor MCP server (module 27).

`require_mcp_credential` is a deliberately different gate from every other
router's `Depends(get_current_user)` at the mount, not an omission: two
different credential shapes can legitimately arrive in `Authorization` here.
A human/service caller sends a real Raptor JWT — that gets the exact same
`get_current_user` check every other route gets. An autonomous agent sends an
opaque `mcp-agent-*` token (module 27's own credential, issued by
`POST /mcp/auth/register`, resolved against module 27's Redis store) — that
is not a JWT `verify_jwt()` can parse at all, so bolting plain
`Depends(get_current_user)` onto this mount would reject every legitimate
agent call. What this gate still enforces, deny-by-default, is that *some*
bearer credential is present — closing the gap where a request with no
Authorization header at all used to reach module 27, which (before its own
fix — see module 27's `app/tools/__init__.py::_resolve_token`) silently
executed the tool as the MCP server's own service account instead of being
rejected. This is a byte-for-byte reverse proxy, not a JSON-in/JSON-out proxy
like the other routers — the MCP Streamable HTTP transport returns either
`application/json` or `text/event-stream`, and both must pass through
unmodified.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.dependencies import get_current_user, get_http_client
from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)

router = APIRouter()

# Root-level, unversioned — same pattern as agent_protocol.py's
# well_known_router (A2A spec). RFC 9728 requires this exact path with no
# version prefix so generic OAuth clients can find it without knowing
# anything about Raptor's own /api/{version}/ convention.
mcp_well_known_router = APIRouter()

# Hop-by-hop headers that must not be forwarded either direction — Starlette/
# httpx recompute these themselves based on the actual streamed body.
_HOP_BY_HOP = {"content-length", "transfer-encoding", "connection", "keep-alive"}


def _public_base(request: Request, settings: Settings) -> str:
    """The gateway's base URL as clients reach it: GATEWAY_PUBLIC_URL when set (behind a TLS proxy),
    else the request's own."""
    return (settings.public_url or str(request.base_url)).rstrip("/")


@mcp_well_known_router.get(
    "/.well-known/oauth-protected-resource",
    include_in_schema=False,
)
async def oauth_protected_resource_metadata(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """RFC 9728 Protected Resource Metadata — tells an MCP client (e.g.
    Claude Code) where the real Authorization Server (Keycloak) lives, so it
    can run authorization_code + PKCE directly against Keycloak. Module 13
    never handles /authorize, /callback, or token exchange itself — see
    docs/MCP_OAUTH_REQUIREMENTS.md for the full design and why.

    `resource` is derived from the actual Host header the client used to
    reach this gateway (works correctly regardless of which hostname/IP a
    given deployment is reached at); `authorization_servers` comes from
    keycloak_public_url, which — unlike keycloak_url — must be configured to
    an address this MCP client can actually resolve, not an internal Docker
    hostname.
    """
    resource = _public_base(request, settings) + f"/api/{settings.api_version}/mcp"
    issuer = f"{settings.keycloak_public_url.rstrip('/')}/realms/{settings.keycloak_realm}"
    return {
        "resource": resource,
        "authorization_servers": [issuer],
    }

# MCP-specific timeout: some tools (raptor_chat, raptor_a2a_agent) document up
# to 120s latency, well beyond the gateway's default request_timeout.
_MCP_TIMEOUT = httpx.Timeout(150.0)


async def _stream_and_close(resp: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    finally:
        await resp.aclose()


async def require_mcp_credential(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Deny-by-default gate for the raw MCP transport — see module docstring
    for why this isn't just `Depends(get_current_user)`."""
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Missing bearer credential")
    if token.startswith("mcp-agent-"):
        return  # opaque agent credential — module 27 resolves/validates this itself
    await get_current_user(request, token=token, settings=settings)


_MCP_REQUIRED_ACCEPT = ("application/json", "text/event-stream")


def _ensure_mcp_accept(accept: str | None) -> str:
    """MCP's Streamable HTTP transport requires both application/json and
    text/event-stream in Accept, or module 27's SDK returns 406. A caller
    that only sends one (e.g. Swagger UI's fixed Accept: application/json)
    gets rejected -- add whichever is missing instead of forwarding as-is.
    """
    if not accept:
        return ", ".join(_MCP_REQUIRED_ACCEPT)
    types = [t.strip() for t in accept.split(",") if t.strip()]
    present = {t.split(";")[0].strip().lower() for t in types}
    types += [req for req in _MCP_REQUIRED_ACCEPT if req not in present]
    return ", ".join(types)


# openapi_extra is docs-only (Swagger UI showed "No parameters" and sent an
# empty body) -- runtime still reads request.body() raw, no Pydantic model.
_MCP_INITIALIZE_EXAMPLE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "swagger-ui", "version": "1.0"},
    },
}


@router.post(
    "",
    include_in_schema=True,
    tags=["mcp"],
    dependencies=[Depends(require_mcp_credential)],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"},
                    "example": _MCP_INITIALIZE_EXAMPLE,
                }
            },
        }
    },
)
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
        "accept": _ensure_mcp_accept(request.headers.get("accept")),
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

    if resp.status_code == 401:
        # RFC 9728 §5.1: point the client at the discovery metadata so a
        # compliant MCP client can find the authorize_code+PKCE flow instead
        # of needing it hardcoded. Module 27 (the actual source of this 401 —
        # see this file's module docstring) never sets its own
        # WWW-Authenticate header, so there's nothing to merge with here.
        metadata_url = _public_base(request, settings) + "/.well-known/oauth-protected-resource"
        response_headers["www-authenticate"] = (
            f'Bearer error="invalid_token", resource_metadata="{metadata_url}"'
        )

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
    scope: str | None = None


class RegisterResponse(BaseModel):
    agent_token: str
    expires_in: int


@router.post("/auth/register", response_model=RegisterResponse, tags=["mcp"])
async def register_agent(
    request: Request,
    body: RegisterRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
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
    current_user: dict[str, Any] = Depends(get_current_user),
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
