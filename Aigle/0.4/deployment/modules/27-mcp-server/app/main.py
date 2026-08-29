from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version as _pkg_version
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from app.context import current_bearer
from app.core.config import get_settings
from app.mcp_server import mcp
from app.services.token_manager import TokenManager

_logger = logging.getLogger(__name__)


class _AlreadySentResponse(Response):
    """No-op FastAPI response — session_manager already sent via raw ASGI send."""

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        pass


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # FastMCP lifespan (yields settings + token_manager into lifespan_context)
    # is bound to mcp._mcp_server via lifespan_wrapper; StreamableHTTPSessionManager
    # runs it automatically when session_manager.run() is entered.
    session_manager = StreamableHTTPSessionManager(mcp._mcp_server)
    app.state.session_manager = session_manager
    # Token manager for the agent-auth router. Shares Redis with the
    # tool-side instance, so agent credentials are visible to both.
    app.state.token_manager = TokenManager(get_settings())

    async with session_manager.run():
        _logger.info("Raptor MCP Server ready — Streamable HTTP on /mcp, SSE alias on /mcp/events")
        yield

    _logger.info("MCP Server shutdown")


app = FastAPI(
    title="Raptor MCP Server",
    description="Exposes Raptor 0.4 APIs as MCP tools, resources, and prompts.",
    version="0.4.0",
    lifespan=lifespan,
)

from app.routers.auth import router as auth_router  # noqa: E402
app.include_router(auth_router, prefix="/mcp/auth")


async def _dispatch_mcp(request: Request) -> _AlreadySentResponse:
    bearer = _extract_bearer(request)
    token = current_bearer.set(bearer)
    try:
        await request.app.state.session_manager.handle_request(
            request.scope, request.receive, request._send
        )
    finally:
        current_bearer.reset(token)
    return _AlreadySentResponse()


@app.post("/mcp")
async def mcp_post(request: Request) -> _AlreadySentResponse:
    """JSON-RPC over HTTP — MCP 1.1 Streamable HTTP transport."""
    return await _dispatch_mcp(request)


@app.get("/mcp")
async def mcp_get(request: Request) -> _AlreadySentResponse:
    """Open SSE stream — MCP 1.1 Streamable HTTP GET channel."""
    return await _dispatch_mcp(request)


@app.delete("/mcp")
async def mcp_delete(request: Request) -> _AlreadySentResponse:
    """Explicit session termination — MCP 1.1 Streamable HTTP DELETE channel.

    Clients send this on close. Without the route FastAPI answered 405, which
    the spec allows but leaves the session's streams to be reclaimed only when
    the connection drops; handing it to the session manager closes them at once.
    """
    return await _dispatch_mcp(request)


@app.get("/mcp/events")
async def mcp_events(request: Request) -> _AlreadySentResponse:
    """SSE stream alias for clients that connect via /mcp/events."""
    return await _dispatch_mcp(request)


@app.get("/mcp/capabilities", tags=["mcp"])
async def mcp_capabilities() -> JSONResponse:
    """MCP server capabilities document for client discovery."""
    from app.core.config import get_settings
    s = get_settings()
    return JSONResponse({
        "protocolVersion": "2024-11-05",
        "mcp_version": "1.1",
        "serverInfo": {"name": s.server_name, "version": s.server_version},
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
            "logging": {},
        },
        "transports": {
            "streamableHttp": {"endpoint": "/mcp", "methods": ["GET", "POST", "DELETE"]},
            "sse": {"endpoint": "/mcp/events"},
            "stdio": {"entrypoint": "python -m app.stdio_main", "envFlag": "MCP_TRANSPORT=stdio"},
        },
    })


@app.get("/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "mcp_version": "1.1",
        "mcp_sdk_version": _pkg_version("mcp"),
    }


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "mcp_endpoint": "/mcp",
        "mcp_events_endpoint": "/mcp/events",
        "mcp_capabilities": "/mcp/capabilities",
        "agent_register": "/mcp/auth/register",
        "health": "/health",
        "docs": "/docs",
    }
