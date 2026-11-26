"""CF-3 / VIE01-233: strip identity and internal-trust headers at the gateway's ingress.

Module 13 is the only HTTP entry point for clients, and several routers
forward the incoming headers downstream as-is (agent_protocol `_proxy`, the
guardrail / training / aiml_life_cycle / docagent / benchmark proxies). A
client could therefore send its own `X-User-ID` (Module 25 trusts it),
`X-A2A-Internal: true` or `X-A2A-Source-Agent: <a known sub-agent id>`
(either one makes Module 21 skip its guardrail) or `X-Internal-Token`,
and have it reach an internal service.

This pure ASGI middleware runs outermost and removes those headers from every
http and websocket request, so identity downstream only ever comes from the
JWT: routers that need `X-User-ID` keep setting it from the token's `sub`
(agent_protocol rag_query, memory, inference, benchmark, gdpr, ...).

One exception: `X-Internal-Token` is kept on /admin/maintenance/*, whose only
caller is Module 01's NFS freeze script and which checks the token itself
(`require_internal_token` in app.main). PR E (GRD-X2) replaces that shared
token with workload identity.
"""
from __future__ import annotations

import logging

from starlette.types import ASGIApp, Receive, Scope, Send

_logger = logging.getLogger(__name__)

STRIPPED_HEADERS = frozenset({
    b"x-user-id",
    b"x-branch-id",
    b"x-internal",
    b"x-a2a-internal",
    b"x-a2a-source-agent",
    b"x-internal-token",
})
_MAINTENANCE_PATH = "/admin/maintenance"
_MAINTENANCE_KEPT = frozenset({b"x-internal-token"})


def _is_maintenance_path(path: str) -> bool:
    return path == _MAINTENANCE_PATH or path.startswith(_MAINTENANCE_PATH + "/")


class IngressHeaderStripMiddleware:
    """Drops client-supplied identity/internal headers before any router or proxy sees them."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        strip = STRIPPED_HEADERS
        if _is_maintenance_path(scope.get("path", "")):
            strip = STRIPPED_HEADERS - _MAINTENANCE_KEPT

        headers = scope.get("headers") or []
        kept = [(name, value) for name, value in headers if name.lower() not in strip]
        if len(kept) != len(headers):
            dropped = sorted({name.lower().decode("latin-1") for name, _ in headers if name.lower() in strip})
            _logger.debug("Stripped client-supplied headers %s on %s", dropped, scope.get("path"))
            scope = dict(scope)
            scope["headers"] = kept

        await self.app(scope, receive, send)
