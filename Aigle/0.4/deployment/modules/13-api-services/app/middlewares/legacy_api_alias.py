"""ASGI middleware providing the /api/0.3 -> /api/0.4 compatibility alias.
"""
from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LEGACY_PREFIX = "/api/0.3/"
ALIASED_GROUPS = ("chat", "search", "asset", "a2a", "sso", "processing")


class LegacyApiAliasMiddleware:
    """Rewrites /api/0.3/{chat,search,asset,a2a}/* to the canonical prefix."""

    def __init__(self, app: ASGIApp, canonical_prefix: str) -> None:
        self.app = app
        self.canonical_prefix = canonical_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(LEGACY_PREFIX):
            await self.app(scope, receive, send)
            return

        rest = scope["path"][len(LEGACY_PREFIX):]
        group = rest.split("/", 1)[0]
        if group not in ALIASED_GROUPS:
            await self.app(scope, receive, send)
            return

        canonical_path = self.canonical_prefix + rest
        scope = dict(scope)
        scope["path"] = canonical_path
        if scope.get("raw_path"):
            scope["raw_path"] = canonical_path.encode("utf-8")

        async def send_with_deprecation_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("Deprecation", "true")
                headers.append("Link", f'<{canonical_path}>; rel="successor-version"')
            await send(message)

        await self.app(scope, receive, send_with_deprecation_headers)
