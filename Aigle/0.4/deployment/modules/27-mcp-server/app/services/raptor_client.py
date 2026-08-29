from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_DEFAULT_TIMEOUT   = _settings.timeout_default
_UPLOAD_TIMEOUT    = _settings.timeout_upload
_MAX_ATTEMPTS      = _settings.max_attempts  # 1 initial + (max_attempts - 1) retries
_RETRY_BACKOFF     = _settings.retry_backoff_seconds


class MCPAuthError(Exception):
    """Raised on 401 from Module 13 — invalid or expired JWT."""


class MCPToolError(Exception):
    """Raised when Module 13 returns 5xx after all retries are exhausted."""


class RaptorClient:
    """Shared async HTTP client for all MCP tools.

    Usage:
        async with RaptorClient(base_url, jwt_token) as client:
            data = await client.post_json("/search/hybrid", body, tool_name="raptor_search")
    """

    def __init__(self, base_url: str, jwt_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = jwt_token
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def __aenter__(self) -> "RaptorClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        tool_name: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        **kwargs,
    ) -> httpx.Response:
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_ATTEMPTS):
            t0 = time.monotonic()
            try:
                r = await self._http.request(
                    method, path,
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=timeout,
                    **kwargs,
                )
                elapsed = time.monotonic() - t0
                logger.debug("[%s] %s %s → %d (%.2fs)", tool_name, method, path, r.status_code, elapsed)

                if r.status_code == 401:
                    # Downstream response bodies may contain internal error
                    # text, stack traces, etc. — log the full body server-side
                    # only; the client always gets a bounded, generic message.
                    logger.warning("[%s] %s %s → 401: %s", tool_name, method, path, r.text[:500])
                    raise MCPAuthError("Unauthorized (401) — invalid or expired token")

                if r.status_code >= 500:
                    if attempt < _MAX_ATTEMPTS - 1:
                        logger.warning("[%s] %s %s → %d, retry %d/%d in %.1fs",
                                       tool_name, method, path, r.status_code, attempt + 1, _MAX_ATTEMPTS - 1, _RETRY_BACKOFF)
                        await asyncio.sleep(_RETRY_BACKOFF)
                        continue
                    logger.error("[%s] %s %s → %d after %d attempts: %s",
                                 tool_name, method, path, r.status_code, _MAX_ATTEMPTS, r.text[:500])
                    raise MCPToolError(f"Downstream service error ({r.status_code}) after {_MAX_ATTEMPTS} attempts")

                if r.status_code >= 400:
                    # Any other non-2xx (400/403/404/422/...). Not using
                    # httpx's raise_for_status() here on purpose — its default
                    # message embeds the full request URL, which would leak
                    # the internal gateway hostname to the client.
                    logger.warning("[%s] %s %s → %d: %s", tool_name, method, path, r.status_code, r.text[:500])
                    raise MCPToolError(f"Downstream service error ({r.status_code})")

                return r

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                elapsed = time.monotonic() - t0
                logger.debug("[%s] %s %s → network error (%.2fs): %s", tool_name, method, path, elapsed, exc)
                last_exc = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    logger.warning("[%s] network error, retry %d/%d in %.1fs: %s", tool_name, attempt + 1, _MAX_ATTEMPTS - 1, _RETRY_BACKOFF, exc)
                    await asyncio.sleep(_RETRY_BACKOFF)

        logger.error("[%s] %s %s failed after %d attempts: %s", tool_name, method, path, _MAX_ATTEMPTS, last_exc)
        raise MCPToolError(f"Downstream service unavailable after {_MAX_ATTEMPTS} attempts")

    async def post_json(
        self,
        path: str,
        body: Dict[str, Any],
        tool_name: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.debug("[%s] → POST %s %s", tool_name, path, str(body)[:200])
        r = await self._request("POST", path, tool_name=tool_name, timeout=timeout, json=body, params=params)
        return r.json()

    async def get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        tool_name: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        logger.debug("[%s] → GET %s params=%s", tool_name, path, params)
        r = await self._request("GET", path, tool_name=tool_name, timeout=timeout, params=params)
        return r.json()

    async def upload_file(
        self,
        path: str,
        field_name: str,
        filename: str,
        content: bytes,
        content_type: str,
        extra_data: Optional[Dict[str, str]] = None,
        tool_name: str = "",
        timeout: float = _UPLOAD_TIMEOUT,
    ) -> Dict[str, Any]:
        logger.debug("[%s] → POST %s file=%s (%d bytes)", tool_name, path, filename, len(content))
        r = await self._request(
            "POST", path,
            tool_name=tool_name,
            timeout=timeout,
            files={field_name: (filename, content, content_type)},
            data=extra_data or {},
        )
        return r.json()
