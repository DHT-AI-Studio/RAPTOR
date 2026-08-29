"""Downstream error responses must never reach the MCP client verbatim —
only a bounded, generic message. Full details still go to server logs."""
from __future__ import annotations

import httpx
import pytest
from unittest.mock import AsyncMock, patch

from app.services.raptor_client import MCPAuthError, MCPToolError, RaptorClient

_LEAKY_BODY = (
    'Traceback (most recent call last):\n'
    '  File "/app/routes/memory.py", line 42, in retrieve\n'
    '    conn = psycopg2.connect(host="raptor-personal-db-service", '
    'port=5432, user="admin", password="S3cr3t!")\n'
    'psycopg2.OperationalError: could not connect to server'
)


def _client() -> RaptorClient:
    return RaptorClient("http://raptor-api-gateway:8012/api/0.4", "fake-jwt")


@pytest.mark.asyncio
async def test_5xx_body_not_forwarded_to_client():
    resp = httpx.Response(500, text=_LEAKY_BODY,
                           request=httpx.Request("POST", "http://x/memory/retrieve"))
    client = _client()
    with patch.object(client._http, "request", new=AsyncMock(return_value=resp)):
        with pytest.raises(MCPToolError) as exc_info:
            await client.post_json("/memory/retrieve", {}, tool_name="memory_retrieve")

    message = str(exc_info.value)
    assert "psycopg2" not in message
    assert "S3cr3t!" not in message
    assert "raptor-personal-db-service" not in message
    assert "500" in message


@pytest.mark.asyncio
async def test_401_body_not_forwarded_to_client():
    resp = httpx.Response(401, text=_LEAKY_BODY,
                           request=httpx.Request("GET", "http://x/asset/users/commits"))
    client = _client()
    with patch.object(client._http, "request", new=AsyncMock(return_value=resp)):
        with pytest.raises(MCPAuthError) as exc_info:
            await client.get_json("/asset/users/commits", tool_name="raptor_list_assets")

    message = str(exc_info.value)
    assert "psycopg2" not in message
    assert "S3cr3t!" not in message


@pytest.mark.asyncio
async def test_other_4xx_does_not_leak_internal_url():
    """r.raise_for_status() would normally embed the full request URL in its
    default message — confirm we're not relying on it."""
    resp = httpx.Response(422, text='{"detail": "bad request"}',
                           request=httpx.Request("POST", "http://raptor-api-gateway:8012/api/0.4/asset/fileupload_analysis"))
    client = _client()
    with patch.object(client._http, "request", new=AsyncMock(return_value=resp)):
        with pytest.raises(MCPToolError) as exc_info:
            await client.post_json("/asset/fileupload_analysis", {}, tool_name="raptor_upload_asset")

    message = str(exc_info.value)
    assert "raptor-api-gateway" not in message
    assert "8012" not in message
    assert "422" in message


@pytest.mark.asyncio
async def test_network_error_does_not_leak_raw_exception():
    client = _client()
    boom = httpx.ConnectError("[Errno 111] Connection refused to raptor-memory-service:8099")
    with patch.object(client._http, "request", new=AsyncMock(side_effect=boom)):
        with pytest.raises(MCPToolError) as exc_info:
            await client.post_json("/memory/retrieve", {}, tool_name="memory_retrieve")

    message = str(exc_info.value)
    assert "raptor-memory-service" not in message
    assert "8099" not in message
