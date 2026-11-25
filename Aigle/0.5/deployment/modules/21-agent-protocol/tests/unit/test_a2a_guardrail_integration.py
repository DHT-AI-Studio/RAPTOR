# tests/unit/test_a2a_guardrail_integration.py
"""
GB-21 Integration tests — POST /a2a/jsonrpc guardrail pre-check
Uses httpx.AsyncClient against the real FastAPI app (no Docker needed).
Module 23 is mocked via respx.
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient
from main import app  # path set up by tests/conftest.py

# ── Helpers ───────────────────────────────────────────────────────────────────
GUARDRAIL_URL = "http://guardrail-mock/guardrail/policy/decide"


def _jsonrpc(text: str = "hello", method: str = "agent.query") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            "message": {
                "parts": [{"type": "text", "text": text}]
            }
        },
    }


def _guardrail_cfg(mock_cfg):
    mock_cfg.return_value.guardrail_active = True
    mock_cfg.return_value.guardrail_url = "http://guardrail-mock"


# ── Tests ─────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_harmful_task_blocked():
    """block decision → JSON-RPC -32600 Policy violation, rpc_handler never called."""
    respx.post(GUARDRAIL_URL).mock(
        return_value=httpx.Response(200, json={
            "decision": "block",
            "categories": ["hate_speech"],
            "reason": "harmful content detected",
        })
    )

    with patch("handlers.task_handler.get_settings") as mock_cfg, \
         patch("main.rpc_handler") as mock_rpc:
        _guardrail_cfg(mock_cfg)
        mock_rpc.handle = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": "ok"}
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/a2a/jsonrpc", json=_jsonrpc(text="I hate everyone"))

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32600
    assert "Policy violation" in body["error"]["message"]
    assert "hate_speech" in body["error"]["data"]["category"]
    mock_rpc.handle.assert_not_called()


@pytest.mark.asyncio
@respx.mock
async def test_safe_task_passes_through():
    """allow decision → task forwarded to rpc_handler, response transparent."""
    respx.post(GUARDRAIL_URL).mock(
        return_value=httpx.Response(200, json={
            "decision": "allow",
            "categories": [],
            "reason": "",
        })
    )

    with patch("handlers.task_handler.get_settings") as mock_cfg, \
         patch("main.rpc_handler") as mock_rpc:
        _guardrail_cfg(mock_cfg)
        mock_rpc.handle = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": {"answer": "42"}}
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/a2a/jsonrpc", json=_jsonrpc(text="What is 6x7?"))

    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body
    assert body["result"]["answer"] == "42"
    mock_rpc.handle.assert_called_once()


@pytest.mark.asyncio
@respx.mock
async def test_internal_call_skips_guardrail():
    """X-A2A-Internal: true → guardrail endpoint never hit, task forwarded directly."""
    guardrail_route = respx.post(GUARDRAIL_URL)

    with patch("handlers.task_handler.get_settings") as mock_cfg, \
         patch("main.rpc_handler") as mock_rpc:
        _guardrail_cfg(mock_cfg)
        mock_rpc.handle = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": "internal ok"}
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/a2a/jsonrpc",
                json=_jsonrpc(text="internal agent message"),
                headers={"X-A2A-Internal": "true"},
            )

    assert resp.status_code == 200
    assert not guardrail_route.called
    mock_rpc.handle.assert_called_once()
