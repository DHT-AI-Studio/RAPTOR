"""
Unit tests for Module 26 (Memory Service) long-term memory retrieval integration.

Run from the 15-chat-service/ directory:
    python -m pytest test_retrieve_long_term.py -v

No external services needed — Module 26 HTTP calls are mocked via httpx.MockTransport.
"""
import json
import os
import sys

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")

_MODULE_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_MODULE_ROOT, "src"))
sys.path.insert(0, _MODULE_ROOT)

import httpx
import pytest

from services.memory_client import MemoryClient
# See test_compact_context.py's _FakeService docstring for why this nested
# import (not the flat `services.chat_service`) is needed here.
from src.services.chat_service import UnifiedChatService


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── MemoryClient.search_longterm ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_longterm_returns_hits():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/memory/longterm/search"
        assert request.headers["X-User-ID"] == "user-1"
        body = json.loads(request.content)
        assert body["query"] == "語言偏好"
        assert body["top_k"] == 3
        return httpx.Response(200, json=[
            {
                "text": "用戶偏好繁體中文回覆", "score": 0.9,
                "timestamp": 123.0, "session_id": "s1", "frame_type": "preference",
            },
        ])

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_longterm("user-1", "語言偏好")
        assert len(hits) == 1
        assert hits[0]["text"] == "用戶偏好繁體中文回覆"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_longterm_fails_open_on_http_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_longterm("user-1", "query")
        assert hits == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_longterm_fails_open_on_connection_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_longterm("user-1", "query")
        assert hits == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_longterm_skips_empty_query():
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=[])

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_longterm("user-1", "   ")
        assert hits == []
        assert called is False
    finally:
        await client.aclose()


# ── Long-term context formatting (mirrors ChatService._retrieve_long_term) ────
# Extracted so the mapping can be tested without constructing the full
# UnifiedChatService (LangGraph + LLM client), matching the pattern used in
# test_compact_context.py for _compact_context.

def _format_long_term_context(hits: list[dict]) -> list[str]:
    return [
        f"[{hit.get('frame_type', 'fact')}] {hit.get('text', '')}"
        for hit in hits
        if hit.get("text")
    ]


def test_format_long_term_context_basic():
    hits = [
        {"text": "用戶偏好繁體中文回覆", "frame_type": "preference"},
        {"text": "用戶的車型是 Toyota Camry 2022", "frame_type": "fact"},
    ]
    result = _format_long_term_context(hits)
    assert result == [
        "[preference] 用戶偏好繁體中文回覆",
        "[fact] 用戶的車型是 Toyota Camry 2022",
    ]


def test_format_long_term_context_skips_empty_text():
    hits = [{"text": "", "frame_type": "fact"}, {"text": "kept", "frame_type": "fact"}]
    assert _format_long_term_context(hits) == ["[fact] kept"]


def test_format_long_term_context_defaults_frame_type():
    hits = [{"text": "no type given"}]
    assert _format_long_term_context(hits) == ["[fact] no type given"]


def test_format_long_term_context_empty_hits():
    assert _format_long_term_context([]) == []


# ── Graph node: _retrieve_long_term (503 must not affect chat) ─────────────────

class _FakeService:
    """Minimal stand-in with the one attribute _retrieve_long_term reads —
    see test_compact_context.py's _FakeService for why this avoids
    constructing a full UnifiedChatService (ChatOpenAI, search_service, etc)."""
    def __init__(self, memory_client):
        self.memory_client = memory_client


@pytest.mark.asyncio
async def test_retrieve_long_term_node_survives_module26_503():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _client(handler)
    try:
        fake = _FakeService(MemoryClient("http://memory:8026", client))
        result = await UnifiedChatService._retrieve_long_term(
            fake, {"user_id": "user-1", "current_query": "hi"},
        )
    finally:
        await client.aclose()

    # Must not raise, and chat proceeds with empty long-term context.
    assert result == {"long_term_context": []}


@pytest.mark.asyncio
async def test_retrieve_long_term_node_no_memory_client():
    fake = _FakeService(memory_client=None)
    result = await UnifiedChatService._retrieve_long_term(
        fake, {"user_id": "user-1", "current_query": "hi"},
    )
    assert result == {"long_term_context": []}
