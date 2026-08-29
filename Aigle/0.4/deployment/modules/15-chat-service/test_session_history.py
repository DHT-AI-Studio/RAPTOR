"""
Unit tests for Module 26 (Memory Service) session-history retrieval integration.

Run from the 15-chat-service/ directory:
    python -m pytest test_session_history.py -v

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
from src.services.chat_service import UnifiedChatService


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _FakeService:
    def __init__(self, memory_client):
        self.memory_client = memory_client


# ── MemoryClient.search_session ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_session_returns_hits():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/memory/sessions/sess1/search"
        assert request.headers["X-User-ID"] == "user-1"
        body = json.loads(request.content)
        assert body["query"] == "冷卻系統"
        assert body["top_k"] == 5
        return httpx.Response(200, json={
            "hits": [
                {"text": "...", "score": 0.8, "timestamp": "2026-01-01T00:00:00Z",
                 "turn_index": 3, "user_message": "冷卻系統怎麼運作？",
                 "assistant_response": "透過循環冷卻液"},
            ],
            "total_frames_searched": 10,
        })

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_session("user-1", "sess1", "冷卻系統")
        assert len(hits) == 1
        assert hits[0]["user_message"] == "冷卻系統怎麼運作？"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_session_fails_open_on_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_session("user-1", "sess1", "query")
        assert hits == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_search_session_skips_empty_query():
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"hits": [], "total_frames_searched": 0})

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        hits = await mc.search_session("user-1", "sess1", "   ")
        assert hits == []
        assert called is False
    finally:
        await client.aclose()


# ── ChatService._retrieve_session_history ──────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_session_history_formats_hits():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "hits": [
                {"user_message": "問題A", "assistant_response": "答案A", "turn_index": 1},
            ],
            "total_frames_searched": 5,
        })

    client = _client(handler)
    try:
        service = _FakeService(MemoryClient("http://memory:8026", client))
        state = {"user_id": "user-1", "session_id": "sess1", "current_query": "問題A"}
        result = await UnifiedChatService._retrieve_session_history(service, state)
        assert result == {"session_history_context": ["用戶: 問題A\n助手: 答案A"]}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_retrieve_session_history_no_memory_client():
    service = _FakeService(memory_client=None)
    state = {"user_id": "user-1", "session_id": "sess1", "current_query": "問題A"}
    result = await UnifiedChatService._retrieve_session_history(service, state)
    assert result == {"session_history_context": []}


@pytest.mark.asyncio
async def test_retrieve_session_history_fails_open_on_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        service = _FakeService(MemoryClient("http://memory:8026", client))
        state = {"user_id": "user-1", "session_id": "sess1", "current_query": "問題A"}
        result = await UnifiedChatService._retrieve_session_history(service, state)
        assert result == {"session_history_context": []}
    finally:
        await client.aclose()


# ── Restart survival: continuity comes from Module 26, not in-process state ────

@pytest.mark.asyncio
async def test_turn_survives_service_restart():
    """A turn archived by one UnifiedChatService "process" must be retrievable
    by a completely separate instance with no shared Python state — proving
    continuity comes from Module 26's persistence, not anything cached
    in-process (which is exactly what a Module 15 restart would wipe)."""
    store: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/memory/sessions/s1/turns":
            store.append(json.loads(request.content))
            return httpx.Response(200, json={"turn_index": len(store)})
        if request.url.path == "/memory/sessions/s1/search":
            hits = [
                {"user_message": t["user_message"], "assistant_response": t["assistant_response"]}
                for t in store
            ]
            return httpx.Response(200, json={"hits": hits})
        return httpx.Response(404)

    # "Before restart": one process instance archives a turn.
    client_before = _client(handler)
    try:
        old_process = _FakeService(MemoryClient("http://memory:8026", client_before))
        await old_process.memory_client.append_turn(
            user_id="user-1", session_id="s1",
            user_message="上次討論的冷卻系統", assistant_response="用循環冷卻液降溫",
        )
    finally:
        await client_before.aclose()

    # "After restart": brand-new instance/client, nothing shared but the
    # backing store — standing in for Module 26 surviving the restart.
    client_after = _client(handler)
    try:
        new_process = _FakeService(MemoryClient("http://memory:8026", client_after))
        result = await UnifiedChatService._retrieve_session_history(
            new_process,
            {"user_id": "user-1", "session_id": "s1", "current_query": "冷卻系統"},
        )
    finally:
        await client_after.aclose()

    assert any("冷卻系統" in c for c in result["session_history_context"])
