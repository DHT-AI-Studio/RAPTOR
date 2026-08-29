"""
Unit tests for the compact_context LangGraph node in ChatService (MV-12).

Run from the 15-chat-service/ directory:
    python -m pytest test_compact_context.py -v

Module 26 calls are mocked via httpx.MockTransport; the local fallback trim
runs with no external services needed.
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

from core.config import settings
from services.memory_client import MemoryClient
# chat_service.py uses `from ..core.config import ...` internally, which only
# resolves correctly when it's loaded as a submodule of a real `src` package
# (matching how the Dockerfile actually runs it: `uvicorn src.main:app` with
# PYTHONPATH=/app) — hence the nested import instead of the flat one above.
from src.services.chat_service import UnifiedChatService


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _FakeService:
    """Minimal stand-in with the one attribute _compact_context reads,
    avoiding the need to construct a full UnifiedChatService (ChatOpenAI,
    search_service, etc)."""
    def __init__(self, memory_client):
        self.memory_client = memory_client


# ── Fallback: local char-based trim (Module 26 unavailable/declines) ──────────

async def _run_compact(context_window: list[str], budget_tokens: int = 20000) -> list[str]:
    """Mirror of the fallback trim branch in ChatService._compact_context()."""
    budget_chars = budget_tokens * 4
    total_chars = sum(len(s) for s in context_window)
    if total_chars <= budget_chars:
        return context_window
    trimmed: list[str] = []
    remaining = budget_chars
    for entry in reversed(context_window):
        if remaining <= 0:
            break
        trimmed.insert(0, entry)
        remaining -= len(entry)
    return trimmed


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_trim_when_under_budget():
    cw = ["short message"] * 5
    result = await _run_compact(cw, budget_tokens=20000)
    assert result == cw


@pytest.mark.asyncio
async def test_trims_oldest_entries_first():
    # Budget = 10 tokens = 40 chars; entries are 20 chars each
    entries = [f"entry_{i:02d}_{'x'*15}" for i in range(10)]  # each ~20 chars
    result = await _run_compact(entries, budget_tokens=10)  # 40 chars budget
    assert len(result) < len(entries)
    # Most-recent entries are preserved
    assert entries[-1] in result
    assert entries[-2] in result


@pytest.mark.asyncio
async def test_empty_context_window_returns_empty():
    result = await _run_compact([], budget_tokens=20000)
    assert result == []


@pytest.mark.asyncio
async def test_single_entry_never_removed():
    # Even if 1 entry exceeds budget, it must not be silently dropped
    # (the loop exits immediately after inserting it into trimmed)
    big = "x" * 10000
    result = await _run_compact([big], budget_tokens=1)
    assert result == [big]


@pytest.mark.asyncio
async def test_preserves_order():
    entries = [f"msg_{i}" for i in range(20)]
    result = await _run_compact(entries, budget_tokens=5)
    # Result must be a contiguous suffix of the input
    if result:
        idx = entries.index(result[0])
        assert entries[idx:idx + len(result)] == result


@pytest.mark.asyncio
async def test_config_budget_default():
    # Verify the setting is present and sane
    assert settings.COMPACT_CONTEXT_BUDGET == 20000


# ── MemoryClient.compact_session / get_latest_summary ──────────────────────────

@pytest.mark.asyncio
async def test_compact_session_posts_expected_request():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/memory/sessions/sess1/compact"
        assert request.headers["X-User-ID"] == "user-1"
        body = json.loads(request.content)
        assert body == {"trigger": "auto", "context_window": 128000}
        return httpx.Response(200, json={"compacted": True, "source": "session_memory"})

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        result = await mc.compact_session("user-1", "sess1", context_window=128000)
        assert result == {"compacted": True, "source": "session_memory"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_compact_session_fails_open_on_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        result = await mc.compact_session("user-1", "sess1", context_window=128000)
        assert result is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_latest_summary_picks_most_recent():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/memory/sessions/sess1/summaries"
        return httpx.Response(200, json=[
            {"summary_id": "a", "created_at": "2026-01-01T00:00:00Z", "text": "old"},
            {"summary_id": "b", "created_at": "2026-06-01T00:00:00Z", "text": "new"},
        ])

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        summary = await mc.get_latest_summary("user-1", "sess1")
        assert summary["summary_id"] == "b"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_latest_summary_returns_none_when_empty():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _client(handler)
    try:
        mc = MemoryClient("http://memory:8026", client)
        assert await mc.get_latest_summary("user-1", "sess1") is None
    finally:
        await client.aclose()


# ── ChatService._compact_context (exercised via the actual method) ────────────

@pytest.mark.asyncio
async def test_compact_context_prepends_summary_when_module26_compacts():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/compact"):
            return httpx.Response(200, json={
                "compacted": True, "source": "llm_compact",
                "pre_compact_tokens": 90000, "post_compact_tokens": 15000,
            })
        return httpx.Response(200, json=[
            {"summary_id": "s1", "created_at": "2026-07-01T00:00:00Z", "text": "壓縮後的摘要內容"},
        ])

    client = _client(handler)
    try:
        service = _FakeService(MemoryClient("http://memory:8026", client))
        state = {
            "user_id": "user-1", "session_id": "sess1",
            "context_window": ["用戶: 你好\n助手: 你好！"],
        }
        result = await UnifiedChatService._compact_context(service, state)
        assert result["context_window"][0] == "[歷史摘要]\n壓縮後的摘要內容"
        assert result["context_window"][1:] == state["context_window"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_compact_context_falls_back_to_local_trim_when_module26_declines():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"compacted": False, "source": "under_budget"})

    client = _client(handler)
    try:
        service = _FakeService(MemoryClient("http://memory:8026", client))
        state = {
            "user_id": "user-1", "session_id": "sess1",
            "context_window": ["用戶: 你好\n助手: 你好！"],
        }
        result = await UnifiedChatService._compact_context(service, state)
        # Under local budget too => no change
        assert result == {}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_compact_context_falls_back_to_local_trim_on_module26_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        service = _FakeService(MemoryClient("http://memory:8026", client))
        entries = [f"entry_{i:02d}_{'x'*15}" for i in range(10)]
        state = {"user_id": "user-1", "session_id": "sess1", "context_window": entries}
        result = await UnifiedChatService._compact_context(service, state)
        # Module 26 unreachable => local fallback trim still runs
        assert result == {}  # entries total well under default 20000-token budget
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_compact_context_no_memory_client_uses_local_trim():
    service = _FakeService(memory_client=None)
    cw = ["short message"] * 5
    state = {"user_id": "user-1", "session_id": "sess1", "context_window": cw}
    result = await UnifiedChatService._compact_context(service, state)
    assert result == {}


@pytest.mark.asyncio
async def test_compact_context_empty_context_window_short_circuits():
    service = _FakeService(memory_client=None)
    state = {"user_id": "user-1", "session_id": "sess1", "context_window": []}
    result = await UnifiedChatService._compact_context(service, state)
    assert result == {}


@pytest.mark.asyncio
async def test_compact_context_empty_context_window_still_calls_module26():
    """Regression test: `context_window` is Redis's local short-term cache
    (populated only by `_load_memory`), separate from Module 26's own
    archived session. A session compacted for the first time via
    /api/v1/chat — or one whose history was seeded directly into Module 26 —
    can have an empty local context_window while Module 26 already holds a
    session well past its compact threshold. Module 26 must still be asked;
    the empty local cache should not silently skip the check."""
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/compact"):
            return httpx.Response(200, json={
                "compacted": True, "source": "llm_compact",
                "pre_compact_tokens": 90000, "post_compact_tokens": 15000,
            })
        return httpx.Response(200, json=[
            {"summary_id": "s1", "created_at": "2026-07-01T00:00:00Z", "text": "壓縮後的摘要內容"},
        ])

    client = _client(handler)
    try:
        service = _FakeService(MemoryClient("http://memory:8026", client))
        state = {"user_id": "user-1", "session_id": "sess1", "context_window": []}
        result = await UnifiedChatService._compact_context(service, state)
        assert any(p.endswith("/compact") for p in calls), (
            "compact_context must call Module 26 even when the local Redis "
            "context_window cache is empty"
        )
        assert result == {"context_window": ["[歷史摘要]\n壓縮後的摘要內容"]}
    finally:
        await client.aclose()
