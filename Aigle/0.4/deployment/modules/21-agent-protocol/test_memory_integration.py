"""
Unit tests for memory integration in Module 21 (Agent Protocol).

Run from the 21-agent-protocol/ directory:
    conda run -n CIE python -m pytest test_memory_integration.py -v

Tests MemoryClient fail-open behaviour and the grounded-prompt memory block
in isolation — no A2A agents, LLM, or live Module 26 needed.
"""
import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent / "app"))

from memory_client import MemoryClient, set_auth_token  # noqa: E402
from pipeline import assemble_grounded_prompt  # noqa: E402

FAKE_BEARER = "Bearer test-jwt-token"


@pytest.fixture(autouse=True)
def _auth_token():
    """MemoryClient reads the caller's JWT from a ContextVar (set once per
    inbound request in main.py), not a per-call parameter — set it here so
    every test gets a token, matching a real authenticated request."""
    set_auth_token(FAKE_BEARER)
    yield
    set_auth_token("")


# ── MemoryClient ──────────────────────────────────────────────────────────────

def _client_with_handler(handler) -> MemoryClient:
    transport = httpx.MockTransport(handler)
    return MemoryClient(
        memory_service_url="http://module26:8026",
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.asyncio
async def test_search_longterm_returns_hits():
    def handler(request):
        assert request.url.path == "/memory/retrieve"
        assert request.headers["Authorization"] == FAKE_BEARER
        assert request.url.params["query"] == "語言偏好"
        return httpx.Response(200, json={
            "sessions": [],
            "longterm": [{"frame_type": "preference", "text": "偏好繁體中文", "score": 0.9}],
            "multimedia": [],
            "total_frames_searched": 1,
        })

    client = _client_with_handler(handler)
    hits = await client.search_longterm("u1", "語言偏好", top_k=3)
    assert hits == [{"frame_type": "preference", "text": "偏好繁體中文", "score": 0.9}]


@pytest.mark.asyncio
async def test_search_longterm_fails_open_on_error():
    def handler(request):
        return httpx.Response(500)

    client = _client_with_handler(handler)
    assert await client.search_longterm("u1", "anything") == []


@pytest.mark.asyncio
async def test_search_longterm_empty_query_short_circuits():
    def handler(request):  # pragma: no cover — must never be called
        raise AssertionError("no HTTP call expected for empty query")

    client = _client_with_handler(handler)
    assert await client.search_longterm("u1", "   ") == []


@pytest.mark.asyncio
async def test_append_turn_posts_expected_request():
    import json as _json
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"turn_index": 1})

    client = _client_with_handler(handler)
    await client.append_turn(
        user_id="u1", session_id="s1",
        user_message="q", assistant_response="a",
        search_results=[{"doc_id": "d1", "score": 0.9}],
    )
    assert captured["path"] == "/memory/archive"
    assert captured["auth"] == FAKE_BEARER
    assert captured["body"]["mode"] == "agent"


@pytest.mark.asyncio
async def test_append_turn_swallows_errors():
    def handler(request):
        return httpx.Response(503)

    client = _client_with_handler(handler)
    # Must not raise
    await client.append_turn(user_id="u1", session_id="s1",
                             user_message="q", assistant_response="a")


@pytest.mark.asyncio
async def test_append_turn_omits_session_id_param_when_none():
    """No session_id → no query param sent at all; Module 13's /archive
    alias falls back to the caller's own `default` session server-side."""
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={})

    client = _client_with_handler(handler)
    await client.append_turn(user_id="u1", session_id=None,
                             user_message="q", assistant_response="a")
    assert captured["path"] == "/memory/archive"
    assert "session_id" not in captured["params"]


@pytest.mark.asyncio
async def test_append_turn_forwards_explicit_session_id_as_query_param():
    """A real session_id must reach Module 13 so /archive compacts the
    caller's actual conversation, not a shared `default` bucket."""
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={})

    client = _client_with_handler(handler)
    await client.append_turn(user_id="u1", session_id="my-conversation-42",
                             user_message="q", assistant_response="a")
    assert captured["path"] == "/memory/archive"
    assert captured["params"]["session_id"] == "my-conversation-42"


@pytest.mark.asyncio
async def test_search_multimedia_returns_hits():
    def handler(request):
        assert request.url.path == "/memory/multimedia/search"
        return httpx.Response(200, json=[
            {"media_type": "video", "asset_path": "videos/a.mp4", "transcription": "冷卻系統"},
        ])

    client = _client_with_handler(handler)
    hits = await client.search_multimedia("u1", "冷卻系統影片 a.mp4", top_k=2)
    assert hits[0]["asset_path"] == "videos/a.mp4"


@pytest.mark.asyncio
async def test_search_multimedia_fails_open_on_error():
    def handler(request):
        return httpx.Response(500)

    client = _client_with_handler(handler)
    assert await client.search_multimedia("u1", "a.mp4") == []


@pytest.mark.asyncio
async def test_evaluate_compact_posts_expected_request():
    import json as _json
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"should_compact": True, "token_count": 90000})

    client = _client_with_handler(handler)
    result = await client.evaluate_compact(user_id="u1", session_id="s1", context_window=128000)
    assert captured["path"] == "/memory/compact/evaluate"
    assert captured["body"]["session_id"] == "s1"
    assert captured["body"]["context_window"] == 128000
    assert result == {"should_compact": True, "token_count": 90000}


@pytest.mark.asyncio
async def test_evaluate_compact_fails_open_on_error():
    def handler(request):
        return httpx.Response(500)

    client = _client_with_handler(handler)
    assert await client.evaluate_compact(user_id="u1", session_id="s1", context_window=128000) is None


@pytest.mark.asyncio
async def test_compact_session_posts_expected_request():
    import json as _json
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"compacted": False})

    client = _client_with_handler(handler)
    result = await client.compact_session(user_id="u1", session_id="s1", context_window=128000)
    assert captured["path"] == "/memory/compact"
    assert captured["params"]["session_id"] == "s1"
    assert captured["body"]["context_window"] == 128000
    assert result == {"compacted": False}


@pytest.mark.asyncio
async def test_compact_session_fails_open_on_error():
    def handler(request):
        return httpx.Response(500)

    client = _client_with_handler(handler)
    assert await client.compact_session(user_id="u1", session_id="s1", context_window=128000) is None


@pytest.mark.asyncio
async def test_compact_session_fails_open_on_503():
    def handler(request):
        return httpx.Response(503)

    client = _client_with_handler(handler)
    assert await client.compact_session(user_id="u1", session_id="s1", context_window=128000) is None


@pytest.mark.asyncio
async def test_get_latest_summary_picks_most_recent():
    def handler(request):
        return httpx.Response(200, json=[
            {"summary_id": "old", "created_at": "2026-01-01T00:00:00Z", "text": "old"},
            {"summary_id": "new", "created_at": "2026-06-01T00:00:00Z", "text": "new"},
        ])

    client = _client_with_handler(handler)
    summary = await client.get_latest_summary("u1", "s1")
    assert summary["summary_id"] == "new"


@pytest.mark.asyncio
async def test_get_latest_summary_returns_none_when_empty():
    def handler(request):
        return httpx.Response(200, json=[])

    client = _client_with_handler(handler)
    assert await client.get_latest_summary("u1", "s1") is None


# ── Media reference detection ─────────────────────────────────────────────────

def test_media_ref_regex_matches_extensions_and_asset_path():
    from pipeline import _MEDIA_REF_RE
    assert _MEDIA_REF_RE.search("上次那部 cooling.mp4 講什麼？")
    assert _MEDIA_REF_RE.search("圖 engine.PNG 的內容")
    assert _MEDIA_REF_RE.search("asset_path=videos/x 的片段")
    assert not _MEDIA_REF_RE.search("冷卻系統怎麼運作？")


# ── Grounded prompt memory block ──────────────────────────────────────────────

def test_prompt_includes_memory_block_when_present():
    prompt = assemble_grounded_prompt(
        question="Q", top_chunks=[], graph_context="", doc_sources=[],
        memory_context="[preference] 偏好繁體中文",
    )
    assert "USER MEMORY" in prompt
    assert "[preference] 偏好繁體中文" in prompt
    # Memory precedes retrieved chunks
    assert prompt.index("USER MEMORY") < prompt.index("CONTEXT CHUNKS")


def test_prompt_omits_memory_block_when_empty():
    prompt = assemble_grounded_prompt(
        question="Q", top_chunks=[], graph_context="", doc_sources=[],
    )
    assert "USER MEMORY" not in prompt


# ── mode=tool session_id threading (smolagents_host) ───────────────────────────

def test_rag_search_tool_forwards_session_id_to_pipeline(monkeypatch):
    """RAGSearchTool.forward must pass session_id through to run_rag_pipeline,
    otherwise mode=tool falls back to user_id as the session key (main.py's
    /query session_id param would silently do nothing for tool mode)."""
    import smolagents_host
    from models import OrchestratorResult

    captured = {}

    async def fake_run_rag_pipeline(question, top_k=5, request_id=None,
                                      user_id="", branch_id="", session_id="", mode="direct"):
        captured["user_id"]    = user_id
        captured["branch_id"]  = branch_id
        captured["session_id"] = session_id
        return OrchestratorResult(answer="冷卻系統靠循環冷卻液帶走熱量", sources=[], graph_context="", chunks_used=0)

    monkeypatch.setattr(smolagents_host, "run_rag_pipeline", fake_run_rag_pipeline)

    tool = smolagents_host.RAGSearchTool(user_id="u1", branch_id="b1", session_id="s1")
    tool.forward("冷卻系統怎麼運作？")
    assert tool.last_answer == "冷卻系統靠循環冷卻液帶走熱量"

    assert captured == {"user_id": "u1", "branch_id": "b1", "session_id": "s1"}


def test_build_tool_agent_propagates_session_id():
    import smolagents_host

    _agent, rag_tool = smolagents_host.build_tool_agent(
        user_id="u1", branch_id="b1", session_id="s1",
    )
    assert rag_tool.session_id == "s1"


# ── mode=agent memory integration (pipeline helpers + smolagents_host) ─────────

class _FakeMemoryClient:
    """Stand-in for memory_client.get_memory_client() — records calls, returns
    canned responses, no network."""

    def __init__(self, longterm_hits=None, should_compact=False, summary_text=None):
        self._longterm_hits  = longterm_hits or []
        self._should_compact = should_compact
        self._summary_text   = summary_text
        self.append_turn_calls: list[dict] = []

    async def search_longterm(self, user_id, question, top_k=3):
        return self._longterm_hits

    async def search_multimedia(self, user_id, question, top_k=2):
        return []

    async def evaluate_compact(self, user_id, session_id, context_window):
        return {"should_compact": self._should_compact}

    async def compact_session(self, user_id, session_id, context_window):
        if not self._should_compact:
            return None
        return {"compacted": True, "pre_compact_tokens": 90000, "post_compact_tokens": 20000}

    async def get_latest_summary(self, user_id, session_id):
        return {"text": self._summary_text} if self._summary_text else None

    async def append_turn(self, user_id, session_id, user_message, assistant_response,
                            search_results=None, tool_calls=None, mode="agent"):
        self.append_turn_calls.append({
            "user_id": user_id, "session_id": session_id,
            "user_message": user_message, "assistant_response": assistant_response,
            "search_results": search_results, "mode": mode,
        })


@pytest.mark.asyncio
async def test_gather_memory_context_empty_without_user_id():
    import pipeline
    assert await pipeline.gather_memory_context("", "s1", "Q") == ""


@pytest.mark.asyncio
async def test_gather_memory_context_includes_longterm_facts(monkeypatch):
    import pipeline

    fake = _FakeMemoryClient(longterm_hits=[{"frame_type": "preference", "text": "偏好繁體中文"}])
    import memory_client
    monkeypatch.setattr(memory_client, "get_memory_client", lambda: fake)

    ctx = await pipeline.gather_memory_context("u1", "s1", "冷卻系統怎麼運作？")
    assert "[preference] 偏好繁體中文" in ctx


@pytest.mark.asyncio
async def test_gather_memory_context_prepends_compact_summary(monkeypatch):
    import pipeline
    import memory_client

    fake = _FakeMemoryClient(should_compact=True, summary_text="上次討論了引擎冷卻系統")
    monkeypatch.setattr(memory_client, "get_memory_client", lambda: fake)

    ctx = await pipeline.gather_memory_context("u1", "s1", "Q")
    assert ctx.startswith("[歷史摘要]")
    assert "上次討論了引擎冷卻系統" in ctx


@pytest.mark.asyncio
async def test_gather_memory_context_continues_when_compaction_503s(monkeypatch):
    """Module 26's compaction endpoint returning 503 must not surface as an
    error to the agent — gather_memory_context still returns normally."""
    import pipeline
    import memory_client

    class _FlakyCompactClient(_FakeMemoryClient):
        async def evaluate_compact(self, user_id, session_id, context_window):
            raise httpx.HTTPStatusError("503", request=None, response=httpx.Response(503))

        async def compact_session(self, user_id, session_id, context_window):
            raise httpx.HTTPStatusError("503", request=None, response=httpx.Response(503))

    fake = _FlakyCompactClient(longterm_hits=[{"frame_type": "fact", "text": "上次的討論"}])
    monkeypatch.setattr(memory_client, "get_memory_client", lambda: fake)

    ctx = await pipeline.gather_memory_context("u1", "s1", "Q")
    assert "[歷史摘要]" not in ctx
    assert "上次的討論" in ctx


@pytest.mark.asyncio
async def test_archive_turn_noop_without_user_id(monkeypatch):
    import pipeline
    import memory_client

    fake = _FakeMemoryClient()
    monkeypatch.setattr(memory_client, "get_memory_client", lambda: fake)

    await pipeline.archive_turn("", "s1", "Q", "A")
    assert fake.append_turn_calls == []


@pytest.mark.asyncio
async def test_archive_turn_posts_expected_fields(monkeypatch):
    import pipeline
    import memory_client

    fake = _FakeMemoryClient()
    monkeypatch.setattr(memory_client, "get_memory_client", lambda: fake)

    await pipeline.archive_turn("u1", "", "Q", "A")
    assert fake.append_turn_calls == [{
        "user_id": "u1", "session_id": "u1",  # falls back to user_id when session_id empty
        "user_message": "Q", "assistant_response": "A",
        "search_results": [], "mode": "agent",
    }]


@pytest.mark.asyncio
async def test_build_host_agent_injects_memory_context_as_instructions(monkeypatch):
    """mode=agent must surface memory via the `instructions` system-prompt
    channel (not mixed into the task string) — see smolagents_host.py's
    build_host_agent docstring for why."""
    import smolagents_host

    async def fake_get_tools(self):
        return []

    monkeypatch.setattr(smolagents_host.A2AToolBootstrap, "get_tools", fake_get_tools)

    host, _tools = await smolagents_host.build_host_agent(
        branch_id="b1", memory_context="[preference] 偏好繁體中文",
    )
    assert host.instructions is not None
    assert "[preference] 偏好繁體中文" in host.instructions


@pytest.mark.asyncio
async def test_build_host_agent_no_instructions_when_memory_empty(monkeypatch):
    import smolagents_host

    async def fake_get_tools(self):
        return []

    monkeypatch.setattr(smolagents_host.A2AToolBootstrap, "get_tools", fake_get_tools)

    host, _tools = await smolagents_host.build_host_agent(branch_id="b1")
    assert host.instructions is None


# ── fire_and_forget: must survive a throwaway asyncio.run() loop ───────────

def test_fire_and_forget_survives_caller_loop_teardown():
    """Regression test for the archive-never-runs bug: a plain
    asyncio.create_task() scheduled inside asyncio.run() is discarded when
    that loop closes before the task runs (exactly what happens in
    RAGSearchTool.forward's asyncio.run(...) call). fire_and_forget must
    complete the coroutine on its own persistent loop regardless."""
    import memory_client

    result: dict = {}

    async def _mark_done():
        result["done"] = True

    async def _caller():
        memory_client.fire_and_forget(_mark_done())

    asyncio.run(_caller())  # caller's loop is closed by the time this returns

    for _ in range(50):
        if result.get("done"):
            break
        time.sleep(0.02)
    assert result.get("done") is True


# ── Cross-scope isolation: retrieval must never mix across users ───────────

@pytest.mark.asyncio
async def test_search_longterm_scoped_to_callers_jwt_no_cross_contamination():
    """Two different users must never see each other's long-term memory.
    Module 13 derives identity from the caller's JWT (Authorization header),
    not any client-supplied user_id — so isolation here means: whichever
    token is set in the ContextVar at call time is what the backend sees."""
    fake_store = {
        "Bearer token-for-A": {"sessions": [], "longterm": [{"frame_type": "fact", "text": "A's secret", "score": 0.9}], "multimedia": []},
        "Bearer token-for-B": {"sessions": [], "longterm": [{"frame_type": "fact", "text": "B's secret", "score": 0.9}], "multimedia": []},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        caller = request.headers["Authorization"]
        assert caller in fake_store, f"unexpected caller {caller!r}"
        return httpx.Response(200, json=fake_store[caller])

    client = _client_with_handler(handler)
    set_auth_token("Bearer token-for-A")
    hits_a = await client.search_longterm("irrelevant-user-id", "secret")
    set_auth_token("Bearer token-for-B")
    hits_b = await client.search_longterm("irrelevant-user-id", "secret")

    assert hits_a == [{"frame_type": "fact", "text": "A's secret", "score": 0.9}]
    assert hits_b == [{"frame_type": "fact", "text": "B's secret", "score": 0.9}]
    assert hits_a != hits_b


@pytest.mark.asyncio
async def test_gather_memory_context_never_leaks_across_users(monkeypatch):
    """gather_memory_context must forward exactly the user_id it was called
    with to every downstream call — no cached/shared client state that could
    bleed one user's context into another user's turn."""
    import pipeline

    class _PerUserMemoryClient(_FakeMemoryClient):
        async def search_longterm(self, user_id, question, top_k=3):
            assert user_id in ("u1", "u2")
            return [{"frame_type": "fact", "text": f"secret belonging to {user_id}"}]

    fake = _PerUserMemoryClient()
    import memory_client
    monkeypatch.setattr(memory_client, "get_memory_client", lambda: fake)

    ctx_u1 = await pipeline.gather_memory_context("u1", "s1", "Q")
    ctx_u2 = await pipeline.gather_memory_context("u2", "s2", "Q")

    assert "secret belonging to u1" in ctx_u1
    assert "secret belonging to u2" not in ctx_u1
    assert "secret belonging to u2" in ctx_u2
    assert "secret belonging to u1" not in ctx_u2
