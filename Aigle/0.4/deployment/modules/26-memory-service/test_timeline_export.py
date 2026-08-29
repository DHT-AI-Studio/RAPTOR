"""
MV-8: Timeline / Time-Travel API + Memory Export — unit + integration tests.

Run:
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/mv8_test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from services.session_memory import SessionMemoryService, TurnAppendRequest
from services.long_term_memory import LongTermMemoryService, FactAddRequest
from routers.management import ManagementService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def redis():
    r = FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def session_svc(tmp_path, redis):
    yield SessionMemoryService(redis=redis, storage_root=str(tmp_path))


@pytest_asyncio.fixture
async def mgmt_svc(tmp_path):
    from pathlib import Path
    svc = ManagementService.__new__(ManagementService)
    svc._root = Path(str(tmp_path))
    yield svc


# ── Timeline unit tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeline_chronological_order(session_svc):
    user_id, session_id = "u1", "sess_tl"
    base_ts = 1_700_000_000.0
    for i in range(5):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message=f"Q{i}", assistant_response=f"A{i}", timestamp=base_ts + i,
        ))

    resp = await session_svc.get_timeline(user_id, session_id, page=1, page_size=20)

    assert resp.total == 5
    assert len(resp.entries) == 5
    assert resp.has_next is False
    for j in range(len(resp.entries) - 1):
        assert resp.entries[j].timestamp <= resp.entries[j + 1].timestamp


@pytest.mark.asyncio
async def test_timeline_pagination(session_svc):
    user_id, session_id = "u2", "sess_page"
    for i in range(10):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message=f"msg{i}", assistant_response=f"ans{i}",
        ))

    page1 = await session_svc.get_timeline(user_id, session_id, page=1, page_size=4)
    page2 = await session_svc.get_timeline(user_id, session_id, page=2, page_size=4)
    page3 = await session_svc.get_timeline(user_id, session_id, page=3, page_size=4)

    assert page1.total == 10
    assert len(page1.entries) == 4
    assert page1.has_next is True
    assert len(page2.entries) == 4
    assert page2.has_next is True
    assert len(page3.entries) == 2
    assert page3.has_next is False

    ids_p1 = {e.turn_index for e in page1.entries}
    ids_p2 = {e.turn_index for e in page2.entries}
    assert ids_p1.isdisjoint(ids_p2)


@pytest.mark.asyncio
async def test_timeline_time_travel_at(session_svc):
    from datetime import datetime, timezone
    user_id, session_id = "u3", "sess_tt"
    base_ts = 1_700_100_000.0

    for i in range(6):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message=f"turn{i}", assistant_response=f"resp{i}",
            timestamp=base_ts + i * 3600,
        ))

    at_iso = datetime.fromtimestamp(base_ts + 2 * 3600, tz=timezone.utc).isoformat()
    resp = await session_svc.get_timeline(user_id, session_id, page=1, page_size=20, at=at_iso)

    assert resp.total == 3
    for entry in resp.entries:
        entry_ts = datetime.fromisoformat(entry.timestamp).timestamp()
        assert entry_ts <= base_ts + 2 * 3600


@pytest.mark.asyncio
async def test_timeline_empty_session(session_svc):
    resp = await session_svc.get_timeline("nobody", "ghost", page=1, page_size=20)
    assert resp.total == 0
    assert resp.entries == []
    assert resp.has_next is False


@pytest.mark.asyncio
async def test_timeline_entry_has_required_fields(session_svc):
    user_id, session_id = "u4", "sess_fields"
    await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
        user_message="what is AI?",
        assistant_response="AI is artificial intelligence.",
        tool_calls=[{"name": "search", "args": {}}],
    ))

    resp = await session_svc.get_timeline(user_id, session_id, page=1, page_size=10)
    e = resp.entries[0]
    assert e.user_message == "what is AI?"
    assert e.assistant_response == "AI is artificial intelligence."
    assert isinstance(e.media_refs, list)
    assert isinstance(e.tool_calls, list)
    assert "T" in e.timestamp


# ── User-wide timeline (across sessions) ────────────────────────────────────

@pytest.mark.asyncio
async def test_user_timeline_interleaves_sessions_chronologically(session_svc):
    user_id = "u_multi"
    base_ts = 1_700_000_000.0
    # Interleaved timestamps across two sessions — session B's turn lands
    # between session A's two turns.
    await session_svc.append_turn(user_id, "sess_a", TurnAppendRequest(
        user_message="A0", assistant_response="a0", timestamp=base_ts,
    ))
    await session_svc.append_turn(user_id, "sess_b", TurnAppendRequest(
        user_message="B0", assistant_response="b0", timestamp=base_ts + 1,
    ))
    await session_svc.append_turn(user_id, "sess_a", TurnAppendRequest(
        user_message="A1", assistant_response="a1", timestamp=base_ts + 2,
    ))

    resp = await session_svc.get_user_timeline(user_id, page=1, page_size=20)

    assert resp.total == 3
    assert [e.user_message for e in resp.entries] == ["A0", "B0", "A1"]
    assert [e.session_id for e in resp.entries] == ["sess_a", "sess_b", "sess_a"]


@pytest.mark.asyncio
async def test_user_timeline_pagination(session_svc):
    user_id = "u_multi_page"
    for i in range(3):
        await session_svc.append_turn(user_id, f"sess_{i}", TurnAppendRequest(
            user_message=f"msg{i}", assistant_response=f"ans{i}", timestamp=1_700_000_000.0 + i,
        ))

    page1 = await session_svc.get_user_timeline(user_id, page=1, page_size=2)
    page2 = await session_svc.get_user_timeline(user_id, page=2, page_size=2)

    assert page1.total == 3
    assert len(page1.entries) == 2
    assert page1.has_next is True
    assert len(page2.entries) == 1
    assert page2.has_next is False


@pytest.mark.asyncio
async def test_user_timeline_empty_user(session_svc):
    resp = await session_svc.get_user_timeline("nobody", page=1, page_size=20)
    assert resp.total == 0
    assert resp.entries == []
    assert resp.has_next is False


@pytest.mark.asyncio
async def test_user_timeline_ignores_other_users_sessions(session_svc):
    await session_svc.append_turn("u_owner", "sess_x", TurnAppendRequest(
        user_message="mine", assistant_response="mine-reply",
    ))
    await session_svc.append_turn("u_other", "sess_y", TurnAppendRequest(
        user_message="not mine", assistant_response="not-mine-reply",
    ))

    resp = await session_svc.get_user_timeline("u_owner", page=1, page_size=20)
    assert resp.total == 1
    assert resp.entries[0].user_message == "mine"


# ── Stats unit tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_empty_user(mgmt_svc):
    stats = await mgmt_svc.get_stats("nobody")
    assert stats.session_count == 0
    assert stats.total_turns == 0
    assert stats.summary_frame_count == 0
    assert stats.total_media_items == 0
    assert stats.long_term_frame_count == 0
    assert stats.storage_bytes_used == 0


@pytest.mark.asyncio
async def test_stats_session_and_turn_counts(tmp_path, redis):
    from pathlib import Path
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    for s in range(3):
        for t in range(4):
            await svc.append_turn("u_stats", f"sess_{s}", TurnAppendRequest(
                user_message=f"q{t}", assistant_response=f"a{t}",
            ))

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    stats = await mgmt.get_stats("u_stats")

    assert stats.session_count == 3
    assert stats.total_turns == 12
    assert stats.summary_frame_count == 0
    assert stats.storage_bytes_used > 0


@pytest.mark.asyncio
async def test_stats_excludes_summary_and_boundary_frames_from_total_turns(tmp_path, redis):
    """A compacted session's summary/compact_boundary bookkeeping frames must not
    inflate total_turns — they're counted separately via summary_frame_count."""
    from pathlib import Path
    from core.config import settings as cfg
    from services.compact_memory import CompactMemoryService

    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    for t in range(5):
        await svc.append_turn("u_compact_stats", "sess_compact", TurnAppendRequest(
            user_message="x" * 800, assistant_response="y" * 800,
        ))

    compact = CompactMemoryService(storage_root=str(tmp_path))
    original_max_tail = cfg.compact_max_tail_tokens
    original_keep_turns = cfg.compact_keep_turns
    cfg.compact_max_tail_tokens = 200  # force turns_to_summarize to be non-empty
    # This session only has 5 turns — below compact_keep_turns' default (10),
    # which would otherwise force all 5 into the tail via the mandatory-floor
    # guarantee and leave nothing to summarize. Not what this test exercises.
    cfg.compact_keep_turns = 1
    try:
        with patch(
            "services.compact_memory._llm_summarize",
            new_callable=AsyncMock,
            return_value="# Current State\nStats test summary.",
        ):
            result = await compact.compact_session("u_compact_stats", "sess_compact", context_window=4096)
    finally:
        cfg.compact_max_tail_tokens = original_max_tail
        cfg.compact_keep_turns = original_keep_turns
    assert result.compacted is True

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    stats = await mgmt.get_stats("u_compact_stats")

    assert stats.summary_frame_count == 1
    assert stats.total_turns == 5  # original turns only — summary/boundary excluded


@pytest.mark.asyncio
async def test_stats_longterm_frame_count(tmp_path):
    from pathlib import Path
    lt_svc = LongTermMemoryService(storage_root=str(tmp_path))
    for i in range(5):
        await lt_svc.add_fact("u_lt_stats", FactAddRequest(text=f"fact {i}", frame_type="fact"))

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    stats = await mgmt.get_stats("u_lt_stats")

    assert stats.long_term_frame_count == 5


# ── Export unit tests ─────────────────────────────────────────────────────────

async def _collect_export(mgmt_svc, user_id: str) -> dict:
    chunks = []
    async for chunk in mgmt_svc.export_generator(user_id):
        chunks.append(chunk)
    return json.loads(b"".join(chunks))


@pytest.mark.asyncio
async def test_export_is_valid_json(tmp_path, redis):
    from pathlib import Path
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await svc.append_turn("u_exp", "sess_e1", TurnAppendRequest(
        user_message="hello", assistant_response="world",
    ))

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    doc = await _collect_export(mgmt, "u_exp")

    assert doc["export_schema_version"] == "1.0"
    assert doc["user_id"] == "u_exp"
    assert "exported_at" in doc
    assert isinstance(doc["sessions"], list)
    assert isinstance(doc["longterm"], list)
    assert isinstance(doc["multimedia"], list)


@pytest.mark.asyncio
async def test_export_contains_session_turns(tmp_path, redis):
    from pathlib import Path
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await svc.append_turn("u_exp2", "sess_x", TurnAppendRequest(
        user_message="export me", assistant_response="exported",
    ))

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    doc = await _collect_export(mgmt, "u_exp2")

    assert len(doc["sessions"]) == 1
    assert doc["sessions"][0]["session_id"] == "sess_x"
    assert len(doc["sessions"][0]["turns"]) == 1


@pytest.mark.asyncio
async def test_export_contains_longterm_facts(tmp_path):
    from pathlib import Path
    lt_svc = LongTermMemoryService(storage_root=str(tmp_path))
    await lt_svc.add_fact("u_exp3", FactAddRequest(text="user likes dark mode", frame_type="preference"))

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    doc = await _collect_export(mgmt, "u_exp3")

    assert len(doc["longterm"]) == 1
    assert "dark mode" in doc["longterm"][0].get("text", "")


@pytest.mark.asyncio
async def test_export_empty_user_is_valid(mgmt_svc):
    doc = await _collect_export(mgmt_svc, "nobody")
    assert doc["export_schema_version"] == "1.0"
    assert doc["sessions"] == []
    assert doc["longterm"] == []
    assert doc["multimedia"] == []


# ── Delete unit tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_all_removes_files_and_redis_keys(tmp_path, redis):
    from pathlib import Path
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await svc.append_turn("u_del", "sess_d", TurnAppendRequest(
        user_message="bye", assistant_response="gone",
    ))

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))

    result = await mgmt.delete_all("u_del", redis)
    assert result is True
    assert not (Path(str(tmp_path)) / "user_u_del").exists()
    assert await svc.list_sessions("u_del") == []


@pytest.mark.asyncio
async def test_delete_all_removes_module15_chat_cache(tmp_path, redis):
    """DELETE /memory must also clear Module 15's short-term chat_memory
    cache — it lives on the same shared Redis cluster and would otherwise
    survive a GDPR erasure until MEMORY_TTL (default 1h) expires."""
    from pathlib import Path
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await svc.append_turn("u_del2", "sess_d2", TurnAppendRequest(
        user_message="bye", assistant_response="gone",
    ))

    await redis.set("chat_memory:u_del2", "[]")
    await redis.set("chat_memory:u_del2:sess_d2", "[]")
    await redis.set("chat_memory:someone_else", "[]")  # different user — must survive

    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    await mgmt.delete_all("u_del2", redis)

    assert await redis.exists("chat_memory:u_del2") == 0
    assert await redis.exists("chat_memory:u_del2:sess_d2") == 0
    assert await redis.exists("chat_memory:someone_else") == 1


@pytest.mark.asyncio
async def test_delete_all_nonexistent_user(tmp_path, redis):
    from pathlib import Path
    mgmt = ManagementService.__new__(ManagementService)
    mgmt._root = Path(str(tmp_path))
    assert await mgmt.delete_all("no_such_user", redis) is False


# ── API integration tests ─────────────────────────────────────────────────────

@pytest.fixture
def api_client(tmp_path):
    from main import app
    from core.dependencies import get_current_user, get_redis
    from routers.sessions import get_session_service
    import core.config

    fake_redis = FakeAsyncRedis(decode_responses=True)
    original_root = core.config.settings.storage_root
    core.config.settings.storage_root = str(tmp_path)

    # Snapshot so teardown restores rather than wipes — app.dependency_overrides
    # is a module-level singleton shared with test_api.py, whose overrides are
    # set once at import time. Clearing unconditionally here strands test_api.py
    # with no override (→ real get_current_user → 401) if this fixture's tests
    # run afterward in the same pytest session.
    original_overrides = app.dependency_overrides.copy()

    app.dependency_overrides[get_redis] = lambda: fake_redis
    app.dependency_overrides[get_session_service] = lambda: SessionMemoryService(
        redis=fake_redis, storage_root=str(tmp_path)
    )
    app.dependency_overrides[get_current_user] = lambda: "api_user"

    client = TestClient(app)
    yield client, tmp_path, fake_redis

    core.config.settings.storage_root = original_root
    app.dependency_overrides.clear()
    app.dependency_overrides.update(original_overrides)
    asyncio.get_event_loop().run_until_complete(fake_redis.aclose())


def test_api_timeline_200(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/tl_sess/turns", json={
        "user_message": "hello", "assistant_response": "hi",
    })
    r = client.get("/memory/sessions/tl_sess/timeline")
    assert r.status_code == 200
    body = r.json()
    for field in ("entries", "total", "page", "page_size", "has_next"):
        assert field in body


def test_api_timeline_pagination(api_client):
    client, *_ = api_client
    for i in range(7):
        client.post("/memory/sessions/pg_sess/turns", json={
            "user_message": f"q{i}", "assistant_response": f"a{i}",
        })
    r = client.get("/memory/sessions/pg_sess/timeline?page=1&page_size=3")
    body = r.json()
    assert body["total"] == 7
    assert len(body["entries"]) == 3
    assert body["has_next"] is True


def test_api_timeline_time_travel(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/tt_sess/turns", json={
        "user_message": "recent turn", "assistant_response": "yes",
    })
    r = client.get("/memory/sessions/tt_sess/timeline?at=2000-01-01T00:00:00Z")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_api_timeline_entry_shape(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/shape_sess/turns", json={
        "user_message": "shape check", "assistant_response": "ok",
        "tool_calls": [{"name": "lookup"}],
    })
    r = client.get("/memory/sessions/shape_sess/timeline")
    e = r.json()["entries"][0]
    for field in ("turn_index", "timestamp", "user_message", "assistant_response", "media_refs", "tool_calls"):
        assert field in e


def test_api_stats_200(api_client):
    client, *_ = api_client
    r = client.get("/memory/stats")
    assert r.status_code == 200
    body = r.json()
    for field in (
        "session_count", "total_turns", "summary_frame_count",
        "total_media_items", "long_term_frame_count", "storage_bytes_used",
    ):
        assert field in body


def test_api_stats_reflects_written_data(api_client):
    client, *_ = api_client
    for i in range(2):
        client.post(f"/memory/sessions/st_sess_{i}/turns", json={
            "user_message": f"q{i}", "assistant_response": f"a{i}",
        })
    body = client.get("/memory/stats").json()
    assert body["session_count"] >= 2
    assert body["total_turns"] >= 2


def test_api_export_valid_json_and_schema_version(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/exp_sess/turns", json={
        "user_message": "export test", "assistant_response": "ok",
    })
    r = client.get("/memory/export")
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    doc = r.json()
    assert doc["export_schema_version"] == "1.0"
    assert "sessions" in doc and "longterm" in doc and "multimedia" in doc and "exported_at" in doc


def test_api_delete_all_204(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/del_sess/turns", json={
        "user_message": "bye", "assistant_response": "gone",
    })
    assert client.delete("/memory").status_code == 204


def test_api_stats_zero_after_delete(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/pre_del/turns", json={"user_message": "x", "assistant_response": "y"})
    client.delete("/memory")
    body = client.get("/memory/stats").json()
    assert body["session_count"] == 0
    assert body["total_turns"] == 0
