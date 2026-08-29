"""
Unit tests for SessionMemoryService.
Run from the 26-memory-service/ directory:

    pip install memvid-sdk pydantic-settings fakeredis pytest pytest-asyncio
"""
import os
import sys
import time

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from services.session_memory import SessionMemoryService, TurnAppendRequest


@pytest_asyncio.fixture
async def svc(tmp_path):
    redis = FakeAsyncRedis(decode_responses=True)
    yield SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await redis.aclose()


# ── Append & retrieve ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_append_10_retrieve_last_5(svc):
    user_id = "u1"
    session_id = "sess_abc"
    base_ts = time.time()

    for i in range(10):
        req = TurnAppendRequest(
            user_message=f"Q{i}",
            assistant_response=f"A{i}",
            timestamp=base_ts + i,
        )
        resp = await svc.append_turn(user_id, session_id, req)
        assert resp.turn_index == i + 1
        assert resp.session_id == session_id
        assert resp.frame_id is not None

    recent = await svc.get_recent(user_id, session_id, n=5)
    assert len(recent) == 5

    # Chronological order — each entry's timestamp must be <= the next
    for j in range(len(recent) - 1):
        assert recent[j].get("timestamp", 0) <= recent[j + 1].get("timestamp", 0)


@pytest.mark.asyncio
async def test_append_returns_sequential_turn_index(svc):
    user_id = "u2"
    session_id = "sess_seq"
    for i in range(3):
        resp = await svc.append_turn(
            user_id, session_id,
            TurnAppendRequest(user_message="hi", assistant_response="hello"),
        )
        assert resp.turn_index == i + 1


@pytest.mark.asyncio
async def test_get_recent_empty_session(svc):
    result = await svc.get_recent("u_nobody", "sess_ghost", n=5)
    assert result == []


@pytest.mark.asyncio
async def test_get_recent_fewer_than_n(svc):
    user_id = "u3"
    session_id = "sess_few"
    for _ in range(3):
        await svc.append_turn(
            user_id, session_id,
            TurnAppendRequest(user_message="x", assistant_response="y"),
        )
    recent = await svc.get_recent(user_id, session_id, n=10)
    assert len(recent) == 3


# ── Session listing ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_sessions(svc):
    user_id = "u4"
    for s in ["s1", "s2", "s3"]:
        await svc.append_turn(
            user_id, s,
            TurnAppendRequest(user_message="ping", assistant_response="pong"),
        )

    sessions = await svc.list_sessions(user_id)
    assert len(sessions) == 3
    ids = {sess.session_id for sess in sessions}
    assert ids == {"s1", "s2", "s3"}
    for sess in sessions:
        assert sess.turn_count == 1
        assert sess.created_at > 0
        assert sess.last_active_at > 0


@pytest.mark.asyncio
async def test_list_sessions_empty_user(svc):
    result = await svc.list_sessions("no_such_user")
    assert result == []


# ── Delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_session(svc):
    user_id = "u5"
    session_id = "sess_del"
    await svc.append_turn(
        user_id, session_id,
        TurnAppendRequest(user_message="bye", assistant_response="goodbye"),
    )

    deleted = await svc.delete_session(user_id, session_id)
    assert deleted is True

    # After deletion the session list should be empty
    sessions = await svc.list_sessions(user_id)
    assert sessions == []

    # Second delete returns False (file gone)
    deleted_again = await svc.delete_session(user_id, session_id)
    assert deleted_again is False


@pytest.mark.asyncio
async def test_delete_nonexistent_session(svc):
    result = await svc.delete_session("ghost_user", "ghost_sess")
    assert result is False
