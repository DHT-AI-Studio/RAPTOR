"""
Unit tests for session .mv2 shard rollover (services/session_shards.py).

A single session's .mv2 file has no cap on how long a conversation can run,
and MV-12 compaction never shrinks it (summarized turns are archived, not
deleted) — so without rollover, a long-running session eventually hits the
memvid-sdk free-tier's 50MB-per-file capacity and every subsequent
append_turn starts failing.

sync_get_remaining_capacity is monkeypatched to report "nearly full" so
these tests force rollover deterministically on the first write, rather
than depending on the SDK's actual (non-linear) capacity accounting to
guess how many real frames it takes to get there.

Run from the 26-memory-service/ directory:
    pip install memvid-sdk pydantic-settings fakeredis pytest pytest-asyncio
    python -m pytest test_session_shards.py -v
"""
import os
import sys
import time
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from services.session_memory import SessionMemoryService, SessionSearchRequest, TurnAppendRequest


@pytest_asyncio.fixture
async def svc(tmp_path):
    redis = FakeAsyncRedis(decode_responses=True)
    yield SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await redis.aclose()


@pytest.fixture
def force_rollover_every_write():
    """Report 0 bytes remaining on every capacity check, so
    get_active_shard_path rolls over to a new shard on every append —
    deterministic and fast, unlike waiting on real SDK capacity accounting."""
    with patch("services.session_shards.sync_get_remaining_capacity", return_value=0):
        yield


async def _append_n(svc, user_id, session_id, n, size=200):
    with (
        patch.object(svc, "_trigger_extraction", new=AsyncMock(return_value=None)),
        patch.object(svc, "_trigger_compact", new=AsyncMock(return_value=None)),
    ):
        base_ts = time.time()
        for i in range(n):
            await svc.append_turn(user_id, session_id, TurnAppendRequest(
                user_message="q" * size, assistant_response="a" * size, timestamp=base_ts + i,
            ))


@pytest.mark.asyncio
async def test_rollover_creates_a_new_shard_per_write(svc, force_rollover_every_write):
    await _append_n(svc, "u1", "s1", 3)
    shards = svc._shard_paths("u1", "s1")
    assert len(shards) == 3
    assert shards[0].endswith("session_s1_0.mv2")
    assert shards[1].endswith("session_s1_1.mv2")
    assert shards[2].endswith("session_s1_2.mv2")


@pytest.mark.asyncio
async def test_recent_spans_shards_after_rollover(svc, force_rollover_every_write):
    await _append_n(svc, "u1", "s1", 5)
    recent = await svc.get_recent("u1", "s1", n=5)
    assert [e["turn_index"] for e in recent] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_timeline_spans_shards_after_rollover(svc, force_rollover_every_write):
    await _append_n(svc, "u1", "s1", 5)
    resp = await svc.get_timeline("u1", "s1", page=1, page_size=20)
    assert resp.total == 5
    assert [e.turn_index for e in resp.entries] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_timeline_pagination_still_works_across_shards(svc, force_rollover_every_write):
    await _append_n(svc, "u1", "s1", 5)
    page1 = await svc.get_timeline("u1", "s1", page=1, page_size=2)
    page2 = await svc.get_timeline("u1", "s1", page=2, page_size=2)
    assert page1.total == 5
    assert [e.turn_index for e in page1.entries] == [1, 2]
    assert page1.has_next is True
    assert [e.turn_index for e in page2.entries] == [3, 4]


@pytest.mark.asyncio
async def test_search_finds_hits_written_to_a_different_shard(svc, force_rollover_every_write):
    with (
        patch.object(svc, "_trigger_extraction", new=AsyncMock(return_value=None)),
        patch.object(svc, "_trigger_compact", new=AsyncMock(return_value=None)),
    ):
        base_ts = time.time()
        await svc.append_turn("u1", "s1", TurnAppendRequest(
            user_message="冷卻系統維修計畫", assistant_response="a" * 200, timestamp=base_ts,
        ))
        # This write lands on a different shard than the one above, thanks
        # to force_rollover_every_write.
        await svc.append_turn("u1", "s1", TurnAppendRequest(
            user_message="不相關的問題", assistant_response="a" * 200, timestamp=base_ts + 1,
        ))

    assert len(svc._shard_paths("u1", "s1")) == 2
    hits = await svc.search("u1", "s1", SessionSearchRequest(query="冷卻系統維修計畫", top_k=5))
    assert any("冷卻系統" in h.text for h in hits.hits)


@pytest.mark.asyncio
async def test_delete_session_removes_all_shards(svc, force_rollover_every_write):
    await _append_n(svc, "u1", "s1", 3)
    shards_before = svc._shard_paths("u1", "s1")
    assert len(shards_before) == 3

    deleted = await svc.delete_session("u1", "s1")
    assert deleted is True
    assert svc._shard_paths("u1", "s1") == []
    for p in shards_before:
        assert not os.path.exists(p)


@pytest.mark.asyncio
async def test_next_turn_index_resync_counts_all_shards(svc, force_rollover_every_write):
    """If Redis loses the turn_count key, resync must sum frames across every
    shard, not just shard 0."""
    await _append_n(svc, "u1", "s1", 4)
    assert len(svc._shard_paths("u1", "s1")) == 4

    await svc._redis.delete("memory:session:u1:s1:turn_count")
    resp = await svc.append_turn("u1", "s1", TurnAppendRequest(
        user_message="one more", assistant_response="ok",
    ))
    assert resp.turn_index == 5


@pytest.mark.asyncio
async def test_no_rollover_when_capacity_is_healthy(svc):
    """Sanity check: without force_rollover_every_write, ordinary small
    writes stay on a single shard."""
    await _append_n(svc, "u1", "s1", 5)
    assert len(svc._shard_paths("u1", "s1")) == 1
