"""
Unit tests for LongTermMemoryService (MV-3).
Run from the 26-memory-service/ directory:

"""
import os
import sys
import time

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio

from services.long_term_memory import (
    FactAddRequest,
    LongTermMemoryService,
    SearchRequest,
)
from services.memvid_store import sync_append_batch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def svc(tmp_path):
    yield LongTermMemoryService(storage_root=str(tmp_path))


def _lt_path(svc: LongTermMemoryService, user_id: str) -> str:
    return svc._lt_path(user_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _seed_100_frames(svc: LongTermMemoryService, user_id: str, base_ts: float) -> None:
    """Write 100 frames: 34 conversation, 33 preference, 33 fact."""
    path = _lt_path(svc, user_id)
    frames = []
    for i in range(34):
        frames.append({
            "title": f"conversation {i}",
            "label": "conversation",
            "text": f"We discussed topic alpha number {i} in detail.",
            "timestamp": base_ts + i,
            "session_id": f"sess_{i}",
            "frame_type": "conversation",
        })
    for i in range(33):
        frames.append({
            "title": f"preference {i}",
            "label": "preference",
            "text": f"User prefers beta option {i} over the default.",
            "timestamp": base_ts + 34 + i,
            "session_id": "",
            "frame_type": "preference",
        })
    for i in range(33):
        frames.append({
            "title": f"fact {i}",
            "label": "fact",
            "text": f"Established fact gamma {i}: this is a known truth.",
            "timestamp": base_ts + 67 + i,
            "session_id": "",
            "frame_type": "fact",
        })
    import asyncio
    await asyncio.to_thread(sync_append_batch, path, frames)


# ── add_fact ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_fact_returns_response(svc):
    resp = await svc.add_fact("u1", FactAddRequest(text="User speaks Mandarin.", frame_type="preference"))
    assert resp.frame_id is not None
    assert resp.frame_type == "preference"
    assert resp.timestamp > 0


@pytest.mark.asyncio
async def test_add_fact_all_types(svc):
    for ft in ("conversation", "preference", "entity", "fact"):
        resp = await svc.add_fact("u2", FactAddRequest(text=f"Sample {ft}.", frame_type=ft))
        assert resp.frame_type == ft


# ── search ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_empty_store_returns_empty(svc):
    result = await svc.search("u_nobody", SearchRequest(query="anything"))
    assert result == []


@pytest.mark.asyncio
async def test_search_finds_relevant_frames(svc, tmp_path):
    user_id = "u_search"
    base_ts = time.time()
    await _seed_100_frames(svc, user_id, base_ts)

    hits = await svc.search(user_id, SearchRequest(query="alpha topic", top_k=5))
    assert len(hits) >= 1
    # At least one hit should be from the "conversation" frames that mention "alpha"
    texts = " ".join(h.text for h in hits)
    assert "alpha" in texts.lower()


@pytest.mark.asyncio
async def test_search_respects_top_k(svc, tmp_path):
    user_id = "u_topk"
    base_ts = time.time()
    await _seed_100_frames(svc, user_id, base_ts)

    hits = await svc.search(user_id, SearchRequest(query="topic", top_k=3))
    assert len(hits) <= 3


@pytest.mark.asyncio
async def test_search_date_filter_from_date(svc, tmp_path):
    user_id = "u_date_from"
    base_ts = 1_000_000.0

    await _seed_100_frames(svc, user_id, base_ts)

    # Frames 0–33 have timestamps base_ts to base_ts+33
    # Frames 34–66 have timestamps base_ts+34 to base_ts+66
    # Frames 67–99 have timestamps base_ts+67 to base_ts+99
    cutoff = base_ts + 67  # only "fact" frames should pass

    hits = await svc.search(
        user_id,
        SearchRequest(query="gamma established fact", top_k=10, from_date=cutoff),
    )
    assert len(hits) >= 1
    for h in hits:
        assert h.timestamp >= cutoff


@pytest.mark.asyncio
async def test_search_date_filter_to_date(svc, tmp_path):
    user_id = "u_date_to"
    base_ts = 2_000_000.0
    await _seed_100_frames(svc, user_id, base_ts)

    cutoff = base_ts + 33  # only "conversation" frames (indices 0–33)

    hits = await svc.search(
        user_id,
        SearchRequest(query="alpha topic detail", top_k=10, to_date=cutoff),
    )
    assert len(hits) >= 1
    for h in hits:
        assert h.timestamp <= cutoff


@pytest.mark.asyncio
async def test_search_date_filter_range(svc, tmp_path):
    user_id = "u_date_range"
    base_ts = 3_000_000.0
    await _seed_100_frames(svc, user_id, base_ts)

    # "preference" frames span base_ts+34 to base_ts+66
    from_ts = base_ts + 34
    to_ts = base_ts + 66

    hits = await svc.search(
        user_id,
        SearchRequest(query="beta preference option", top_k=10, from_date=from_ts, to_date=to_ts),
    )
    assert len(hits) >= 1
    for h in hits:
        assert from_ts <= h.timestamp <= to_ts


# ── get_facts ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_facts_empty_store(svc):
    result = await svc.get_facts("u_empty")
    assert result == []


@pytest.mark.asyncio
async def test_get_facts_only_preference_and_fact(svc, tmp_path):
    user_id = "u_getfacts"
    base_ts = time.time()
    await _seed_100_frames(svc, user_id, base_ts)

    facts = await svc.get_facts(user_id)
    assert len(facts) == 66  # 33 preference + 33 fact
    for f in facts:
        ft = f.get("frame_type") or f.get("label")
        assert ft in ("preference", "fact")


@pytest.mark.asyncio
async def test_get_facts_sorted_by_recency(svc, tmp_path):
    user_id = "u_sorted"
    base_ts = time.time()
    await _seed_100_frames(svc, user_id, base_ts)

    facts = await svc.get_facts(user_id)
    for i in range(len(facts) - 1):
        assert facts[i].get("timestamp", 0) >= facts[i + 1].get("timestamp", 0)


# ── delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_removes_file(svc):
    user_id = "u_del"
    await svc.add_fact(user_id, FactAddRequest(text="To be deleted.", frame_type="fact"))
    assert await svc.delete(user_id) is True
    # File gone
    from pathlib import Path
    assert not Path(svc._lt_path(user_id)).exists()


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(svc):
    assert await svc.delete("u_ghost") is False


@pytest.mark.asyncio
async def test_delete_idempotent(svc):
    user_id = "u_idem"
    await svc.add_fact(user_id, FactAddRequest(text="Fact.", frame_type="fact"))
    assert await svc.delete(user_id) is True
    assert await svc.delete(user_id) is False  # second call returns False


# ── long_term.mv2 created automatically ──────────────────────────────────────

@pytest.mark.asyncio
async def test_file_created_on_first_write(svc, tmp_path):
    user_id = "u_auto"
    from pathlib import Path
    path = Path(svc._lt_path(user_id))
    assert not path.exists()
    await svc.add_fact(user_id, FactAddRequest(text="Hello.", frame_type="fact"))
    assert path.exists()
