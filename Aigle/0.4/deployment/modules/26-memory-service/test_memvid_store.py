"""
Standalone test for MemvidStore wrapper.
Run from the 26-memory-service/ directory:

    pip install memvid-sdk pydantic-settings pytest
"""
import os
import sys
import time
import tempfile

# Disable Ollama embedder so test runs without a live Ollama instance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
from services.memvid_store import MemvidStore


@pytest.fixture
def store_with_tmp(tmp_path):
    """MemvidStore opened on a fresh temp .mv2 file."""
    store = MemvidStore()
    store.open(str(tmp_path / "test.mv2"))
    yield store
    store.close()


def test_open_creates_file(tmp_path):
    store = MemvidStore()
    path = tmp_path / "new.mv2"
    assert not path.exists()
    store.open(str(path))
    assert path.exists()
    store.close()


def test_open_existing_file(tmp_path):
    path = tmp_path / "existing.mv2"
    # Create first
    s1 = MemvidStore()
    s1.open(str(path))
    s1.append({"title": "seed", "label": "test", "text": "hello world"})
    s1.close()
    # Re-open
    s2 = MemvidStore()
    s2.open(str(path))
    s2.close()


def test_append_returns_frame_id(store_with_tmp):
    frame_id = store_with_tmp.append({
        "title": "Meeting note",
        "label": "note",
        "text": "Discussed Q3 roadmap with the team.",
    })
    assert frame_id is not None
    assert str(frame_id).isdigit()


def test_append_multiple_frames(store_with_tmp):
    frames = [
        {"title": "Event A", "label": "event", "text": "User logged in from Taipei."},
        {"title": "Event B", "label": "event", "text": "User uploaded a video file."},
        {"title": "Event C", "label": "event", "text": "User searched for meeting notes."},
    ]
    ids = [store_with_tmp.append(f) for f in frames]
    assert len(ids) == 3
    assert len(set(ids)) == 3  # all unique


def test_search_returns_hits(store_with_tmp):
    store_with_tmp.append({"title": "Memory 1", "label": "note", "text": "I love eating ramen in Tokyo."})
    store_with_tmp.append({"title": "Memory 2", "label": "note", "text": "The project deadline is Friday."})
    store_with_tmp.append({"title": "Memory 3", "label": "note", "text": "My favourite food is ramen."})

    hits = store_with_tmp.search("ramen", top_k=5)
    assert isinstance(hits, list)
    assert len(hits) >= 1
    # At least one hit should mention ramen
    snippets = " ".join(h.get("snippet", "") for h in hits)
    assert "ramen" in snippets.lower()


def test_search_respects_top_k(store_with_tmp):
    for i in range(10):
        store_with_tmp.append({"title": f"Doc {i}", "label": "doc", "text": f"Content about topic number {i}."})

    hits = store_with_tmp.search("topic", top_k=3)
    assert len(hits) <= 3


def test_timeline_range(store_with_tmp):
    now = int(time.time())
    store_with_tmp.append({
        "title": "Past event",
        "label": "event",
        "text": "Something happened.",
        "timestamp": now - 3600,  # 1 hour ago
    })
    store_with_tmp.append({
        "title": "Recent event",
        "label": "event",
        "text": "Something happened recently.",
        "timestamp": now,
    })

    entries = store_with_tmp.timeline(from_ts=now - 7200, to_ts=now + 60)
    assert isinstance(entries, list)


def test_close_and_reopen(tmp_path):
    path = tmp_path / "persist.mv2"
    store = MemvidStore()
    store.open(str(path))
    store.append({"title": "Persist test", "label": "test", "text": "This should survive a close."})
    store.close()

    store2 = MemvidStore()
    store2.open(str(path))
    hits = store2.search("survive", top_k=5)
    store2.close()
    assert len(hits) >= 1


def test_require_open_raises():
    store = MemvidStore()
    with pytest.raises(RuntimeError, match="open\\(\\) first"):
        store.append({"title": "x", "label": "x", "text": "x"})
