"""CM-3 (VIE01-406): a video moment's keyframe moderation verdict -> its Chunk's
moderation_flags; the video's rollup -> its Source (the CM-2 summary path).

Pure unit test (no ArcadeDB). tests/integration/test_moment_moderation_flags.py
checks the same on a real ArcadeDB.

Run from the 25-personal-db-service/ directory:
    python -m pytest tests/unit/test_moment_moderation_flags.py -v
"""
from __future__ import annotations

import pytest
from app.services import kafka_consumer

MOMENT = {"verdict": "unsafe", "model": "shieldgemma-2", "decision": "block", "flagged": ["nsfw"],
          "scores": {"nsfw": 0.95}, "frames_checked": 2, "frames_failed": 0}
ROLLUP = {"verdict": "unsafe", "model": "shieldgemma-2", "modality": "video_frame", "flagged": ["nsfw"],
          "flagged_moments": [{"moment_index": 1, "start_time": 10.0, "end_time": 20.0,
                               "decision": "block", "flagged": ["nsfw"]}],
          "latency_ms": 912.0, "checked_at": "2026-11-05T04:05:06+00:00"}


class FakeRedis:
    def __init__(self):
        self.keys = {}

    async def exists(self, key):
        return int(key in self.keys)

    async def set(self, key, value, ex=None):
        self.keys[key] = value

    async def sismember(self, key, member):
        return False


@pytest.fixture
def indexed(monkeypatch):
    """(chunks, summaries) the consumer hands to index_chunk / index_source_summary."""
    chunks, summaries = [], []

    async def index_chunk(_client, _branch, req):
        chunks.append(req)
        return "#1:0"

    async def index_source_summary(_client, _branch, req):
        summaries.append(req)
        return "#1:1"

    async def nothing(*_a, **_kw):
        return None

    async def claim(*_a, **_kw):
        return True

    async def embed(texts):
        return [[0.0] * kafka_consumer.settings.vector_dim for _ in texts]

    monkeypatch.setattr(kafka_consumer, "index_chunk", index_chunk)
    monkeypatch.setattr(kafka_consumer, "index_source_summary", index_source_summary)
    monkeypatch.setattr(kafka_consumer, "_ensure_db", nothing)
    monkeypatch.setattr(kafka_consumer, "claim_event", claim)
    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    return chunks, summaries


def _video_message(moment_flags: list, rollup=None):
    """A Module 11 personal-index event: a summary entry plus one moment per flag."""
    summary = {"type": "videos", "filename": "clip.mp4", "version_id": "v1", "asset_path": "video/mp4/clip",
               "branch_id": "u1", "embedding_type": "summary", "summary": "A short clip."}
    if rollup is not None:
        summary["moderation_flags"] = rollup
    moments = []
    for i, flags in enumerate(moment_flags):
        m = {"moment_index": i, "start_time": str(i * 10.0), "end_time": str(i * 10.0 + 10), "type": "videos",
             "filename": "clip.mp4", "text": f"moment {i}", "branch_id": "u1"}
        if flags is not ...:
            m["moderation_flags"] = flags
        moments.append(m)
    return {"event_id": "e-cm3", "source_module": "11-video", "payload": {
        "branch_id": "u1", "correlation_id": "c", "parameters": {
            "version_id": "v1", "asset_path": "video/mp4/clip",
            "chunks": [{"id": "s1", "payload": summary}], "moments": moments}}}


async def _consume(message):
    await kafka_consumer._handle_message(None, FakeRedis(), message)


async def test_each_moment_verdict_reaches_its_chunk(indexed):
    chunks, _ = indexed
    clean = {**MOMENT, "verdict": "safe", "flagged": [], "decision": "allow"}
    await _consume(_video_message([clean, MOMENT]))

    by_index = {c.chunk_index: c for c in chunks}
    assert by_index[0].moderation_flags == clean
    assert by_index[1].moderation_flags == MOMENT


@pytest.mark.parametrize("flags", [..., None, "unsafe", ["nsfw"]])
async def test_no_verdict_or_a_malformed_one_writes_nothing(indexed, flags):
    chunks, _ = indexed
    await _consume(_video_message([flags]))

    [chunk] = chunks
    assert chunk.moderation_flags is None


async def test_the_rollup_reaches_the_source(indexed):
    _, summaries = indexed
    await _consume(_video_message([MOMENT], rollup=ROLLUP))

    [summary] = summaries
    assert summary.moderation_flags == ROLLUP


def test_index_chunk_sends_the_flags_only_when_set():
    req = kafka_consumer._map_entry({"type": "videos", "moment_index": 3, "moderation_flags": MOMENT,
                                     "text": "x"}, "c1", [0.0] * kafka_consumer.settings.vector_dim)
    assert req.model_dump(exclude_none=True)["moderation_flags"] == MOMENT
    req = kafka_consumer._map_entry({"type": "videos", "moment_index": 3, "text": "x"}, "c2",
                                    [0.0] * kafka_consumer.settings.vector_dim)
    assert "moderation_flags" not in req.model_dump(exclude_none=True)
