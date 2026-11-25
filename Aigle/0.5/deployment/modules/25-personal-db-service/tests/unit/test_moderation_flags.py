"""CM-2: Module 10's image moderation verdict -> the image's Source.moderation_flags.

Pure unit test (no ArcadeDB): the Kafka consumer's summary path hands the
summary entry's moderation_flags to index_source_summary(), which writes it in
the Source upsert. tests/integration/test_moderation_flags.py checks the same
on a real ArcadeDB.

Run from the 25-personal-db-service/ directory:
    python -m pytest tests/unit/test_moderation_flags.py -v
"""
from __future__ import annotations

import pytest
from app.models.index import SourceSummaryIndexRequest
from app.services import indexer, kafka_consumer

FLAGS = {"verdict": "unsafe", "model": "shieldgemma-2", "policies": {"sexual": 0.97},
         "latency_ms": 812.4, "checked_at": "2026-11-03T04:05:06.123456+00:00"}


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
def summaries(monkeypatch):
    """index_source_summary() requests the consumer makes; nothing else runs."""
    captured = []

    async def index_source_summary(_client, _branch, req):
        captured.append(req)
        return "#1:0"

    async def nothing(*_a, **_kw):
        return None

    async def claim(*_a, **_kw):
        return True

    async def embed(texts):
        return [[0.0] * kafka_consumer.settings.vector_dim for _ in texts]

    monkeypatch.setattr(kafka_consumer, "index_source_summary", index_source_summary)
    monkeypatch.setattr(kafka_consumer, "index_chunk", nothing)
    monkeypatch.setattr(kafka_consumer, "_ensure_db", nothing)
    monkeypatch.setattr(kafka_consumer, "claim_event", claim)
    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    return captured


def _image_message(summary_extra=None):
    base = {"type": "images", "filename": "harbour.jpg", "version_id": "p1", "asset_path": "image/jpg/harbour",
            "branch_id": "u1"}
    return {"event_id": "e-cm2", "source_module": "10-image", "payload": {
        "branch_id": "u1", "correlation_id": "c", "parameters": {"version_id": "p1", "asset_path": "image/jpg/harbour",
                                                                  "chunks": [
            {"id": "s1", "payload": {**base, "embedding_type": "summary", "summary": "Two people at a harbour.",
                                     **(summary_extra or {})}},
            {"id": "t1", "payload": {**base, "embedding_type": "text", "text": ""}}]}}}


async def _consume(message):
    await kafka_consumer._handle_message(None, FakeRedis(), message)


async def test_the_summary_entry_verdict_reaches_index_source_summary(summaries):
    await _consume(_image_message({"moderation_flags": FLAGS}))

    [req] = summaries
    assert req.moderation_flags == FLAGS


@pytest.mark.parametrize("extra", [None, {"moderation_flags": None}, {"moderation_flags": "unsafe"}])
async def test_no_verdict_or_a_malformed_one_writes_nothing(summaries, extra):
    await _consume(_image_message(extra))

    [req] = summaries
    assert req.moderation_flags is None


class FakeArcade:
    def __init__(self):
        self.commands = []

    async def command(self, db, sql, params=None):
        self.commands.append((sql, params))
        return [{"@rid": "#1:0"}]


@pytest.fixture
def arcade(monkeypatch):
    async def ready(_client, _branch):
        return "db"

    async def nothing(*_a, **_kw):
        return None

    monkeypatch.setattr(indexer, "_ensure_ready", ready)
    monkeypatch.setattr(indexer, "_link_waiting_face_traces", nothing)
    return FakeArcade()


def _summary(**extra):
    return SourceSummaryIndexRequest(version_id="p1", summary="Two people at a harbour.",
                                     embedding=[0.0] * kafka_consumer.settings.vector_dim, media_type="images", **extra)


async def test_index_source_summary_writes_the_flags(arcade):
    await indexer.index_source_summary(arcade, "u1", _summary(moderation_flags=FLAGS))

    [(sql, params)] = arcade.commands
    assert "moderation_flags = :moderation_flags" in sql
    assert params["moderation_flags"] == FLAGS


async def test_index_source_summary_without_flags_leaves_them_alone(arcade):
    await indexer.index_source_summary(arcade, "u1", _summary())

    [(sql, params)] = arcade.commands
    assert "moderation_flags" not in sql and "moderation_flags" not in params
