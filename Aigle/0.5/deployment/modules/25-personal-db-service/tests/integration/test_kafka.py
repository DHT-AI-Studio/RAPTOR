"""PA-10 — Kafka path: a personal-index-requests message drives an index into the
user's DB and stats increment.

We test the consumer's message handler directly (no live Kafka broker): it is a
pure function `_handle_message(client, redis, raw)`. The `redis` dependency is a
tiny in-memory fake, and the embedder is mocked (so the local BGE-M3 model never
loads), so this needs neither a Kafka broker nor a Redis server — just ArcadeDB
(skips if unreachable).
"""
from __future__ import annotations

import sys
import types

import pytest

# The consumer imports `redis.asyncio` at module load. If the redis package is
# not installed, stub it so the import succeeds — the test supplies its own fake
# redis object to the handler, so no real client is ever used.
try:  # pragma: no cover
    import redis.asyncio  # noqa: F401
except ImportError:  # pragma: no cover
    _stub = types.ModuleType("redis")
    _asy = types.ModuleType("redis.asyncio")
    _asy.Redis = object
    _asy.from_url = lambda *a, **k: None
    _stub.asyncio = _asy
    sys.modules["redis"] = _stub
    sys.modules["redis.asyncio"] = _asy

from app.services import kafka_consumer
from app.services.arcadedb_client import db_name_for

from tests.integration.conftest import fake_vector

pytestmark = pytest.mark.asyncio


class FakeRedis:
    """Minimal async stand-in for what the consumer uses: the dedup keys, and the
    gdpr:erased_users set it checks before indexing anything (GD-1)."""
    def __init__(self):
        self._keys: set[str] = set()
        self._sets: dict[str, set[str]] = {}

    async def exists(self, key: str) -> int:
        return 1 if key in self._keys else 0

    async def set(self, key: str, value, ex=None) -> None:
        self._keys.add(key)

    async def sismember(self, key: str, member: str) -> bool:
        return member in self._sets.get(key, set())

    def sadd(self, key: str, member: str) -> None:   # test setup only
        self._sets.setdefault(key, set()).add(member)


def _message(branch: str) -> dict:
    # version_id at the top of parameters, where 09-12's personal_index_publisher put it
    # and where the consumer reads it for the event claim and the completion payload
    return {"payload": {"branch_id": branch, "parameters": {"version_id": "vk", "chunks": [
        {"id": "k1", "payload": {"type": "documents", "text": "kafka indexed doc",
                                 "version_id": "vk"}}]}}}


async def _chunk_count(client, db) -> int:
    rows = await client.query(db, "SELECT count(*) AS c FROM Chunk")
    return rows[0]["c"] if rows else 0


@pytest.fixture(autouse=True)
def _claims(monkeypatch):
    """In-memory stand-in for the VIE01-190 event claim (PostgreSQL
    personal_index_events), keeping this suite's "just ArcadeDB" promise. Same
    semantics as the real INSERT ... ON CONFLICT: the first claim of an event wins,
    later ones are told it was already handled."""
    claimed: set[str] = set()

    async def _claim(event_id, *args, **kwargs) -> bool:
        if event_id in claimed:
            return False
        claimed.add(event_id)
        return True

    async def _mark_failed(event_id, error) -> None:
        claimed.discard(event_id)

    monkeypatch.setattr(kafka_consumer, "claim_event", _claim)
    monkeypatch.setattr(kafka_consumer, "mark_event_failed", _mark_failed)
    return claimed


@pytest.fixture
def _embed(monkeypatch):
    async def _e(texts):
        return [fake_vector(t) for t in texts]
    monkeypatch.setattr("app.services.kafka_consumer.embed_texts", _e)
    return _e


async def test_message_indexes_chunk_and_dedupes(client, make_db, _embed):
    branch = await make_db("ittest_kafka")
    db = db_name_for(branch)

    redis = FakeRedis()
    completion = await kafka_consumer._handle_message(client, redis, _message(branch))
    assert await _chunk_count(client, db) == 1               # stats incremented
    # a real indexing pass returns a completion payload -- 09-12's orchestrators
    # publish `active` off this, not off merely enqueueing the index-request.
    assert completion == {
        "correlation_id": "",
        "payload": {"branch_id": branch, "asset_path": "", "version_id": "vk"},
    }

    # same message again → the whole-event claim (personal_index_events) sees it
    # as already processed and returns early with no completion -- the first
    # pass already published one for this event.
    completion = await kafka_consumer._handle_message(client, redis, _message(branch))
    assert await _chunk_count(client, db) == 1
    assert completion is None


async def test_auto_creates_db_for_new_user(arcade, _embed):
    """A message for a branch with no DB yet must create it (no manual /db/init)."""
    branch = "ittest_kafka_new"
    db = db_name_for(branch)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    # This test does not use make_db, so it clears its own VIE01-190 event claim —
    # otherwise the second run of the suite drops the message as a duplicate.
    from tests.integration.conftest import _clear_index_claims
    await _clear_index_claims(branch)
    try:
        assert not await arcade.database_exists(db)
        await kafka_consumer._handle_message(arcade, FakeRedis(), _message(branch))
        assert await arcade.database_exists(db)              # auto-created
        assert await _chunk_count(arcade, db) == 1
    finally:
        await arcade.drop_database(db)
        await _clear_index_claims(branch)


async def test_routes_entities_to_graph_indexer(client, make_db, _embed):
    branch = await make_db("ittest_kafka_graph")
    db = db_name_for(branch)
    msg = {"payload": {"branch_id": branch, "parameters": {
        "chunks": [{"id": "k1", "payload": {"type": "documents", "text": "Acme news"}}],
        "entities": [{"payload": {"entity_id": "acme", "name": "Acme", "type": "ORG",
                                  "source_chunk_id": "k1"}}]}}}
    await kafka_consumer._handle_message(client, FakeRedis(), msg)

    assert await _chunk_count(client, db) == 1
    ent = await client.query(db, "SELECT count(*) AS c FROM Entity")
    men = await client.query(db, "SELECT count(*) AS c FROM MENTIONS")
    assert ent[0]["c"] == 1 and men[0]["c"] == 1             # entity + Chunk->Entity edge


async def test_erased_user_messages_are_dropped(client, make_db, _embed):
    """GD-1: a user in gdpr:erased_users must not get new content indexed."""
    branch = await make_db("ittest_kafka_erased")
    db = db_name_for(branch)
    redis = FakeRedis()
    redis.sadd("gdpr:erased_users", branch)
    await kafka_consumer._handle_message(client, redis, _message(branch))
    assert await _chunk_count(client, db) == 0
