"""The consumer's per-chunk Redis dedup against a real ArcadeDB (skips if unreachable).

Found with the P12 demo: the key was `personal:indexed:{chunk_id}` for every tenant, so a second user's
upload whose chunk ids matched the first's (content-derived ids) was skipped as "already indexed" -- nothing
written to their database, and their temporal job cancelled for want of a source. The same happened to a
re-upload after delete-by-version, because the key outlives the rows. Redis and the embedder are stubs here;
the event claim (PostgreSQL) is bypassed.
"""
from __future__ import annotations

import pytest
from app.services import indexer, kafka_consumer, temporal_jobs
from app.services.arcadedb_client import db_name_for

from tests.integration.conftest import fake_vector

pytestmark = pytest.mark.asyncio


class FakeRedis:
    def __init__(self):
        self.keys: dict[str, str] = {}

    async def exists(self, key):
        return int(key in self.keys)

    async def set(self, key, value, ex=None):
        self.keys[key] = value

    async def sismember(self, key, member):
        return False


@pytest.fixture
def consumer(monkeypatch):
    async def claim(*_a, **_kw):
        return True

    async def embed(texts):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(kafka_consumer, "claim_event", claim)
    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    return FakeRedis()


def _message(branch: str, version: str = "v1") -> dict:
    texts = ["Android Inc. was founded in Palo Alto in October 2003.", "Google acquired the company in July 2005."]
    return {"event_id": f"e-{branch}-{version}", "source_module": "12-document", "payload": {
        "branch_id": branch, "correlation_id": "c", "parameters": {
            "version_id": version, "asset_path": f"{branch}/a.pdf",
            "chunks": [{"id": f"doc-same-c{i:03d}", "payload": {"text": t, "type": "documents", "chunk_index": i,
                                                               "version_id": version, "branch_id": branch}}
                       for i, t in enumerate(texts)]}}}


async def _chunks(client, branch: str) -> int:
    rows = await client.query(db_name_for(branch), "SELECT count(*) AS c FROM Chunk")
    return int(rows[0]["c"])


async def test_same_chunk_ids_in_two_tenants_are_both_indexed(client, make_db, consumer):
    a, b = await make_db("ittest_dedup_a"), await make_db("ittest_dedup_b")
    await kafka_consumer._handle_message(client, consumer, _message(a))
    await kafka_consumer._handle_message(client, consumer, _message(b))
    assert (await _chunks(client, a), await _chunks(client, b)) == (2, 2)
    assert (await temporal_jobs.status(client, b, "v1"))["segments_total"] == 2


async def test_redelivery_is_still_deduplicated(client, make_db, consumer, monkeypatch):
    a = await make_db("ittest_dedup_redeliver")
    await kafka_consumer._handle_message(client, consumer, _message(a))
    calls = []
    real = kafka_consumer.index_chunk

    async def counting(*args, **kw):
        calls.append(args[2].chunk_id)
        return await real(*args, **kw)

    monkeypatch.setattr(kafka_consumer, "index_chunk", counting)
    await kafka_consumer._handle_message(client, consumer, _message(a))
    assert calls == []                                    # rows exist: nothing embedded or written again
    assert await _chunks(client, a) == 2


async def test_reupload_after_delete_is_indexed_again(client, make_db, consumer):
    a = await make_db("ittest_dedup_reupload")
    await kafka_consumer._handle_message(client, consumer, _message(a))
    await indexer.delete_by_version(client, a, "v1")
    assert await _chunks(client, a) == 0
    await kafka_consumer._handle_message(client, consumer, _message(a))
    assert await _chunks(client, a) == 2
    assert (await temporal_jobs.status(client, a, "v1"))["status"] == "pending"
