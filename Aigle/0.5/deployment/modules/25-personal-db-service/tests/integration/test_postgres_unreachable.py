"""PostgreSQL that cannot be reached at all, against a real ArcadeDB (skips if unreachable).

An unresolvable host or a refused connection makes asyncpg raise OSError (socket.gaierror,
ConnectionRefusedError) -- not asyncpg.PostgresError. #294 narrowed the handlers to asyncpg's own errors, so:

- the two deletes (database, GDPR) answered 500 instead of the documented 503 "retry later";
- the Kafka consumer's event claim raised, every attempt failed, and the DLQ step's own Postgres write
  raised again outside the retry loop -- stopping the consumer for good (seen in the P12 test env).

Each test points the service at a PostgreSQL that is not there; nothing is mocked on the Postgres side.
"""
from __future__ import annotations

import httpx
import pytest
from app.core.config import settings
from app.services import audit, index_events, kafka_consumer
from app.services.arcadedb_client import db_name_for
from app.services.schema_init import initialize_schema

from tests.integration.conftest import fake_vector

pytestmark = pytest.mark.asyncio

NOWHERE = [
    "postgresql://raptor:raptor@no-such-postgres.invalid:5432/personal_db",   # DNS fails: socket.gaierror
    "postgresql://raptor:raptor@127.0.0.1:1/personal_db",                     # refused: ConnectionRefusedError
]


@pytest.fixture(params=NOWHERE, ids=["unresolvable", "refused"])
def postgres_down(request, monkeypatch):
    monkeypatch.setattr(settings, "postgres_dsn", request.param)
    monkeypatch.setattr(audit, "_pool", None)
    yield
    audit._pool = None


async def _fresh_db(arcade, user_id: str) -> str:
    db = db_name_for(user_id)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    await arcade.create_database(db)
    await initialize_schema(arcade, db)
    return db


async def _delete(path: str, headers: dict) -> httpx.Response:
    from app.main import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        return await http.delete(path, headers=headers)


async def test_delete_database_answers_503_and_keeps_the_data(arcade, postgres_down, monkeypatch):
    monkeypatch.setattr(settings, "audit_required", True)
    db = await _fresh_db(arcade, "ittest-pg-down-db")
    try:
        resp = await _delete("/internal/db", {"X-User-ID": "ittest-pg-down-db"})
        assert resp.status_code == 503, resp.text
        assert await arcade.database_exists(db), "dropped without an audit row"
    finally:
        await arcade.drop_database(db)


async def test_gdpr_erasure_answers_503_and_keeps_the_data(arcade, postgres_down, monkeypatch):
    monkeypatch.setattr(settings, "audit_required", True)
    monkeypatch.setattr(settings, "gdpr_internal_token", "t")
    db = await _fresh_db(arcade, "ittest-pg-down-gdpr")
    try:
        resp = await _delete("/internal/gdpr/user/ittest-pg-down-gdpr", {"X-Internal-Token": "t"})
        assert resp.status_code == 503, resp.text
        assert await arcade.database_exists(db), "erased without an audit row"
    finally:
        await arcade.drop_database(db)


async def test_event_claim_and_dlq_mark_do_not_raise(postgres_down):
    assert await index_events.claim_event("e1", "u1", "a", "v1", "12-document") is True   # processed anyway
    await index_events.mark_event_failed("e1", "boom")                                     # logged, not raised


class FakeRedis:
    def __init__(self):
        self.keys: dict[str, str] = {}

    async def exists(self, key):
        return int(key in self.keys)

    async def set(self, key, value, ex=None):
        self.keys[key] = value

    async def sismember(self, key, member):
        return False


async def test_consumer_still_indexes_a_message(client, make_db, postgres_down, monkeypatch):
    """The message that stopped the consumer in the P12 test env: now indexed, its completion returned."""
    async def embed(texts):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    branch = await make_db("ittest_pg_down_consumer")
    message = {"event_id": "e-pg-down", "source_module": "12-document", "payload": {
        "branch_id": branch, "correlation_id": "corr", "parameters": {
            "version_id": "v1", "asset_path": "a.pdf",
            "chunks": [{"id": "c1", "payload": {"text": "Android Inc. was founded in October 2003.",
                                                "type": "documents", "version_id": "v1", "branch_id": branch}}]}}}
    completion = await kafka_consumer._handle_message(client, FakeRedis(), message)
    assert completion["correlation_id"] == "corr"
    rows = await client.query(db_name_for(branch), "SELECT count(*) AS c FROM Chunk")
    assert rows[0]["c"] == 1
