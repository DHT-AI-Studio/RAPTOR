"""CM-2: an image's moderation verdict on its Source record, on a real ArcadeDB --
the MAP property the schema declares, the Kafka consumer writing the summary
entry's moderation_flags, reading them back (and filtering on them), and a
dedup clone carrying them to the new version.

Needs a reachable ArcadeDB (see conftest.py):
    PD_ARCADEDB_URL=http://localhost:2480 PD_ARCADEDB_PASSWORD=... \
        python -m pytest tests/integration/test_moderation_flags.py -v
"""
from __future__ import annotations

from app.models.index import CloneVersionRequest
from app.services import indexer, kafka_consumer
from app.services.arcadedb_client import db_name_for

from tests.integration.conftest import fake_vector

FLAGS = {"verdict": "unsafe", "model": "shieldgemma-2",
         "policies": {"dangerous": 0.01, "sexual": 0.97, "violence": 0.02},
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


def _image_message(branch, version_id, summary_extra):
    base = {"type": "images", "filename": f"{version_id}.jpg", "version_id": version_id,
            "asset_path": f"image/jpg/{version_id}", "branch_id": branch}
    return {"event_id": f"e-{version_id}", "source_module": "10-image", "payload": {
        "branch_id": branch, "correlation_id": "c", "parameters": {
            "version_id": version_id, "asset_path": base["asset_path"], "chunks": [
                {"id": f"s-{version_id}", "payload": {**base, "embedding_type": "summary",
                                                      "summary": "Two people at a harbour.", **summary_extra}},
                {"id": f"t-{version_id}", "payload": {**base, "embedding_type": "text", "text": "PIER 7"}}]}}}


async def _consume(monkeypatch, client, message):
    async def claim(*_a, **_kw):
        return True

    async def embed(texts):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(kafka_consumer, "claim_event", claim)
    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(indexer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    await kafka_consumer._handle_message(client, FakeRedis(), message)


async def test_schema_declares_the_property_as_a_map(client, make_db):
    db = db_name_for(await make_db("ittest_cm2_schema"))

    rows = await client.query(db, "SELECT properties FROM schema:types WHERE name = 'Source'")
    [prop] = [p for p in rows[0]["properties"] if p["name"] == "moderation_flags"]
    assert prop["type"] == "MAP"


async def test_the_consumer_writes_the_verdict_onto_the_source(client, make_db, monkeypatch):
    branch = await make_db("ittest_cm2_consume")
    db = db_name_for(branch)

    await _consume(monkeypatch, client, _image_message(branch, "p1", {"moderation_flags": FLAGS}))
    await _consume(monkeypatch, client, _image_message(branch, "p2", {}))      # moderation off

    [flagged] = await client.query(db, "SELECT moderation_flags FROM Source WHERE version_id = 'p1'")
    assert flagged["moderation_flags"] == FLAGS
    [clean] = await client.query(db, "SELECT moderation_flags FROM Source WHERE version_id = 'p2'")
    assert clean.get("moderation_flags") is None
    # A stored MAP, so it can be filtered on.
    unsafe = await client.query(db, "SELECT version_id FROM Source WHERE moderation_flags.verdict = 'unsafe'")
    assert [r["version_id"] for r in unsafe] == ["p1"]


async def test_a_dedup_clone_keeps_the_verdict(client, make_db, monkeypatch):
    branch = await make_db("ittest_cm2_clone")
    db = db_name_for(branch)
    await _consume(monkeypatch, client, _image_message(branch, "p1", {"moderation_flags": FLAGS}))

    await indexer.clone_version(client, branch, CloneVersionRequest(
        source_version_id="p1", source_asset_path="image/jpg/p1",
        target_version_id="p9", target_asset_path="image/jpg/copy"))

    [cloned] = await client.query(db, "SELECT moderation_flags FROM Source WHERE version_id = 'p9'")
    assert cloned["moderation_flags"] == FLAGS
