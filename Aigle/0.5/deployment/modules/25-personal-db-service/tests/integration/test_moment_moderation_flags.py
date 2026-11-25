"""CM-3 (VIE01-406): a video moment's keyframe moderation verdict on its Chunk, on a
real ArcadeDB -- the MAP property the schema declares, the Kafka consumer writing
each moment's moderation_flags (and the video's rollup on the Source), filtering
moments on it with their moment index and times, and a dedup clone keeping it.

Needs a reachable ArcadeDB (see conftest.py):
    PD_ARCADEDB_URL=http://localhost:2480 PD_ARCADEDB_PASSWORD=... \
        python -m pytest tests/integration/test_moment_moderation_flags.py -v
"""
from __future__ import annotations

from app.models.index import CloneVersionRequest
from app.services import indexer, kafka_consumer
from app.services.arcadedb_client import db_name_for

from tests.integration.conftest import fake_vector

UNSAFE = {"verdict": "unsafe", "model": "shieldgemma-2", "decision": "block", "flagged": ["nsfw"],
          "scores": {"nsfw": 0.95, "violence": 0.02}, "frames_checked": 2, "frames_failed": 0}
SAFE = {"verdict": "safe", "model": "shieldgemma-2", "decision": "allow", "flagged": [],
        "scores": {"nsfw": 0.01}, "frames_checked": 2, "frames_failed": 0}
ROLLUP = {"verdict": "unsafe", "model": "shieldgemma-2", "modality": "video_frame", "flagged": ["nsfw"],
          "flagged_moments": [{"moment_index": 1, "start_time": 10.0, "end_time": 20.0,
                               "decision": "block", "flagged": ["nsfw"]}],
          "moments_checked": 3, "moments_failed": 0, "latency_ms": 912.0,
          "checked_at": "2026-11-05T04:05:06+00:00"}


class FakeRedis:
    def __init__(self):
        self.keys = {}

    async def exists(self, key):
        return int(key in self.keys)

    async def set(self, key, value, ex=None):
        self.keys[key] = value

    async def sismember(self, key, member):
        return False


def _video_message(branch, version_id, moment_flags, rollup=None):
    asset_path = f"video/mp4/{version_id}"
    summary = {"type": "videos", "filename": f"{version_id}.mp4", "version_id": version_id,
               "asset_path": asset_path, "branch_id": branch, "embedding_type": "summary",
               "summary": "A short clip at a harbour."}
    if rollup is not None:
        summary["moderation_flags"] = rollup
    moments = []
    for i, flags in enumerate(moment_flags):
        m = {"moment_index": i, "start_time": str(i * 10.0), "end_time": str(i * 10.0 + 10), "type": "videos",
             "filename": f"{version_id}.mp4", "text": f"moment {i} at the harbour", "branch_id": branch}
        if flags is not None:
            m["moderation_flags"] = flags
        moments.append(m)
    return {"event_id": f"e-{version_id}", "source_module": "11-video", "payload": {
        "branch_id": branch, "correlation_id": "c", "parameters": {
            "version_id": version_id, "asset_path": asset_path,
            "chunks": [{"id": f"s-{version_id}", "payload": summary}], "moments": moments}}}


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


async def test_schema_declares_the_chunk_property_as_a_map(client, make_db):
    db = db_name_for(await make_db("ittest_cm3_schema"))

    rows = await client.query(db, "SELECT properties FROM schema:types WHERE name = 'Chunk'")
    [prop] = [p for p in rows[0]["properties"] if p["name"] == "moderation_flags"]
    assert prop["type"] == "MAP"


async def test_moment_verdicts_land_on_their_chunks_and_can_be_filtered(client, make_db, monkeypatch):
    branch = await make_db("ittest_cm3_consume")
    db = db_name_for(branch)

    await _consume(monkeypatch, client, _video_message(branch, "v1", [SAFE, UNSAFE, SAFE], rollup=ROLLUP))
    await _consume(monkeypatch, client, _video_message(branch, "v2", [None, None]))     # moderation off

    rows = await client.query(db, "SELECT chunk_index, moderation_flags FROM Chunk WHERE version_id = 'v1' "
                                  "ORDER BY chunk_index")
    assert [r["moderation_flags"]["verdict"] for r in rows] == ["safe", "unsafe", "safe"]
    assert rows[1]["moderation_flags"] == UNSAFE
    off = await client.query(db, "SELECT moderation_flags FROM Chunk WHERE version_id = 'v2'")
    assert [r.get("moderation_flags") for r in off] == [None, None]

    # Search can filter moments on it: the flagged moment, with its id and times.
    flagged = await client.query(db, "SELECT version_id, chunk_index, start_sec, end_sec FROM Chunk "
                                     "WHERE moderation_flags.verdict = 'unsafe'")
    assert [(r["version_id"], r["chunk_index"], r["start_sec"], r["end_sec"]) for r in flagged] == [("v1", 1, 10.0, 20.0)]

    [source] = await client.query(db, "SELECT moderation_flags FROM Source WHERE version_id = 'v1'")
    assert source["moderation_flags"] == ROLLUP


async def test_a_dedup_clone_keeps_the_moment_verdicts(client, make_db, monkeypatch):
    branch = await make_db("ittest_cm3_clone")
    db = db_name_for(branch)
    await _consume(monkeypatch, client, _video_message(branch, "v1", [SAFE, UNSAFE], rollup=ROLLUP))

    await indexer.clone_version(client, branch, CloneVersionRequest(
        source_version_id="v1", source_asset_path="video/mp4/v1",
        target_version_id="v9", target_asset_path="video/mp4/copy"))

    rows = await client.query(db, "SELECT chunk_index, moderation_flags FROM Chunk WHERE version_id = 'v9' "
                                  "ORDER BY chunk_index")
    assert [r["moderation_flags"] for r in rows] == [SAFE, UNSAFE]
