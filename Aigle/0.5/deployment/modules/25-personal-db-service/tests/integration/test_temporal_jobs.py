"""P12 -- durable temporal jobs against a real ArcadeDB (skips if unreachable).

The model is replaced by a stub (no Module 07 here); everything else -- job and segment rows, the
extractor over stored replies, fact writes with their evidence, generation checks, delete -- is real.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.models.graph_index import TemporalFactIndexRequest
from app.models.index import ChunkIndexRequest
from app.services import graph_indexer, indexer, temporal_jobs
from app.services.arcadedb_client import db_name_for
from app.temporal.extract import EXTRACTOR_VERSION, Segment, extract

from tests.integration.conftest import fake_vector

pytestmark = pytest.mark.asyncio

_FIXTURE = Path(__file__).parents[1] / "unit" / "fixtures" / "android_history_segments.json"
_ALL = json.loads(_FIXTURE.read_text(encoding="utf-8"))["segments"]
_ANDROID = _ALL[:4]
_REPLIES = {"c00": {"facts": [{"entity": "Andy Rubin", "relation": "founded", "value": "Android Inc.",
                               "date_text": "October 2003", "extent": "point", "confidence": 0.9},
                              {"entity": "Rich Miner", "relation": "joined", "value": "Android Inc.", "date_text": "later",
                               "links": [{"relation": "after", "other": "Android Inc. was founded"}]}]}}
_JAPANESE = "アンドロイドは二〇〇三年にパロアルトで設立された会社であり、のちにグーグルに買収されたと言われている。"


async def _source(client, branch: str, version: str, texts: list[tuple[str, str]]) -> list[temporal_jobs.SourceSegment]:
    """Index the chunks (so the source is live) and return them as the job's segments."""
    out = []
    for i, (cid, text) in enumerate(texts):
        await indexer.index_chunk(client, branch, ChunkIndexRequest(
            chunk_id=cid, type="documents", embedding_type="text", embedding=fake_vector(text), text=text,
            version_id=version, chunk_index=i))
        out.append(temporal_jobs.SourceSegment(chunk_id=cid, text=text, seq=i))
    return out


def _android():
    return [(s["chunk_id"], s["text"]) for s in _ANDROID]


@pytest.fixture
def model(monkeypatch):
    """A stub model that records calls. It answers c00 with one fact the rules do not find (Andy Rubin as
    founder) and every other segment with none; `fail` holds chunk ids whose call fails."""
    calls: list[str] = []
    fail: set[str] = set()

    async def call(messages):
        text = messages[-1]["content"]
        cid = next((s["chunk_id"] for s in _ALL if s["text"] in text), "?") if "<<<SEGMENT>>>" in text else "?"
        calls.append(cid)
        if cid in fail:
            raise temporal_jobs.ModelError("stub: down")
        return _REPLIES.get(cid, {"facts": []})

    monkeypatch.setattr(temporal_jobs, "call_model", call)
    return calls, fail


async def _facts(client, db: str, version: str) -> list[dict]:
    """The source's extracted facts -- not the ones reasoning derived from them (P15)."""
    return await client.query(db, "SELECT FROM TemporalFact WHERE source_version_id = :v "
                                  "AND (derived IS NULL OR derived = false) ORDER BY fact_id", params={"v": version})


async def test_job_writes_the_extractors_facts_with_their_evidence(client, make_db, model):
    branch = await make_db("ittest_tjob_basic")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())

    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    assert await temporal_jobs.enqueue(client, branch, "v1", "documents", segments) == key    # idempotent
    assert await temporal_jobs.run_job(client, branch, key) == "complete"

    expected, _ = extract([Segment(c, t) for c, t in _android()], replies=_REPLIES)
    assert any(c.entity == "Andy Rubin" and c.origin == "llm" for c in expected)   # the stored reply is used
    facts = await _facts(client, db, "v1")
    assert len(facts) == len(expected) > 0
    assert {(f["date_text"], f["time_start_raw"]) for f in facts} == {(c.date_text, c.time_start) for c in expected}
    f = next(f for f in facts if f["date_text"] == "October 2003")
    offsets = json.loads(f["evidence_offsets"])
    text = dict(_android())[next(c.chunk_id for c in expected if c.date_text == "October 2003")]
    assert text[offsets["date"][0]:offsets["date"][1]] == "October 2003"
    assert (f["extractor_version"], f["status"], f["precision"], f["s_state"]) == (EXTRACTOR_VERSION, "active", "month", "known")
    assert f["fact_id"].startswith("tf-") and f["source_sentence"]

    status = await temporal_jobs.status(client, branch, "v1")
    assert (status["status"], status["segments_total"], status["segments_done"], status["facts"]) == (
        "complete", 4, 4, len(expected))
    assert status["segments_with_problems"] == []


async def test_restart_only_calls_the_model_for_unfinished_segments(client, make_db, model):
    calls, fail = model
    branch = await make_db("ittest_tjob_resume")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)

    fail.add(_ANDROID[2]["chunk_id"])
    assert await temporal_jobs.run_job(client, branch, key) == "partial"
    status = await temporal_jobs.status(client, branch, "v1")
    assert (status["segments_done"], status["segments_failed"], status["reason"]) == (3, 1, "segments_failed")
    assert status["segments_with_problems"][0]["chunk_id"] == _ANDROID[2]["chunk_id"]

    fail.clear()
    calls.clear()
    assert await temporal_jobs.enqueue(client, branch, "v1", "documents", segments) == key    # reopens it
    assert await temporal_jobs.run_job(client, branch, key) == "complete"
    assert calls == [_ANDROID[2]["chunk_id"]]


async def test_a_segment_failing_every_run_stays_failed(client, make_db, model, monkeypatch):
    _, fail = model
    monkeypatch.setattr(temporal_jobs.settings, "temporal_segment_max_attempts", 2)
    branch = await make_db("ittest_tjob_maxattempts")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    fail.add(_ANDROID[0]["chunk_id"])
    for _ in range(2):
        assert await temporal_jobs.run_job(client, branch, key) == "partial"
        await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    job = await temporal_jobs.status(client, branch, "v1")
    assert job["status"] == "partial"                          # not reopened: attempts used up


async def test_unsupported_language(client, make_db, model):
    branch = await make_db("ittest_tjob_lang")
    mixed = await _source(client, branch, "v1", [*_android()[:2], ("jp", _JAPANESE)])
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", mixed)
    assert await temporal_jobs.run_job(client, branch, key) == "partial"
    status = await temporal_jobs.status(client, branch, "v1")
    assert (status["reason"], status["segments_unsupported"]) == ("unsupported_language", 1)
    assert status["segments_with_problems"] == [
        {"chunk_id": "jp", "status": "unsupported", "reason": "unsupported_language", "attempts": 0}]

    only = await _source(client, branch, "v2", [("jp2", _JAPANESE)])
    key = await temporal_jobs.enqueue(client, branch, "v2", "documents", only)
    assert await temporal_jobs.run_job(client, branch, key) == "failed"      # never "complete with zero facts"
    assert (await temporal_jobs.status(client, branch, "v2"))["reason"] == "unsupported_language"


async def test_archived_source_cancels_the_job_and_writes_nothing(client, make_db, model):
    branch = await make_db("ittest_tjob_cancel")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await indexer.set_status_by_version(client, branch, "v1", "archived")
    assert await temporal_jobs.run_job(client, branch, key) == "cancelled"
    assert await _facts(client, db_name_for(branch), "v1") == []
    assert model[0] == []                                      # not one model call for a dead source


async def test_cancelled_job_reopens_when_its_source_is_indexed_again(client, make_db, model):
    branch = await make_db("ittest_tjob_reopen")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await indexer.set_status_by_version(client, branch, "v1", "archived")
    assert await temporal_jobs.run_job(client, branch, key) == "cancelled"
    await indexer.set_status_by_version(client, branch, "v1", "active")
    assert await temporal_jobs.enqueue(client, branch, "v1", "documents", segments) == key
    assert (await temporal_jobs.status(client, branch, "v1"))["status"] == "pending"
    assert await temporal_jobs.run_job(client, branch, key) == "complete"


async def test_source_archived_while_the_model_runs_gets_no_facts(client, make_db, monkeypatch):
    branch = await make_db("ittest_tjob_latecancel")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)

    async def call(messages):                     # the source is archived during the first model call
        await indexer.set_status_by_version(client, branch, "v1", "archived")
        return {"facts": []}

    monkeypatch.setattr(temporal_jobs, "call_model", call)
    assert await temporal_jobs.run_job(client, branch, key) == "cancelled"
    assert await _facts(client, db_name_for(branch), "v1") == []


async def test_job_replaces_the_sources_older_facts(client, make_db, model):
    branch = await make_db("ittest_tjob_replace")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(   # the retired video path's kind of row
        fact_id="tf-legacy", entity="Android", relation="ACQUIRED_BY", value="Google", time_start="2005-01-01",
        source_version_id="v1"))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(   # another source: untouched
        fact_id="tf-other", entity="X", relation="r", value="y", time_start="2001", source_version_id="v9"))
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)
    ids = {f["fact_id"] for f in await _facts(client, db, "v1")}
    assert "tf-legacy" not in ids and ids
    assert [f["fact_id"] for f in await _facts(client, db, "v9")] == ["tf-other"]


async def test_rerun_is_idempotent(client, make_db, model):
    branch = await make_db("ittest_tjob_rerun")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)
    first = [f["fact_id"] for f in await _facts(client, db, "v1")]
    await client.command(db, "UPDATE TemporalJob SET status = 'running' WHERE job_key = :k", params={"k": key})
    assert await temporal_jobs.run_job(client, branch, key) == "complete"     # as after a crash mid-write
    assert [f["fact_id"] for f in await _facts(client, db, "v1")] == first


async def test_delete_by_version_removes_the_job(client, make_db, model):
    branch = await make_db("ittest_tjob_delete")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)
    result = await indexer.delete_by_version(client, branch, "v1")
    assert result["temporal_jobs"] == 1 and result["temporal_facts"] > 0
    assert await temporal_jobs.status(client, branch, "v1") is None
    assert await client.query(db, "SELECT FROM TemporalJobSegment") == []


async def test_recovery_requeues_unfinished_jobs(client, make_db, model):
    branch = await make_db("ittest_tjob_recover")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    runner = temporal_jobs.TemporalJobRunner(client)
    assert await runner.recover() >= 1
    assert (branch, key) in runner.queued


async def test_named_subjects_become_entities_the_tkg_search_finds(client, make_db, model):
    """P12 writes an Entity for every named subject (MENTIONS from its chunk, HAS_TEMPORAL_FACT to its facts),
    so "What was Hugo Barra doing in 2010?" is answered by entity retrieval -- documents get no entity graph
    from anywhere else."""
    from app.models.graph_index import EntityIndexRequest
    from app.models.graph_search import TKGRequest
    from app.services import searcher

    branch = await make_db("ittest_tjob_entities")
    db = db_name_for(branch)
    texts = [(s["chunk_id"], s["text"]) for s in _ALL if "Hugo Barra served" in s["text"]][:1] + _android()[:1]
    segments = await _source(client, branch, "v1", texts)
    await graph_indexer.index_entity(client, branch, EntityIndexRequest(    # made earlier by another source
        entity_id=temporal_jobs.entity_id_for("Android Inc."), name="Android Inc.", type="ORG"))
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    assert await temporal_jobs.run_job(client, branch, key) == "complete"

    ents = {r["entity_id"]: r for r in await client.query(db, "SELECT entity_id, name, type, mention_count FROM Entity")}
    assert ents[temporal_jobs.entity_id_for("Hugo Barra")]["mention_count"] >= 1
    assert ents[temporal_jobs.entity_id_for("Android Inc.")]["type"] == "ORG"            # not overwritten
    linked = await client.query(db, "SELECT count(*) AS c FROM HAS_TEMPORAL_FACT WHERE outV().entity_id = :e",
                                params={"e": temporal_jobs.entity_id_for("Hugo Barra")})
    assert linked[0]["c"] >= 1
    assert not [e for e in ents.values() if e["name"].casefold().startswith(("the ", "users"))]

    reply = await searcher.tkg_search(client, branch, TKGRequest(
        query="What was Hugo Barra doing in 2010?", time_start="2010-01-01", time_end="2010-12-31",
        match="overlaps", keywords=["Hugo", "Barra", "doing"], score_threshold=0.0))
    assert reply.retrieval == "entity"
    assert any(f["entity"] == "Hugo Barra" and f.get("when") == "2008 to 2013" for f in reply.temporal_facts)

    await indexer.delete_by_version(client, branch, "v1")
    left = {r["entity_id"] for r in await client.query(db, "SELECT entity_id FROM Entity")}
    assert temporal_jobs.entity_id_for("Hugo Barra") not in left                       # pruned with its source


async def test_undated_candidates_are_kept_with_their_links_but_never_returned_as_facts(client, make_db, model):
    from app.models.graph_search import TKGRequest
    from app.services import searcher

    branch = await make_db("ittest_tjob_unresolved")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)

    _, expected = extract([Segment(c, t) for c, t in _android()], replies=_REPLIES)
    rows = await client.query(db, "SELECT FROM TemporalCandidate WHERE source_version_id = 'v1'")
    status = await temporal_jobs.status(client, branch, "v1")
    assert status["unresolved"] == len(rows) > 0 and len(rows) <= len(expected)
    miner = next(r for r in rows if r["entity"] == "Rich Miner")
    assert (miner["date_text"], miner["reason"], miner["origin"]) == ("later", "date_text_not_normalised", "llm")
    assert json.loads(miner["links"]) == [{"relation": "after", "other": "Android Inc. was founded"}]
    text = dict(_android())["c00"]
    a, b = json.loads(miner["evidence_offsets"])["entity"]
    assert text[a:b] == "Rich Miner"

    reply = await searcher.tkg_search(client, branch, TKGRequest(query="Rich Miner", score_threshold=0.0))
    # the undated candidate itself is never a fact; what reasoning derived from it (P15) is, marked as such
    assert all(f.get("derived") is True for f in reply.temporal_facts if f["entity"] == "Rich Miner")

    await client.command(db, "UPDATE TemporalJob SET status = 'running' WHERE job_key = :k", params={"k": key})
    await temporal_jobs.run_job(client, branch, key)                                    # a rerun replaces, not adds
    assert len(await client.query(db, "SELECT FROM TemporalCandidate")) == len(rows)
    await indexer.delete_by_version(client, branch, "v1")
    assert await client.query(db, "SELECT FROM TemporalCandidate") == []


async def test_publish_date_from_the_file_dates_deictic_phrases(client, make_db, monkeypatch):
    """P13 end to end: a chunk arrives with its PDF CreationDate; the Source keeps it as read, the job records
    it validated, and "two years ago" in the text becomes 2013 (kind deictic) instead of staying unresolved."""
    from app.services import kafka_consumer

    class FakeRedis:
        def __init__(self):
            self.keys = {}

        async def exists(self, key):
            return int(key in self.keys)

        async def set(self, key, value, ex=None):
            self.keys[key] = value

        async def sismember(self, key, member):
            return False

    async def claim(*_a, **_kw):
        return True

    async def embed(texts):
        return [fake_vector(t) for t in texts]

    async def call(messages):
        return {"facts": [{"entity": "Google", "relation": "released", "value": "the phone",
                           "date_text": "two years ago", "extent": "point"}]}

    monkeypatch.setattr(kafka_consumer, "claim_event", claim)
    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    monkeypatch.setattr(temporal_jobs, "call_model", call)
    branch = await make_db("ittest_tjob_anchor")
    db = db_name_for(branch)
    message = {"event_id": "e-anchor", "source_module": "12-document", "payload": {
        "branch_id": branch, "correlation_id": "c", "parameters": {"version_id": "v1", "asset_path": "a.pdf", "chunks": [
            {"id": "c1", "payload": {"text": "Google released the phone two years ago.", "type": "documents",
                                     "chunk_index": 0, "version_id": "v1", "branch_id": branch,
                                     "capture_time": "D:20150320121500+08'00'",
                                     "capture_time_source": "pdf_creation_date"}}]}}}
    await kafka_consumer._handle_message(client, FakeRedis(), message)

    [source] = await client.query(db, "SELECT capture_time, capture_time_source FROM Source WHERE version_id = 'v1'")
    assert (source["capture_time"], source["capture_time_source"]) == ("D:20150320121500+08'00'", "pdf_creation_date")
    status = await temporal_jobs.status(client, branch, "v1")
    assert (status["anchor_time"], status["anchor_source"], status["anchor_reason"]) == (
        "2015-03-20T12:15:00+08:00", "pdf_creation_date", None)

    assert await temporal_jobs.run_job(client, branch, status["job_id"]) == "complete"
    [fact] = await client.query(db, "SELECT time_start_raw, time_end_raw, anchor, date_text FROM TemporalFact "
                                    "WHERE source_version_id = 'v1'")
    assert (fact["date_text"], fact["time_start_raw"], fact["anchor"]) == ("two years ago", "2013", "deictic")


async def test_a_photo_is_an_observation_event_at_its_exif_time(client, make_db, monkeypatch):
    """P13: a photo with no OCR text (so no chunk, no job) is still recorded as taken at its EXIF time; the
    fact survives a later job run on the same version and goes with the photo on delete."""
    from app.services import kafka_consumer

    class FakeRedis:
        def __init__(self):
            self.keys = {}

        async def exists(self, key):
            return int(key in self.keys)

        async def set(self, key, value, ex=None):
            self.keys[key] = value

        async def sismember(self, key, member):
            return False

    async def claim(*_a, **_kw):
        return True

    async def embed(texts):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(kafka_consumer, "claim_event", claim)
    monkeypatch.setattr(kafka_consumer, "embed_texts", embed)
    monkeypatch.setattr(kafka_consumer.settings, "graph_extraction_enabled", False)
    branch = await make_db("ittest_tjob_photo")
    db = db_name_for(branch)
    base = {"type": "images", "filename": "IMG_0042.jpg", "version_id": "p1", "asset_path": "IMG_0042.jpg",
            "branch_id": branch, "capture_time": "2015:03:20 12:15:00", "capture_time_source": "exif_datetime_original"}
    message = {"event_id": "e-photo", "source_module": "10-image", "payload": {
        "branch_id": branch, "correlation_id": "c", "parameters": {"version_id": "p1", "asset_path": "IMG_0042.jpg", "chunks": [
            {"id": "s1", "payload": {**base, "embedding_type": "summary", "summary": "Two people at a harbour."}},
            {"id": "t1", "payload": {**base, "embedding_type": "text", "text": ""}}]}}}
    await kafka_consumer._handle_message(client, FakeRedis(), message)

    [fact] = await client.query(db, "SELECT entity, relation, value, date_text, anchor, precision, time_start "
                                    "FROM TemporalFact WHERE source_version_id = 'p1'")
    assert (fact["entity"], fact["relation"], fact["anchor"], fact["date_text"]) == (
        "IMG_0042.jpg", "was taken", "capture", "2015:03:20 12:15:00")
    assert (fact["precision"], fact["time_start"], fact["value"]) == ("second", "2015-03-20", "Two people at a harbour.")
    assert await temporal_jobs.status(client, branch, "p1") is None          # no text: no job, observation anyway
    [source] = await client.query(db, "SELECT capture_time, capture_time_source FROM Source WHERE version_id = 'p1'")
    assert (source["capture_time"], source["capture_time_source"]) == ("2015:03:20 12:15:00", "exif_datetime_original")

    segments = await _source(client, branch, "p1", [("ocr1", "SALE ends 2015-04-01")])
    key = await temporal_jobs.enqueue(client, branch, "p1", "images", segments)

    async def call(messages):
        return {"facts": []}
    monkeypatch.setattr(temporal_jobs, "call_model", call)
    await temporal_jobs.run_job(client, branch, key)
    anchors = sorted(r["anchor"] for r in await client.query(db, "SELECT anchor FROM TemporalFact"))
    assert "capture" in anchors                                               # the job does not own it

    await indexer.delete_by_version(client, branch, "p1")
    assert await client.query(db, "SELECT FROM TemporalFact") == []


# ---------------------------------------------------------------- P14: lifecycle

async def test_archive_cancels_and_reactivation_reopens(client, make_db, model):
    branch = await make_db("ittest_p14_archive")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    assert await temporal_jobs.run_job(client, branch, key) == "complete"
    assert await client.query(db, "SELECT FROM TemporalCandidate WHERE status = 'active'")

    await temporal_jobs.enqueue(client, branch, "v1", "documents",
                                [*segments, temporal_jobs.SourceSegment("late", "Added in 2015.", 9)])
    result = await indexer.set_status_by_version(client, branch, "v1", "archived")
    assert (result["temporal_jobs_cancelled"], result["temporal_candidates"] > 0) == (1, True)
    job = await temporal_jobs.status(client, branch, "v1")
    assert (job["status"], job["reason"]) == ("cancelled", "source_archived")
    assert not await client.query(db, "SELECT FROM TemporalCandidate WHERE status = 'active'")

    result = await indexer.set_status_by_version(client, branch, "v1", "active")
    assert result["temporal_jobs_reopened"] == 1
    assert (await temporal_jobs.status(client, branch, "v1"))["status"] == "pending"
    assert await temporal_jobs.run_job(client, branch, key) == "complete"
    assert await client.query(db, "SELECT FROM TemporalCandidate WHERE status = 'active'")


async def test_work_started_before_an_archive_never_writes_after_reactivation(client, make_db, monkeypatch):
    """Archive + reactivate while the model is busy: the source is active again, but the run belongs to the old
    generation -- it writes nothing and leaves the reopened job to its rerun, which completes."""
    branch = await make_db("ittest_p14_generation")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    calls = []

    async def call(messages):
        calls.append(1)
        if len(calls) == 1:
            await indexer.set_status_by_version(client, branch, "v1", "archived")
            await indexer.set_status_by_version(client, branch, "v1", "active")
        return {"facts": []}

    monkeypatch.setattr(temporal_jobs, "call_model", call)
    assert await temporal_jobs.run_job(client, branch, key) == "superseded"
    assert await _facts(client, db, "v1") == []
    assert (await temporal_jobs.status(client, branch, "v1"))["status"] == "pending"
    assert await temporal_jobs.run_job(client, branch, key) == "complete"
    assert await _facts(client, db, "v1")


async def test_clone_takes_the_job_and_candidates_along(client, make_db, model):
    from app.models.index import CloneVersionRequest
    calls, _ = model
    branch = await make_db("ittest_p14_clone")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)
    src = await temporal_jobs.status(client, branch, "v1")
    calls.clear()

    result = await indexer.clone_version(client, branch, CloneVersionRequest(
        source_version_id="v1", source_asset_path="a.pdf", target_version_id="v2", target_asset_path="b.pdf"))
    assert (result["temporal_jobs"], result["temporal_candidates"]) == (1, src["unresolved"])
    dst = await temporal_jobs.status(client, branch, "v2")
    assert (dst["status"], dst["facts"], dst["segments_done"]) == ("complete", src["facts"], src["segments_done"])
    chunks = {r["chunk_id"] for r in await client.query(db, "SELECT chunk_id FROM Chunk WHERE version_id = 'v2'")}
    cand = await client.query(db, "SELECT chunk_id FROM TemporalCandidate WHERE source_version_id = 'v2'")
    assert cand and {c["chunk_id"] for c in cand} <= chunks                     # remapped onto the clone's chunks
    segs = await client.query(db, "SELECT chunk_id FROM TemporalJobSegment WHERE job_key = :k",
                              params={"k": dst["job_id"]})
    assert {s["chunk_id"] for s in segs} <= chunks
    await indexer.delete_by_version(client, branch, "v1")                         # the clone keeps its own
    assert (await temporal_jobs.status(client, branch, "v2"))["facts"] == src["facts"]
    assert len(await client.query(db, "SELECT FROM TemporalCandidate")) == src["unresolved"]
    assert calls == []                                                            # no model call for the clone


async def test_reextraction_clears_obsolete_evidence_fields(client, make_db):
    branch = await make_db("ittest_p14_clear")
    db = db_name_for(branch)
    base = {"fact_id": "tf-x", "entity": "Android", "relation": "r", "value": "v", "time_start": "2005",
            "source_version_id": "v1", "extractor_version": "1.1.0"}
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        **base, extent="span", date_text="2005", media_start_ms=1000))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(**base, date_text="in 2005"))
    [row] = await client.query(db, "SELECT extent, date_text, media_start_ms, source_version_id FROM TemporalFact")
    assert (row["extent"], row["date_text"], row["media_start_ms"], row["source_version_id"]) == (
        None, "in 2005", None, "v1")
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(        # v1 writer: unchanged
        fact_id="tf-y", entity="A", relation="r", value="v", time_start="2005", extent="point"))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="tf-y", entity="A", relation="r", value="v", time_start="2005"))
    [row] = await client.query(db, "SELECT extent FROM TemporalFact WHERE fact_id = 'tf-y'")
    assert row["extent"] == "point"


# ---------------------------------------------------------------- P15: reasoning at ingestion

async def test_reasoning_derives_bounds_for_undated_events(client, make_db, model):
    """'Rich Miner and Nick Sears joining later' + the model's link (after 'Android Inc. was founded') -> Rich
    Miner joined after October 2003: a derived fact, with the facts it rests on, found by the TKG search."""
    from app.models.graph_search import TKGRequest
    from app.services import searcher
    calls, _ = model
    branch = await make_db("ittest_p15_derive")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    assert await temporal_jobs.run_job(client, branch, key) == "complete"

    result = await temporal_jobs.reasoning_of(client, branch, "v1")
    assert (result["status"], result["stale"]) == ("consistent", False)
    [miner] = [f for f in result["derived_facts"] if f["entity"] == "Rich Miner"]
    assert miner["date_text"].startswith("after 2003-10-01") and miner["date_text"].endswith("(derived)")
    [support] = await client.query(db, "SELECT entity, relation FROM TemporalFact WHERE fact_id = :f",
                                   params={"f": miner["support_ids"][0]})
    assert support["relation"] == "founded"
    [row] = await client.query(db, "SELECT derived, anchor, s_state, migration_version FROM TemporalFact "
                                   "WHERE fact_id = :f", params={"f": miner["fact_id"]})
    assert (row["derived"], row["anchor"], row["s_state"]) == (True, "derived", "unknown")   # unbounded above

    reply = await searcher.tkg_search(client, branch, TKGRequest(
        query="Rich Miner", keywords=["Rich Miner"], time_start="2004", time_end="2004", match="overlaps",
        score_threshold=0))
    [found] = [f for f in reply.temporal_facts if f["fact_id"] == miner["fact_id"]]
    assert found["derived"] is True and found["when"] == miner["date_text"]

    calls.clear()
    assert await temporal_jobs.recompute(client, branch, "v1")
    assert (await temporal_jobs.reasoning_of(client, branch, "v1"))["stale"] is True
    assert await temporal_jobs.run_job(client, branch, key) == "complete"
    assert calls == []                                                         # rebuilt from stored replies
    assert [f["fact_id"] for f in (await temporal_jobs.reasoning_of(client, branch, "v1"))["derived_facts"]
            if f["entity"] == "Rich Miner"] == [miner["fact_id"]]


async def test_contradicting_statements_are_reported_not_hidden(client, make_db, monkeypatch):
    branch = await make_db("ittest_p15_conflict")
    segments = await _source(client, branch, "v1", _android()[:1])

    async def call(messages):
        return {"facts": [{"entity": "Rich Miner", "relation": "joined", "value": "Android Inc.", "date_text": "later",
                           "links": [{"relation": "after", "other": "Android Inc. was founded"},
                                     {"relation": "before", "other": "Android Inc. was founded"}]}]}

    monkeypatch.setattr(temporal_jobs, "call_model", call)
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    assert await temporal_jobs.run_job(client, branch, key) == "complete"     # a conflict is a finding, not a failure
    result = await temporal_jobs.reasoning_of(client, branch, "v1")
    assert result["status"] == "inconsistent" and result["derived_facts"] == []
    assert any("after" in step["because"] for step in result["witness"])
    assert any("before" in step["because"] for step in result["witness"])


async def test_reason_operation_over_http(client, make_db, model):
    import httpx
    from app.main import app
    branch = await make_db("ittest_p15_http")
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        r = await http.get("/personal/temporal/reason/v1", headers={"X-Branch-ID": branch})
        assert r.status_code == 200 and r.json()["status"] == "consistent" and r.json()["derived_facts"]
        r = await http.post("/personal/temporal/reason/v1/recompute", headers={"X-Branch-ID": branch})
        assert (r.status_code, r.json()["queued"]) == (202, True)
        r = await http.get("/personal/temporal/reason/nope", headers={"X-Branch-ID": branch})
        assert r.status_code == 404
    await indexer.delete_by_version(client, branch, "v1")
    assert await temporal_jobs.reasoning_of(client, branch, "v1") is None


async def test_backfill_never_renormalises_a_derived_fact(client, make_db, model):
    """A grammar upgrade re-normalises facts from their date strings; a derived fact has none -- its bounds come
    from reasoning -- so the backfill must leave it alone (a job rerun recomputes it instead)."""
    from app.services import temporal_backfill
    branch = await make_db("ittest_p15_backfill")
    db = db_name_for(branch)
    segments = await _source(client, branch, "v1", _android())
    key = await temporal_jobs.enqueue(client, branch, "v1", "documents", segments)
    await temporal_jobs.run_job(client, branch, key)
    # a derived fact bounded on both sides: [2005-06-30, 2005-07-31], projected as time_start "2005-07-01" --
    # re-normalising that string would replace the bounds with one day
    lo, hi = 1120060800000, 1122854399999
    await client.command(db, "INSERT INTO TemporalFact SET fact_id = 'tf-der-x', entity = 'staff', relation = 'joined', "
                             "value = 'Google', source_version_id = 'v1', derived = true, s_state = 'known', "
                             "s_lo = :lo, s_hi = :hi, e_state = 'known', e_lo = :lo, e_hi = :hi, "
                             "time_start = '2005-07-01', time_end = '2005-07-31', precision = 'derived'",
                         params={"lo": lo, "hi": hi})
    await client.command(db, "UPDATE TemporalFact SET migration_version = 'old'")      # as after a grammar bump
    cols = "fact_id, s_state, s_lo, s_hi, e_state, e_lo, e_hi, precision, date_text"
    before = await client.query(db, f"SELECT {cols} FROM TemporalFact WHERE derived = true")
    assert len(before) >= 2
    await temporal_backfill.backfill_database(client, db)
    after = await client.query(db, f"SELECT {cols} FROM TemporalFact WHERE derived = true")
    assert sorted(map(str, before)) == sorted(map(str, after))
