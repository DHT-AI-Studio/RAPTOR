"""P12 -- the pure parts of the durable temporal job: the consumer's segment hand-off for every medium,
the language check, the Module 07 client (retries, tolerant JSON), stored-reply extraction and fact
identity. No ArcadeDB here -- tests/integration/test_temporal_jobs.py runs the job itself.
"""
from __future__ import annotations

import dataclasses
import json

import httpx
import pytest
from app.services import temporal_jobs
from app.services.kafka_consumer import _tag_entry_type, _temporal_segments
from app.temporal.extract import (
    CANDIDATE_SCHEMA,
    EXTRACTOR_VERSION,
    Segment,
    extract,
    segment_messages,
)

from .test_temporal_extract import _segments

# ---------------------------------------------------------------- segments from the consumer's entries

def _doc(cid, text, idx, **extra):
    return {"id": cid, "payload": {"text": text, "type": "documents", "chunk_index": idx, "version_id": "v1",
                                   "branch_id": "u1", **extra}}


def test_every_medium_becomes_segments_in_document_order():
    entries = [
        {"id": "sum", "payload": {"summary": "A summary.", "embedding_type": "summary", "version_id": "v1"}},
        _doc("c2", "Second, in 2010.", 2, page_numbers=[3, 4], section_heading="2010s"),
        _doc("c0", "First, in 2003.", 0),
        _doc("c1", "Never indexed.", 1),
        _doc("c3", "   ", 3),
    ]
    out = _temporal_segments(entries, "u1", "v1", indexed={"c0", "c2", "c3", "sum"})
    media, segments, _ = out[("u1", "v1")]
    assert media == "documents"
    assert [(s.chunk_id, s.seq, s.page, s.heading) for s in segments] == [("c0", 0, "", ""), ("c2", 1, "3,4", "2010s")]


def test_video_moments_need_no_summary_and_keep_their_media_offsets():
    moment = _tag_entry_type({"payload": {"moment_index": 0, "asr_text": "We launched in May 2013.",
                                          "start_time": 12.5, "end_time": 20}}, "videos")
    out = _temporal_segments([moment], "u1", "v9", indexed={_moment_id(moment, "v9")})
    media, [seg], _ = out[("u1", "v9")]
    assert media == "videos"
    assert "May 2013" in seg.text
    assert (seg.media_start_ms, seg.media_end_ms) == (12500, 20000)


def test_video_context_written_by_a_model_is_not_evidence():
    """Module 11's contextual prefix is an LLM's summary, not what the recording says: the segment is what
    follows it, and offsets into the segment map back into the Chunk's text."""
    from app.services.kafka_consumer import _extract_content
    moment = _tag_entry_type({"payload": {"moment_index": 3, "asr_text": "We shipped it in May 2013.",
                                          "contextual_prefix": "A 2019 keynote recap of the launch"}}, "videos")
    chunk_text, _, _ = _extract_content(moment["payload"])
    _, [seg], _ = _temporal_segments([moment], "u1", "v9", indexed={_moment_id(moment, "v9")})[("u1", "v9")]
    assert "2019" not in seg.text and seg.text == "asr:{We shipped it in May 2013.}"
    assert chunk_text[seg.text_offset:] == seg.text
    accepted, _ = extract([Segment(seg.chunk_id, seg.text)])
    [req] = temporal_jobs.fact_requests("v9", accepted, {}, {seg.chunk_id: seg.text_offset})
    a, b = json.loads(req.evidence_offsets)["date"]
    assert chunk_text[a:b] == "May 2013"


def test_module_12_wrapping_is_not_evidence():
    """Module 12's contextual chunking: the file name and the LLM's [CONTEXT] sentence carry years the document
    never states; only the [CHUNK] text is read, and offsets map back into the Chunk's text."""
    raw = "Google acquired the company in July 2005 for at least $50 million."
    wrapped = ("[FILENAME] annual_report_2019.pdf [/FILENAME]\n"
               "[CONTEXT] This 2021 review covers Android's early history. [/CONTEXT]\n"
               f"[CHUNK] {raw} [/CHUNK]")
    _, [seg], _ = _temporal_segments([_doc("c0", wrapped, 0)], "u1", "v1", indexed={"c0"})[("u1", "v1")]
    assert seg.text == raw and wrapped[seg.text_offset:seg.text_offset + len(raw)] == raw
    accepted, _ = extract([Segment(seg.chunk_id, seg.text)])
    assert {c.time_start for c in accepted} == {"2005-07"}                 # not 2019, not 2021
    [req] = [r for r in temporal_jobs.fact_requests("v1", accepted, {}, {"c0": seg.text_offset})][:1]
    a, b = json.loads(req.evidence_offsets)["date"]
    assert wrapped[a:b] == "July 2005"
    no_context = f"[FILENAME] a.pdf [/FILENAME]\n[CHUNK] {raw} [/CHUNK]"
    _, [seg2], _ = _temporal_segments([_doc("c1", no_context, 0)], "u1", "v1", indexed={"c1"})[("u1", "v1")]
    assert seg2.text == raw


def test_a_document_chunk_is_evidence_as_a_whole():
    doc = _doc("c0", "contextual:{not a video} / founded in 2003", 0)
    _, [seg], _ = _temporal_segments([doc], "u1", "v1", indexed={"c0"})[("u1", "v1")]
    assert (seg.text, seg.text_offset) == ("contextual:{not a video} / founded in 2003", 0)


def _moment_id(entry, version):
    from app.services.kafka_consumer import _stable_chunk_id
    return _stable_chunk_id(entry["payload"], version)


# ---------------------------------------------------------------- language

@pytest.mark.parametrize("text, ok", [
    ("Android Inc. was founded in October 2003.", True),
    ("2011年5月，谷歌發布了新版本。", True),
    ("Google 在 2005 年收購了 Android。", True),
    ("2019-08-22", True),                                         # no letters: dates are language-neutral
    ("アンドロイドは二〇〇三年に設立された。", False),
    ("Андроид был основан в 2003 году.", False),
    ("Android (안드로이드) was founded in 2003.", True),           # mostly Latin
])
def test_language_supported(text, ok):
    assert temporal_jobs.language_supported(text) is ok


# ---------------------------------------------------------------- Module 07 reply parsing / client

@pytest.mark.parametrize("raw", [
    '{"facts": []}',
    '```json\n{"facts": []}\n```',
    '<think>let me see {not json}</think>{"facts": []}',
    'Here you go: {"facts": []} -- done',
])
def test_parse_reply_tolerates_wrappers(raw):
    assert temporal_jobs.parse_reply(raw) == {"facts": []}


def test_parse_reply_allows_raw_newlines_in_strings():
    assert temporal_jobs.parse_reply('{"facts": [{"value": "two\nlines"}]}')["facts"][0]["value"] == "two\nlines"


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2]", '{"facts": [}'])
def test_parse_reply_never_guesses(raw):
    with pytest.raises(ValueError):
        temporal_jobs.parse_reply(raw)


def _ok(body: str) -> httpx.Response:
    return httpx.Response(200, json={"result": {"response": body}})


@pytest.fixture
def module07(monkeypatch):
    """Replies served in order; records the requests."""
    replies: list = []
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    real = httpx.AsyncClient
    monkeypatch.setattr(temporal_jobs.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(temporal_jobs.settings, "temporal_llm_retry_delay", 0)
    monkeypatch.setattr(temporal_jobs.settings, "temporal_llm_retries", 3)
    return replies, seen


@pytest.mark.asyncio
async def test_call_model_retries_transient_failures(module07):
    replies, seen = module07
    replies += [httpx.Response(503), httpx.ConnectError("refused"), _ok("not json"), _ok('{"facts": [1]}')]
    assert await temporal_jobs.call_model([{"role": "user", "content": "x"}]) == {"facts": [1]}
    assert len(seen) == 4
    assert seen[0]["options"] == {"temperature": 0.0, "think": False, "format": CANDIDATE_SCHEMA}
    assert seen[0]["task"] == "text-generation"


@pytest.mark.asyncio
async def test_call_model_gives_up_after_its_retries(module07):
    replies, seen = module07
    replies += [httpx.Response(502)] * 4
    with pytest.raises(temporal_jobs.ModelError, match="502"):
        await temporal_jobs.call_model([])
    assert len(seen) == 4


@pytest.mark.asyncio
async def test_call_model_does_not_retry_a_refusal(module07):
    replies, seen = module07
    replies += [httpx.Response(400, json={"detail": {"error_type": "PolicyViolationError"}})]
    with pytest.raises(temporal_jobs.ModelError, match="400"):
        await temporal_jobs.call_model([])
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_call_model_takes_module_07s_validated_reply(module07):
    replies, _ = module07
    replies += [httpx.Response(200, json={"result": {"response": '{"facts": []}', "parsed": {"facts": [2]}}})]
    assert await temporal_jobs.call_model([]) == {"facts": [2]}


@pytest.mark.asyncio
async def test_call_model_does_not_retry_an_output_that_misses_the_schema(module07):
    replies, seen = module07
    replies += [httpx.Response(502, json={"detail": {"code": "structured_output_error", "violations": ["$: x"]}})]
    with pytest.raises(temporal_jobs.ModelError, match="structured_output_error"):
        await temporal_jobs.call_model([])
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_call_model_without_schema_support_reads_the_old_way(module07):
    replies, seen = module07
    replies += [httpx.Response(400, json={"detail": {"code": "unsupported_capability"}}), _ok('```{"facts": [3]}```')]
    assert await temporal_jobs.call_model([]) == {"facts": [3]}
    assert "format" in seen[0]["options"] and "format" not in seen[1]["options"]


# ---------------------------------------------------------------- stored replies

def test_stored_replies_give_the_same_result_as_live_calls():
    segs = _segments()[:6]
    canned = {s.chunk_id: {"facts": [{"entity": "Android Inc.", "relation": "was founded in", "value": "Palo Alto",
                                      "date_text": "October 2003", "extent": "point", "confidence": 0.9}]}
              if s.chunk_id == "c00" else {"facts": []} for s in segs}
    by_message = {json.dumps(m): canned[s.chunk_id] for s, m in zip(segs, segment_messages(segs), strict=True)}

    def llm(messages, schema):
        assert schema is CANDIDATE_SCHEMA
        return by_message[json.dumps(messages)]          # also proves segment_messages() == what extract() sends

    live = extract(segs, llm=llm)
    stored = extract(segs, replies=canned)
    assert [[dataclasses.asdict(c) for c in part] for part in live] == \
           [[dataclasses.asdict(c) for c in part] for part in stored]
    assert any(c.origin != "rules" for c in stored[0])


def test_a_segment_without_a_stored_reply_keeps_its_rule_candidates():
    segs = _segments()[:3]
    rules_only = extract(segs)
    assert extract(segs, replies={}) == rules_only


# ---------------------------------------------------------------- fact identity

def _candidate():
    accepted, _ = extract([Segment("c00", "Android Inc. was founded in Palo Alto, California, in October 2003.")])
    return accepted[0]


def test_fact_id_is_stable_and_source_specific():
    c = _candidate()
    [a] = temporal_jobs.fact_requests("v1", [c], {})
    [b] = temporal_jobs.fact_requests("v1", [c], {})
    [other_source] = temporal_jobs.fact_requests("v2", [c], {})
    [other_date] = temporal_jobs.fact_requests("v1", [dataclasses.replace(c, time_start="2004-10")], {})
    [other_evidence] = temporal_jobs.fact_requests("v1", [dataclasses.replace(c, date_offsets=(0, 1))], {})
    assert a.fact_id == b.fact_id
    assert len({a.fact_id, other_source.fact_id, other_date.fact_id, other_evidence.fact_id}) == 4


def test_fact_request_carries_the_evidence():
    c = _candidate()
    [req] = temporal_jobs.fact_requests("v1", [c], {"c00": (1000, 2000)})
    offsets = json.loads(req.evidence_offsets)
    assert c.sentence[offsets["date"][0]:offsets["date"][1]] == "October 2003"
    assert (req.date_text, req.time_start, req.anchor, req.extractor_version) == ("October 2003", "2003-10", c.kind, EXTRACTOR_VERSION)
    assert (req.media_start_ms, req.media_end_ms, req.source_version_id, req.chunk_id) == (1000, 2000, "v1", "c00")
    assert req.entity_id == temporal_jobs.entity_id_for(c.entity)


# ---------------------------------------------------------------- entities for named subjects

@pytest.mark.parametrize("entity, coref, named", [
    ("Hugo Barra", False, True),
    ("Android Inc.", False, True),
    ("谷歌", False, True),
    ("the company", False, False),
    ("The acquisition", False, False),
    ("users", False, False),
    ("Rubin", True, False),                 # resolved coreference ("he"): not stated as a name in that sentence
    ("", False, False),
])
def test_is_named(entity, coref, named):
    c = dataclasses.replace(_candidate(), entity=entity, coreference=coref)
    assert temporal_jobs.is_named(c) is named


def test_named_subjects_get_one_entity_per_chunk_and_their_facts_link_to_it():
    c = _candidate()
    unnamed = dataclasses.replace(c, entity="the company", relation="pivoted")
    again = dataclasses.replace(c, relation="moved")
    ents = temporal_jobs.entity_requests([c, unnamed, again])
    assert [(e.entity_id, e.name, e.source_chunk_id, e.modality) for e in ents] == [
        (temporal_jobs.entity_id_for("Android Inc."), "Android Inc.", "c00", "text")]
    facts = temporal_jobs.fact_requests("v1", [c, unnamed], {})
    assert [f.entity_id for f in facts] == [temporal_jobs.entity_id_for("Android Inc."), None]


def test_candidate_rows_keep_evidence_links_and_reason():
    seg = Segment("c1", "Rich Miner and Nick Sears joined later, after Android Inc. was founded.")
    reply = {"c1": {"facts": [{"entity": "Rich Miner", "relation": "joined", "value": "Nick Sears", "date_text": "later",
                               "links": [{"relation": "after", "other": "Android Inc. was founded"}]}]}}
    _, unresolved = extract([seg], replies=reply)
    rows = temporal_jobs.candidate_rows("v1", unresolved, {"c1": 7})
    [row] = [r for r in rows if r["entity"] == "Rich Miner"]
    assert row["reason"] == "date_text_not_normalised" and json.loads(row["links"])[0]["relation"] == "after"
    assert json.loads(row["evidence_offsets"])["entity"] == [7, 17]                    # shifted into Chunk.text
    again = temporal_jobs.candidate_rows("v1", unresolved, {"c1": 7})
    assert [r["candidate_id"] for r in rows] == [r["candidate_id"] for r in again]


# ---------------------------------------------------------------- runner

@pytest.mark.asyncio
async def test_a_short_job_is_not_stuck_behind_a_long_one(monkeypatch):
    import asyncio
    monkeypatch.setattr(temporal_jobs.settings, "temporal_job_workers", 2)
    release, done = asyncio.Event(), []

    async def run(client, branch_id, key):
        if key == "long":
            await release.wait()
        done.append(key)

    async def no_recovery(self):
        return 0

    monkeypatch.setattr(temporal_jobs, "run_job", run)
    monkeypatch.setattr(temporal_jobs.TemporalJobRunner, "recover", no_recovery)
    runner = temporal_jobs.TemporalJobRunner(None)
    await runner.start()
    runner.submit("b", "long")
    runner.submit("b", "short")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if done:
            break
    assert done == ["short"]                    # finished while the long one still runs
    release.set()
    await asyncio.sleep(0.05)
    assert done == ["short", "long"]
    await runner.stop()


@pytest.mark.asyncio
async def test_a_job_never_runs_twice_at_once_and_reruns_once_if_resubmitted(monkeypatch):
    import asyncio
    monkeypatch.setattr(temporal_jobs.settings, "temporal_job_workers", 3)
    release, live, peak, runs = asyncio.Event(), [0], [0], []

    async def run(client, branch_id, key):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        runs.append(key)
        if len(runs) == 1:
            await release.wait()
        live[0] -= 1

    async def no_recovery(self):
        return 0

    monkeypatch.setattr(temporal_jobs, "run_job", run)
    monkeypatch.setattr(temporal_jobs.TemporalJobRunner, "recover", no_recovery)
    runner = temporal_jobs.TemporalJobRunner(None)
    await runner.start()
    runner.submit("b", "k")
    await asyncio.sleep(0.02)
    runner.submit("b", "k")                     # new segments arrived while it runs
    runner.submit("b", "k")
    await asyncio.sleep(0.02)
    assert runs == ["k"]
    release.set()
    await asyncio.sleep(0.05)
    assert runs == ["k", "k"] and peak[0] == 1
    await runner.stop()
