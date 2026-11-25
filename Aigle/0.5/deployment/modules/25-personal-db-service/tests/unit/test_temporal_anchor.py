"""P13: a source's capture / publish time (file metadata) as the anchor of its deictic phrases.

- app.temporal.anchor validates what a worker read: PDF, EXIF and ISO forms; placeholders and impossible
  dates are refused (the source then has no anchor, never the upload time);
- the extractor dates "two years ago" / "yesterday" / 去年 from it, marks them `deictic`, and leaves them
  unresolved without it; explicit dates are unaffected;
- the Kafka consumer hands the validated anchor (or why it was refused) to the temporal job.
"""
from __future__ import annotations

import datetime as dt

import pytest
from app.services.kafka_consumer import _temporal_segments
from app.temporal import anchor
from app.temporal.extract import Segment, extract

NOW = dt.datetime(2026, 10, 19, tzinfo=dt.timezone.utc)


@pytest.mark.parametrize("value, source, expected", [
    ("D:20150320121500+08'00'", "pdf_creation_date", "2015-03-20T12:15:00+08:00"),
    ("D:20150320121500Z", "pdf_creation_date", "2015-03-20T12:15:00+00:00"),
    ("D:20150320", "pdf_creation_date", "2015-03-20T00:00:00+08:00"),          # no zone: the deployment's
    ("2015:03:20 12:15:00", "exif_datetime_original", "2015-03-20T12:15:00+08:00"),
    ("2015-03-20T04:15:00Z", "container_creation_time", "2015-03-20T04:15:00+00:00"),
    ("2015-03-20T12:15:00", "docx_core_created", "2015-03-20T12:15:00+08:00"),
])
def test_metadata_forms(value, source, expected):
    assert anchor.validate(value, source, "Asia/Taipei", NOW) == (expected, None)


@pytest.mark.parametrize("value, source, reason", [
    ("1970-01-01T00:00:00Z", "container_creation_time", "placeholder_capture_time"),    # Unix epoch
    ("1904-01-01T00:00:00Z", "container_creation_time", "placeholder_capture_time"),    # QuickTime epoch
    ("1980-01-01T00:00:00", "docx_core_created", "placeholder_capture_time"),           # DOS / ZIP epoch
    ("0000:00:00 00:00:00", "exif_datetime_original", "unparseable_capture_time"),
    ("2031-01-01T00:00:00Z", "pdf_creation_date", "implausible_capture_time"),          # the future
    ("1850-06-01T00:00:00Z", "pdf_creation_date", "implausible_capture_time"),
    ("D:20151345", "pdf_creation_date", "unparseable_capture_time"),                    # month 13
    ("2015-03-20", "upload_time", "unknown_capture_time_source"),                       # never the upload
])
def test_refused(value, source, reason):
    assert anchor.validate(value, source, "UTC", NOW) == (None, reason)


def test_nothing_supplied_is_not_an_error():
    assert anchor.validate(None, None) == (None, None)


def test_anchor_day_is_the_deployments_calendar_day():
    assert anchor.anchor_day("2015-03-20T20:30:00+00:00", "Asia/Taipei") == "2015-03-21"


# ---------------------------------------------------------------- the extractor

TEXT = "Google released the phone two years ago. The team shipped an update yesterday. 去年公司推出了平板。"


def _reply(*items):
    return {"c1": {"facts": [{"entity": e, "relation": r, "value": v, "date_text": d} for e, r, v, d in items]}}


REPLY = _reply(("Google", "released", "the phone", "two years ago"),
               ("The team", "shipped", "an update", "yesterday"),
               ("公司", "推出了", "平板", "去年"))


def test_deictic_phrases_are_dated_from_the_publish_day():
    accepted, _ = extract([Segment("c1", TEXT)], replies=REPLY, published="2015-03-20")
    got = {c.date_text: (c.time_start, c.time_end, c.kind) for c in accepted if c.origin == "llm"}
    assert got == {"two years ago": ("2013", "2013", "deictic"),
                   "yesterday": ("2015-03-19", "2015-03-19", "deictic"),
                   "去年": ("2014", "2014", "deictic")}


def test_without_an_anchor_they_stay_unresolved():
    accepted, unresolved = extract([Segment("c1", TEXT)], replies=REPLY)
    assert not [c for c in accepted if c.origin == "llm"]
    assert {c.date_text for c in unresolved} >= {"two years ago", "yesterday", "去年"}


def test_explicit_dates_ignore_the_anchor_and_deictic_ones_count_for_less():
    text = "Google acquired Android in July 2005. Google released the phone two years ago."
    reply = _reply(("Google", "acquired", "Android", "July 2005"), ("Google", "released", "the phone", "two years ago"))
    accepted, _ = extract([Segment("c1", text)], replies=reply, published="2015-03-20")
    by = {c.date_text: c for c in accepted}
    assert (by["July 2005"].time_start, by["July 2005"].kind) == ("2005-07", "explicit")
    assert by["two years ago"].confidence < by["July 2005"].confidence


# ---------------------------------------------------------------- the consumer hand-off

def _doc(cid, idx, **extra):
    return {"id": cid, "payload": {"text": TEXT, "type": "documents", "chunk_index": idx, "version_id": "v1",
                                   "branch_id": "u1", **extra}}


def test_consumer_passes_the_validated_anchor_to_the_job():
    entries = [_doc("c0", 0), _doc("c1", 1, capture_time="D:20150320121500+08'00'",
                                    capture_time_source="pdf_creation_date")]
    _, _, got = _temporal_segments(entries, "u1", "v1", indexed={"c0", "c1"})[("u1", "v1")]
    assert (got.time, got.source, got.reason) == ("2015-03-20T12:15:00+08:00", "pdf_creation_date", None)


def test_consumer_reports_why_an_anchor_was_refused():
    entries = [_doc("c0", 0, capture_time="1970-01-01T00:00:00Z", capture_time_source="container_creation_time")]
    _, _, got = _temporal_segments(entries, "u1", "v1", indexed={"c0"})[("u1", "v1")]
    assert (got.time, got.reason) == (None, "placeholder_capture_time")


# ---------------------------------------------------------------- recordings and photos (P13, step 2)

def test_quicktime_offsets_without_a_colon():
    assert anchor.validate("2015-03-20T12:15:00+0800", "container_creation_time", "UTC", NOW) == (
        "2015-03-20T12:15:00+08:00", None)
    assert anchor.validate("2015-03-20T04:15:00.000000Z", "container_creation_time", "UTC", NOW) == (
        "2015-03-20T04:15:00+00:00", None)


def test_a_video_moment_carries_the_recording_time_to_the_job():
    from app.services.kafka_consumer import _stable_chunk_id, _tag_entry_type
    moment = _tag_entry_type({"payload": {"moment_index": 0, "asr_text": "We met yesterday.", "start_time": 0,
                                          "end_time": 10, "capture_time": "2015-03-20T12:15:00+0800",
                                          "capture_time_source": "container_creation_time"}}, "videos")
    cid = _stable_chunk_id(moment["payload"], "v9")
    _, _, got = _temporal_segments([moment], "u1", "v9", indexed={cid})[("u1", "v9")]
    assert (got.time, got.source) == ("2015-03-20T12:15:00+08:00", "container_creation_time")


def _photo(**extra):
    base = {"type": "images", "filename": "IMG_0042.jpg", "version_id": "p1", "branch_id": "u1"}
    return [{"id": "s", "payload": {**base, "embedding_type": "summary", "summary": "Two people at a harbour.", **extra}},
            {"id": "t", "payload": {**base, "embedding_type": "text", "text": "", **extra}}]


def test_a_photo_with_exif_becomes_an_observation():
    from app.services.kafka_consumer import _photo_observations
    obs = _photo_observations(_photo(capture_time="2015:03:20 12:15:00", capture_time_source="exif_datetime_original"),
                              "u1", "p1")
    name, caption, raw, got = obs[("u1", "p1")]
    assert (name, caption, raw) == ("IMG_0042.jpg", "Two people at a harbour.", "2015:03:20 12:15:00")
    assert got.time == "2015-03-20T12:15:00+08:00" and got.reason is None       # camera local time, deployment zone


def test_a_photo_without_exif_or_with_a_placeholder():
    from app.services.kafka_consumer import _photo_observations
    assert _photo_observations(_photo(), "u1", "p1") == {}
    obs = _photo_observations(_photo(capture_time="1970:01:01 00:00:00", capture_time_source="exif_datetime_original"),
                              "u1", "p1")
    assert (obs[("u1", "p1")][3].time, obs[("u1", "p1")][3].reason) == (None, "placeholder_capture_time")
