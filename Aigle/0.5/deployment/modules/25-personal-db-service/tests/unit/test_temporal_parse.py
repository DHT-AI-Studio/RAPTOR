"""TMP-7: `parse` -- the time a question asks about, read by Module 25's own normaliser (temporal plan §5.11 / P02)."""
import asyncio

import pytest
from app.routers.temporal import ParseRequest, temporal_parse
from app.temporal.query import parse_query
from fastapi import HTTPException

TZ = "Asia/Taipei"


def _window(q, anchor=None):
    r = parse_query(q, TZ, anchor)
    w = r["window"] or {}
    return r["status"], w.get("start"), w.get("end"), r["match"]


@pytest.mark.parametrize("q, start, end, match", [
    # the TMP-2 AC, now served by the normaliser
    ("What happened after 2011?", "2012-01-01T00:00:00.000+08:00", None, "inside"),        # strict: 2011 excluded
    ("before 2013", None, "2012-12-31T23:59:59.999+08:00", "inside"),
    ("since 2010", "2010-01-01T00:00:00.000+08:00", None, "inside"),
    ("until 2012", None, "2012-12-31T23:59:59.999+08:00", "inside"),
    ("2011年以後發生什麼", "2012-01-01T00:00:00.000+08:00", None, "inside"),
    ("2013年之前", None, "2012-12-31T23:59:59.999+08:00", "inside"),
    ("2010年以來", "2010-01-01T00:00:00.000+08:00", None, "inside"),
    ("到2012年為止", None, "2012-12-31T23:59:59.999+08:00", "inside"),
    ("After the 1990s", "2000-01-01T00:00:00.000+08:00", None, "inside"),
    ("1990年代以後", "2000-01-01T00:00:00.000+08:00", None, "inside"),
    # a period named in full: what was going on then; "within": the fact inside it
    ("What was Hugo Barra doing in 2010?", "2010-01-01T00:00:00.000+08:00", "2010-12-31T23:59:59.999+08:00", "overlaps"),
    ("within 2010", "2010-01-01T00:00:00.000+08:00", "2010-12-31T23:59:59.999+08:00", "inside"),
    ("What did Samsung do between 2010 and 2016?", "2010-01-01T00:00:00.000+08:00", "2016-12-31T23:59:59.999+08:00",
     "overlaps"),
    ("LG phone 2010-2012", "2010-01-01T00:00:00.000+08:00", "2012-12-31T23:59:59.999+08:00", "overlaps"),
    ("2010年到2016年", "2010-01-01T00:00:00.000+08:00", "2016-12-31T23:59:59.999+08:00", "overlaps"),
    # periods named apart, out of order: still the earliest to the latest, never an inverted window
    ("What happened in 2016 and 2010?", "2010-01-01T00:00:00.000+08:00", "2016-12-31T23:59:59.999+08:00", "overlaps"),
    ("2016年和2010年發生什麼", "2010-01-01T00:00:00.000+08:00", "2016-12-31T23:59:59.999+08:00", "overlaps"),
    ("June 2012 and 2011", "2011-01-01T00:00:00.000+08:00", "2012-06-30T23:59:59.999+08:00", "overlaps"),
    # forms Module 18's regex never read: the normaliser's grammar
    ("June 2010", "2010-06-01T00:00:00.000+08:00", "2010-06-30T23:59:59.999+08:00", "overlaps"),
    ("2010 年 6 月", "2010-06-01T00:00:00.000+08:00", "2010-06-30T23:59:59.999+08:00", "overlaps"),
    ("民國 100 年", "2011-01-01T00:00:00.000+08:00", "2011-12-31T23:59:59.999+08:00", "overlaps"),
    ("early 2011", "2011-01-01T00:00:00.000+08:00", "2011-04-30T23:59:59.999+08:00", "overlaps"),
    ("since 2010-11", "2010-11-01T00:00:00.000+08:00", None, "inside"),
])
def test_windows(q, start, end, match):
    assert _window(q) == ("RESOLVED", start, end, match)


def test_millisecond_bounds_are_given_too():
    w = parse_query("after 2011", "UTC")["window"]
    assert (w["start"], w["start_ms"], w["end_ms"]) == ("2012-01-01T00:00:00.000+00:00", 1325376000000, None)


@pytest.mark.parametrize("q, reason", [
    ("2011-13 release", "invalid_calendar"),            # a calendar error, never the year 2011
    ("What happened after 2011-02-30?", "invalid_calendar"),
    ("What happened yesterday?", "missing_anchor"),      # deictic: never guessed without the query's time
    ("去年發生什麼", "missing_anchor"),
    ("between 2016 and 2010", "reversed_range"),        # one range, written backwards
    ("2016年到2010年", "reversed_range"),
])
def test_a_date_the_grammar_cannot_read_is_unresolved(q, reason):
    r = parse_query(q, TZ)
    assert (r["status"], r["reason"], r["window"]) == ("UNRESOLVED", reason, None)


def test_deictic_phrases_are_read_from_the_anchor():
    anchor = "2015-03-20T10:00:00+08:00"
    assert _window("What happened last year?", anchor)[1:3] == ("2014-01-01T00:00:00.000+08:00",
                                                                "2014-12-31T23:59:59.999+08:00")
    assert _window("去年發生什麼", anchor)[1:3] == ("2014-01-01T00:00:00.000+08:00", "2014-12-31T23:59:59.999+08:00")


@pytest.mark.parametrize("q", ["What is Android?", "iPhone 20110 sales", "Galaxy S2010", "after that"])
def test_no_temporal_condition(q):
    r = parse_query(q, TZ)
    assert (r["status"], r["window"], r["reference"]) == ("NO_TEMPORAL", None, None)


@pytest.mark.parametrize("q, ref, relations", [
    ("What happened after Google acquired Android?", "Google acquired Android", ["after"]),
    ("Who joined immediately after Android was founded?", "Android was founded", ["met-by"]),
    ("Google收購Android之後發生了什麼？", "Google收購Android", ["after"]),
])
def test_reference_events(q, ref, relations):
    r = parse_query(q, TZ)
    assert (r["status"], r["reference"], r["reference_relations"], r["window"]) == ("RESOLVED", ref, relations, None)


def test_keywords_leave_out_the_date_and_the_question_words():
    assert parse_query("What did Samsung do between 2010 and 2016?", TZ)["keywords"] == ["Samsung"]
    assert parse_query("What did Rubin do after the acquisition of Android Inc.", TZ)["keywords"] == ["Rubin"]


def test_route_reads_in_the_deployment_zone_and_checks_its_inputs():
    out = asyncio.run(temporal_parse(ParseRequest(text="after 2011")))
    assert out["window"]["start"] == "2012-01-01T00:00:00.000+08:00" and out["timezone"] == "Asia/Taipei"
    for bad in (ParseRequest(text="after 2011", timezone="Mars/Olympus"),
                ParseRequest(text="last year", anchor="2015-03-20")):          # an anchor needs its offset
        with pytest.raises(HTTPException) as exc:
            asyncio.run(temporal_parse(bad))
        assert exc.value.status_code == 400
