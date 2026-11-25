"""Grammar coverage the temporal plan (§5.2) lists and the normaliser now reads (version 1.1.0).

- calendar forms: early/mid/late YYYY, "the summer of YYYY", "the third quarter of YYYY", the early/late
  decades, "between X and Y", centuries, 年初/年中/年底, 月初/月中/月底 and 上/中/下旬, "2010–11";
- anaphoric phrases, dated from the event they refer back to (`anchor`, §5.3 / P05): "two years later",
  "the same day", "within a year", "less than six months earlier", "July of that year", "two years after
  <clause>", N年後 -- never from the publish date, which stays the anchor of deictic phrases only
  ("two years ago", 去年, 三年前, yesterday).
"""
import pytest
from app.temporal import normalize

from .test_temporal_grammar import milliseconds


def _days(result):
    from datetime import datetime, timezone
    def day(ms):
        return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    return day(result["start"]["lo_ms"]), day(result["end"]["hi_ms"]), result["precision"]


def _interval(text):
    r = normalize(text)
    return r["start"]["lo_ms"], r["end"]["hi_ms"]


@pytest.mark.parametrize("text, lo, hi, precision", [
    ("early 2011", "2011-01-01", "2011-04-30", "part_year"),
    ("mid 2011", "2011-05-01", "2011-08-31", "part_year"),
    ("mid-2011", "2011-05-01", "2011-08-31", "part_year"),
    ("late 2021", "2021-09-01", "2021-12-31", "part_year"),          # the plan's own §5.1 example
    ("the summer of 2019", "2019-06-01", "2019-08-31", "season"),    # = "summer 2019"
    ("summer of 2019", "2019-06-01", "2019-08-31", "season"),
    ("the third quarter of 2016", "2016-07-01", "2016-09-30", "quarter"),
    ("the 2nd quarter 2016", "2016-04-01", "2016-06-30", "quarter"),
    ("the early 2000s", "2000-01-01", "2003-12-31", "decade"),
    ("the mid 1990s", "1994-01-01", "1996-12-31", "decade"),
    ("the late 1990s", "1997-01-01", "1999-12-31", "decade"),
    ("between 2008 and 2013", "2008-01-01", "2013-12-31", "year"),
    ("between May 2008 and June 2009", "2008-05-01", "2009-06-30", "month"),
    ("the 20th century", "1901-01-01", "2000-12-31", "century"),     # as Module 18 reads 20世紀
    ("20世紀", "1901-01-01", "2000-12-31", "century"),
    ("二十世紀", "1901-01-01", "2000-12-31", "century"),
    ("2011年初", "2011-01-01", "2011-04-30", "part_year"),
    ("2011年中", "2011-05-01", "2011-08-31", "part_year"),
    ("2011年底", "2011-09-01", "2011-12-31", "part_year"),
    ("2011年5月初", "2011-05-01", "2011-05-10", "part_month"),
    ("2011年5月中", "2011-05-11", "2011-05-20", "part_month"),
    ("2011年5月底", "2011-05-21", "2011-05-31", "part_month"),
    ("2024年2月下旬", "2024-02-21", "2024-02-29", "part_month"),       # leap year
    ("2011年二月上旬", "2011-02-01", "2011-02-10", "part_month"),
    ("2010–11", "2010-01-01", "2011-12-31", "year"),                 # en dash: a year range
])
def test_calendar_forms(text, lo, hi, precision):
    assert _days(normalize(text)) == (lo, hi, precision)


@pytest.mark.parametrize("text", ["2010-11", "2024-02"])
def test_iso_year_month_stays_a_month(text):
    """ISO YYYY-MM is a month; only an en/em dash makes "2010–11" a year range."""
    lo, hi, precision = _days(normalize(text))
    assert precision == "month" and lo[:7] == hi[:7] == text


@pytest.mark.parametrize("text, reason", [
    ("2011-13", "invalid_calendar"),          # a hyphen with an impossible month is a calendar error, not a range
    ("2011–09", "reversed_range"),            # en dash, end before start
    ("the 99th century", "implausible_century"),
])
def test_never_guessed(text, reason):
    r = normalize(text)
    assert (r["extent"], r["reason"]) == ("UNRESOLVED", reason)


# --- anaphoric phrases: dated from the event they refer back to

@pytest.mark.parametrize("text, anchor, lo, hi, precision", [
    ("July of that year", "2005", "2005-07-01", "2005-07-31", "month"),
    ("In September", "June 2014", "2014-09-01", "2014-09-30", "month"),
    ("a week later", "August 2015", "2015-08-08", "2015-09-07", "day"),
    ("Within a year", "November 5, 2007", "2007-11-05", "2008-11-05", "day"),
    ("Less than six months earlier", "August 2013", "2013-02-01", "2013-08-01", "day"),
    ("less than a year later", "2010", "2010-12-31", "2011-12-31", "day"),
    ("On the same day", "August 22, 2019", "2019-08-22", "2019-08-22", "day"),
    ("two years later", "2005", "2007-01-01", "2007-12-31", "year"),
    ("a year earlier", "May 2013", "2012-05-01", "2012-05-31", "month"),
    ("two years after Google bought it", "2014", "2012-01-01", "2012-12-31", "year"),   # P05: inverted, 2012
    ("a year before the merger", "2014", "2015-01-01", "2015-12-31", "year"),
    ("3年後", "2005年", "2008-01-01", "2008-12-31", "year"),
    ("三個月後", "2005年7月", "2005-10-01", "2005-10-31", "month"),
    ("one month later", "2024-01-31", "2024-02-29", "2024-02-29", "day"),   # month-end policy: clamped
])
def test_anaphoric_phrases_use_the_anchor(text, anchor, lo, hi, precision):
    r = normalize(text, anchor=_interval(anchor), source_published="2019-05-01")   # the publish date is not used
    assert _days(r) == (lo, hi, precision)
    assert r["anchor"]["type"] in ("previous_event", "same_sentence_event")


@pytest.mark.parametrize("text", ["two years later", "the same day", "within a year", "3年後", "July of that year"])
def test_anaphoric_phrase_without_an_anchor_is_unresolved(text):
    """P05: no previous event, no date -- never the publish date instead."""
    r = normalize(text, source_published="2013-08-15")
    assert (r["extent"], r["reason"]) == ("UNRESOLVED", "missing_anchor")


@pytest.mark.parametrize("text, reason", [
    ("about two years later", "approximate_relative"),
    ("more than a year after", "approximate_relative"),
    ("later", "relation_only:after"),
    ("afterwards", "relation_only:after"),
    ("後來", "relation_only:after"),
    ("at the time", "relation_only:during"),
    ("當時", "relation_only:during"),
])
def test_no_date_from_vague_or_relation_only_phrases(text, reason):
    r = normalize(text, anchor=_interval("2005"))
    assert (r["extent"], r["reason"]) == ("UNRESOLVED", reason)


def test_invalid_anchor_is_unresolved():
    r = normalize("two years later", anchor=(milliseconds("2006-01-01T00:00:00Z"), milliseconds("2005-01-01T00:00:00Z")))
    assert r["reason"] == "invalid_anchor"


@pytest.mark.parametrize("text, lo", [
    ("two years ago", "2011-01-01"),
    ("三年前", "2010-01-01"),
    ("去年", "2012-01-01"),
    ("yesterday", "2013-08-14"),
])
def test_deictic_phrases_still_use_the_publish_date(text, lo):
    r = normalize(text, source_published="2013-08-15", anchor=_interval("2005"))   # the anchor is not used
    assert _days(r)[0] == lo and r["anchor"]["type"] == "source_published"
