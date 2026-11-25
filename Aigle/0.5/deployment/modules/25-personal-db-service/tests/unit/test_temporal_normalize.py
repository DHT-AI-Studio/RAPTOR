"""Canonical temporal dates (TMP-3): temporal_normalize.py and date_utils' precision.

Pure stdlib, no ArcadeDB: every form a date can arrive in becomes a UTC epoch-ms range
at the precision it was given, or a recorded reason -- never a raw string to compare.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from app.services.date_utils import extract_first_date
from app.services.temporal_normalize import (
    NORMALIZER_VERSION,
    DateParseError,
    iso_utc,
    normalize_fact_dates,
    parse_bound,
)

TZ = "Asia/Taipei"


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


# --- parse_bound: each form -> a range at its own precision

@pytest.mark.parametrize("raw, lo, hi_exclusive, precision", [
    ("2011", "2011-01-01T00:00:00+08:00", "2012-01-01T00:00:00+08:00", "year"),
    ("2011-05", "2011-05-01T00:00:00+08:00", "2011-06-01T00:00:00+08:00", "month"),
    ("2011-12", "2011-12-01T00:00:00+08:00", "2012-01-01T00:00:00+08:00", "month"),
    ("2011-5", "2011-05-01T00:00:00+08:00", "2011-06-01T00:00:00+08:00", "month"),
    ("2012-02-29", "2012-02-29T00:00:00+08:00", "2012-03-01T00:00:00+08:00", "day"),
    # TMP-5's grammar, now the module's only parser: forms the old ISO-only parser quarantined
    ("2011年", "2011-01-01T00:00:00+08:00", "2012-01-01T00:00:00+08:00", "year"),
    ("民國100年", "2011-01-01T00:00:00+08:00", "2012-01-01T00:00:00+08:00", "year"),
    ("2011年5月", "2011-05-01T00:00:00+08:00", "2011-06-01T00:00:00+08:00", "month"),
    ("2011年上半年", "2011-01-01T00:00:00+08:00", "2011-07-01T00:00:00+08:00", "half_year"),
    ("Q3 2011", "2011-07-01T00:00:00+08:00", "2011-10-01T00:00:00+08:00", "quarter"),
    ("May 2011", "2011-05-01T00:00:00+08:00", "2011-06-01T00:00:00+08:00", "month"),
    ("the 1990s", "1990-01-01T00:00:00+08:00", "2000-01-01T00:00:00+08:00", "decade"),
    ("2010-2012", "2010-01-01T00:00:00+08:00", "2013-01-01T00:00:00+08:00", "year"),
    ("since 2008", "2008-01-01T00:00:00+08:00", "2009-01-01T00:00:00+08:00", "year"),  # end ongoing: the start's range
])
def test_calendar_forms_are_ranges_in_the_interpreting_zone(raw, lo, hi_exclusive, precision):
    b = parse_bound(raw, TZ)
    assert (b.lo_ms, b.hi_ms, b.precision, b.raw) == (_ms(lo), _ms(hi_exclusive) - 1, precision, raw)


@pytest.mark.parametrize("raw, instant, precision, width", [
    ("2011-05-03T10:00", "2011-05-03T10:00:00+08:00", "minute", 0),
    ("2011-05-03T10:00:00", "2011-05-03T10:00:00+08:00", "second", 0),
    ("2011-05-03 10:00:00", "2011-05-03T10:00:00+08:00", "second", 0),
    ("2011-05-03T10:00:00.250", "2011-05-03T10:00:00.250+08:00", "millisecond", 0),
    ("2011-05-03T10:00:00.25", "2011-05-03T10:00:00.250+08:00", "millisecond", 0),
    ("2011-05-03T10:00:00.5Z", "2011-05-03T10:00:00.500+00:00", "millisecond", 0),
    ("2011-05-03T10:00:00Z", "2011-05-03T10:00:00+00:00", "second", 0),
    ("2011-05-03T10:00:00+09:00", "2011-05-03T10:00:00+09:00", "second", 0),
])
def test_date_times_keep_their_zone_or_take_the_default(raw, instant, precision, width):
    # a timestamp is a point (TMP-5): width is always 0
    b = parse_bound(raw, TZ)
    assert (b.lo_ms, b.hi_ms, b.precision) == (_ms(instant), _ms(instant) + width, precision)


def test_calendar_projection_is_the_date_in_the_interpreting_zone():
    # 2011-05-03 20:00 UTC is already 4 May in Taipei
    b = parse_bound("2011-05-03T20:00:00Z", TZ)
    assert (b.lo_date, b.hi_date) == ("2011-05-04", "2011-05-04")
    y = parse_bound("2011", TZ)
    assert (y.lo_date, y.hi_date) == ("2011-01-01", "2011-12-31")


@pytest.mark.parametrize("raw", [None, "", "  ", "null", "None", "N/A", "unknown", "-"])
def test_placeholders_mean_no_date(raw):
    assert parse_bound(raw, TZ) is None


@pytest.mark.parametrize("raw", ["2011-13", "2011-02-30", "2011-13-01", "0000", "2011-05-03T25:00",
                                 "2011-05-03T10:61", "20110503", "yesterday",
                                 "2011/05/03", "2011.5.3", "2011-05-03T10:00:00+0900", "about 2011"])
def test_anything_else_is_an_error_with_a_reason(raw):
    with pytest.raises(DateParseError):
        parse_bound(raw, TZ)


# --- normalize_fact_dates: every property a TemporalFact stores

def test_fact_with_both_endpoints():
    d = normalize_fact_dates("2011", "2013-06", TZ)
    assert d["s_state"] == d["e_state"] == "known"
    assert (d["s_lo"], d["s_hi"]) == (_ms("2011-01-01T00:00:00+08:00"), _ms("2012-01-01T00:00:00+08:00") - 1)
    assert (d["e_lo"], d["e_hi"]) == (_ms("2013-06-01T00:00:00+08:00"), _ms("2013-07-01T00:00:00+08:00") - 1)
    assert (d["time_start"], d["time_end"]) == ("2011-01-01", "2013-06-30")   # v1 projections
    assert (d["time_start_raw"], d["time_end_raw"]) == ("2011", "2013-06")
    assert d["precision"] == "year"                                            # the coarser endpoint
    assert (d["timezone"], d["normalizer_version"], d["migration_version"]) == (TZ, NORMALIZER_VERSION,
                                                                               NORMALIZER_VERSION)
    assert d["quarantine_reason"] is None and d["requires_reextraction"] is False


def test_missing_endpoint_is_unknown_not_ongoing():
    d = normalize_fact_dates("2011-05-03", None, TZ)
    assert (d["s_state"], d["e_state"]) == ("known", "unknown")
    assert d["e_lo"] is d["e_hi"] is d["time_end"] is d["time_end_raw"] is None
    assert d["precision"] == "day"


def test_unusable_endpoint_is_quarantined_and_its_raw_value_kept():
    d = normalize_fact_dates("2023-13-01", "2024", TZ)
    assert d["s_state"] == "unknown" and d["s_lo"] is None and d["time_start"] is None
    assert d["time_start_raw"] == "2023-13-01"
    assert d["e_state"] == "known"
    assert "time_start '2023-13-01'" in d["quarantine_reason"]
    assert d["precision"] == "unknown" and d["requires_reextraction"] is True   # P03


def test_reversed_interval_is_kept_but_flagged():
    d = normalize_fact_dates("2015", "2012", TZ)
    assert d["s_state"] == d["e_state"] == "known"
    assert d["quarantine_reason"] == "time_start is after time_end"
    assert d["requires_reextraction"] is True and d["precision"] == "year"     # dates parsed, but contradict


def test_same_coarse_date_on_both_sides_is_not_reversed():
    assert normalize_fact_dates("2011", "2011", TZ)["quarantine_reason"] is None


def test_no_dates_at_all():
    d = normalize_fact_dates(None, "null", TZ)
    assert d["s_state"] == d["e_state"] == "unknown"
    assert d["precision"] is None and d["quarantine_reason"] is None
    assert d["requires_reextraction"] is False        # no dates is not an error
    assert d["time_end_raw"] == "null"


def test_normalising_twice_is_stable():
    once = normalize_fact_dates("2011-05", "2011-05-03T10:00:00Z", TZ)
    again = normalize_fact_dates(once["time_start_raw"], once["time_end_raw"], TZ)
    assert once == again


def test_window_bound_rejects_a_bad_date():
    with pytest.raises(DateParseError):
        parse_bound("2011-02-30", TZ)


def test_iso_utc():
    assert iso_utc(_ms("2011-01-01T00:00:00+08:00")) == "2010-12-31T16:00:00.000+00:00"
    assert iso_utc(None) is None


# --- date_utils keeps the precision the text gave (D4: "2011年" is not 1 January)

@pytest.mark.parametrize("text, expected", [
    ("他在1987年2月21日出生", "1987-02-21"),
    ("2017年3月上市", "2017-03"),
    ("1995年成立", "1995"),
    ("民國76年3月", "1987-03"),
    ("民國76年", "1987"),
    ("沒有日期", None),
])
def test_extracted_dates_keep_their_precision(text, expected):
    assert extract_first_date(text) == expected
    if expected:
        assert parse_bound(expected, TZ).precision == {10: "day", 7: "month", 4: "year"}[len(expected)]


# --- padded dates (TMP-8 follow-up): a midnight timestamp is a day; a padded period is doubtful

@pytest.mark.parametrize("raw, tz, lo, lo_date, hi_date", [
    # the day is the written one, in the timestamp's own offset; the projections are in the
    # interpreting timezone like every other value, so a UTC day can span two local dates
    ("2012-12-31T00:00:00Z", TZ, "2012-12-31T00:00:00+00:00", "2012-12-31", "2013-01-01"),
    ("2012-12-31T00:00:00Z", "America/New_York", "2012-12-31T00:00:00+00:00", "2012-12-30", "2012-12-31"),
    ("2012-12-31T00:00:00Z", "UTC", "2012-12-31T00:00:00+00:00", "2012-12-31", "2012-12-31"),
    ("2012-12-31T00:00:00.000+08:00", TZ, "2012-12-31T00:00:00+08:00", "2012-12-31", "2012-12-31"),
    ("2012-12-31 00:00", TZ, "2012-12-31T00:00:00+08:00", "2012-12-31", "2012-12-31"),   # no offset: read in TZ
])
def test_midnight_timestamp_is_the_whole_day(raw, tz, lo, lo_date, hi_date):
    b = parse_bound(raw, tz)
    assert (b.precision, b.lo_date, b.hi_date) == ("day", lo_date, hi_date)
    assert (b.lo_ms, b.hi_ms - b.lo_ms) == (_ms(lo), 86_400_000 - 1)


def test_a_timestamp_with_a_time_of_day_stays_a_point():
    b = parse_bound("2014-06-25T10:00:00", TZ)
    assert (b.precision, b.lo_ms) == ("second", b.hi_ms)


def test_midnight_on_an_impossible_date_is_still_rejected():
    with pytest.raises(DateParseError):
        parse_bound("2023-02-30T00:00:00Z", TZ)


@pytest.mark.parametrize("start, end, kind", [
    ("2012-01-01T00:00:00Z", "2012-12-31T00:00:00Z", "years"),     # "2012", written by the Temporal Demo
    ("2008-01-01T00:00:00Z", "2013-12-31T00:00:00Z", "years"),     # "2008 to 2013"
    ("2013-08-01T00:00:00Z", "2013-08-31T00:00:00Z", "months"),    # "August 2013"
    ("2010-03-01", "2010-06-30", "months"),
])
def test_padded_period_is_kept_but_marked_for_reextraction(start, end, kind):
    d = normalize_fact_dates(start, end, "UTC")
    assert (d["s_state"], d["e_state"], d["time_start"], d["time_end"]) == ("known", "known", start[:10], end[:10])
    assert d["precision"] == "unknown" and d["requires_reextraction"] is True
    assert f"whole {kind}" in d["reextraction_reason"] and d["quarantine_reason"] is None   # doubtful, not unusable


@pytest.mark.parametrize("tz", [TZ, "America/New_York", "UTC"])
def test_padding_is_judged_on_the_written_dates_in_any_timezone(tz):
    """In Asia/Taipei the UTC day 2012-12-31 ends on 1 January, in New York it starts on 30 December:
    the projections move, the evidence does not."""
    d = normalize_fact_dates("2012-01-01T00:00:00Z", "2012-12-31T00:00:00Z", tz)
    assert d["requires_reextraction"] is True and "whole years" in d["reextraction_reason"]
    lone = normalize_fact_dates("2011-01-01T00:00:00Z", None, tz)
    assert lone["requires_reextraction"] is True and "whole year 2011" in lone["reextraction_reason"]
    assert normalize_fact_dates("2011-05-03T00:00:00Z", None, tz)["requires_reextraction"] is False


def test_lone_first_of_january_is_marked_but_a_lone_first_of_month_is_not():
    year = normalize_fact_dates("2011-01-01", None, TZ)
    assert (year["precision"], year["requires_reextraction"]) == ("unknown", True)
    assert "whole year 2011" in year["reextraction_reason"]
    month = normalize_fact_dates("2013-08-01", None, TZ)
    assert (month["precision"], month["requires_reextraction"], month["reextraction_reason"]) == ("day", False, None)


@pytest.mark.parametrize("start, end", [
    ("2011", None), ("2011-05", "2011-06"), ("2012", "2012"),      # coarse values carry their own precision
    ("2010-01-01", "2010-12-30"),                                  # does not end on a month's last day
    ("2011-05-03", "2011-05-31"),                                  # does not start on the 1st
    ("2011-01-01T10:00:00Z", None),                                # a real time of day
])
def test_values_that_carry_their_precision_are_not_marked(start, end):
    d = normalize_fact_dates(start, end, TZ)
    assert d["requires_reextraction"] is False and d["reextraction_reason"] is None


def test_a_quarantined_row_keeps_its_quarantine_reason_only():
    d = normalize_fact_dates("2023-13-01", "2024-12-31", TZ)
    assert d["quarantine_reason"] and d["reextraction_reason"] is None and d["requires_reextraction"] is True
