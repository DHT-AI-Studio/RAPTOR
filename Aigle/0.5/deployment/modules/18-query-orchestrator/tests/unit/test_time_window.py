"""Canonical query-window values (TMP-3): whole days, explicit offset."""
# signal_extractor emits naive calendar days; these tests feed it exactly that
from datetime import datetime

from app.services.time_window import window_end, window_start


def test_year_window_is_whole_days_with_offset():
    assert window_start(datetime(2011, 1, 1), "Asia/Taipei") == "2011-01-01T00:00:00.000+08:00"
    assert window_end(datetime(2011, 12, 31), "Asia/Taipei") == "2011-12-31T23:59:59.999+08:00"


def test_offset_follows_the_zone_including_dst():
    assert window_start(datetime(2011, 7, 1), "Europe/London") == "2011-07-01T00:00:00.000+01:00"
    assert window_end(datetime(2011, 12, 31), "UTC") == "2011-12-31T23:59:59.999+00:00"


def test_end_of_month_and_leap_day():
    assert window_end(datetime(2012, 2, 29), "Asia/Taipei") == "2012-02-29T23:59:59.999+08:00"


def test_no_bound():
    assert window_start(None, "Asia/Taipei") is None
    assert window_end(None, "Asia/Taipei") is None
