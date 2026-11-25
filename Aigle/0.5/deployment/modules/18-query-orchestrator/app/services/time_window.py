"""Canonical query-window values (TMP-3 / temporal plan D3).

signal_extractor turns "2011年" / "1990年代" / "20世紀" into calendar days (naive
datetimes at midnight). Sent as `2011-01-01T00:00:00` .. `2011-12-31T00:00:00` they
had no zone, and the upper bound stopped at midnight on the last day -- Module 25
compared them against stored dates as raw strings, which only worked by accident.

Here each bound becomes an instant with an explicit offset: the start of the first
day and the last millisecond of the last day, in the deployment's timezone, so the
window covers whole days and every consumer reads the same instants.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


def window_start(day: datetime | None, tz_name: str) -> str | None:
    """First day of the window -> its first instant, e.g. 2011-01-01T00:00:00.000+08:00."""
    if day is None:
        return None
    start = datetime.combine(day.date(), time.min, tzinfo=ZoneInfo(tz_name))
    return start.isoformat(timespec="milliseconds")


def window_end(day: datetime | None, tz_name: str) -> str | None:
    """Last day of the window -> its last millisecond, e.g. 2011-12-31T23:59:59.999+08:00."""
    if day is None:
        return None
    next_day = datetime.combine(day.date() + timedelta(days=1), time.min, tzinfo=ZoneInfo(tz_name))
    return (next_day - timedelta(milliseconds=1)).isoformat(timespec="milliseconds")
