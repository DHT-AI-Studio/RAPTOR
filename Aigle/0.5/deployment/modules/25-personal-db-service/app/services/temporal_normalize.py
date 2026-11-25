"""Canonical temporal dates (TMP-3; temporal plan D3/D4, §5.1, §5.8, P03).

Every date a TemporalFact stores, and every date a TKG query filters by, goes
through here, so nothing is ever compared as a raw string again. Before this,
Module 18 sent `2011-01-01T00:00:00` while facts held `2011-01-01` or `2011`,
and lexicographically "2011-01-01" < "2011-01-01T00:00:00": a fact starting on
the window's first day failed its own `>= :ts` check (D3).

A date becomes a **range** in UTC epoch milliseconds, at the precision the text
actually gave. "2011" is all of 2011 -- [2011-01-01 00:00, 2011-12-31 23:59:59.999]
in the interpreting timezone -- never the single day 2011-01-01 (D4).

Parsing itself is app.temporal.normalize (TMP-5's EN/ZH grammar): one grammar for
the whole module. This file maps its result onto the storage contract -- one
stored endpoint as a range, the s_lo/s_hi/e_lo/e_hi columns, the legacy
projections -- and does not parse dates itself.

Padded dates (TMP-8 follow-up). A writer that only knew "2012" or "August 2013" often
stores a full date or a midnight timestamp -- "2012-01-01T00:00:00Z .. 2012-12-31T00:00:00Z"
(the Temporal Demo does, because the index API has no precision field). Two rules:
  - a timestamp at exactly 00:00 says nothing about the time of day: it is read as that
    whole calendar day (in its own offset), not as one millisecond -- the wider reading,
    never a narrower one, so a value is never made more precise than its evidence;
  - a pair that looks like a padded period -- the 1st of a month to the last day of a
    month, or a lone start on 1 January -- cannot establish its own precision: the range
    is kept as written, `precision` becomes "unknown" and the row is marked for
    re-extraction with `reextraction_reason` (P03). A lone 1st of a month is not marked:
    real events on the 1st are too common to call every one of them doubtful.

Pure stdlib (no ArcadeDB, no settings import) so it is unit-tested in CI.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.temporal import NORMALIZER_VERSION as GRAMMAR_VERSION
from app.temporal import normalize

# Bump when this mapping changes; the grammar's own version is part of it, so a grammar
# change re-normalises stored rows too (the backfill re-runs rows whose
# TemporalFact.migration_version differs).
NORMALIZER_VERSION = f"tmp8-2+grammar-{GRAMMAR_VERSION}"    # tmp8-2: midnight = day; padded periods

# Every fact so far is a calendar ("world") date; media-time facts would get their
# recording's own timeline (temporal plan §5.1). Values on different timelines are
# never compared.
WORLD_TIMELINE = "world"
WORLD_AXIS = "world"                    # derived from the timeline: world | media
TIME_UNIT = "ms"                        # unit of s_lo / s_hi / e_lo / e_hi

# The grammar's precision names, coarsest last; a fact's `precision` is the coarser of
# its two endpoints. "mixed": a range whose ends have different precisions.
PRECISIONS = ("millisecond", "second", "minute", "day", "part_month", "month", "quarter", "season",
              "part_year", "half_year", "year", "decade", "century", "mixed")

# Literal placeholders an LLM writes for "no date" (D4): absent, not an error.
_ABSENT = {"", "null", "none", "nil", "nan", "n/a", "na", "unknown", "undefined", "-"}


class DateParseError(ValueError):
    """The value is not a date the grammar resolves; the message is its reason."""


@dataclass(frozen=True)
class Bound:
    """One endpoint as a range of UTC epoch ms, inclusive at both ends."""
    lo_ms: int
    hi_ms: int
    precision: str
    raw: str
    lo_date: str            # calendar date of lo / hi in the interpreting timezone,
    hi_date: str            # used for the legacy time_start / time_end projections


def _local_date(ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=tz).date().isoformat()


# "2012-12-31T00:00:00Z", "2012-12-31 00:00:00.000+08:00", "2012-12-31T00:00": a date at midnight
_MIDNIGHT = re.compile(r"(\d{4}-\d{2}-\d{2})[T ]00:00(?::00(?:\.0+)?)?(Z|[+-]\d{2}:?\d{2})?", re.IGNORECASE)


def _midnight_day(raw: str, tz_name: str) -> Bound | None:
    """A timestamp at exactly 00:00 -> that whole calendar day, in the timestamp's own offset
    (the interpreting timezone when it has none). None when `raw` is not one."""
    m = _MIDNIGHT.fullmatch(raw)
    if not m:
        return None
    try:
        day = date.fromisoformat(m[1])
    except ValueError as exc:
        raise DateParseError("invalid_calendar") from exc
    offset = m[2]
    if offset is None:
        tz: Any = ZoneInfo(tz_name)
    elif offset.upper() == "Z":
        tz = timezone.utc
    else:
        sign = -1 if offset[0] == "-" else 1
        digits = offset[1:].replace(":", "")
        tz = timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
    lo = datetime.combine(day, time.min, tzinfo=tz)
    hi = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz) - timedelta(milliseconds=1)
    lo_ms, hi_ms = int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)
    # projections in the interpreting timezone, like every other Bound: a UTC day can span two local dates
    local = ZoneInfo(tz_name)
    return Bound(lo_ms, hi_ms, "day", raw, _local_date(lo_ms, local), _local_date(hi_ms, local))


def parse_bound(value: Any, tz_name: str) -> Bound | None:
    """One stored date value -> the range it can denote; None when it means "no date".
    Raises DateParseError, with the grammar's reason, for anything it doesn't resolve --
    the caller records the reason rather than guessing.

    Whatever the grammar resolves becomes [start of its start, end of its end]: "2011",
    "2011年", "民國100年", "2011年上半年", "Q3 2011", "2010-2012"; a timestamp is a point. A phrase with no end
    ("since 2008", end ongoing) gives its start's range. Calendar dates and zone-less
    times are read in `tz_name`.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if raw.lower() in _ABSENT:
        return None
    day = _midnight_day(raw, tz_name)
    if day is not None:
        return day
    result = normalize(raw, tz_name)
    if result["extent"] == "UNRESOLVED":
        raise DateParseError(result.get("reason") or "unresolved")
    start, end = result["start"], result["end"]
    lo = start["lo_ms"]
    hi = end["hi_ms"] if end["hi_ms"] is not None else start["hi_ms"]
    tz = ZoneInfo(tz_name)
    return Bound(lo, hi, result["precision"], raw, _local_date(lo, tz), _local_date(hi, tz))


def normalize_fact_dates(time_start: Any, time_end: Any, tz_name: str) -> dict[str, Any]:
    """A fact's two endpoints -> every TemporalFact date property to store.

    - s_lo/s_hi, e_lo/e_hi: each endpoint's range (UTC epoch ms); NULL when unknown
    - s_state / e_state: "known" or "unknown" -- a missing endpoint is unknown,
      never ongoing (§5.1)
    - time_start / time_end: legacy projections for v1 readers -- the first day of
      the start's range and the last day of the end's range, YYYY-MM-DD. Lossy for
      coarse dates by design; the ranges carry the precision
    - time_start_raw / time_end_raw: the values as given, never altered
    - quarantine_reason: why an endpoint could not be used (it is then unknown) or
      the interval is reversed; the row is kept, never dropped
    - requires_reextraction: set on every quarantined row (P03); precision is then
      "unknown" when an endpoint could not be parsed. Also set, with precision "unknown"
      and `reextraction_reason`, on a padded-looking period (see the module docstring)
    """
    reasons = []
    unparseable = False
    bounds: dict[str, Bound | None] = {}
    for side, value in (("start", time_start), ("end", time_end)):
        try:
            bounds[side] = parse_bound(value, tz_name)
        except DateParseError as exc:
            bounds[side] = None
            unparseable = True
            reasons.append(f"time_{side} {value!r}: {exc}")
    s, e = bounds["start"], bounds["end"]
    if s and e and s.lo_ms > e.hi_ms:
        reasons.append("time_start is after time_end")
    known = [b.precision for b in (s, e) if b]
    # P03: where the evidence can't establish precision, precision is "unknown" and the
    # fact is marked for re-extraction. That is certain only for a quarantined row: an
    # unparseable endpoint has no precision at all; a reversed interval has precision
    # but contradicts itself.
    if unparseable:
        precision = "unknown"
    else:
        precision = max(known, key=PRECISIONS.index) if known else None
    padded = None if reasons else _padded_period(s, e)
    if padded:
        precision = "unknown"
    return {
        "time_start": s.lo_date if s else None,
        "time_end": e.hi_date if e else None,
        "time_start_raw": None if time_start is None else str(time_start),
        "time_end_raw": None if time_end is None else str(time_end),
        "s_state": "known" if s else "unknown",
        "e_state": "known" if e else "unknown",
        "s_lo": s.lo_ms if s else None,
        "s_hi": s.hi_ms if s else None,
        "e_lo": e.lo_ms if e else None,
        "e_hi": e.hi_ms if e else None,
        "precision": precision,
        "timezone": tz_name,
        "timeline_id": WORLD_TIMELINE,
        "axis": WORLD_AXIS,
        "time_unit": TIME_UNIT,
        "normalizer_version": NORMALIZER_VERSION,
        "migration_version": NORMALIZER_VERSION,
        "quarantine_reason": "; ".join(reasons) or None,
        "requires_reextraction": bool(reasons) or bool(padded),
        "reextraction_reason": padded,
    }


_WRITTEN_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _written_day(b: Bound) -> date:
    """The calendar date the value was written as -- the evidence a padding check is about. Not
    lo_date / hi_date: those are projections into the interpreting timezone, where the UTC day
    "2012-12-31T00:00:00Z" ends on 1 January in Asia/Taipei and starts on 30 December in New York."""
    m = _WRITTEN_DAY.match(b.raw)
    return date(int(m[1]), int(m[2]), int(m[3])) if m else date.fromisoformat(b.lo_date)


def _padded_period(s: Bound | None, e: Bound | None) -> str | None:
    """Why the stored values may be a coarser period padded to days, or None.

    Only day-precision endpoints can be padding; a value that already says "2012" or
    "2013-08" carries its own precision."""
    if s is None or s.precision != "day":
        return None
    start = _written_day(s)
    if start.day != 1:
        return None
    if e is None:
        if start.month == 1:
            return f"time_start {s.raw!r} is 1 January: may stand for the whole year {start.year}"
        return None
    if e.precision != "day":
        return None
    end = _written_day(e)
    if end.day != calendar.monthrange(end.year, end.month)[1] or end < start:
        return None
    kind = "years" if (start.month, end.month) == (1, 12) else "months"
    return (f"{s.raw!r} .. {e.raw!r} runs from the 1st of a month to the last day of a month: "
            f"may stand for whole {kind}, not exact days")


def iso_utc(ms: int | None) -> str | None:
    """UTC epoch ms -> ISO 8601 with an explicit zone, for logs and responses."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")


def _range_text(lo_ms: int, hi_ms: int, tz: ZoneInfo) -> str:
    """One endpoint's range at the precision it has: "2024" (a whole year), "2024-07" (a whole
    month), "2024-07-15" (a day), "2019-06..2019-08" (some time in those months), or the
    instants of a timestamp. Never more precise than the range: a month is not its 1st day."""
    lo = datetime.fromtimestamp(lo_ms / 1000, tz)
    end = datetime.fromtimestamp((hi_ms + 1) / 1000, tz)          # first instant after the range
    if lo.time() == time.min and end.time() == time.min:           # whole calendar days
        last = (end - timedelta(days=1)).date()
        if lo.day == 1 and end.day == 1:                            # whole months
            if lo.month == 1 and end.month == 1:                    # whole years
                a, b = str(lo.year), str(last.year)
            else:
                a, b = f"{lo:%Y-%m}", f"{last:%Y-%m}"
        else:
            a, b = lo.date().isoformat(), last.isoformat()
    else:
        a, b = lo.isoformat(timespec="seconds"), datetime.fromtimestamp(hi_ms / 1000, tz).isoformat(timespec="seconds")
    return a if a == b else f"{a}..{b}"


def describe_dates(start: tuple[int, int] | None, end: tuple[int, int] | None, tz_name: str) -> str | None:
    """A fact's dates as an answer may state them (temporal plan D8): each endpoint at its own
    precision, e.g. "2024-07", "2008 to 2013", "2010-06-01 (no end date stated)". `start` /
    `end` are (lo_ms, hi_ms) ranges, None when unknown; None when neither is known.

    The legacy time_start / time_end projections pad a month or a year to a day ("2024-07" is
    stored as 2024-07-01), and a model reading them states a day the source never gave."""
    tz = ZoneInfo(tz_name)
    s = _range_text(*start, tz) if start else None
    e = _range_text(*end, tz) if end else None
    if s and e:
        return s if s == e else f"{s} to {e}"
    if s:
        return f"{s} (no end date stated)"
    if e:
        return f"until {e} (no start date stated)"
    return None
