"""Conservative EN/ZH calendar grammar with UTC integer-millisecond bounds.

Calendar phrases denote closed periods. Resolved period endpoints are exact;
uncertainty in a 'since YEAR' start is retained as the whole year's domain.
No clock, relative phrase or short year acquires an implicit current-date anchor.
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from datetime import timezone as utc_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

NORMALIZER_VERSION = "1.1.1"          # 1.1.1: spaces around 年 / 月 / 日 (「2010 年 6 月」) read as without them
_UTC = utc_timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=_UTC)
_DIGITS = dict(zip("零〇一二三四五六七八九兩", "001234567892"))
_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if index}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if index})
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_ZH_NUMBER = r"[0-9零〇一二三四五六七八九十百千兩]+"
# A space between a number and a Chinese date unit, or after 民國 / 第: 「2010 年 6 月」, 「民國 113 年」, 「第 3 季」.
# Never between two numbers.
_ZH_UNIT_SPACE = re.compile(r"(?<=[0-9零〇一二三四五六七八九十百千兩])\s+(?=[年月日號季])"
                            r"|(?<=[年月國第])\s+(?=[0-9零〇一二三四五六七八九十百千兩])")
_ISO_TIMESTAMP = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?"
_ISO_DATE = r"\d{4}-\d{1,2}-\d{1,2}"
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
# Thirds of a year (temporal plan §5.1: "late 2021" = [2021-09-01, 2021-12-31]); 年初/年中/年底 alike
_YEAR_PARTS = {"early": (1, 4), "mid": (5, 4), "late": (9, 4), "初": (1, 4), "中": (5, 4), "底": (9, 4)}
# Thirds of a decade, as the Temporal Demo reads "the early 2000s"
_DECADE_PARTS = {"early": (0, 4), "mid": (4, 3), "late": (7, 3)}
# 旬: 月初/上旬 = days 1-10, 月中/中旬 = 11-20, 月底/下旬 = 21 to the month's end
_MONTH_PARTS = {"初": 1, "上旬": 1, "中": 11, "中旬": 11, "底": 21, "下旬": 21}


class _Unresolved(ValueError):
    def __init__(self, reason: str):
        self.reason = reason


def _number(text: str) -> int:
    if text.isdigit():
        return int(text)
    if text.lower() in _NUMBERS:
        return _NUMBERS[text.lower()]
    if not any(unit in text for unit in "十百千"):
        return int("".join(_DIGITS[c] for c in text))
    total = current = 0
    for char in text:
        if char in "十百千":
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            current = 0
        else:
            current = int(_DIGITS[char])
    return total + current


def _ms(value: datetime) -> int:
    delta = value.astimezone(_UTC) - _EPOCH
    return delta.days * 86400000 + delta.seconds * 1000 + delta.microseconds // 1000


def _local(value: datetime, zone: ZoneInfo) -> datetime:
    """Reject nonexistent and ambiguous civil clock times, including DST folds."""
    first = value.replace(tzinfo=zone, fold=0)
    second = value.replace(tzinfo=zone, fold=1)
    valid = [item for item in (first, second)
             if item.astimezone(_UTC).astimezone(zone).replace(tzinfo=None) == value]
    if not valid:
        raise _Unresolved("nonexistent_local_time")
    if len(valid) == 2 and first.utcoffset() != second.utcoffset():
        raise _Unresolved("ambiguous_local_time")
    return valid[0]


def _period(year: int, month: int, day: int, unit: str, zone: ZoneInfo, count: int = 1):
    begin = datetime(year, month, day)
    if unit == "year":
        finish = datetime(year + count, 1, 1)
    elif unit == "month":
        total = year * 12 + month - 1 + count
        finish = datetime(total // 12, total % 12 + 1, 1)
    else:
        finish = begin + timedelta(days=count)
    return _ms(_local(begin, zone)), _ms(_local(finish, zone)) - 1


def _fromisoformat(text: str) -> datetime:
    """Python 3.10's fromisoformat takes only 3 or 6 fraction digits; pad to 6 so every
    version reads ".25" the way 3.11 does."""
    text = re.sub(r"\.(\d{1,6})", lambda m: "." + m[1].ljust(6, "0"), text.replace("Z", "+00:00"), count=1)
    return datetime.fromisoformat(text)


def _timestamp(text: str, zone: ZoneInfo):
    value = _fromisoformat(text)
    if value.tzinfo is None:
        value = _local(value, zone)
    return _ms(value)


def _calendar_phrase(text: str, zone: ZoneInfo):
    """Return lo, hi, precision for a full calendar phrase."""
    text = text.strip()
    lower = text.lower()
    if re.fullmatch(_ISO_TIMESTAMP, text):
        value = _timestamp(text, zone)
        return value, value, "millisecond" if "." in text else "second" if text.count(":") >= 2 else "minute"
    if re.fullmatch(_ISO_DATE, text):
        year, month, day = map(int, text.split("-"))
        return (*_period(year, month, day, "day", zone), "day")
    if match := re.fullmatch(r"(\d{4})-(\d{1,2})", text):
        year, month = map(int, match.groups())
        return (*_period(year, month, 1, "month", zone), "month")
    if re.fullmatch(r"\d{4}", text):
        return (*_period(int(text), 1, 1, "year", zone), "year")
    if match := re.fullmatch(r"(?:the\s+)?(\d{3}0)(?:s|'s)", lower):
        return (*_period(int(match[1]), 1, 1, "year", zone, 10), "decade")
    if match := re.fullmatch(r"(?:q([1-4])\s+(\d{4})|(\d{4})\s+q([1-4]))", lower):
        quarter = int(match[1] or match[4])
        year = int(match[2] or match[3])
        return (*_period(year, (quarter - 1) * 3 + 1, 1, "month", zone, 3), "quarter")
    if match := re.fullmatch(r"(spring|summer|autumn|fall|winter)\s+(\d{4})", lower):
        month = {"spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12}[match[1]]
        return (*_period(int(match[2]), month, 1, "month", zone, 3), "season")
    if (match := re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:,)?\s+(\d{4})", lower)) and match[1] in _MONTHS:
        return (*_period(int(match[3]), _MONTHS[match[1]], int(match[2]), "day", zone), "day")
    if (match := re.fullmatch(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", lower)) and match[2] in _MONTHS:
        return (*_period(int(match[3]), _MONTHS[match[2]], int(match[1]), "day", zone), "day")
    if (match := re.fullmatch(r"([a-z]+)\s+(\d{4})", lower)) and match[1] in _MONTHS:
        return (*_period(int(match[2]), _MONTHS[match[1]], 1, "month", zone), "month")
    if match := re.fullmatch(r"(?:民國)?(" + _ZH_NUMBER + r")年(?:(" + _ZH_NUMBER + r")月)?(?:(" + _ZH_NUMBER + r")[日號])?", text):
        year = _number(match[1]) + (1911 if text.startswith("民國") else 0)
        if text.startswith("民國") and _number(match[1]) < 1:
            raise ValueError("invalid ROC year")
        month, day = _number(match[2]) if match[2] else 1, _number(match[3]) if match[3] else 1
        if match[3] and not match[2]:
            raise _Unresolved("incomplete_date")
        precision = "day" if match[3] else "month" if match[2] else "year"
        return (*_period(year, month, day, precision, zone), precision)
    if match := re.fullmatch(r"(" + _ZH_NUMBER + r")年代", text):
        year = _number(match[1])
        if year < 1000 or year % 10:
            raise _Unresolved("ambiguous_decade")
        return (*_period(year, 1, 1, "year", zone, 10), "decade")
    if match := re.fullmatch(r"(\d{4})年([上下])半年", text):
        return (*_period(int(match[1]), 1 if match[2] == "上" else 7, 1, "month", zone, 6), "half_year")
    if match := re.fullmatch(r"(\d{4})年第?(" + _ZH_NUMBER + r")(?:季|季度)", text):
        quarter = _number(match[2])
        if not 1 <= quarter <= 4:
            raise ValueError("invalid quarter")
        return (*_period(int(match[1]), (quarter - 1) * 3 + 1, 1, "month", zone, 3), "quarter")
    return _calendar_parts(text, lower, zone)


def _calendar_parts(text: str, lower: str, zone: ZoneInfo):
    """Parts of periods and the longer forms of §5.2: early/mid/late YYYY, "the summer of YYYY",
    "the third quarter of YYYY", the early/late decades, centuries, 年初/年中/年底, 月初/月中/月底,
    and short year ranges written with an en dash ("2010–11")."""
    if match := re.fullmatch(r"(early|mid|late)[-\s]+(\d{4})", lower):
        month, count = _YEAR_PARTS[match[1]]
        return (*_period(int(match[2]), month, 1, "month", zone, count), "part_year")
    if match := re.fullmatch(r"(?:the\s+)?(spring|summer|autumn|fall|winter)\s+of\s+(\d{4})", lower):
        return _calendar_phrase(f"{match[1]} {match[2]}", zone)
    if match := re.fullmatch(r"(?:the\s+)?(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(\d{4})", lower):
        return _calendar_phrase(f"Q{_ORDINALS[match[1]]} {match[2]}", zone)
    if match := re.fullmatch(r"(?:the\s+)?(early|mid|late)[-\s]*(\d{3}0)(?:s|'s)", lower):
        offset, count = _DECADE_PARTS[match[1]]
        return (*_period(int(match[2]) + offset, 1, 1, "year", zone, count), "decade")
    if (match := re.fullmatch(r"(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\s+century", lower)) \
            or (match := re.fullmatch(r"(" + _ZH_NUMBER + r")世紀", text)):
        century = _number(match[1])
        if not 1 <= century <= 30:
            raise _Unresolved("implausible_century")
        # the 20th century is 1901-2000, as Module 18 reads "20世紀"
        return (*_period((century - 1) * 100 + 1, 1, 1, "year", zone, 100), "century")
    if match := re.fullmatch(r"(\d{4})年(初|中|底)", text):
        month, count = _YEAR_PARTS[match[2]]
        return (*_period(int(match[1]), month, 1, "month", zone, count), "part_year")
    if match := re.fullmatch(r"(\d{4})年(" + _ZH_NUMBER + r")月(初|中|底|上旬|中旬|下旬)", text):
        year, month, first = int(match[1]), _number(match[2]), _MONTH_PARTS[match[3]]
        last = 10 if first == 1 else 20 if first == 11 else calendar.monthrange(year, month)[1]
        return (*_period(year, month, first, "day", zone, last - first + 1), "part_month")
    if match := re.fullmatch(r"(\d{2})(\d{2})\s*[\u2013\u2014]\s*(\d{2})", text):
        # "2010–11" with an en/em dash is a year range; "2010-11" with a hyphen stays ISO year-month
        first, second = int(match[1] + match[2]), int(match[1] + match[3])
        if second <= first:
            raise _Unresolved("reversed_range")
        return _period(first, 1, 1, "year", zone)[0], _period(second, 1, 1, "year", zone)[1], "year"
    raise _Unresolved("unsupported_expression")


def _anchor(source: str | datetime | None, zone: ZoneInfo) -> datetime:
    if source is None:
        raise _Unresolved("missing_anchor")
    if isinstance(source, datetime):
        return source.astimezone(zone) if source.tzinfo else _local(source, zone)
    value = _fromisoformat(source)
    return value.astimezone(zone) if value.tzinfo else _local(value, zone)


def _shift(anchor: datetime, count: int, unit: str) -> datetime:
    if unit == "year":
        return anchor.replace(year=anchor.year + count)
    if unit == "month":
        total = anchor.year * 12 + anchor.month - 1 + count
        return anchor.replace(year=total // 12, month=total % 12 + 1)
    return anchor + timedelta(days=count * (7 if unit == "week" else 1))


_UNITS = r"(years?|months?|weeks?|days?)"
_ZH_UNITS = {"年": "year", "個月": "month", "月": "month", "週": "week", "星期": "week", "天": "day", "日": "day"}


def _clamped_shift(value: datetime, count: int, unit: str) -> datetime:
    """Calendar arithmetic with the documented month-end policy (§5.2): a day the target month does not
    have is clamped to its last day -- 31 January + 1 month = 28/29 February."""
    if unit in ("year", "month"):
        total = value.year * 12 + value.month - 1 + count * (12 if unit == "year" else 1)
        year, month = total // 12, total % 12 + 1
        return value.replace(year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1]))
    return value + timedelta(days=count * (7 if unit == "week" else 1))


def _anchor_interval(anchor, zone: ZoneInfo) -> tuple[datetime, datetime]:
    """The previous event's interval -- (lo_ms, hi_ms) or {"lo_ms", "hi_ms"} in UTC ms -- as local times."""
    lo, hi = (anchor["lo_ms"], anchor["hi_ms"]) if isinstance(anchor, dict) else anchor
    if lo is None or hi is None or lo > hi:
        raise _Unresolved("invalid_anchor")
    def local(ms):
        return (_EPOCH + timedelta(milliseconds=ms)).astimezone(zone).replace(tzinfo=None)
    return local(lo), local(hi)


def _span_precision(lo: datetime, hi: datetime) -> str:
    """Precision of a computed interval from its shape: whole years, whole months, or days."""
    end = hi + timedelta(milliseconds=1)
    if (lo.month, lo.day, lo.time(), end.month, end.day, end.time()) == (1, 1, datetime.min.time(), 1, 1, datetime.min.time()):
        return "year"
    if (lo.day, lo.time(), end.day, end.time()) == (1, datetime.min.time(), 1, datetime.min.time()):
        return "month"
    return "day"


def _anaphoric(text: str, zone: ZoneInfo, anchor):
    """Phrases dated from another event (§5.3, P05): the previous one ("two years later", "the same day",
    "within a year", "less than six months earlier", "July of that year") or, for "two years after
    <clause>", the explicit date of its own sentence. *anchor* is that event's interval. Returns None
    when the phrase is not one of these; raises when it is but has no anchor or would need a guess."""
    lower = re.sub(r"\s+", " ", text.strip().lower())
    lower = re.sub(r"^(?:on|in|at) (?=the same day)", "", lower)
    words = r"(an?|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,3})"
    if re.fullmatch(r"(?:later|afterwards?|subsequently|後來|之後|此後)", lower):
        raise _Unresolved("relation_only:after")
    if re.fullmatch(r"(?:at (?:that|the) time|then|當時|那時)", lower):
        raise _Unresolved("relation_only:during")
    if re.fullmatch(r"(?:about|around|almost|nearly|roughly|more than|over) " + words + r" " + _UNITS
                    + r" (?:later|after|earlier|before)(?: .*)?", lower):
        raise _Unresolved("approximate_relative")
    form = None
    if m := re.fullmatch(words + r" " + _UNITS + r" (?:later|afterwards?)", lower):
        form = ("shift", _number("1" if m[1] in ("a", "an") else m[1]), m[2].rstrip("s"))
    elif m := re.fullmatch(words + r" " + _UNITS + r" (?:earlier|before)", lower):
        form = ("shift", -_number("1" if m[1] in ("a", "an") else m[1]), m[2].rstrip("s"))
    elif m := re.fullmatch(words + r" " + _UNITS + r" (after|before) (?!\d{4}$)(?!.* in \d{4}$).+", lower):
        # "sold in 2014, two years after Google bought it": the clause is n units before its sentence's date
        n = _number("1" if m[1] in ("a", "an") else m[1])
        form = ("inverted", -n if m[3] == "after" else n, m[2].rstrip("s"))
    elif m := re.fullmatch(r"less than " + words + r" " + _UNITS + r" (earlier|before|later|after)", lower):
        form = ("less_than", _number("1" if m[1] in ("a", "an") else m[1]) * (-1 if m[3] in ("earlier", "before") else 1),
                m[2].rstrip("s"))
    elif m := re.fullmatch(r"within " + words + r" " + _UNITS, lower):
        form = ("within", _number("1" if m[1] in ("a", "an") else m[1]), m[2].rstrip("s"))
    elif lower in ("the same day", "same day", "that day", "同一天", "當天", "同日"):
        form = ("same", 0, "day")
    elif ((m := re.fullmatch(r"([a-z]+) of that year", lower))
          or (m := re.fullmatch(r"(?:in |during |by )?([a-z]+)", lower))) and m[1] in _MONTHS and len(m[1]) > 2:
        form = ("month", _MONTHS[m[1]], "month")         # "July of that year", "in September": the anchor's year
    elif m := re.fullmatch(r"(" + _ZH_NUMBER + r")(個月|年|週|星期|天|日)(?:之)?後", text.strip()):
        form = ("shift", _number(m[1]), _ZH_UNITS[m[2]])
    if form is None:
        return None
    if anchor is None:
        raise _Unresolved("missing_anchor")
    lo, hi = _anchor_interval(anchor, zone)
    kind, count, unit = form
    if kind in ("shift", "inverted"):
        lo, hi = _clamped_shift(lo, count, unit), _clamped_shift(hi, count, unit)
    elif kind == "less_than":
        lo, hi = (_clamped_shift(lo, count, unit), lo) if count < 0 else (hi, _clamped_shift(hi, count, unit))
    elif kind == "within":
        hi = _clamped_shift(hi, count, unit)
    elif kind == "month":
        lo_ms, hi_ms = _period(lo.year, count, 1, "month", zone)
        return lo_ms, hi_ms, "month", {"type": "previous_event", "year": lo.year, "evidence": text}
    lo_ms, hi_ms = _ms(_local(lo, zone)), _ms(_local(hi, zone))
    info = {"type": "same_sentence_event" if kind == "inverted" else "previous_event",
            "lo_ms": anchor["lo_ms"] if isinstance(anchor, dict) else anchor[0],
            "hi_ms": anchor["hi_ms"] if isinstance(anchor, dict) else anchor[1], "evidence": text}
    return lo_ms, hi_ms, _span_precision(lo, hi), info


def _relative(text: str, zone: ZoneInfo, source, anchor=None):
    found = _anaphoric(text, zone, anchor)
    if found is not None:
        return found
    lower = text.lower()
    # This ticket's explicit inverted same-sentence anchor describes the later event's year.
    inverted = re.fullmatch(r"(\w+) years? after (.+) in (\d{4})", lower)
    if inverted:
        amount = _number(inverted[1])
        year = int(inverted[3]) - amount
        return (*_period(year, 1, 1, "year", zone), "year",
                {"type": "inverted_same_sentence", "year": int(inverted[3]), "evidence": text})
    direct = re.fullmatch(r"(\w+) years? (after|before) (\d{4})", lower)
    if direct:
        year = int(direct[3]) + _number(direct[1]) * (1 if direct[2] == "after" else -1)
        return (*_period(year, 1, 1, "year", zone), "year",
                {"type": "same_sentence", "year": int(direct[3]), "evidence": text})
    simple = {"today": (0, "day"), "yesterday": (-1, "day"), "tomorrow": (1, "day"),
              "今天": (0, "day"), "昨天": (-1, "day"), "明天": (1, "day"),
              "今年": (0, "year"), "去年": (-1, "year"), "明年": (1, "year"),
              "上個月": (-1, "month"), "下個月": (1, "month")}
    if lower in simple:
        count, unit = simple[lower]
    elif match := re.fullmatch(r"(last|next|this) (year|month|day)", lower):
        count, unit = {"last": -1, "next": 1, "this": 0}[match[1]], match[2]
    elif match := re.fullmatch(r"(\w+) (years?|months?|weeks?|days?) ago", lower):
        count, unit = -_number(match[1]), match[2].rstrip("s")
    elif match := re.fullmatch(r"(" + _ZH_NUMBER + r")([年月天日週])前", text):
        count = -_number(match[1])
        unit = {"年": "year", "月": "month", "天": "day", "日": "day", "週": "week"}[match[2]]
    else:
        raise _Unresolved("unsupported_expression")
    anchor = _anchor(source, zone)
    # A year/month phrase refers to the whole period, so the source's leap day
    # or month-end day must not make an otherwise valid target period invalid.
    period_anchor = (anchor.replace(month=1, day=1) if unit == "year"
                     else anchor.replace(day=1) if unit == "month" else anchor)
    shifted = _shift(period_anchor, count, unit)
    precision = "day" if unit == "week" else unit
    month = 1 if precision == "year" else shifted.month
    day = 1 if precision in {"year", "month"} else shifted.day
    return (*_period(shifted.year, month, day, precision, zone), precision,
            {"type": "source_published", "value": anchor.isoformat()})


def _endpoint(lo: int | None, hi: int | None, state: str = "resolved") -> dict:
    return {"state": state, "lo_ms": lo, "hi_ms": hi}


def normalize(text: str, timezone: str = "UTC", *, source_published: str | datetime | None = None,
              anchor: tuple[int, int] | dict | None = None) -> dict:
    """Normalize one phrase, preserving evidence and unresolved reasons.

    Args:
        text: Calendar phrase in EN/ZH or an embedded structured ISO date/timestamp.
        timezone: IANA civil timezone for calendar dates and clocks without offsets.
        source_published: Explicit reference for deictic phrases -- "yesterday", "last year",
            "two years ago", 去年, 三年前 -- for clocks and for ongoing confirmation.
        anchor: The interval of the event an anaphoric phrase refers back to -- the previous
            event for "two years later", "the same day", "within a year", "July of that year";
            the sentence's own explicit date for "two years after <clause>". (lo_ms, hi_ms) or
            {"lo_ms", "hi_ms"}, UTC ms. Never replaced by source_published (P05).
    """
    result = {"start": _endpoint(None, None, "unresolved"),
              "end": _endpoint(None, None, "unresolved"), "extent": "UNRESOLVED",
              "precision": None, "anchor": None, "raw": text, "timezone": timezone,
              "normalizer_version": NORMALIZER_VERSION}
    try:
        zone = ZoneInfo(timezone)
        phrase = _ZH_UNIT_SPACE.sub("", text.strip())          # result["raw"] keeps the text as written
        if not phrase:
            raise _Unresolved("empty_expression")
        anchor_info = None
        if match := re.fullmatch(r"(?:since\s+|自|從)(.+?)(?:起|以來)?", phrase, re.IGNORECASE):
            lo, hi, precision = _calendar_phrase(match[1], zone)
            result.update(start=_endpoint(lo, hi), end=_endpoint(None, None, "ongoing"),
                          extent="ongoing", precision=precision)
            if source_published is not None:
                published = _anchor(source_published, zone)
                if _ms(published) < lo:
                    raise _Unresolved("anchor_precedes_start")
                result["start"]["hi_ms"] = min(hi, _ms(published))
                result["confirmed_active_at"] = _ms(published)
                result["anchor"] = {"type": "source_published", "value": published.isoformat()}
            return result
        if match := (re.fullmatch(r"between\s+(.+?)\s+and\s+(.+)", phrase, re.IGNORECASE)
                     or re.fullmatch(r"(?:from\s+)?(.+?)(?:\s+(?:to|through|until)\s+|\s*[至到]\s*)(.+)", phrase, re.IGNORECASE)):
            lo, _, p1 = _calendar_phrase(match[1], zone)
            _, hi, p2 = _calendar_phrase(match[2], zone)
            precision = p1 if p1 == p2 else "mixed"
        elif match := re.fullmatch(r"(\d{4})[–—-](\d{4})", phrase):
            lo, _, _ = _calendar_phrase(match[1], zone)
            _, hi, _ = _calendar_phrase(match[2], zone)
            precision = "year"
        elif match := re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})[–-](\d{1,2}),?\s+(\d{4})", phrase):
            lo, _, _ = _calendar_phrase(f"{match[1]} {match[2]}, {match[4]}", zone)
            _, hi, _ = _calendar_phrase(f"{match[1]} {match[3]}, {match[4]}", zone)
            precision = "day"
        elif re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", phrase):
            published = _anchor(source_published, zone)
            clock = ":".join(f"{int(part):02d}" for part in phrase.split(":"))
            lo = hi = _timestamp(published.strftime("%Y-%m-%d") + "T" + clock, zone)
            precision = "second" if phrase.count(":") == 2 else "minute"
            anchor_info = {"type": "source_published", "value": published.isoformat()}
        else:
            # ISO evidence is parsed in any surrounding language, but multiple dates need a range grammar.
            matches = list(re.finditer(_ISO_TIMESTAMP + "|" + _ISO_DATE, phrase))
            if matches:
                if len(matches) != 1:
                    raise _Unresolved("multiple_temporal_spans")
                span = matches[0]
                before, after = phrase[:span.start()], phrase[span.end():]
                # Never salvage a valid prefix from a malformed ISO token.
                # Non-ASCII surrounding text and ordinary sentence punctuation
                # are allowed, while trailing digits/offsets/fractions are not.
                if (re.search(r"[A-Za-z0-9_]$", before)
                        or re.match(r"[A-Za-z0-9_:+/\-]|\.\d", after)
                        or (re.fullmatch(_ISO_DATE, span.group()) and re.match(r"\s+\d{1,2}:", after))):
                    raise _Unresolved("invalid_calendar")
                lo, hi, precision = _calendar_phrase(span.group(), zone)
            else:
                try:
                    lo, hi, precision = _calendar_phrase(phrase, zone)
                except _Unresolved as exc:
                    if exc.reason != "unsupported_expression":
                        raise
                    try:
                        lo, hi, precision, anchor_info = _relative(phrase, zone, source_published, anchor)
                    except _Unresolved as relative_exc:
                        if relative_exc.reason == "unsupported_expression":
                            known = re.search(r"[年月日天週季]|\b(?:year|month|day|early|late|around|sometime|ago|after|before|since)\b", phrase, re.IGNORECASE)
                            if not known:
                                raise _Unresolved("unsupported_language") from relative_exc
                        raise
        if lo > hi:
            raise _Unresolved("reversed_range")
        result.update(start=_endpoint(lo, lo), end=_endpoint(hi, hi),
                      extent="point" if lo == hi else "interval", precision=precision, anchor=anchor_info)
        return result
    except ZoneInfoNotFoundError:
        reason = "invalid_timezone"
    except _Unresolved as exc:
        reason = exc.reason
    except (ValueError, KeyError, OverflowError):
        reason = "invalid_calendar"
    result.update(start=_endpoint(None, None, "unresolved"), end=_endpoint(None, None, "unresolved"),
                  extent="UNRESOLVED", precision=None, anchor=None, reason=reason)
    result.pop("confirmed_active_at", None)
    return result
