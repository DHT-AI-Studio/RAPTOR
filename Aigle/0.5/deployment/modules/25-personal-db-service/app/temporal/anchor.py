"""Capture / publish time of a source: the anchor of its deictic phrases (temporal plan §5.2, P05, P13).

"Yesterday", "two years ago", "last month", 去年 mean nothing without the moment the source was written or
recorded. That moment comes from the file's own metadata, read by the worker that opened it:

    capture_time_source        where it came from                        what it means
    pdf_creation_date          PDF info dictionary /CreationDate         publish / authoring date
    docx_core_created          DOCX/PPTX docProps/core.xml created       publish / authoring date
    exif_datetime_original     image EXIF DateTimeOriginal               capture time
    container_creation_time    video / audio container creation_time     capture time

Metadata is often wrong or a placeholder, so a value is used only when it validates: it parses, it is not a
known default (the Unix and QuickTime epochs, the DOS/ZIP epoch, EXIF's all-zero date), and it lies between
1900 and tomorrow. Anything else leaves the source without an anchor -- its deictic phrases stay unresolved
(`missing_anchor`), never dated from the upload time (§9: upload anchoring is an explicit opt-in only).

Pure: no I/O.
"""
from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo

SOURCES = ("pdf_creation_date", "docx_core_created", "exif_datetime_original", "container_creation_time")

_PLACEHOLDERS = {dt.date(1970, 1, 1), dt.date(1904, 1, 1), dt.date(1980, 1, 1), dt.date(1601, 1, 1)}
_PDF = re.compile(r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(Z|[+-]\d{2}'?\d{2}'?)?")
_ISO_OFFSET = re.compile(r"(T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?[+-]\d{2})(\d{2})$")
_EXIF = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


def _parse(value: str, tz: str) -> dt.datetime | None:
    """The instant a metadata string names, or None. Naive values are read in *tz*."""
    v = value.strip()
    if m := _PDF.match(v):                                      # D:20150320121500+08'00'
        y, mo, d, h, mi, s, off = m.groups()
        if off and off != "Z":
            digits = off.replace("'", "")
            sign = 1 if digits[0] == "+" else -1
            zone: dt.tzinfo = dt.timezone(sign * dt.timedelta(hours=int(digits[1:3]), minutes=int(digits[3:5])))
        else:
            zone = dt.timezone.utc if off == "Z" else ZoneInfo(tz)
        try:
            return dt.datetime(int(y), int(mo or 1), int(d or 1), int(h or 0), int(mi or 0), int(s or 0), tzinfo=zone)
        except ValueError:
            return None
    if m := _EXIF.match(v):                                     # 2015:03:20 12:15:00 (no zone: local)
        try:
            return dt.datetime(*map(int, m.groups()), tzinfo=ZoneInfo(tz))
        except ValueError:
            return None
    iso = _ISO_OFFSET.sub(r"\1:\2", v.replace("Z", "+00:00"))     # "+0800" (QuickTime) -> "+08:00"
    try:
        t = dt.datetime.fromisoformat(iso)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=ZoneInfo(tz))


def validate(value: object, source: object, tz: str = "UTC",
             now: dt.datetime | None = None) -> tuple[str | None, str | None]:
    """(ISO 8601 instant, None) for a usable capture / publish time, else (None, reason)."""
    if value in (None, ""):
        return None, None
    if source not in SOURCES:
        return None, "unknown_capture_time_source"
    t = _parse(str(value), tz)
    if t is None:
        return None, "unparseable_capture_time"
    now = now or dt.datetime.now(dt.timezone.utc)
    if t.date() in _PLACEHOLDERS:
        return None, "placeholder_capture_time"
    if t.year < 1900 or t > now + dt.timedelta(days=1):
        return None, "implausible_capture_time"
    return t.isoformat(), None


def anchor_day(iso: str, tz: str) -> str:
    """The calendar day of an anchor instant in the deployment's zone -- what "today" means for the source."""
    return dt.datetime.fromisoformat(iso).astimezone(ZoneInfo(tz)).date().isoformat()
