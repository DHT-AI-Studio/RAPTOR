"""When a recording was made, as its container says (temporal plan P13).

Module 25 dates the recording's deictic phrases ("yesterday", "last week") from this, after validating it.
Dropped verbatim into Module 09's and Module 11's orchestrators -- separate containers with no shared
package, so the file is duplicated rather than imported. Keep the two copies identical.

Read with ffprobe, in this order (first usable one wins):
    com.apple.quicktime.creationdate   QuickTime / iPhone: local time with its offset (most accurate)
    creation_time                      MP4 / MOV / M4A container, format or first stream (UTC)
    date + creation_time (time only)   WAV broadcast extension (BWF): ffprobe reports origination_date as
                                       `date` and origination_time as a time-only `creation_time`
An ID3 "date" on its own is deliberately not used: it is a release date, not when the recording was made.

Never raises: a file without a readable date is processed exactly as before, with no anchor.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess  # nosec B404 -- fixed ffprobe argv, no shell

logger = logging.getLogger(__name__)

SOURCE = "container_creation_time"
_OFFSET = re.compile(r"([+-]\d{2})(\d{2})$")
_TIME_ONLY = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _tags(path: str) -> list[dict]:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", path]
    out = subprocess.run(cmd, check=False, capture_output=True, timeout=60)  # nosec B603 -- fixed argv
    if out.returncode != 0:
        return []
    data = json.loads(out.stdout or b"{}")
    return [(data.get("format") or {}).get("tags") or {}] + [s.get("tags") or {} for s in data.get("streams") or []]


def read_capture_time(path: str) -> tuple[str | None, str | None]:
    """(the recording time as the container states it, ISO 8601, SOURCE) or (None, None)."""
    try:
        tags = [{k.lower(): str(v) for k, v in t.items()} for t in _tags(path)]
    except Exception as exc:  # noqa: BLE001 -- ffprobe missing, timeout, odd output: the file only loses its date; logged
        logger.warning("could not read the creation time of %s: %s", path, exc)
        return None, None
    for t in tags:
        if t.get("com.apple.quicktime.creationdate"):
            return _OFFSET.sub(r"\1:\2", t["com.apple.quicktime.creationdate"].strip()), SOURCE   # +0800 -> +08:00
    for t in tags:
        created = (t.get("creation_time") or "").strip()
        if created and not _TIME_ONLY.match(created):
            return created, SOURCE
        day = (t.get("date") or "").strip()
        if created and _FULL_DATE.match(day):                    # BWF as ffprobe reports it
            return f"{day}T{created}", SOURCE
    for t in tags:
        if t.get("origination_date"):
            day = t["origination_date"].strip().replace(":", "-")
            clock = (t.get("origination_time") or "00:00:00").strip().replace("-", ":")
            return f"{day}T{clock}", SOURCE
    return None, None
