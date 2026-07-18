# Deterministic date extraction (Fix 1B).
#
# The LLM relation extractor often leaves time_start = null even when the
# source text clearly states a date. These helpers find dates in the text and
# attach them to timeless relations, so they can become TemporalFacts.
#
# Pure-stdlib (re only) on purpose, so it can be unit-tested without the LLM /
# Neo4j / httpx stack: `python3 test_date_utils.py`.

import re
from typing import Any, Dict, List, Optional

# 1987年2月21日 / 2017年3月 / 1987 年 2 月 21 日
_RE_FULL_DATE = re.compile(r"(1[89]\d{2}|20\d{2})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")
# 民國76年 / 民國 76 年 3 月
_RE_MINGUO = re.compile(r"民國\s*(\d{1,3})\s*年(?:\s*(\d{1,2})\s*月)?(?:\s*(\d{1,2})\s*日)?")
# 1995年 (year only) — guard against matching inside a longer number
_RE_YEAR = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})\s*年")
# split into sentences (keep Chinese commas intact so an event + its date stay together)
_RE_SENT_SPLIT = re.compile(r"[。！？!?;；\n]")


def _iso(year, month=None, day=None) -> str:
    m = int(month) if month else 1
    d = int(day) if day else 1
    return f"{int(year):04d}-{m:02d}-{d:02d}"


def extract_first_date(text: str) -> Optional[str]:
    """Return the first date in `text` as ISO 8601 (YYYY-MM-DD), or None.

    Handles 'YYYY年MM月DD日', 'YYYY年MM月', 'YYYY年', and '民國N年...'.
    """
    if not text:
        return None
    mo = _RE_FULL_DATE.search(text)
    if mo:
        return _iso(mo.group(1), mo.group(2), mo.group(3))
    mo = _RE_MINGUO.search(text)
    if mo:
        return _iso(int(mo.group(1)) + 1911, mo.group(2), mo.group(3))
    mo = _RE_YEAR.search(text)
    if mo:
        return _iso(mo.group(1))
    return None


def _date_in_sentences(sentences: List[str], subj: str, obj: str, require_both: bool) -> Optional[str]:
    for s in sentences:
        has_subj = bool(subj) and subj in s
        has_obj = bool(obj) and obj in s
        hit = (has_subj and has_obj) if require_both else (has_subj or has_obj)
        if hit:
            d = extract_first_date(s)
            if d:
                return d
    return None


def fill_relation_dates(
    text: str,
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> int:
    """Attach time_start to relations the LLM left timeless.

    Strategy per relation (only when time_start is missing):
      1. prefer a sentence mentioning BOTH the subject and object names that
         also contains a date (strongest evidence);
      2. else fall back to a sentence mentioning either endpoint with a date.
    Returns how many relations were filled.
    """
    id2name = {e.get("id"): (e.get("name") or "") for e in entities}
    sentences = [s for s in _RE_SENT_SPLIT.split(text or "") if s.strip()]
    filled = 0
    for rel in relations:
        if rel.get("time_start"):
            continue
        subj = id2name.get(rel.get("subject_id"), "")
        obj = id2name.get(rel.get("object_id"), "")
        date = (
            _date_in_sentences(sentences, subj, obj, require_both=True)
            or _date_in_sentences(sentences, subj, obj, require_both=False)
        )
        if date:
            rel["time_start"] = date
            filled += 1
    return filled
