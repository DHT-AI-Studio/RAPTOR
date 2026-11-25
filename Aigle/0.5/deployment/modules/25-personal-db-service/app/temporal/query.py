"""`parse` -- the time a question asks about (temporal plan §5.9 / §5.11 / P02, TMP-7).

Module 18 calls this for every query instead of reading dates itself, so the normaliser lives in one
place: every date phrase found here is read by app.temporal.normalize, the grammar extraction uses.
What this module adds is the query side: which words are a date phrase, the operator around it
("after 2011", 「2011年以後」), the event a question is relative to ("after Google acquired Android") and
the keywords a TKG search filters the evidence by.

`parse_query(text, timezone, anchor)`:

- `RESOLVED`: a window (one side may be unbounded) and/or a reference event.
  after 2011 -> [2012-01-01T00:00:00.000, unbounded); before 2013 -> (unbounded, 2012-12-31T23:59:59.999];
  since 2010 -> [2010-01-01, unbounded); until 2012 -> (unbounded, 2012-12-31T23:59:59.999]; Chinese
  以後 / 之前 / 以來 / 為止 after the period alike. A period named in full ("in 2010") -> `overlaps`; one open
  side, or "within 2010" -> `inside`.
- `NO_TEMPORAL`: nothing in the question constrains time.
- `UNRESOLVED`: a date phrase the grammar cannot read ("2011-13", "yesterday" with no `anchor`), with
  normalize's reason. The caller must not run the question without it (Appendix D F2).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.temporal.normalize import NORMALIZER_VERSION, normalize

_NUM = r"(?:19|20)\d{2}"
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|"
          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_ZH_DAY = r"(?:民國\s*\d{1,3}|" + _NUM + r")\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*[日號])?)?"
# Date phrases, longest forms first; each match is read by normalize(). A year is not taken from inside a
# longer number or a name ("iPhone 20110", "Galaxy S2010").
_PERIOD = re.compile("|".join([
    r"between\s+(?:" + _MONTH + r"\s+)?" + _NUM + r"\s+and\s+(?:" + _MONTH + r"\s+)?" + _NUM,
    r"from\s+(?:" + _MONTH + r"\s+)?" + _NUM + r"\s+(?:to|through|until)\s+(?:" + _MONTH + r"\s+)?" + _NUM,
    _ZH_DAY + r"\s*[到至]\s*" + _ZH_DAY,
    _NUM + r"\s*[到至]\s*" + _NUM + r"\s*年",
    r"(?<![0-9A-Za-z])" + _NUM + r"[–—-]" + _NUM + r"(?![0-9A-Za-z])",
    # ISO dates, read whole: 「2011-13」 or 「2011-02-30」 is a calendar error, never the year 2011
    r"(?<![0-9A-Za-z])\d{4}-\d{1,2}(?:-\d{1,2})?(?![0-9A-Za-z])",
    _MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+" + _NUM,
    r"\d{1,2}\s+" + _MONTH + r"\.?,?\s+" + _NUM,
    _MONTH + r"\.?,?\s+" + _NUM,
    r"(?:early|mid|late)[\s-]+" + _NUM + r"(?![0-9A-Za-z])",
    r"(?:the\s+)?(?:early\s+|mid\s+|late\s+)?\d{3}0s\b",
    r"\d{3}0\s*年代",
    r"\d{1,2}(?:st|nd|rd|th)\s+century",
    r"(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*世紀",
    _NUM + r"\s*年\s*(?:初|中|底|上半年|下半年)",
    _ZH_DAY,
    r"(?<![0-9A-Za-z])" + _NUM + r"(?![0-9A-Za-z])",
    # deictic: read from the query's own time (`anchor`), never guessed without it
    (r"\b(?:yesterday|today|last\s+(?:year|month|week)|this\s+(?:year|month)|next\s+(?:year|month)|"
     r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years?|months?|weeks?|days?)\s+ago)\b"),
    r"去年|前年|今年|明年|昨天|今天|前天|上個月|上週|本月|\d+\s*(?:年|個月|天|週)前",
]), re.IGNORECASE)

# Operators: in English before the period, in Chinese after it.
_OPERATOR_EN = re.compile(r"\b(after|before|since|until|till|within)\s+(?:the\s+)?(?:year\s+)?$", re.IGNORECASE)
_OPERATOR_ZH = re.compile(r"\s*(以後|之後|之前|以前|以來|起|開始|為止)")
_OPERATORS = {"after": "after", "before": "before", "since": "since", "until": "until", "till": "until",
              "within": "within", "以後": "after", "之後": "after", "之前": "before", "以前": "before",
              "以來": "since", "起": "since", "開始": "since", "為止": "until"}

# P16 (§5.11): a relation word followed by an EVENT, not a date, names a reference event. Relations, strictest
# reading (§5.9): "after" is the Allen `after`; "immediately after" is `met-by`.
_DURING = ["starts", "during", "finishes", "equals"]
_REFERENCE_EN = re.compile(
    r"\b(?P<op>immediately after|right after|immediately before|right before|after|before|during|while)\s+"
    r"(?P<ref>[^?.!,;]+)", re.IGNORECASE)
_REFERENCE_ZH = re.compile(r"(?:在|當)?(?P<ref>[^，。？！,.?!；;]+?)(?P<op>之後|以後|之前|以前|期間)")
_EN_RELATIONS = {"after": ["after"], "before": ["before"], "during": _DURING, "while": _DURING,
                 "immediately after": ["met-by"], "right after": ["met-by"],
                 "immediately before": ["meets"], "right before": ["meets"]}
_ZH_RELATIONS = {"之後": ["after"], "以後": ["after"], "之前": ["before"], "以前": ["before"], "期間": _DURING}
# A pronoun after the relation word ("after that", 「在那之後」) points back into the question: no event.
_ANAPHORA_EN = frozenset({"that", "this", "it", "then", "these", "those", "which", "there", "them", "him", "her",
                          "so", "now"})
_ANAPHORA_ZH = frozenset({"那", "這", "此", "其", "那時", "這時", "那個", "這個", "那件事", "這件事"})

# TMP-4: the words a TKG search should find in a fact's evidence. Latin-script terms only; Chinese is not
# segmented here (a wrongly cut term would silently empty the result).
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9&'.+-]*[A-Za-z0-9+]|[A-Za-z]")
_STOPWORDS = frozenset({
    "a", "about", "after", "all", "an", "and", "any", "are", "as", "at", "be", "been", "before", "between",
    "by", "can", "could", "did", "do", "does", "done", "during", "event", "events", "for", "from", "give",
    "had", "has", "have", "how", "i", "in", "is", "it", "its", "list", "me", "news", "of", "on", "or",
    "please", "s", "show", "since", "tell", "that", "the", "their", "there", "these", "thing", "things",
    "this", "those", "till", "to", "until", "was", "were", "what", "whats", "when", "where", "which", "who",
    "whom", "why", "will", "with", "within", "would", "year", "years", "decade", "decades", "century", "centuries",
    "happen", "happened", "happening", "happens", "occur", "occurred", "going", "go", "went",
    "st", "nd", "rd", "th", "early", "mid", "late", "ago", "last", "next", "yesterday", "today",
})


def keywords(text: str) -> list[str]:
    """Latin-script terms of *text*, without date phrases, question words or the temporal vocabulary."""
    seen: dict[str, str] = {}
    for term in _TERM.findall(_PERIOD.sub(" ", text)):
        key = term.casefold().strip(".'")
        if key and key not in _STOPWORDS and key not in seen:
            seen[key] = term.strip(".'")
    return list(seen.values())


def _names_an_event(ref: str) -> bool:
    words = re.findall(r"[A-Za-z]+", ref.casefold())
    if words and all(w in _ANAPHORA_EN for w in words) and not re.search(r"[一-鿿]", ref):
        return False
    if ref in _ANAPHORA_ZH:
        return False
    return bool(keywords(ref)) or bool(re.search(r"[一-鿿]{2,}", ref))


def reference_event(text: str) -> tuple[str | None, list[str] | None, tuple[int, int] | None]:
    """(event text, Allen relations, span of the phrase) -- or Nones. A date after the relation word is a
    window, not an event; a pronoun names nothing, so the next relation phrase is tried."""
    for m in _REFERENCE_EN.finditer(text):
        ref = m.group("ref").strip()
        if ref and not _PERIOD.match(re.sub(r"^(?:the\s+)?(?:year\s+)?", "", ref, flags=re.IGNORECASE)) and _names_an_event(ref):
            return ref, list(_EN_RELATIONS[m.group("op").lower()]), m.span()
    for m in _REFERENCE_ZH.finditer(text):
        ref = m.group("ref").strip()
        if ref and not _PERIOD.fullmatch(ref.strip()) and _names_an_event(ref):
            return ref, list(_ZH_RELATIONS[m.group("op")]), m.span()
    return None, None, None


def _iso(ms: int | None, zone: ZoneInfo) -> str | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, zone).isoformat(timespec="milliseconds")


def _operator(text: str, span: tuple[int, int]) -> str | None:
    if m := _OPERATOR_EN.search(text[:span[0]]):
        return _OPERATORS[m.group(1).lower()]
    if m := _OPERATOR_ZH.match(text, span[1]):
        return _OPERATORS[m.group(1)]
    return None


def parse_query(text: str, timezone: str = "UTC", anchor: str | None = None) -> dict[str, Any]:
    """The time *text* asks about. *timezone*: the zone calendar dates are read in; *anchor*: the query's
    own time (ISO 8601), which deictic phrases ("last year", 去年) are read from -- without it they are
    UNRESOLVED, never guessed."""
    zone = ZoneInfo(timezone)
    reference, relations, ref_span = reference_event(text)
    rest = text if ref_span is None else text[:ref_span[0]] + " " + text[ref_span[1]:]
    out: dict[str, Any] = {"status": "NO_TEMPORAL", "reason": None, "phrase": None, "operator": None,
                           "window": None, "precision": None, "match": None,
                           "reference": reference, "reference_relations": relations,
                           "keywords": keywords(rest), "timezone": timezone, "grammar_version": NORMALIZER_VERSION}
    periods = []
    for m in _PERIOD.finditer(text):
        if ref_span and ref_span[0] <= m.start() < ref_span[1]:
            continue                                    # a date inside the reference phrase belongs to the event
        r = normalize(m.group(0), timezone, source_published=anchor)
        if r["extent"] == "UNRESOLVED":
            out.update(status="UNRESOLVED", reason=r.get("reason") or "unresolved", phrase=m.group(0))
            return out
        periods.append((m, r))
    if not periods:
        if reference:
            out["status"] = "RESOLVED"
        return out

    first, last = periods[0], periods[-1]
    # periods named apart ("in 2016 and 2010") come in text order: the window spans the earliest to the latest
    lo, hi = min(r["start"]["lo_ms"] for _, r in periods), max(r["end"]["hi_ms"] for _, r in periods)
    op = _operator(text, first[0].span()) if len(periods) == 1 else None
    if op == "after":
        start, end = hi + 1, None                       # successor tick: 2011 -> 2012-01-01T00:00:00.000
    elif op == "before":
        start, end = None, lo - 1                       # predecessor tick: 2013 -> 2012-12-31T23:59:59.999
    elif op == "since":
        start, end = lo, None
    elif op == "until":
        start, end = None, hi
    else:
        start, end = lo, hi
    phrase = text[first[0].start():last[0].end()]
    out.update(status="RESOLVED", phrase=phrase, operator=None if op == "within" else op,
               precision=first[1]["precision"] if len(periods) == 1 else "mixed",
               window={"start": _iso(start, zone), "end": _iso(end, zone), "start_ms": start, "end_ms": end},
               # P07: a period named in full asks what was going on then; one open side, or "within", the fact inside
               match="overlaps" if start is not None and end is not None and op != "within" else "inside")
    return out
