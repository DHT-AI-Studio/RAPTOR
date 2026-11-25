"""TKG contract v2 (temporal plan §5.9, P08 / P09): request checks, ranking and cursor pagination.

A request with `"version": 2` gets
  - **strict checking**: unknown fields, and fields of the plan's contract this deployment does not serve
    yet (`as_of`: see `capabilities()`), are refused by name instead of ignored; a
    `reference` (P16) must be well formed and come with its `reference_relations`;
  - **ranking over the whole result** (`sort`: `time`, or `relevance` -- every match is scored, never
    only the first `limit`), then **keyset pagination**: `limit` facts and a `cursor` naming the last one.
    Facts written between two pages are neither skipped nor repeated, and a cursor only works for the
    query it came from. `complete` is true only on a page with nothing after it.
A request without `version` (v1) is untouched -- Module 21 and older clients keep today's behaviour.

Pure: no I/O.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import zoneinfo
from typing import Any

from app.models.graph_search import V2_UNSUPPORTED, TKGRequest
from app.temporal.allen import RELATIONS

CONTRACT_VERSIONS = (1, 2)
SORTS = ("time", "relevance")
TIMELINES = ("world",)


class ContractError(ValueError):
    """A v2 request this deployment refuses: `code` and the offending names, for a structured 400."""

    def __init__(self, code: str, detail: dict[str, Any]):
        super().__init__(code)
        self.code, self.detail = code, detail

    def body(self) -> dict[str, Any]:
        return {"error": self.code, **self.detail}


def check_v2(req: TKGRequest) -> None:
    """Refuse what a v2 request asks for but this deployment would not do. Raises ContractError."""
    unknown = sorted((req.model_extra or {}).keys())
    if unknown:
        raise ContractError("UNKNOWN_FIELDS", {"fields": unknown})
    unsupported = [f for f in V2_UNSUPPORTED if getattr(req, f) not in (None, [], {})]
    if req.timeline_id not in (None, *TIMELINES):
        unsupported.append("timeline_id")
    if unsupported:
        raise ContractError("UNSUPPORTED_FEATURES", {"features": unsupported,
                                                     "see": "/personal/temporal/capabilities"})
    if req.window is not None and (req.time_start or req.time_end):
        raise ContractError("CONFLICTING_FIELDS", {"fields": ["window", "time_start/time_end"]})
    if req.timezone:
        try:
            zoneinfo.ZoneInfo(req.timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise ContractError("INVALID_TIMEZONE", {"timezone": req.timezone}) from exc
    check_reference(req)
    check_source_filter(req)


MAX_SOURCE_FILTER = 100


def check_source_filter(req: TKGRequest) -> None:
    """P21: `source_filter` is a non-empty list of source version ids -- the scope a question about
    particular sources (ingest_and_query) is answered from."""
    sf = req.source_filter
    if sf is None:
        return
    if not (isinstance(sf, list) and 0 < len(sf) <= MAX_SOURCE_FILTER
            and all(isinstance(v, str) and v.strip() for v in sf)):
        raise ContractError("INVALID_SOURCE_FILTER", {"expected": "a list of 1 to 100 source version ids"})


def check_reference(req: TKGRequest) -> None:
    """P16: `reference` is {"fact_id": str} or {"text": str}, and comes with a non-empty list of Allen
    relation names -- there is no default meaning of "relative to X"."""
    ref, rels = req.reference, req.reference_relations
    if ref is None and rels is None:
        return
    if ref is None or rels is None:
        raise ContractError("INCOMPLETE_REFERENCE", {"fields": ["reference", "reference_relations"]})
    if not (isinstance(ref, dict) and len(ref) == 1 and next(iter(ref)) in ("fact_id", "text")
            and isinstance(next(iter(ref.values())), str) and next(iter(ref.values())).strip()):
        raise ContractError("INVALID_REFERENCE", {"expected": [{"fact_id": "str"}, {"text": "str"}]})
    if not (isinstance(rels, list) and rels and all(r in RELATIONS for r in rels)):
        raise ContractError("INVALID_REFERENCE_RELATIONS", {"allowed": list(RELATIONS)})


def window_of(req: TKGRequest) -> tuple[str | None, str | None]:
    """The window as (start, end) whichever form the request used."""
    if req.window is not None:
        return req.window.start, req.window.end
    return req.time_start, req.time_end


# ---------------------------------------------------------------- ranking

_STOP = {"a", "an", "and", "the", "of", "in", "on", "at", "to", "for", "what", "who", "when", "did", "do", "does",
         "was", "were", "is", "are", "be", "been", "how", "which", "between", "after", "before", "since", "until",
         "during", "with", "by", "from", "happen", "happened", "doing", "about"}
_WORD = re.compile(r"[a-z0-9][a-z0-9'.+-]*")
_HAN_RUN = re.compile(r"[㐀-䶿一-鿿]+")


def query_terms(*texts: str | None) -> list[str]:
    """The query's words: Latin words (lower-cased, minus function words and years) and, for Chinese, every
    two-character piece of each run of Han characters (an unsegmented question still matches)."""
    terms: list[str] = []
    for text in texts:
        t = (text or "").casefold()
        terms += [w.strip(".'") for w in _WORD.findall(t) if w.strip(".'") not in _STOP and not re.fullmatch(r"\d{4}", w)]
        for run in _HAN_RUN.findall(t):
            terms += [run] if len(run) < 2 else [run[i:i + 2] for i in range(len(run) - 1)]
    return list(dict.fromkeys(t for t in terms if t))


def relevance(fact: dict[str, Any], terms: list[str]) -> float:
    """How well a fact answers the query's words: a word naming its subject counts 3, its relation or value 2,
    its source sentence or evidence 1; averaged over the words. Deterministic, and computed for every match."""
    if not terms:
        return 0.0
    fields = ((3.0, fact.get("entity")), (2.0, f"{fact.get('relation') or ''} {fact.get('value') or ''}"),
              (1.0, f"{fact.get('source_sentence') or ''} {fact.get('evidence') or ''}"))
    score = 0.0
    for term in terms:
        score += max((w for w, text in fields if term in (text or "").casefold()), default=0.0)
    return round(score / len(terms), 6)


def time_key(fact: dict[str, Any]) -> list:
    """Same order as v1's time sort: dated facts first, by start, then fact_id."""
    return [fact.get("time_start") is None, fact.get("time_start") or "", fact.get("fact_id") or ""]


# ---------------------------------------------------------------- pagination

def fingerprint(req: TKGRequest) -> str:
    """What a cursor is bound to: the request minus `cursor` and `limit` (a page size may change)."""
    body = req.model_dump(exclude={"cursor", "limit"}, mode="json")
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def encode_cursor(key: list, fp: str) -> str:
    raw = json.dumps({"k": key, "q": fp}, separators=(",", ":"), ensure_ascii=False).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str, fp: str) -> list:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        key, q = data["k"], data["q"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ContractError("INVALID_CURSOR", {"cursor": cursor}) from exc
    if q != fp:
        raise ContractError("CURSOR_FOR_ANOTHER_QUERY", {"cursor": cursor})
    return key


def page(facts: list[dict[str, Any]], req: TKGRequest) -> tuple[list[dict[str, Any]], str | None, int]:
    """(this page, the next page's cursor or None, how many matched). Every fact is ranked first; the page
    is what follows the cursor's key."""
    sort = req.sort or "time"
    if sort == "relevance":
        terms = query_terms(req.query, req.entity, *(req.keywords or []))
        keyed = [([-relevance(f, terms), *time_key(f)], f) for f in facts]
    else:
        keyed = [(time_key(f), f) for f in facts]
    keyed.sort(key=lambda kf: kf[0])
    fp = fingerprint(req)
    if req.cursor:
        after = decode_cursor(req.cursor, fp)
        keyed = [kf for kf in keyed if kf[0] > after]
    size = max(1, req.limit)
    chunk = keyed[:size]
    nxt = encode_cursor(chunk[-1][0], fp) if len(keyed) > size else None
    return [f for _, f in chunk], nxt, len(facts)
