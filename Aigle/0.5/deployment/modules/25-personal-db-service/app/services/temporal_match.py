"""Allen classification of TKG facts against a query window (temporal plan P07, §5.9).

The SQL read is only a necessary prefilter; this module decides. Each fact's two endpoints
are independent ranges of UTC ms (TMP-3/TMP-8), and allen.window_relations() returns every
relation the fact can have to the window. A fact is kept when that set meets the requested
relation set (`certainty: possible`), or lies inside it (`certainty: certain`).
"""
from __future__ import annotations

from typing import Any

from app.services.temporal_normalize import DateParseError, parse_bound
from app.temporal.allen import (
    RELATIONS,
    EndpointRange,
    IntervalDomain,
    window_relations,
)

MATCH_RELATIONS: dict[str, frozenset[str]] = {
    "inside": frozenset({"starts", "during", "finishes", "equals"}),
    "contains": frozenset({"contains", "started-by", "finished-by", "equals"}),
    # inclusive intersection: s <= window end AND e >= window start, boundary points included
    "overlaps": frozenset(RELATIONS) - {"before", "after"},
    "any": frozenset(RELATIONS),
}

# what the SQL read needs to classify a row; dropped from the response
STATE_COLUMNS = ("s_state", "s_lo", "s_hi", "e_state", "e_lo", "e_hi", "extent", "confirmed_active_at")


def relation_set(match: str | None, relations: list[str] | None) -> frozenset[str]:
    """Explicit `relations` override `match`; no match at all means today's `inside`."""
    if relations:
        return frozenset(relations)
    return MATCH_RELATIONS[match or "inside"]


def needs_intersection(accepted: frozenset[str]) -> bool:
    """Inclusive intersection is a necessary condition for every relation except before/after,
    so it is a safe SQL prefilter for a set without them. Anything else is read unfiltered."""
    return not accepted & {"before", "after"}


def _endpoint(row: dict, p: str, side: str, tz: str) -> tuple[int | None, int | None] | None:
    state = row.get(f"{p}_state")
    if state == "known":
        return row.get(f"{p}_lo"), row.get(f"{p}_hi")
    if state in ("unknown", "ongoing"):
        return None, None
    # not normalised yet (written before TMP-3, the backfill hasn't reached it): read the
    # legacy string the same way the backfill would
    try:
        b = parse_bound(row.get(f"time_{side}"), tz)
    except DateParseError:
        return None
    return (None, None) if b is None else (b.lo_ms, b.hi_ms)


def fact_domain(row: dict, tz: str) -> IntervalDomain | None:
    """The fact's endpoint ranges; None when a legacy date can't be read (it can't be classified)."""
    start, end = _endpoint(row, "s", "start", tz), _endpoint(row, "e", "end", tz)
    if start is None or end is None:
        return None
    if row.get("e_state") == "ongoing":
        extent = "ongoing"
    elif row.get("extent") == "point":
        extent = "point"
    else:
        extent = "interval"
    return IntervalDomain(EndpointRange(*start), EndpointRange(*end), extent, row.get("confirmed_active_at"))


def classify(domain: IntervalDomain, w0: int | None, w1: int | None) -> frozenset[str]:
    """Every Allen relation the fact can have to the window [w0, w1].

    An unbounded window side lies beyond every instant, so beyond every finite bound here: it
    is placed past the furthest one, and a fact's own unbounded bounds on that side (an
    unknown end: some instant, but not a known one) are brought just inside it. Relations
    depend only on endpoint order, so this is exact: an unknown end never passes the end of
    "from 2012 on", and a fact that started then is certainly inside it."""
    if w0 is not None and w1 is not None:
        return window_relations(domain, w0, w1)
    finite = [v for v in (domain.start.lo_ms, domain.start.hi_ms, domain.end.lo_ms, domain.end.hi_ms,
                          domain.confirmed_active_at, w0, w1) if v is not None]
    lo, hi = min(finite) - 2, max(finite) + 2

    def inside(r: EndpointRange) -> EndpointRange:
        return EndpointRange(lo + 1 if r.lo_ms is None and w0 is None else r.lo_ms,
                             hi - 1 if r.hi_ms is None and w1 is None else r.hi_ms)

    domain = IntervalDomain(inside(domain.start), inside(domain.end), domain.extent, domain.confirmed_active_at)
    return window_relations(domain, lo if w0 is None else w0, hi if w1 is None else w1)


def window_ms(time_start: str | None, time_end: str | None, tz: str) -> tuple[int | None, int | None]:
    """The window as UTC ms: start of time_start's range .. end of time_end's range."""
    lo, hi = parse_bound(time_start, tz), parse_bound(time_end, tz)
    return (lo.lo_ms if lo else None), (hi.hi_ms if hi else None)


def judge(row: dict, w0: int | None, w1: int | None, accepted: frozenset[str], certainty: str,
          tz: str) -> tuple[bool, dict[str, Any] | None]:
    """(keep, allen) for one fact. `allen` is None when the fact can't be classified: an
    unreadable legacy date, or endpoint ranges that contradict each other -- such a fact
    is kept only when every relation is acceptable (`match: any`)."""
    domain = fact_domain(row, tz)
    possible = classify(domain, w0, w1) if domain is not None else frozenset()
    if not possible:
        return accepted == MATCH_RELATIONS["any"] and certainty == "possible", None
    certain = possible <= accepted
    keep = certain if certainty == "certain" else bool(possible & accepted)
    return keep, {"vs_window": [r for r in RELATIONS if r in possible],
                  "certainty": "certain" if certain else "possible"}
