"""Point-inclusive Allen relations and exact integer endpoint-domain feasibility.

Equality/shared-start/shared-end take precedence over endpoint touching. Thus a
point at an interval's end finishes it, and a point at its start starts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import inf
from types import MappingProxyType

RELATIONS = (
    "before", "meets", "overlaps", "starts", "during", "finishes", "equals",
    "finished-by", "contains", "started-by", "overlapped-by", "met-by", "after",
)
INVERSE = dict(zip(RELATIONS, reversed(RELATIONS)))


def relate(s1: int, e1: int, s2: int, e2: int) -> str:
    """Classify closed intervals (including points); reject reversed endpoints."""
    if s1 > e1 or s2 > e2:
        raise ValueError("interval start exceeds end")
    if s1 == s2:
        return "equals" if e1 == e2 else "starts" if e1 < e2 else "started-by"
    if e1 == e2:
        return "finishes" if s1 > s2 else "finished-by"
    if e1 < s2:
        return "before"
    if s1 > e2:
        return "after"
    if e1 == s2:
        return "meets"
    if s1 == e2:
        return "met-by"
    if s1 < s2:
        return "overlaps" if e1 < e2 else "contains"
    return "overlapped-by" if e1 > e2 else "during"


@lru_cache(maxsize=1)
def composition_table():
    """Derive all 169 entries on a six-point line with s <= e; cache immutably."""
    intervals = [(s, e) for s in range(6) for e in range(s, 6)]
    table = {(left, right): set() for left in RELATIONS for right in RELATIONS}
    for a, b, c in product(intervals, repeat=3):
        table[relate(*a, *b), relate(*b, *c)].add(relate(*a, *c))
    return MappingProxyType({pair: frozenset(values) for pair, values in table.items()})


@dataclass(frozen=True)
class EndpointRange:
    """Closed integer-ms domain; None represents an unbounded side."""

    lo_ms: int | None
    hi_ms: int | None


@dataclass(frozen=True)
class IntervalDomain:
    """Independent endpoint domains, additionally constrained by start <= end."""

    start: EndpointRange
    end: EndpointRange
    extent: str = "interval"
    confirmed_active_at: int | None = None


def _bounds(fact):
    confirmed = None
    if isinstance(fact, IntervalDomain):
        items = (fact.start.lo_ms, fact.start.hi_ms, fact.end.lo_ms, fact.end.hi_ms)
        extent = fact.extent
        confirmed = fact.confirmed_active_at
    elif isinstance(fact, dict):
        endpoints = [fact["start"], fact["end"]]
        if any(item.get("state") == "unresolved" for item in endpoints):
            return None
        items = tuple(value for item in endpoints for value in (item["lo_ms"], item["hi_ms"]))
        extent = fact.get("extent", "interval")
        confirmed = fact.get("confirmed_active_at")
    else:
        items = tuple(fact)
        extent = items[4] if len(items) == 5 else "interval"
        items = items[:4] if len(items) == 5 else items
        if len(items) != 4:
            raise ValueError("expected (start_lo,start_hi,end_lo,end_hi[,extent])")
    if extent not in {"interval", "point", "ongoing"}:
        return None
    if any(value is not None and (not isinstance(value, int) or isinstance(value, bool)) for value in items):
        raise TypeError("endpoint bounds must be integer milliseconds or None")
    bounds = tuple((-inf if i % 2 == 0 else inf) if value is None else value
                   for i, value in enumerate(items))
    if confirmed is not None:
        if not isinstance(confirmed, int) or isinstance(confirmed, bool):
            raise TypeError("confirmation must be integer milliseconds")
        if extent == "ongoing":
            sl, sh, el, eh = bounds
            bounds = (sl, min(sh, confirmed), max(el, confirmed), eh)
    return (*bounds, extent)


def _feasible(sl, sh, el, eh):
    return sl <= sh and el <= eh and sl <= eh


@lru_cache(maxsize=8192)
def _window(bounds, w0, w1) -> frozenset[str]:
    sl, sh, el, eh, extent = bounds
    if w0 > w1 or not _feasible(sl, sh, el, eh):
        return frozenset()
    candidates = {
        "before": (sl, sh, el, min(eh, w0 - 1)),
        "after": (max(sl, w1 + 1), sh, el, eh),
        "equals": (max(sl, w0), min(sh, w0), max(el, w1), min(eh, w1)),
        "starts": (max(sl, w0), min(sh, w0), el, min(eh, w1 - 1)),
        "started-by": (max(sl, w0), min(sh, w0), max(el, w1 + 1), eh),
        "finishes": (max(sl, w0 + 1), sh, max(el, w1), min(eh, w1)),
        "finished-by": (sl, min(sh, w0 - 1), max(el, w1), min(eh, w1)),
        "during": (max(sl, w0 + 1), sh, el, min(eh, w1 - 1)),
        "contains": (sl, min(sh, w0 - 1), max(el, w1 + 1), eh),
    }
    if w0 < w1:
        candidates.update({
            "meets": (sl, min(sh, w0 - 1), max(el, w0), min(eh, w0)),
            "met-by": (max(sl, w1), min(sh, w1), max(el, w1 + 1), eh),
            "overlaps": (sl, min(sh, w0 - 1), max(el, w0 + 1), min(eh, w1 - 1)),
            "overlapped-by": (max(sl, w0 + 1), min(sh, w1 - 1), max(el, w1 + 1), eh),
        })
    def feasible(ranges):
        a, b, c, d = ranges
        return _feasible(*ranges) and (extent != "point" or max(a, c) <= min(b, d))
    return frozenset(name for name, ranges in candidates.items() if feasible(ranges))


def window_relations(fact, w0: int, w1: int) -> frozenset[str]:
    """Exact constant-time intersection of each relation with a fixed query window."""
    if not isinstance(w0, int) or not isinstance(w1, int):
        raise TypeError("window endpoints must be integer milliseconds")
    bounds = _bounds(fact)
    return _window(bounds, w0, w1) if bounds is not None else frozenset()


# constraints: x - y <= c. Strict integer ordering is <= -1.
def _lt(x, y):
    return [(x, y, -1)]


def _eq(x, y):
    return [(x, y, 0), (y, x, 0)]


_CONSTRAINTS = {
    "before": _lt(1, 2),
    "after": _lt(3, 0),
    "equals": _eq(0, 2) + _eq(1, 3),
    "starts": _eq(0, 2) + _lt(1, 3),
    "started-by": _eq(0, 2) + _lt(3, 1),
    "finishes": _eq(1, 3) + _lt(2, 0),
    "finished-by": _eq(1, 3) + _lt(0, 2),
    "during": _lt(2, 0) + _lt(1, 3),
    "contains": _lt(0, 2) + _lt(3, 1),
    "meets": _eq(1, 2) + _lt(0, 1) + _lt(2, 3),
    "met-by": _eq(0, 3) + _lt(2, 3) + _lt(0, 1),
    "overlaps": _lt(0, 2) + _lt(2, 1) + _lt(1, 3),
    "overlapped-by": _lt(2, 0) + _lt(0, 3) + _lt(3, 1),
}


def _satisfiable(ranges, relation, a_point=False, b_point=False):
    edges = [(1, 0, 0), (3, 2, 0)]  # end >= start in both domains
    for index, (lo, hi) in enumerate(ranges):
        if lo > hi:
            return False
        if hi != inf:
            edges.append((4, index, hi))
        if lo != -inf:
            edges.append((index, 4, -lo))
    edges.extend((y, x, c) for x, y, c in _CONSTRAINTS[relation])
    if a_point:
        edges.extend((y, x, c) for x, y, c in _eq(0, 1))
    if b_point:
        edges.extend((y, x, c) for x, y, c in _eq(2, 3))
    # A super-source at zero reaches all nodes; a negative cycle is infeasible.
    distance = [0] * 5
    for _ in range(5):
        changed = False
        for source, target, weight in edges:
            if distance[target] > distance[source] + weight:
                distance[target] = distance[source] + weight
                changed = True
        if not changed:
            return True
    return False


@lru_cache(maxsize=8192)
def _possible(a, b):
    ranges = list(zip(a[:4:2] + b[:4:2], a[1:4:2] + b[1:4:2]))
    return frozenset(relation for relation in RELATIONS
                     if _satisfiable(ranges, relation, a[4] == "point", b[4] == "point"))


def possible_relations(a, b) -> frozenset[str]:
    """Solve all thirteen relations exactly using integer difference constraints."""
    a_bounds, b_bounds = _bounds(a), _bounds(b)
    if a_bounds is None or b_bounds is None:
        return frozenset()
    return _possible(a_bounds, b_bounds)


def certainty(relations) -> str:
    """A singleton feasible relation is certain; the empty set is never certain."""
    if not relations:
        raise ValueError("certainty is undefined for infeasible domains")
    if not set(relations) <= set(RELATIONS):
        raise ValueError("unknown Allen relation")
    return "certain" if len(relations) == 1 else "possible"
