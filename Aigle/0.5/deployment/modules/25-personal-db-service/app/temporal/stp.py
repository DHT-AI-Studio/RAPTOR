"""Sound metric reasoning over interval endpoints: a Simple Temporal Problem (temporal plan §5.5, P15).

Every interval X has two variables, its start sX and end eX, in integer UTC milliseconds, plus a zero point
Z. Every constraint is a difference `x - y <= c`:

    an endpoint range [lo, hi]   x - Z <= hi,  Z - x <= -lo         (an unbounded side adds nothing)
    an interval                  sX - eX <= 0                       (start <= end)
    a point event                eX - sX <= 0 as well               (start == end)
    a qualitative link           see LINKS -- each is a conjunction of endpoint inequalities

Floyd-Warshall gives the tightest bound of every variable and every difference the constraints entail. A
negative cycle proves the constraints inconsistent: it is returned as a witness -- the constraints along it,
each with what supports it -- which is a conflict but not necessarily a minimal one.

The links are chosen so that each is *exactly* a conjunction of endpoint inequalities (so the network is
fully represented by the STP, and a feasible STP is a consistent network -- §5.5's condition for calling it
consistent):

    before      eA < sB               A is over before B starts      (Allen: before)
    after       sA > eB               A starts after B is over       (after)
    during      sB <= sA, eA <= eB    A lies within B                (during, starts, finishes, equals)
    same_time   sA <= eB, sB <= eA    A and B share time             (every relation but before / after)

"Before" / "after" are strict, as in the plan's worked example: strict order is one millisecond, so an undated
X after the July 2005 acquisition starts at 2005-07-01T00:00:00.001Z at the earliest. Point-inclusive: a
point event can stand at another interval's endpoint. Pure: no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any

LINKS = ("before", "after", "during", "same_time")


@dataclass
class Interval:
    id: str
    start: tuple[int | None, int | None] = (None, None)    # (lo_ms, hi_ms); None: unbounded
    end: tuple[int | None, int | None] = (None, None)
    point: bool = False


@dataclass
class Link:
    a: str
    relation: str          # one of LINKS: "A <relation> B"
    b: str
    support: str           # what states it (a fact or candidate id)


@dataclass
class Result:
    consistent: bool
    bounds: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)   # id -> {start, end: (lo, hi)}
    witness: list[dict[str, Any]] = field(default_factory=list)                     # the negative cycle, if any


def _link_edges(link: Link) -> list[tuple[str, str, int]]:
    """(x, y, c) for x - y <= c, on variables "s:<id>" / "e:<id>"."""
    sa, ea, sb, eb = f"s:{link.a}", f"e:{link.a}", f"s:{link.b}", f"e:{link.b}"
    return {
        "before": [(ea, sb, -1)],
        "after": [(eb, sa, -1)],
        "during": [(sb, sa, 0), (ea, eb, 0)],
        "same_time": [(sa, eb, 0), (sb, ea, 0)],
    }[link.relation]


def solve(intervals: list[Interval], links: list[Link]) -> Result:
    """Tightest endpoint bounds, or the witness of an inconsistency."""
    names = ["Z"] + [f"{p}:{iv.id}" for iv in intervals for p in ("s", "e")]
    index = {n: i for i, n in enumerate(names)}
    edges: list[tuple[int, int, float, str]] = []      # (y, x, c, support) for x - y <= c: an edge y -> x

    def add(x: str, y: str, c: float, support: str) -> None:
        edges.append((index[y], index[x], c, support))

    for iv in intervals:
        s, e = f"s:{iv.id}", f"e:{iv.id}"
        add(s, e, 0, f"{iv.id}: start <= end")
        if iv.point:
            add(e, s, 0, f"{iv.id}: a point event")
        for var, (lo, hi) in ((s, iv.start), (e, iv.end)):
            if hi is not None:
                add(var, "Z", hi, f"{iv.id}: stated dates")
            if lo is not None:
                add("Z", var, -lo, f"{iv.id}: stated dates")
    for link in links:
        if link.relation not in LINKS:
            raise ValueError(f"unknown link {link.relation!r}")
        for x, y, c in _link_edges(link):
            add(x, y, c, f"{link.support}: {link.a} {link.relation} {link.b}")

    n = len(names)
    dist = [[0 if i == j else inf for j in range(n)] for i in range(n)]
    for y, x, c, _ in edges:
        dist[y][x] = min(dist[y][x], c)
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == inf:
                continue
            di = dist[i]
            for j in range(n):
                di[j] = min(di[j], dik + dk[j])
    if any(dist[i][i] < 0 for i in range(n)):
        return Result(False, witness=_negative_cycle(n, edges, names))

    z = index["Z"]
    bounds = {}
    for iv in intervals:
        bounds[iv.id] = {side: (-dist[index[f"{p}:{iv.id}"]][z], dist[z][index[f"{p}:{iv.id}"]])
                         for side, p in (("start", "s"), ("end", "e"))}    # lo = -(Z - v), hi = v - Z
    return Result(True, bounds=bounds)


def _negative_cycle(n: int, edges: list[tuple[int, int, float, str]], names: list[str]) -> list[dict[str, Any]]:
    """A negative cycle, by Bellman-Ford from a virtual source: the constraints along it, in order."""
    dist, pred = [0.0] * n, [None] * n
    last = None
    for _ in range(n):
        last = None
        for y, x, c, support in edges:
            if dist[y] + c < dist[x]:
                dist[x], pred[x], last = dist[y] + c, (y, support), x
        if last is None:
            return []
    v = last
    for _ in range(n):                                  # step back into the cycle itself
        v = pred[v][0]
    cycle, u = [], v
    while True:
        y, support = pred[u]
        cycle.append({"from": names[y], "to": names[u], "because": support})
        u = y
        if u == v:
            break
    return list(reversed(cycle))
