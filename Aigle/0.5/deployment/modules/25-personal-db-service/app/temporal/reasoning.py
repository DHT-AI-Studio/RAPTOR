"""Ingestion-time reasoning over one source's temporal assertions (temporal plan §5.5, P15).

Input: the source's dated facts (endpoint ranges) and its undated candidates (P11/P12), each with the
qualitative `links` the extractor found ("joined later, after Android Inc. was founded ..."). Output:

  - `consistent` -- the links and dates are jointly satisfiable. The network is exactly an STP (every link
    is a conjunction of endpoint inequalities, app/temporal/stp.py), so a feasible STP is a consistent
    network; path consistency over the Allen algebra would add nothing it does not already prove.
  - `inconsistent` -- with a witness: the constraints of a negative cycle, each naming what states it.
  - the derived bounds of every undated candidate that the links tie to dated facts: sound (every
    solution of the network lies inside), the tightest the network entails, with the ids they rest on.

A link's `other` names an event in words; it is resolved to one assertion of the same source by the words
they share -- the subject / relation / value first, the source sentence only as a filter. Two candidates
that fit equally well (and are not interchangeable) leave the link unresolved: it is reported, never guessed.

Pure: no I/O.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from math import inf
from typing import Any

from app.temporal import stp

_WORD = re.compile(r"[a-z0-9]+|[㐀-䶿一-鿿]")
_STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "by", "was", "were", "is", "its", "it",
         "with", "as", "that", "this", "from", "had", "has", "have", "be", "been"}
_LINK_RELATIONS = {"before": "before", "after": "after", "during": "during", "same_time": "same_time"}


def _words(text: str | None) -> set[str]:
    return {w for w in _WORD.findall((text or "").casefold()) if w not in _STOP}


@dataclass
class Node:
    id: str
    kind: str                                   # fact | candidate
    label: str                                  # subject relation value
    sentence: str
    start: tuple[int | None, int | None]
    end: tuple[int | None, int | None]
    point: bool
    links: list[dict[str, str]]
    dated: bool


@dataclass
class Derived:
    candidate_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    support_ids: list[str]


@dataclass
class Reasoning:
    status: str                                 # consistent | inconsistent | no_links
    network_version: str
    derived: list[Derived] = field(default_factory=list)
    witness: list[dict[str, Any]] = field(default_factory=list)
    links_used: list[dict[str, str]] = field(default_factory=list)
    links_unresolved: list[dict[str, str]] = field(default_factory=list)


def _side(row: dict, p: str) -> tuple[int | None, int | None]:
    if row.get(f"{p}_state") == "known":
        return row.get(f"{p}_lo"), row.get(f"{p}_hi")
    return None, None


def _links(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [lk for lk in (raw or []) if isinstance(lk, dict) and lk.get("relation") in _LINK_RELATIONS and lk.get("other")]


def nodes_of(facts: list[dict], candidates: list[dict]) -> list[Node]:
    out = []
    for f in facts:
        start, end = _side(f, "s"), _side(f, "e")
        out.append(Node(f["fact_id"], "fact", f"{f.get('entity') or ''} {f.get('relation') or ''} {f.get('value') or ''}",
                        f.get("source_sentence") or "", start, end, f.get("extent") == "point", _links(f.get("links")),
                        dated=any(v is not None for v in (*start, *end))))
    for c in candidates:
        out.append(Node(c["candidate_id"], "candidate",
                        f"{c.get('entity') or ''} {c.get('relation') or ''} {c.get('value') or ''}",
                        c.get("source_sentence") or "", (None, None), (None, None), c.get("extent") == "point",
                        _links(c.get("links")), dated=False))
    return out


def resolve(phrase: str, nodes: list[Node], exclude: str) -> tuple[Node | None, str]:
    """The one node `phrase` names, by shared words with its subject / relation / value; (None, why) when
    none or several fit. Nodes whose sentence contains the phrase are preferred."""
    want = _words(phrase)
    if not want:
        return None, "empty"
    norm = re.sub(r"\s+", " ", phrase.casefold()).strip()
    scored = []
    for n in nodes:
        if n.id == exclude:
            continue
        overlap = len(want & _words(n.label)) / len(want)
        if overlap == 0:
            continue
        in_sentence = norm in re.sub(r"\s+", " ", n.sentence.casefold())
        scored.append((in_sentence, overlap, n))
    if not scored:
        return None, "no_match"
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = [t for t in scored if (t[0], t[1]) == (scored[0][0], scored[0][1])]
    if not best[0][0] and best[0][1] < 0.5:
        return None, "weak_match"
    if len({(t[2].start, t[2].end, t[2].dated) for t in best}) > 1:
        return None, "ambiguous"
    return best[0][2], "ok"                     # several equal ones are interchangeable for the bounds


def _version(nodes: list[Node], links: list[stp.Link]) -> str:
    body = json.dumps({"n": sorted((n.id, n.start, n.end, n.point) for n in nodes),
                       "l": sorted((lk.a, lk.relation, lk.b) for lk in links)}, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _support(start: str, adjacency: dict[str, set[str]], by_id: dict[str, Node]) -> list[str]:
    """What a derived interval rests on: the dated assertions its links reach, and the undated ones on the way
    to them -- not every assertion that merely shares a neighbour with it."""
    parent: dict[str, str | None] = {start: None}
    queue, support = [start], set()
    while queue:
        node = queue.pop(0)
        for nxt in sorted(adjacency.get(node, ())):
            if nxt in parent:
                continue
            parent[nxt] = node
            if by_id[nxt].dated:
                step = nxt
                while step is not None and step != start:
                    support.add(step)
                    step = parent[step]
            queue.append(nxt)
    return sorted(support)


def reason(facts: list[dict], candidates: list[dict]) -> Reasoning:
    nodes = nodes_of(facts, candidates)
    links: list[stp.Link] = []
    used, unresolved = [], []
    for n in nodes:
        for lk in n.links:
            target, why = resolve(lk["other"], nodes, n.id)
            if target is None:
                unresolved.append({"from": n.id, "relation": lk["relation"], "other": lk["other"], "reason": why})
                continue
            links.append(stp.Link(n.id, _LINK_RELATIONS[lk["relation"]], target.id, support=n.id))
            used.append({"from": n.id, "relation": lk["relation"], "to": target.id, "other": lk["other"]})
    if not links:
        return Reasoning("no_links", _version(nodes, links), links_unresolved=unresolved)

    # only the nodes the links reach take part: an unlinked fact constrains nothing
    involved = {lk.a for lk in links} | {lk.b for lk in links}
    by_id = {n.id: n for n in nodes}
    intervals = [stp.Interval(i, by_id[i].start, by_id[i].end, by_id[i].point) for i in sorted(involved)]
    result = stp.solve(intervals, links)
    version = _version([by_id[i] for i in involved], links)
    if not result.consistent:
        return Reasoning("inconsistent", version, witness=result.witness, links_used=used, links_unresolved=unresolved)

    adjacency: dict[str, set[str]] = {}
    for lk in links:
        adjacency.setdefault(lk.a, set()).add(lk.b)
        adjacency.setdefault(lk.b, set()).add(lk.a)
    derived = []
    for i in sorted(involved):
        node = by_id[i]
        if node.kind != "candidate":
            continue
        b = result.bounds[i]
        if all(v in (-inf, inf) for v in (*b["start"], *b["end"])):
            continue                            # linked only to other undated ones: nothing learned
        derived.append(Derived(i, b["start"], b["end"], _support(i, adjacency, by_id)))
    return Reasoning("consistent", version, derived=derived, links_used=used, links_unresolved=unresolved)
