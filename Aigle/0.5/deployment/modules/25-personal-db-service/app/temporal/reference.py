"""Reference-event queries (temporal plan §5.9, P16): "what happened after Google acquired Android?".

A TKG v2 request names the event the answer is relative to -- `reference: {"fact_id": ...}` or
`{"text": ...}` -- and the Allen relations a fact must have to it (`reference_relations`).

**Resolving** the reference to one fact of the tenant:
  - `fact_id`: that fact, if it exists and is active;
  - `text`: every active fact is scored by the reference's content words.
      - A word found in the fact's subject counts 3, in its relation or value 2, only in its sentence 1;
        inflections match by a shared stem ("acquisition" finds "acquired").
      - Each word is weighted by how rare it is in the tenant's facts: in a history of Android, "acquired"
        names the acquisition, while "Google" and "Android" are in almost every fact.
      - A fact is supported when it has at least MIN_COVERAGE of the words (unweighted).
      - Facts derived by reasoning (P15) are considered only when no stated fact is supported: an
        inference is not the event the question names.
    The best-scoring supported facts are the plausible events:
      - one event (several rows with the same dates are one event stated more than once): `resolved`;
      - events with different dates: `ambiguous`, with the candidates -- never silently the top one;
      - none: `not_found`.
A resolved event without a usable date is `undated`: nothing can be placed relative to it.

**Relating** a fact to the resolved event uses allen.possible_relations over both endpoint ranges, so a
date known only to the month or year is honoured on both sides. `certainty: possible` keeps a fact
that can have one of the requested relations; `certain` one whose every possible relation is requested.
Strict `after` is the Allen `after` -- an event that meets or shares the reference's end is not after it.

Pure: no I/O.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.temporal.allen import RELATIONS, IntervalDomain, possible_relations
from app.temporal.contract import query_terms

MIN_COVERAGE = 0.6                       # share of the reference's content words a supporting fact must have
MAX_CANDIDATES = 5                       # reported with an ambiguous reference
STATUSES = ("resolved", "ambiguous", "not_found", "undated")

_WORD = re.compile(r"[a-z0-9][a-z0-9'.+-]*")
_STEM = 5                                # shared leading letters that make two inflections one word


@dataclass
class Resolution:
    status: str
    fact: dict[str, Any] | None = None                    # the reference event (resolved / undated)
    domain: IntervalDomain | None = None                  # its endpoint ranges (resolved)
    same_event: set[str] = field(default_factory=set)     # fact_ids stating that event
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


def _words(text: str | None) -> list[str]:
    return [w.strip(".'") for w in _WORD.findall((text or "").casefold())]


def _matches(term: str, words: list[str], text: str) -> bool:
    if not term.isascii():                                # a Chinese piece: contained in the text
        return term in text
    for w in words:
        if w == term:
            return True
        n = min(len(w), len(term))
        if n >= _STEM and w[:_STEM] == term[:_STEM]:
            return True
    return False


def _fields(fact: dict[str, Any]) -> list[tuple[float, list[str], str]]:
    texts = [(3.0, fact.get("entity")), (2.0, f"{fact.get('relation') or ''} {fact.get('value') or ''}"),
             (1.0, fact.get("source_sentence"))]
    return [(w, _words(t), (t or "").casefold()) for w, t in texts]


def rarity(terms: list[str], facts: list[dict[str, Any]]) -> dict[str, float]:
    """How much each term tells facts apart in this tenant: 1 + ln((N + 1) / (1 + facts having it)). In a
    history of Android every fact says "Google" and "Android"; "acquired" is what names the acquisition."""
    prepared = [_fields(f) for f in facts]
    n = len(facts)
    return {t: 1.0 + math.log((n + 1) / (1 + sum(any(_matches(t, ws, tx) for _, ws, tx in p) for p in prepared)))
            for t in terms}


def support(fact: dict[str, Any], terms: list[str], weights: dict[str, float] | None = None) -> tuple[float, float]:
    """(coverage, score): the share of the terms the fact has anywhere, and their mean over where each occurs
    (subject 3, relation / value 2, sentence 1) -- weighted by *weights* (rarity) when given."""
    if not terms:
        return 0.0, 0.0
    prepared = _fields(fact)
    found, score = 0, 0.0
    total = sum((weights or {}).get(t, 1.0) for t in terms)
    for term in terms:
        best = max((w for w, words, text in prepared if _matches(term, words, text)), default=0.0)
        found += best > 0
        score += best * (weights or {}).get(term, 1.0)
    return found / len(terms), round(score / total, 6)


def _event_key(domain: IntervalDomain | None) -> tuple | None:
    if domain is None:
        return None
    return (domain.start.lo_ms, domain.start.hi_ms, domain.end.lo_ms, domain.end.hi_ms, domain.extent)


def _known(domain: IntervalDomain | None) -> bool:
    """Some bound of the event is known -- otherwise nothing can be placed relative to it."""
    return domain is not None and any(
        v is not None for v in (domain.start.lo_ms, domain.start.hi_ms, domain.end.lo_ms, domain.end.hi_ms))


def by_fact_id(row: dict[str, Any] | None, domain: IntervalDomain | None) -> Resolution:
    if row is None:
        return Resolution("not_found", reason="no active fact with that fact_id")
    if not _known(domain):
        return Resolution("undated", fact=row, same_event={row["fact_id"]}, reason="the event has no known date")
    return Resolution("resolved", fact=row, domain=domain, same_event={row["fact_id"]})


def by_text(text: str, facts: list[tuple[dict[str, Any], IntervalDomain | None]]) -> Resolution:
    """Resolve reference text among the tenant's active facts (each with its endpoint ranges, None when
    they can't be read)."""
    terms = query_terms(text)
    if not terms:
        return Resolution("not_found", reason="the reference names nothing to look for")
    weights = rarity(terms, [row for row, _ in facts])
    scored = []
    for row, domain in facts:
        coverage, score = support(row, terms, weights)
        if coverage >= MIN_COVERAGE:
            scored.append((score, coverage, row, domain))
    if not scored:
        return Resolution("not_found", reason="no fact states that event")
    # A fact reasoning derived (P15) is an inference worded after its own subject ("the key employees joined
    # Google"), not the event the question names; it names a reference only when no stated fact does.
    stated = [x for x in scored if not x[2].get("derived")]
    if stated:
        scored = stated
    best = max(s for s, _, _, _ in scored)
    # a stated fact names the event before one reasoning derived (P15) with the same dates
    tier = sorted((x for x in scored if x[0] == best), key=lambda x: (bool(x[2].get("derived")), x[2].get("fact_id") or ""))
    events: dict[Any, list] = {}
    for x in tier:
        key = _event_key(x[3]) if _known(x[3]) else ("undated", x[2].get("fact_id"))
        events.setdefault(key, []).append(x)
    ranked = sorted(scored, key=lambda x: (-x[0], -x[1], bool(x[2].get("derived")), x[2].get("fact_id") or ""))
    candidates = [{"fact_id": r.get("fact_id"), "entity": r.get("entity"), "relation": r.get("relation"),
                   "value": r.get("value"), "score": s} for s, _, r, _ in ranked[:MAX_CANDIDATES]]
    if len(events) > 1:
        return Resolution("ambiguous", candidates=candidates,
                          reason=f"{len(events)} events with different dates state it equally well")
    (key, rows), = events.items()
    row, domain = rows[0][2], rows[0][3]
    same = {r.get("fact_id") for _, _, r, _ in rows}
    if not _known(domain):
        return Resolution("undated", fact=row, same_event=same, candidates=candidates,
                          reason="the event has no known date")
    return Resolution("resolved", fact=row, domain=domain, same_event=same, candidates=candidates)


def relate(domain: IntervalDomain | None, reference: IntervalDomain, accepted: frozenset[str],
           certainty: str) -> tuple[bool, dict[str, Any] | None]:
    """(keep, {relations, certainty}) of a fact to the reference event. A fact whose dates can't be read,
    or contradict themselves, is kept only when every relation is accepted."""
    possible = possible_relations(domain, reference) if domain is not None else frozenset()
    if not possible:
        return accepted == frozenset(RELATIONS) and certainty == "possible", None
    certain = possible <= accepted
    keep = certain if certainty == "certain" else bool(possible & accepted)
    return keep, {"relations": [r for r in RELATIONS if r in possible],
                  "certainty": "certain" if certain else "possible"}


def describe(res: Resolution, request: dict[str, Any], fact_out: dict[str, Any] | None) -> dict[str, Any]:
    """The response's `reference` block."""
    out: dict[str, Any] = {"status": res.status, **request}
    if fact_out is not None:
        out["fact"] = fact_out
    if res.status == "ambiguous" or (res.candidates and res.status != "resolved"):
        out["candidates"] = res.candidates
    if res.reason:
        out["reason"] = res.reason
    return out
