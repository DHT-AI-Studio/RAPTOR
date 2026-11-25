"""Verifying an answer's temporal claims (temporal plan §5.12, P17).

The answer stage asks a model for *claims*, not prose; each one cites facts of the evidence it was given.
Every claim must pass, deterministically:

  - **scope**: every cited fact is in the evidence the answer was built from, and is an active fact of the
    tenant. Anything else -- a passage, another tenant's fact, an invented ID -- establishes nothing.
  - **date** (`{"kind": "date", "fact", "date", "endpoint"}`): the cited fact's dates entail the claimed
    date at the claimed precision. The fact's range (its whole span, or one `endpoint`: start / end) must
    lie inside the claimed period, so "2005" is entailed by "July 2005" but "12 July 2005" is not, and a
    2008-2013 role is not "in 2010".
  - **ordering** (`{"kind": "ordering", "subject", "object", "relations", "wording"}`): the relations the
    two facts' dates allow (allen.possible_relations: exact, each relation individually feasible) must be
    non-empty. `certain` wording needs them all among the claimed relations, so {before, equals, after} can
    never become "X happened before Y"; `possible` wording needs one of them -- a feasible witness, since
    every relation in the set is satisfiable on its own.
  - **absence** (`{"kind": "absence"}`, "nothing else", "only"): rejected unless the retrieval was complete.

Pure: no I/O. The caller loads the cited facts.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.temporal.allen import RELATIONS, IntervalDomain, possible_relations

KINDS = ("date", "ordering", "absence")
ENDPOINTS = ("whole", "start", "end")
WORDINGS = ("certain", "possible")
MAX_CLAIMS = 50
MAX_EVIDENCE = 500


def cited(claim: dict[str, Any]) -> list[str]:
    """The fact IDs a claim rests on."""
    kind = claim.get("kind")
    if kind == "date":
        return [claim.get("fact")]
    if kind == "ordering":
        return [claim.get("subject"), claim.get("object")]
    return []


def _date_range(domain: IntervalDomain, endpoint: str) -> tuple[int | None, int | None]:
    if endpoint == "start":
        return domain.start.lo_ms, domain.start.hi_ms
    if endpoint == "end":
        return domain.end.lo_ms, domain.end.hi_ms
    return domain.start.lo_ms, domain.end.hi_ms


def verify(claims: list[dict[str, Any]], evidence: set[str],
           facts: dict[str, tuple[dict[str, Any], IntervalDomain | None]], complete: bool,
           parse: Callable[[str], tuple[int, int] | None]) -> list[dict[str, Any]]:
    """One verdict per claim: {verified, reason, ...what the server found}. *facts*: the cited facts that
    exist and are active, with their endpoint ranges (None when unreadable). *parse*: a claimed date ->
    its (lo_ms, hi_ms), None when it can't be read."""
    out = []
    for i, claim in enumerate(claims):
        verdict = _verify_one(claim if isinstance(claim, dict) else {}, evidence, facts, complete, parse)
        out.append({"index": i, **verdict})
    return out


def _reject(reason: str, **found: Any) -> dict[str, Any]:
    return {"verified": False, "reason": reason, **found}


def _verify_one(claim, evidence, facts, complete, parse) -> dict[str, Any]:
    kind = claim.get("kind")
    if kind not in KINDS:
        return _reject(f"unknown claim kind {kind!r}")
    for fid in cited(claim):
        if not isinstance(fid, str) or fid not in evidence:
            return _reject(f"cites {fid!r}, which is not in the evidence")
        if fid not in facts:
            return _reject(f"cites {fid!r}, which is not an active fact")

    if kind == "absence":
        return {"verified": True, "reason": None} if complete else \
            _reject("the retrieval was incomplete, so absence or completeness cannot be claimed")

    if kind == "date":
        endpoint = claim.get("endpoint") or "whole"
        if endpoint not in ENDPOINTS:
            return _reject(f"unknown endpoint {endpoint!r}")
        row, domain = facts[claim["fact"]]
        when = row.get("when")
        claimed = parse(claim.get("date")) if isinstance(claim.get("date"), str) else None
        if claimed is None:
            return _reject(f"the claimed date {claim.get('date')!r} cannot be read")
        lo, hi = _date_range(domain, endpoint) if domain is not None else (None, None)
        if lo is None or hi is None:
            return _reject("the fact has no known date there", evidence_when=when)
        if lo >= claimed[0] and hi <= claimed[1]:
            return {"verified": True, "reason": None, "evidence_when": when}
        return _reject(f"not entailed: the evidence gives {when or 'other dates'}", evidence_when=when)

    # ordering
    claimed = claim.get("relations")
    if not (isinstance(claimed, list) and claimed and all(r in RELATIONS for r in claimed)):
        return _reject("relations must be a non-empty list of Allen relation names")
    wording = claim.get("wording") or "certain"
    if wording not in WORDINGS:
        return _reject(f"unknown wording {wording!r}")
    (_, a), (_, b) = facts[claim["subject"]], facts[claim["object"]]
    allowed = possible_relations(a, b) if a is not None and b is not None else frozenset()
    found = [r for r in RELATIONS if r in allowed]
    if not allowed:
        return _reject("the two facts' dates allow no relation (unknown or contradictory)", relations=found)
    if wording == "certain" and not allowed <= set(claimed):
        others = [r for r in found if r not in claimed]
        return _reject(f"not certain: the dates also allow {', '.join(others)}", relations=found)
    if wording == "possible" and not allowed & set(claimed):
        return _reject(f"not possible: the dates allow only {', '.join(found)}", relations=found)
    return {"verified": True, "reason": None, "relations": found}
