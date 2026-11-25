"""P17: an answer's temporal claims are verified deterministically (app.temporal.claims)."""
from __future__ import annotations

from app.temporal import claims
from app.temporal.allen import EndpointRange, IntervalDomain

DAY = 86_400_000
JUL_2005 = (1120176000000, 1122854399999)
YEAR_2005 = (1104537600000, 1136073599999)
YEAR_2010 = (1262304000000, 1293839999999)


def dom(s_lo, s_hi, e_lo=None, e_hi=None):
    return IntervalDomain(EndpointRange(s_lo, s_hi), EndpointRange(s_lo if e_lo is None else e_lo,
                                                                   s_hi if e_hi is None else e_hi))


FACTS = {
    "acq": ({"fact_id": "acq", "when": "2005-07"}, dom(*JUL_2005)),
    "g1": ({"fact_id": "g1", "when": "2008-10-22"}, dom(1224633600000, 1224719999999)),
    "staff": ({"fact_id": "staff", "when": "2005-07"}, dom(*JUL_2005)),
    "role": ({"fact_id": "role", "when": "2008 to 2013"},
             dom(1199145600000, 1230767999999, 1356998400000, 1388534399999)),
    "undated": ({"fact_id": "undated", "when": None}, None),
}
EVIDENCE = set(FACTS)
DATES = {"2005": YEAR_2005, "2005-07": JUL_2005, "2005-07-12": (JUL_2005[0] + 11 * DAY, JUL_2005[0] + 12 * DAY - 1),
         "2010": YEAR_2010, "2008": (1199145600000, 1230767999999)}


def run(*cl, complete=True, evidence=EVIDENCE, facts=FACTS):
    return claims.verify(list(cl), evidence, facts, complete, DATES.get)


def test_a_date_is_entailed_only_at_the_evidence_precision_or_coarser():
    ok, coarse, finer = run({"kind": "date", "fact": "acq", "date": "2005-07"},
                            {"kind": "date", "fact": "acq", "date": "2005"},
                            {"kind": "date", "fact": "acq", "date": "2005-07-12"})
    assert ok["verified"] and coarse["verified"]
    assert not finer["verified"] and "the evidence gives 2005-07" in finer["reason"]


def test_a_span_is_not_dated_by_a_year_inside_it_but_its_start_is():
    whole, start = run({"kind": "date", "fact": "role", "date": "2010"},
                       {"kind": "date", "fact": "role", "date": "2008", "endpoint": "start"})
    assert not whole["verified"] and start["verified"]


def test_scope_unreadable_and_undated():
    out = run({"kind": "date", "fact": "chunk-17", "date": "2005"},
              {"kind": "date", "fact": "acq", "date": "sometime"},
              {"kind": "date", "fact": "undated", "date": "2005"},
              {"kind": "guess"})
    assert [v["verified"] for v in out] == [False] * 4
    assert "not in the evidence" in out[0]["reason"] and "cannot be read" in out[1]["reason"]
    assert "no known date" in out[2]["reason"] and "unknown claim kind" in out[3]["reason"]
    gone = run({"kind": "date", "fact": "acq", "date": "2005"}, facts={})[0]          # in evidence, not active
    assert "not an active fact" in gone["reason"]


def test_before_equals_after_never_becomes_certainly_before():
    vague = {"acq": FACTS["acq"], "y": ({"fact_id": "y", "when": "2005"}, dom(*YEAR_2005))}
    certain, possible = run({"kind": "ordering", "subject": "y", "object": "acq", "relations": ["before"]},
                            {"kind": "ordering", "subject": "y", "object": "acq", "relations": ["before"],
                             "wording": "possible"}, evidence=set(vague), facts=vague)
    assert not certain["verified"] and certain["reason"].startswith("not certain")
    assert {"before", "after"} <= set(certain["relations"])
    assert possible["verified"]


def test_ordering_certain_and_impossible():
    ok, wrong = run({"kind": "ordering", "subject": "acq", "object": "g1", "relations": ["before"]},
                    {"kind": "ordering", "subject": "acq", "object": "g1", "relations": ["after"],
                     "wording": "possible"})
    assert ok["verified"] and ok["relations"] == ["before"]
    assert not wrong["verified"] and "only before" in wrong["reason"]
    same_month = run({"kind": "ordering", "subject": "staff", "object": "acq", "relations": ["after"]})[0]
    assert not same_month["verified"]                                     # the same month allows both orders
    bad = run({"kind": "ordering", "subject": "acq", "object": "undated", "relations": ["before"]},
              {"kind": "ordering", "subject": "acq", "object": "g1", "relations": ["later"]})
    assert "no relation" in bad[0]["reason"] and "Allen relation names" in bad[1]["reason"]


def test_absence_needs_a_complete_retrieval():
    assert run({"kind": "absence"})[0]["verified"]
    assert not run({"kind": "absence"}, complete=False)[0]["verified"]
