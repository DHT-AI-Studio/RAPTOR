"""P15: ingestion-time reasoning -- the STP solver (app/temporal/stp.py) and the per-source network built from
facts, undated candidates and their qualitative links (app/temporal/reasoning.py). No ArcadeDB."""
from __future__ import annotations

import datetime as dt
import json
from math import inf

import pytest
from app.services.temporal_jobs import describe_derived
from app.temporal import reasoning, stp


def ms(text: str) -> int:
    return int(dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def iso(v: float) -> str:
    return dt.datetime.fromtimestamp(v / 1000, dt.timezone.utc).isoformat(timespec="milliseconds")


JULY_2005 = (ms("2005-07-01T00:00:00"), ms("2005-07-31T23:59:59.999"))
YEAR_2007 = (ms("2007-01-01T00:00:00"), ms("2007-12-31T23:59:59.999"))


# ---------------------------------------------------------------- the solver

def test_the_plans_worked_example():
    """§5.5: undated X after the July 2005 acquisition and before the 2007 iPhone; strict order is 1 ms."""
    r = stp.solve([stp.Interval("acq", JULY_2005, JULY_2005), stp.Interval("iphone", YEAR_2007, YEAR_2007),
                   stp.Interval("X")],
                  [stp.Link("X", "after", "acq", "c1"), stp.Link("X", "before", "iphone", "c2")])
    assert r.consistent
    assert [iso(v) for v in r.bounds["X"]["start"]] == ["2005-07-01T00:00:00.001+00:00", "2007-12-31T23:59:59.998+00:00"]


def test_a_seeded_negative_cycle_is_reported_with_what_states_it():
    r = stp.solve([stp.Interval("acq", JULY_2005, JULY_2005), stp.Interval("iphone", YEAR_2007, YEAR_2007),
                   stp.Interval("X")],
                  [stp.Link("X", "after", "acq", "c1"), stp.Link("X", "before", "iphone", "c2"),
                   stp.Link("iphone", "before", "acq", "c3")])
    assert not r.consistent and r.bounds == {}
    because = [step["because"] for step in r.witness]
    assert any(b.startswith("c3:") for b in because)                     # the contradicting statement
    assert any("stated dates" in b for b in because)
    assert r.witness[0]["from"] == r.witness[-1]["to"]                   # a cycle


def test_point_events_and_during():
    r = stp.solve([stp.Interval("conf", (ms("2013-05-15T00:00:00"),) * 2, (ms("2013-05-17T00:00:00"),) * 2),
                   stp.Interval("talk", point=True)],
                  [stp.Link("talk", "during", "conf", "c1")])
    assert r.bounds["talk"]["start"] == (ms("2013-05-15T00:00:00"), ms("2013-05-17T00:00:00"))
    r = stp.solve([stp.Interval("a", (0, 0), (10, 10)), stp.Interval("p", point=True)],
                  [stp.Link("p", "same_time", "a", "c")])
    assert r.bounds["p"]["start"] == (0, 10)                            # a point at either endpoint is allowed


def test_valid_point_boundary_networks_survive():
    """X meets Y at an instant (X before? no: X ends exactly where Y starts) -- `during` both of a point."""
    r = stp.solve([stp.Interval("p", (5, 5), (5, 5), point=True), stp.Interval("a", (0, 0), (5, 5)),
                   stp.Interval("b", (5, 5), (9, 9))],
                  [stp.Link("p", "during", "a", "x"), stp.Link("p", "during", "b", "y")])
    assert r.consistent


def test_unknown_link():
    with pytest.raises(ValueError):
        stp.solve([stp.Interval("a"), stp.Interval("b")], [stp.Link("a", "overlaps", "b", "c")])


# ---------------------------------------------------------------- the network of a source

FOUNDED = {"fact_id": "f-founded", "entity": "Android Inc.", "relation": "founded",
           "value": "in Palo Alto, California, by Andy Rubin and Chris White",
           "source_sentence": "Android Inc. was founded in Palo Alto, California, in October 2003 by Andy Rubin and "
                              "Chris White, with Rich Miner and Nick Sears joining later.",
           "s_state": "known", "s_lo": ms("2003-10-01T00:00:00"), "s_hi": ms("2003-10-31T23:59:59.999"),
           "e_state": "known", "e_lo": ms("2003-10-01T00:00:00"), "e_hi": ms("2003-10-31T23:59:59.999")}


def _cand(cid, entity, relation, value, links, sentence=None):
    return {"candidate_id": cid, "entity": entity, "relation": relation, "value": value,
            "source_sentence": sentence or FOUNDED["source_sentence"], "links": json.dumps(links)}


AFTER_FOUNDING = [{"relation": "after", "other": "Android Inc. was founded in Palo Alto, California, in October 2003"}]


def test_a_link_reaches_the_event_it_names_not_a_sibling_from_the_same_sentence():
    miner = _cand("c-miner", "Rich Miner", "joined", "Android Inc.", AFTER_FOUNDING)
    sears = _cand("c-sears", "Nick Sears", "joined", "Android Inc.", AFTER_FOUNDING)
    r = reasoning.reason([FOUNDED], [miner, sears])
    assert r.status == "consistent"
    assert {(u["from"], u["to"]) for u in r.links_used} == {("c-miner", "f-founded"), ("c-sears", "f-founded")}
    got = {d.candidate_id: d for d in r.derived}
    assert got["c-miner"].start[0] == FOUNDED["e_lo"] + 1 and got["c-miner"].start[1] == inf
    assert got["c-miner"].support_ids == ["f-founded"]


def test_unresolvable_or_ambiguous_links_are_reported_not_guessed():
    other = {**FOUNDED, "fact_id": "f-other", "s_lo": ms("2004-01-01T00:00:00"), "e_lo": ms("2004-01-01T00:00:00")}
    vague = _cand("c1", "Someone", "left", "the team", [{"relation": "after", "other": "wider rollout"}])
    twin = _cand("c2", "Someone", "joined", "the team", AFTER_FOUNDING)
    r = reasoning.reason([FOUNDED, other], [vague, twin])
    reasons = {u["from"]: u["reason"] for u in r.links_unresolved}
    assert reasons == {"c1": "no_match", "c2": "ambiguous"}            # two "founded" facts with different dates
    assert r.status == "no_links" and r.derived == []


def test_contradicting_links_make_the_source_inconsistent():
    early = _cand("c-early", "Rich Miner", "joined", "Android Inc.",
                  [*AFTER_FOUNDING, {"relation": "before", "other": "Android Inc. was founded in Palo Alto"}])
    r = reasoning.reason([FOUNDED], [early])
    assert r.status == "inconsistent" and r.witness and not r.derived


def test_no_links():
    r = reasoning.reason([FOUNDED], [])
    assert (r.status, r.derived, r.links_used) == ("no_links", [], [])


def test_describe_derived():
    assert describe_derived((ms("2003-10-01T00:00:00"), inf), (ms("2003-10-01T00:00:00"), inf)).endswith("(derived)")
    assert describe_derived((-inf, inf), (-inf, inf)) == "undated (derived)"
    text = describe_derived((ms("2005-07-01T00:00:00"), ms("2007-12-31T00:00:00")),
                            (ms("2005-07-01T00:00:00"), ms("2007-12-31T00:00:00")))
    assert text.startswith("between 2005-07-01 and 2007-12-31")
