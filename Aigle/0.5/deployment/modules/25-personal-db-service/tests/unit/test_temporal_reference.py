"""P16: resolving a reference event and relating facts to it (app.temporal.reference)."""
from __future__ import annotations

from app.temporal import reference
from app.temporal.allen import EndpointRange, IntervalDomain

MONTH_JUL_2005 = (1120176000000, 1122854399999)              # 2005-07-01 .. 2005-07-31 UTC, ms
DAY = 86_400_000


def span(lo, hi, extent="interval"):
    return IntervalDomain(EndpointRange(lo, hi), EndpointRange(lo, hi), extent)


def fact(fid, entity, relation, value, sentence=""):
    return {"fact_id": fid, "entity": entity, "relation": relation, "value": value, "source_sentence": sentence}


ACQ = fact("f-acq", "Google", "acquired", "Android Inc.", "Google acquired Android Inc. in July 2005.")
STAFF = fact("f-staff", "key employees", "joined", "Google", "Its key employees joined Google as part of the acquisition.")
INSTA = fact("f-insta", "Facebook", "acquired", "Instagram", "Facebook acquired Instagram in 2012.")


def test_inflections_and_chinese_pieces_match():
    assert reference.support(ACQ, ["acquisition", "android"]) == (1.0, 2.0)     # relation 2, value 2
    assert reference.support(fact("z", "谷歌", "收購", "安卓"), ["收購"])[0] == 1.0
    assert reference.support(ACQ, ["ipo"])[0] == 0.0                            # short words match exactly


def test_resolves_the_event_that_states_it_best():
    res = reference.by_text("after Google acquired Android", [(ACQ, span(*MONTH_JUL_2005)),
                                                              (INSTA, span(1325376000000, 1356998399999)),
                                                              (STAFF, None)])
    assert res.status == "resolved" and res.fact["fact_id"] == "f-acq" and res.same_event == {"f-acq"}


def test_a_word_only_in_the_sentence_counts_less():
    res = reference.by_text("the acquisition", [(ACQ, span(*MONTH_JUL_2005)), (STAFF, span(*MONTH_JUL_2005))])
    assert res.status == "resolved" and res.fact["fact_id"] == "f-acq"


def test_two_acquisitions_are_ambiguous_never_the_top_one():
    res = reference.by_text("the acquisition", [(ACQ, span(*MONTH_JUL_2005)),
                                                (INSTA, span(1325376000000, 1356998399999))])
    assert res.status == "ambiguous" and res.fact is None
    assert {c["fact_id"] for c in res.candidates} == {"f-acq", "f-insta"}


def test_the_same_event_stated_twice_is_one_event():
    again = fact("f-acq2", "Google", "acquired", "Android Inc.")
    res = reference.by_text("Google acquired Android", [(ACQ, span(*MONTH_JUL_2005)), (again, span(*MONTH_JUL_2005))])
    assert res.status == "resolved" and res.same_event == {"f-acq", "f-acq2"}


def test_a_stated_fact_names_the_event_before_a_derived_one():
    # as found on the Android history: reasoning derived "the key employees joined Google" in July 2005, and it
    # scores the same as the stated acquisition -- and its id sorts first
    stated = fact("tf-9", "Google", "acquired", "the company for at least $50 million",
                  "Google acquired Android Inc. in July 2005.")
    derived = {**fact("tf-0", "Android's key employees", "joined", "Google",
                      "Its key employees joined Google as part of the acquisition."), "derived": True}
    facts = [(derived, span(*MONTH_JUL_2005)), (stated, span(*MONTH_JUL_2005))]
    assert reference.support(derived, ["google", "acquired", "android"]) == reference.support(
        stated, ["google", "acquired", "android"])
    res = reference.by_text("Google acquired Android", facts)
    assert res.status == "resolved" and res.fact["fact_id"] == "tf-9" and res.same_event == {"tf-9"}
    assert [c["fact_id"] for c in res.candidates] == ["tf-9"]       # a derived fact is no candidate beside a stated one


def test_a_stated_fact_is_preferred_even_when_a_derived_one_scores_higher():
    # the real case on the Android history: the stated fact says "the company", so it lacks "Android" and scores
    # 1.67, while the derived one has all three words and scores 2.0
    stated = fact("tf-7c5", "Google", "acquired", "the company for at least $50 million",
                  "Shortly afterwards, Google acquired the company in July of that year for at least $50 million.")
    derived = {**fact("tf-der-b8f", "Android's key employees, including Rubin, Miner, Sears, and White", "joined",
                      "Google", "Android's key employees, including Rubin, Miner, Sears, and White, joined Google "
                      "as part of the acquisition."), "derived": True}
    terms = ["google", "acquired", "android"]
    assert reference.support(derived, terms)[1] > reference.support(stated, terms)[1]
    res = reference.by_text("Google acquired Android", [(derived, span(*MONTH_JUL_2005)), (stated, span(*MONTH_JUL_2005))])
    assert res.status == "resolved" and res.fact["fact_id"] == "tf-7c5"


def test_a_rare_word_names_the_event_among_facts_that_all_share_the_common_ones():
    # in a history of Android nearly every fact says "Google" and "Android"; "acquired" names the acquisition
    acq = fact("f-acq", "Google", "acquired", "the company", "Google acquired the company in July 2005.")
    others = [(fact(f"f-{i}", "Google", "announced", f"Android {v}", f"Google announced Android {v}."),
               span(MONTH_JUL_2005[0] + (i + 1) * 400 * DAY, MONTH_JUL_2005[0] + (i + 1) * 400 * DAY + DAY))
              for i, v in enumerate(["One", "TV", "Auto", "XR"])]
    res = reference.by_text("Google acquired Android", [(acq, span(*MONTH_JUL_2005)), *others])
    assert res.status == "resolved" and res.fact["fact_id"] == "f-acq"
    plain = [reference.support(r, ["google", "acquired", "android"])[1] for r, _ in [(acq, None), *others]]
    assert len(set(plain)) == 1                     # without the rarity weights they all score the same


def test_a_derived_fact_names_the_event_when_no_stated_one_does():
    derived = {**fact("tf-der-1", "Nick Sears", "joined", "Android Inc."), "derived": True}
    res = reference.by_text("Nick Sears joined Android", [(derived, span(*MONTH_JUL_2005))])
    assert res.status == "resolved" and res.fact["fact_id"] == "tf-der-1"


def test_not_found_undated_and_nothing_to_look_for():
    assert reference.by_text("Apple bought Beats", [(ACQ, span(*MONTH_JUL_2005))]).status == "not_found"
    assert reference.by_text("Google acquired Android", [(ACQ, None)]).status == "undated"
    unknown = IntervalDomain(EndpointRange(None, None), EndpointRange(None, None))
    assert reference.by_text("Google acquired Android", [(ACQ, unknown)]).status == "undated"
    assert reference.by_text("the", [(ACQ, span(*MONTH_JUL_2005))]).reason.startswith("the reference names")
    assert reference.by_fact_id(None, None).status == "not_found"
    assert reference.by_fact_id(ACQ, span(*MONTH_JUL_2005)).status == "resolved"


AFTER = frozenset({"after"})


def exact(s, e):
    return IntervalDomain(EndpointRange(s, s), EndpointRange(e, e))


def test_strict_after_excludes_meeting_and_shared_ends():
    ref = exact(1000, 2000)
    assert reference.relate(span(3000, 3000, "point"), ref, AFTER, "certain") == (
        True, {"relations": ["after"], "certainty": "certain"})
    assert not reference.relate(span(2000, 2000, "point"), ref, AFTER, "possible")[0]          # finishes it
    met_by = IntervalDomain(EndpointRange(2000, 2000), EndpointRange(2500, 2500))
    assert not reference.relate(met_by, ref, AFTER, "possible")[0]
    assert reference.relate(met_by, ref, frozenset({"after", "met-by"}), "possible")[0]


def test_month_precision_is_honoured_on_both_sides():
    jul_2005 = span(*MONTH_JUL_2005)                  # "July 2005": the acquisition is some time in July
    jul_20 = span(MONTH_JUL_2005[0] + 19 * DAY, MONTH_JUL_2005[0] + 20 * DAY - 1)
    keep, vs = reference.relate(jul_20, jul_2005, AFTER, "possible")
    assert keep and vs["certainty"] == "possible" and "before" in vs["relations"]   # may be after, may be before
    assert not reference.relate(jul_20, jul_2005, AFTER, "certain")[0]
    aug = span(MONTH_JUL_2005[1] + 1, MONTH_JUL_2005[1] + 31 * DAY)
    assert reference.relate(aug, jul_2005, AFTER, "certain") == (True, {"relations": ["after"], "certainty": "certain"})
    june = span(MONTH_JUL_2005[0] - 30 * DAY, MONTH_JUL_2005[0] - 1)
    # `certainty` says whether the fact certainly matches the requested relations, as for the window (P07)
    assert reference.relate(june, jul_2005, AFTER, "possible") == (False, {"relations": ["before"],
                                                                           "certainty": "possible"})


def test_unreadable_dates_are_kept_only_when_everything_is_accepted():
    from app.temporal.allen import RELATIONS
    assert reference.relate(None, span(1, 2), AFTER, "possible") == (False, None)
    assert reference.relate(None, span(1, 2), frozenset(RELATIONS), "possible") == (True, None)
