"""TMP-4 / TMP-2: one-sided windows from after / before / since / until and their Chinese forms
(以後 / 之前 / 以來 / 為止; temporal plan P02), and the keywords a TKG search filters the evidence by."""
# signal_extractor works in naive calendar days
import re
from datetime import datetime

import pytest
from app.services.signal_extractor import extract_keywords, extract_signals


@pytest.mark.parametrize("query, start, end", [
    ("What happened after 2011?", datetime(2012, 1, 1), None),          # strict: 2011 itself excluded
    ("before 2013", None, datetime(2012, 12, 31)),
    ("since 2008", datetime(2008, 1, 1), None),
    ("until 2005", None, datetime(2005, 12, 31)),
    ("till the year 2005", None, datetime(2005, 12, 31)),
    ("After the 1990s", datetime(2000, 1, 1), None),
    ("before the 20th century", None, datetime(1900, 12, 31)),
    ("in 2011", datetime(2011, 1, 1), datetime(2011, 12, 31)),            # no operator: the whole year
    ("LG phone 2010-2012", datetime(2010, 1, 1), datetime(2012, 12, 31)),  # a range takes no operator
    ("in 2016 and 2010", datetime(2010, 1, 1), datetime(2016, 12, 31)),    # out of order: never inverted
    ("Android after midnight", None, None),
    # TMP-2: the Chinese operators follow the period
    ("2011年以後發生什麼", datetime(2012, 1, 1), None),
    ("2011年之後", datetime(2012, 1, 1), None),
    ("2011 年以後", datetime(2012, 1, 1), None),
    ("2013年之前", None, datetime(2012, 12, 31)),
    ("2013年以前", None, datetime(2012, 12, 31)),
    ("2010年以來", datetime(2010, 1, 1), None),
    ("2010年起", datetime(2010, 1, 1), None),
    ("到2012年為止", None, datetime(2012, 12, 31)),
    ("1990年代以後", datetime(2000, 1, 1), None),
    ("2011年發生了什麼", datetime(2011, 1, 1), datetime(2011, 12, 31)),  # no operator: the whole year
    ("Android 在 2011 年發生了什麼", datetime(2011, 1, 1), datetime(2011, 12, 31)),
    ("iPhone 20110 sales", None, None),                                   # not a year
])
def test_time_window(query, start, end):
    s = extract_signals(query)
    assert (s.time_from, s.time_to) == (start, end)
    # P07: a period named in full -> facts sharing time with it; one open side -> the fact inside it
    assert s.match == ("overlaps" if start and end else "inside" if start or end else None)


@pytest.mark.parametrize("query, match", [
    ("What was Hugo Barra doing in 2010?", "overlaps"),
    ("What happened between 2010 and 2016?", "overlaps"),
    ("What happened after 2011?", "inside"),
    ("before 2013", "inside"),
    ("2011年以後發生什麼", "inside"),
    ("What is Android?", None),
])
def test_tkg_match(query, match):
    assert extract_signals(query).match == match


def test_classify_returns_the_match():
    from types import SimpleNamespace

    from app.routers.query import ClassifyRequest, classify_endpoint
    classifier = SimpleNamespace(people_intent=lambda q, temporal: None, classify=lambda q, k=3: SimpleNamespace(
        intent_type=["TKG"], confidence=1.0, routing_strategy="rule_based", matched_rules=[]))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(intent_classifier=classifier)))
    closed = classify_endpoint(ClassifyRequest(query="What was Hugo Barra doing in 2010?"), request)
    assert (closed.match, closed.time_from[:10], closed.time_to[:10]) == ("overlaps", "2010-01-01", "2010-12-31")
    after = classify_endpoint(ClassifyRequest(query="What happened after 2011?"), request)
    assert (after.match, after.time_to) == ("inside", None)
    # D3: an instant with its offset, never a bare date
    assert re.fullmatch(r"2012-01-01T00:00:00\.000(?:Z|[+-]\d\d:\d\d)", after.time_from)
    before = classify_endpoint(ClassifyRequest(query="2013年之前"), request)
    assert (before.match, before.time_from) == ("inside", None)
    assert re.fullmatch(r"2012-12-31T23:59:59\.999(?:Z|[+-]\d\d:\d\d)", before.time_to)


@pytest.mark.parametrize("query, keywords", [
    ("What happened after 2011?", []),
    ("LG phone 2010-2012", ["LG", "phone"]),
    ("Hugo Barra 2010-2010", ["Hugo", "Barra"]),
    ("What did Google acquire after 2011", ["Google", "acquire"]),
    ("events in the 20th century", []),
    ("Android 在 2011 年發生了什麼", ["Android"]),                        # Chinese is not segmented
    ("phone, Phone and PHONE", ["phone"]),                                # once, first spelling kept
])
def test_keywords(query, keywords):
    assert extract_keywords(query) == keywords
    assert extract_signals(query).keywords == keywords


# ── P16: reference events ─────────────────────────────────────────────────────

DURING = ["starts", "during", "finishes", "equals"]


@pytest.mark.parametrize("query, reference, relations, keywords", [
    ("What happened after Google acquired Android?", "Google acquired Android", ["after"], []),
    ("After Google acquired Android, what did Rubin do?", "Google acquired Android", ["after"], ["Rubin"]),
    ("What did Rubin do after the acquisition of Android Inc.", "the acquisition of Android Inc", ["after"], ["Rubin"]),
    ("What happened before the G1 launched?", "the G1 launched", ["before"], []),
    ("What happened during the Nexus era?", "the Nexus era", DURING, []),
    ("Who joined immediately after Android was founded?", "Android was founded", ["met-by"], ["joined"]),
    ("Google收購Android之後發生了什麼？", "Google收購Android", ["after"], []),
    ("在Google收購Android之後，Rubin做了什麼", "Google收購Android", ["after"], ["Rubin"]),
    ("Android發布期間有什麼新聞", "Android發布", DURING, []),
])
def test_reference_event(query, reference, relations, keywords):
    s = extract_signals(query)
    assert (s.reference, s.reference_relations, s.keywords) == (reference, relations, keywords)
    assert s.hint_tkg


@pytest.mark.parametrize("query", [
    "What happened after 2011?", "before the 1990s", "after the 20th century", "during 2010",
    "2011年以後發生了什麼", "When was Android founded?", "Android 在 2011 年發生了什麼",
])
def test_a_date_or_no_relation_word_is_not_a_reference(query):
    assert extract_signals(query).reference is None


def test_classify_returns_the_reference():
    from types import SimpleNamespace

    from app.routers.query import ClassifyRequest, classify_endpoint
    classifier = SimpleNamespace(people_intent=lambda q, temporal: None, classify=lambda q, k=3: SimpleNamespace(
        intent_type=["TKG"], confidence=1.0, routing_strategy="rule_based", matched_rules=[]))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(intent_classifier=classifier)))
    plan = classify_endpoint(ClassifyRequest(query="What happened after Google acquired Android?"), request)
    assert (plan.reference, plan.reference_relations) == ("Google acquired Android", ["after"])
    assert classify_endpoint(ClassifyRequest(query="What happened after 2011?"), request).reference is None


@pytest.mark.parametrize("query", [
    "When did Google acquire Android Inc., and what was released after that?",
    "What happened after this?",
    "Google收購了Android，在那之後發生了什麼？",
    "What happened after that event?",
])
def test_a_pronoun_is_not_a_reference(query):
    assert extract_signals(query).reference is None


def test_a_pronoun_is_skipped_for_a_later_reference():
    s = extract_signals("And after that, what changed after Google acquired Android?")
    assert (s.reference, s.reference_relations) == ("Google acquired Android", ["after"])


def test_a_reference_question_goes_to_tkg_whatever_the_classifier_says():
    from types import SimpleNamespace

    from app.routers.query import ClassifyRequest, classify_endpoint
    classifier = SimpleNamespace(people_intent=lambda q, temporal: None, classify=lambda q, k=3: SimpleNamespace(
        intent_type=[], confidence=0.2, routing_strategy="rule_based", matched_rules=[]))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(intent_classifier=classifier)))
    assert classify_endpoint(ClassifyRequest(query="What happened after Google acquired Android?"), request).modes == ["TKG"]
    assert classify_endpoint(ClassifyRequest(query="show me the cooling system"), request).modes == ["VideoRAG"]
