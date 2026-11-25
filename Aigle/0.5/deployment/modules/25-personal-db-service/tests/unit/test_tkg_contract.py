"""P08 / P09: TKG contract v2 -- strict request checks, whole-result ranking, keyset pagination, capabilities.
No ArcadeDB; tests/integration/test_tkg_contract.py runs tkg_search itself."""
from __future__ import annotations

import pytest
from app.models.graph_search import TKGRequest
from app.temporal import contract


def _fact(fid, start, entity="Google", relation="announced", value="something", sentence=""):
    return {"fact_id": fid, "time_start": start, "entity": entity, "relation": relation, "value": value,
            "source_sentence": sentence}


# ---------------------------------------------------------------- strict v2 requests

@pytest.mark.parametrize("extra, code, names", [
    ({"windw": {"start": "2010"}}, "UNKNOWN_FIELDS", {"fields": ["windw"]}),
    ({"reference": {"text": "the acquisition"}}, "INCOMPLETE_REFERENCE", {}),
    ({"reference_relations": ["after"]}, "INCOMPLETE_REFERENCE", {}),
    ({"reference": {"text": " "}, "reference_relations": ["after"]}, "INVALID_REFERENCE", {}),
    ({"reference": {"text": "x", "fact_id": "f"}, "reference_relations": ["after"]}, "INVALID_REFERENCE", {}),
    ({"reference": "the acquisition", "reference_relations": ["after"]}, "INVALID_REFERENCE", {}),
    ({"reference": {"text": "x"}, "reference_relations": []}, "INVALID_REFERENCE_RELATIONS", {}),
    ({"reference": {"text": "x"}, "reference_relations": ["later"]}, "INVALID_REFERENCE_RELATIONS", {}),
    ({"as_of": "2026-01-01"}, "UNSUPPORTED_FEATURES", {"features": ["as_of"]}),
    ({"source_filter": []}, "INVALID_SOURCE_FILTER", {}),                          # P21
    ({"source_filter": "v1"}, "INVALID_SOURCE_FILTER", {}),
    ({"source_filter": ["v1", " "]}, "INVALID_SOURCE_FILTER", {}),
    ({"timeline_id": "media:v9"}, "UNSUPPORTED_FEATURES", {"features": ["timeline_id"]}),
    ({"window": {"start": "2010"}, "time_start": "2011"}, "CONFLICTING_FIELDS", {}),
    ({"timezone": "Mars/Olympus"}, "INVALID_TIMEZONE", {"timezone": "Mars/Olympus"}),
])
def test_v2_refuses_by_name(extra, code, names):
    with pytest.raises(contract.ContractError) as e:
        contract.check_v2(TKGRequest(query="q", version=2, **extra))
    assert e.value.code == code
    assert all(e.value.body()[k] == v for k, v in names.items())


def test_v2_accepts_what_it_serves():
    req = TKGRequest(query="q", version=2, window={"start": "2010", "end": None}, timeline_id="world",
                     timezone="Asia/Taipei", sort="relevance", match="overlaps",
                     reference={"text": "Google acquired Android"}, reference_relations=["after", "met-by"])
    contract.check_v2(req)
    assert contract.window_of(req) == ("2010", None)


def test_v1_keeps_ignoring_unknown_fields():
    req = TKGRequest(query="q", windw={"start": "2010"})
    assert req.version is None and req.model_extra == {"windw": {"start": "2010"}}     # tkg_search never checks v1


# ---------------------------------------------------------------- ranking and pages

FACTS = [_fact("f1", "2003-10-01", "Android Inc.", "founded", "in Palo Alto"),
         _fact("f2", "2005-07-01", "Google", "acquired", "the company"),
         _fact("f3", "2008-01-01", "Hugo Barra", "served", "as product spokesperson"),
         _fact("f4", None, "Google", "said", "nothing dated"),
         _fact("f5", "2013-08-01", "Hugo Barra", "left", "Google to join Xiaomi")]


def _all_pages(facts, **kw):
    req, out, pages = TKGRequest(query=kw.pop("query", "q"), version=2, **kw), [], 0
    while True:
        shown, cursor, total = contract.page(facts, req)
        out += [f["fact_id"] for f in shown]
        pages += 1
        if cursor is None:
            return out, pages, total
        req = req.model_copy(update={"cursor": cursor})


def test_time_order_pages_cover_every_match_once():
    ids, pages, total = _all_pages(FACTS, limit=2)
    assert ids == ["f1", "f2", "f3", "f5", "f4"]          # undated last, as in v1
    assert (pages, total) == (3, 5)


def test_relevance_scores_every_match_not_the_first_page():
    """Hugo Barra's facts are the last in time order; relevance still puts them first."""
    ids, _, _ = _all_pages(FACTS, query="What was Hugo Barra doing?", sort="relevance", limit=2)
    assert ids[:2] == ["f3", "f5"]
    assert sorted(ids) == ["f1", "f2", "f3", "f4", "f5"]


def test_facts_written_between_pages_are_not_skipped_or_repeated():
    req = TKGRequest(query="q", version=2, limit=2)
    first, cursor, _ = contract.page(FACTS, req)
    later = [*FACTS, _fact("f0", "2001-01-01"), _fact("f6", "2010-01-01")]   # f0 sorts before the cursor
    rest, seen = [], [f["fact_id"] for f in first]
    while cursor:
        shown, cursor, _ = contract.page(later, req.model_copy(update={"cursor": cursor}))
        rest += [f["fact_id"] for f in shown]
    assert seen + rest == ["f1", "f2", "f3", "f6", "f5", "f4"]   # f6 picked up in place, f0 was before page 1


def test_a_page_with_more_after_it_is_never_complete():
    shown, cursor, total = contract.page(FACTS, TKGRequest(query="q", version=2, limit=5))
    assert (len(shown), cursor, total) == (5, None, 5)
    shown, cursor, _ = contract.page(FACTS, TKGRequest(query="q", version=2, limit=4))
    assert len(shown) == 4 and cursor


def test_cursors_belong_to_their_query():
    _, cursor, _ = contract.page(FACTS, TKGRequest(query="q", version=2, limit=2))
    with pytest.raises(contract.ContractError) as e:
        contract.page(FACTS, TKGRequest(query="other question", version=2, limit=2, cursor=cursor))
    assert e.value.code == "CURSOR_FOR_ANOTHER_QUERY"
    shown, _, _ = contract.page(FACTS, TKGRequest(query="q", version=2, limit=3, cursor=cursor))   # page size may change
    assert [f["fact_id"] for f in shown] == ["f3", "f5", "f4"]
    with pytest.raises(contract.ContractError) as e:
        contract.page(FACTS, TKGRequest(query="q", version=2, cursor="not-a-cursor"))
    assert e.value.code == "INVALID_CURSOR"


def test_query_terms():
    assert contract.query_terms("What did Google do in 2010?") == ["google"]
    assert contract.query_terms("谷歌收購") == ["谷歌", "歌收", "收購"]


# ---------------------------------------------------------------- capabilities

def test_capabilities_say_what_is_served():
    from app.routers.temporal import capabilities
    caps = capabilities()
    assert caps["contract_versions"] == [1, 2]
    assert set(caps["tkg"]["unsupported"]) == {"as_of"}
    assert caps["tkg"]["reference"]["forms"] == ["fact_id", "text"]                  # P16
    assert len(caps["tkg"]["relations"]) == 13 and caps["tkg"]["sort"] == ["time", "relevance"]
    assert caps["operations"]["status"] and caps["operations"]["parse"] == "/personal/temporal/parse"   # TMP-7
    assert caps["temporal_v2"] is False                                          # PD_TEMPORAL_V2 off by default
    assert caps["operations"]["reason"] == "/personal/temporal/reason/{version_id}"            # P15
    assert {"derived", "support_ids"} <= set(caps["tkg"]["fact_fields"])
    assert caps["extraction"]["media_types"] == ["documents", "videos", "audios", "images"]
    assert "pdf_creation_date" in caps["extraction"]["deictic_anchors"]          # P13
