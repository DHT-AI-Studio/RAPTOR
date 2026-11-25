"""D8: a TKG fact's dates at the precision it has (`when`), for whatever writes an answer from it.

time_start / time_end pad a month or a year to a day: "July 2024" is stored as 2024-07-01, and a
model reading that writes "on 1 July 2024". `when` is built from the stored ranges instead.
"""
import pytest
from app.models.graph_search import TKGRequest
from app.services import searcher
from app.services.temporal_normalize import (
    describe_dates,
    normalize_fact_dates,
    parse_bound,
)

from .test_graph_search import FakeClient, _stub_entity_search


def _when(time_start, time_end, tz="Asia/Taipei"):
    def rng(value):
        b = parse_bound(value, tz)
        return (b.lo_ms, b.hi_ms) if b else None
    return describe_dates(rng(time_start), rng(time_end), tz)


@pytest.mark.parametrize("time_start, time_end, when", [
    ("2024-07", None, "2024-07 (no end date stated)"),           # a month stays a month, never "1 July"
    ("2011", "2011", "2011"),
    ("2008", "2013", "2008 to 2013"),
    ("2010-06-01", "2010-06-01", "2010-06-01"),
    ("2003-10", "2005-08", "2003-10 to 2005-08"),
    (None, "2014-12-31", "until 2014-12-31 (no start date stated)"),
    ("2011年上半年", None, "2011-01..2011-06 (no end date stated)"),   # some time in those months
    ("1990年代", None, "1990..1999 (no end date stated)"),
    ("2011-05-03T10:00:00+08:00", None, "2011-05-03T10:00:00+08:00 (no end date stated)"),
    (None, None, None),
])
def test_dates_at_their_own_precision(time_start, time_end, when):
    assert _when(time_start, time_end) == when


def test_described_in_the_facts_own_timezone():
    assert _when("2012", "2012", "America/New_York") == "2012"
    # the same instants read in another zone are no longer whole days: shown as instants, not rounded
    ny = parse_bound("2012", "America/New_York")
    assert describe_dates((ny.lo_ms, ny.hi_ms), None, "Asia/Taipei").startswith("2012-01-01T13:00:00+08:00..")


@pytest.mark.asyncio
async def test_tkg_facts_carry_when_and_precision(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "s", "name": "Samsung", "type": "ORG"}])
    strike = {"fact_id": "strike", "entity": "Samsung", "relation": "strike", "value": "union",
              **normalize_fact_dates("2024-07", None, "Asia/Taipei")}
    legacy = {"fact_id": "legacy", "entity": "Samsung", "relation": "r", "value": "v",
              "time_start": "2009", "time_end": "2010"}                      # not backfilled yet: no *_state
    client = FakeClient([("entity_id IN :eids", [strike, legacy])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung", entity="Samsung"))
    got = {f["fact_id"]: f for f in resp.temporal_facts}
    assert got["strike"]["time_start"] == "2024-07-01"                       # the v1 projection, unchanged
    assert (got["strike"]["when"], got["strike"]["precision"]) == ("2024-07 (no end date stated)", "month")
    assert got["legacy"]["when"] == "2009 to 2010"
    assert not {"s_lo", "s_hi", "e_lo", "e_hi", "s_state", "e_state", "timezone"} & set(got["strike"])
    sql, _ = next((s, p) for s, p in client.calls if "entity_id IN :eids" in s)
    assert "s_lo" in sql and "timezone" in sql                                # read to build `when`


@pytest.mark.asyncio
async def test_when_survives_the_allen_filter(monkeypatch):
    """P07 classifies every fact from its range columns and drops them; `when` is built from the same
    columns, so it must be built first -- else it falls back to the padded time_start ("2024-07-01")."""
    await _stub_entity_search(monkeypatch, [])
    strike = {"fact_id": "strike", "entity": "Samsung", "relation": "strike", "value": "union",
              **normalize_fact_dates("2024-07", None, "Asia/Taipei")}
    client = FakeClient([("s_state = 'known'", [strike])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(
        query="What happened in 2024?", time_start="2024", time_end="2024", match="overlaps"))
    fact = resp.temporal_facts[0]
    assert fact["when"] == "2024-07 (no end date stated)" and fact["allen"]["certainty"] == "certain"
