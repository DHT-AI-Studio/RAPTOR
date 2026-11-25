"""
P17: a claimed date is read the way TKG writes dates (D8 `when`), since an answer copies them --
"2013-07..2013-09" and a trailing "(no end date stated)" -- besides what the date grammar reads.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.searcher import claimed_range

TZ = "Asia/Taipei"


def ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(TZ)).timestamp() * 1000)


def test_a_range_as_tkg_writes_it_spans_both_ends():
    lo, hi = claimed_range("2013-07..2013-09", TZ)
    assert lo == ms("2013-07-01T00:00:00") and hi == ms("2013-10-01T00:00:00") - 1


def test_the_stated_note_is_not_part_of_the_date():
    assert claimed_range("2010-06 (no end date stated)", TZ) == claimed_range("2010-06", TZ)
    assert claimed_range("until 2014 (no start date stated)", TZ) == claimed_range("until 2014", TZ)


def test_what_the_grammar_reads_is_unchanged_and_nonsense_is_unreadable():
    assert claimed_range("July 2005", TZ) == claimed_range("2005-07", TZ)
    assert claimed_range("2008 to 2013", TZ) is not None
    assert claimed_range("sometime", TZ) is None
    assert claimed_range("2013-07..soon", TZ) is None
