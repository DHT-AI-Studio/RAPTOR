"""Backlog-derived EN/ZH golden cases; original section 5.2 is not supplied."""
import json
from datetime import datetime
from pathlib import Path

import pytest
from app.temporal.normalize import NORMALIZER_VERSION, normalize

_FIXTURES = Path(__file__).parent / "fixtures"


def milliseconds(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


@pytest.mark.parametrize("case", json.loads((_FIXTURES / "temporal_en.json").read_text()))
def test_english_golden(case):
    result = normalize(case["raw"], case.get("timezone", "UTC"), source_published=case.get("source_published"))
    assert result["start"]["lo_ms"] == milliseconds(case["lo"])
    assert result["end"]["hi_ms"] == milliseconds(case["hi"])
    assert result["precision"] == case["precision"]
    assert result["normalizer_version"] == NORMALIZER_VERSION


@pytest.mark.parametrize("case", json.loads((_FIXTURES / "temporal_zh.json").read_text()))
def test_chinese_golden(case):
    test_english_golden(case)


@pytest.mark.parametrize("text", ["2023-13-01", "2023-02-30", "02-30", "2023-02-29",
                                   "2024-00-01", "2024年十三月一日", "民國0年", "2024年第五季"])
def test_invalid_calendar_never_guesses(text):
    result = normalize(text)
    assert result["extent"] == "UNRESOLVED"
    assert result["raw"] == text
    assert result["start"]["lo_ms"] is None


def test_since_2008_only_confirms_activity_at_publication():
    result = normalize("since 2008", source_published="2019-05-01")
    assert result["end"] == {"state": "ongoing", "lo_ms": None, "hi_ms": None}
    assert result["confirmed_active_at"] == milliseconds("2019-05-01T00:00:00Z")
    assert result["start"]["hi_ms"] == milliseconds("2008-12-31T23:59:59.999Z")


def test_since_start_cannot_follow_its_confirmation():
    result = normalize("since 2008", source_published="2008-05-01")
    assert result["start"]["hi_ms"] == result["confirmed_active_at"]
    assert normalize("since 2008", source_published="2007-12-31")["reason"] == "anchor_precedes_start"


@pytest.mark.parametrize("text", ["2024-02-29T12:30:00.1234567Z", "2024-02-29T12:30:00+0800",
                                   "2024-02-29T12:30:00+08:", "2024-02-299", "12024-02-29",
                                   "2024-02-29 12:30:123", "2024-02-29T12:30:00Zbad"])
def test_malformed_iso_tokens_do_not_resolve_a_prefix(text):
    assert normalize(text)["extent"] == "UNRESOLVED"


@pytest.mark.parametrize("text,source,start", [
    ("last year", "2024-02-29", "2023-01-01T00:00:00Z"),
    ("上個月", "2024-03-31", "2024-02-01T00:00:00Z"),
])
def test_relative_periods_ignore_source_day_overflow(text, source, start):
    assert normalize(text, source_published=source)["start"]["lo_ms"] == milliseconds(start)


@pytest.mark.parametrize("text", ["yesterday", "三年前", "09:30", "last year"])
def test_relative_clock_requires_explicit_anchor(text):
    assert normalize(text)["reason"] == "missing_anchor"


@pytest.mark.parametrize("text", ["ayer", "hier", "вчера", "어제"])
def test_unsupported_language_preserves_evidence(text):
    result = normalize(text)
    assert result["reason"] == "unsupported_language"
    assert result["raw"] == text


@pytest.mark.parametrize("text,reason", [
    ("2024-03-10T02:30:00", "nonexistent_local_time"),
    ("2024-11-03T01:30:00", "ambiguous_local_time"),
])
def test_dst_local_clock_does_not_guess(text, reason):
    assert normalize(text, "America/New_York")["reason"] == reason


def test_structured_date_in_other_language():
    result = normalize("Publié le 2024-02-29.")
    assert result["start"]["lo_ms"] == milliseconds("2024-02-29T00:00:00Z")
    assert result["raw"] == "Publié le 2024-02-29."


def test_reversed_range_and_multiple_spans_are_unresolved():
    assert normalize("2024 to 2023")["reason"] == "reversed_range"
    assert normalize("on 2024-01-01 and 2024-02-01")["reason"] == "multiple_temporal_spans"


@pytest.mark.parametrize("text, millisecond", [
    ("2011-05-03T10:00:00.25Z", 250),     # 1-2 / 4-5 fraction digits: Python 3.10's fromisoformat
    ("2011-05-03T10:00:00.5Z", 500),      # rejects them, 3.11 accepts -- same result on both now
    ("2011-05-03T10:00:00.2500Z", 250),
    ("2011-05-03T10:00:00.250Z", 250),
])
def test_fraction_digits_parse_the_same_on_every_python(text, millisecond):
    result = normalize(text)
    assert result["precision"] == "millisecond"
    assert result["start"]["lo_ms"] % 1000 == millisecond


def test_spaced_chinese_date_keeps_its_raw_text():
    """「2010 年 6 月」 reads as 「2010年6月」 (a space around 年 / 月 / 日 is common in Chinese writing); the
    evidence stays as written."""
    result = normalize("2010 年 6 月")
    assert (result["precision"], result["raw"]) == ("month", "2010 年 6 月")


def test_a_space_between_two_numbers_is_not_removed():
    assert normalize("2010 2011")["extent"] == "UNRESOLVED"
