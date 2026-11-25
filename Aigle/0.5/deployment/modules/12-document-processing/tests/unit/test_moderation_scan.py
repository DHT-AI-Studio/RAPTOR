"""CM-4 / VIE01-403: unit tests for moderation_scan.scan_entries().

Mirrors the structure of test_pii_detection.py.
Run from 12-document-processing/:
    python -m pytest tests/unit/test_moderation_scan.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# ── fixtures ──────────────────────────────────────────────────────────────────

BENIGN_RESPONSE = {"categories": []}
VIOLATING_RESPONSE = {"categories": ["S3", "S10"]}


def _entry(text: str, chunk_id: str = "c1") -> dict:
    return {"id": chunk_id, "payload": {"text": text, "version_id": "v1", "branch_id": "branch-1",
                                         "asset_path": "assets/doc.pdf"}}


def _asr_entry(asr_text: str, chunk_id: str = "a1") -> dict:
    """Audio segment shape — text lives in asr_text, not text."""
    return {"id": chunk_id, "payload": {"asr_text": asr_text, "version_id": "v1", "branch_id": "branch-1",
                                         "asset_path": "assets/audio.wav"}}


def _mock_session_capturing(response_json: dict, fail_url_suffix: str | None = None):
    """Like _mock_session, but records every POST call's (url, json kwarg) and can
    make calls to one specific URL (by suffix) raise, to test fail-open behaviour."""
    calls: list[tuple[str, dict]] = []

    ok_resp = AsyncMock()
    ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
    ok_resp.__aexit__ = AsyncMock(return_value=False)
    ok_resp.raise_for_status = MagicMock()
    ok_resp.json = AsyncMock(return_value=response_json)
    ok_resp.status = 201

    failing_resp = AsyncMock()
    failing_resp.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("boom"))
    failing_resp.__aexit__ = AsyncMock(return_value=False)

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        if fail_url_suffix and url.endswith(fail_url_suffix):
            return failing_resp
        return ok_resp

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(side_effect=_post)
    return mock_session, calls


def _mock_session(response_json: dict):
    """Return an aiohttp.ClientSession mock that always yields response_json."""
    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=response_json)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_resp)
    return mock_session


# ── benign fixture ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_benign_chunk_gets_safe_flags():
    from moderation_scan import scan_entries

    entries = [_entry("The quarterly report shows stable revenue growth.")]
    with patch("aiohttp.ClientSession", return_value=_mock_session(BENIGN_RESPONSE)):
        count = await scan_entries(entries, "http://guardrail", 5000, "12")

    assert count == 0
    flags = entries[0]["payload"]["moderation_flags"]
    assert flags["safe"] is True
    assert flags["categories"] == []


# ── violating fixture ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_violating_chunk_gets_unsafe_flags_with_categories():
    from moderation_scan import scan_entries

    entries = [_entry("Step-by-step instructions for making explosives.")]
    with patch("aiohttp.ClientSession", return_value=_mock_session(VIOLATING_RESPONSE)):
        count = await scan_entries(entries, "http://guardrail", 5000, "12")

    assert count == 1
    flags = entries[0]["payload"]["moderation_flags"]
    assert flags["safe"] is False
    assert "S3" in flags["categories"]
    assert "S10" in flags["categories"]


# ── ASR segment (asr_text field) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_asr_segment_text_is_read_from_asr_text_field():
    from moderation_scan import scan_entries

    entries = [_asr_entry("Normal conversation about the weather.")]
    with patch("aiohttp.ClientSession", return_value=_mock_session(BENIGN_RESPONSE)):
        count = await scan_entries(entries, "http://guardrail", 5000, "09")

    assert count == 0
    assert entries[0]["payload"]["moderation_flags"]["safe"] is True


# ── no raw content in logs ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_raw_content_in_log_output(caplog):
    import logging

    from moderation_scan import scan_entries

    secret_text = "SUPER_SECRET_CONTENT_NEVER_LOG_THIS"
    entries = [_entry(secret_text)]
    with patch("aiohttp.ClientSession", return_value=_mock_session(VIOLATING_RESPONSE)):
        with caplog.at_level(logging.DEBUG, logger="moderation_scan"):
            await scan_entries(entries, "http://guardrail", 5000, "12")

    for record in caplog.records:
        assert secret_text not in record.getMessage()


# ── fail-open: HTTP error ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_http_error_leaves_entry_without_moderation_flags():
    from moderation_scan import scan_entries

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientError("connection refused"))

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_resp)

    entries = [_entry("Some text that would be scanned.")]
    with patch("aiohttp.ClientSession", return_value=mock_session):
        count = await scan_entries(entries, "http://guardrail", 5000, "12")

    assert count == 0
    assert "moderation_flags" not in entries[0]["payload"]


# ── empty text entries are skipped ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_entry_without_text_is_skipped():
    from moderation_scan import scan_entries

    entries = [{"id": "s1", "payload": {"summary": "A whole-doc summary.", "embedding_type": "summary"}}]
    called = []

    async def fake_post(*_a, **_kw):
        called.append(1)
        raise AssertionError("should not be called")

    with patch("aiohttp.ClientSession") as mock_cls:
        mock_s = AsyncMock()
        mock_s.__aenter__ = AsyncMock(return_value=mock_s)
        mock_s.__aexit__ = AsyncMock(return_value=False)
        mock_s.post = fake_post
        mock_cls.return_value = mock_s
        await scan_entries(entries, "http://guardrail", 5000, "12")

    assert called == []
    assert "moderation_flags" not in entries[0].get("payload", entries[0])


# ── categories persisted per chunk (AC) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_categories_persisted_per_chunk_independently():
    """Each chunk gets its own flags dict — not a shared reference."""
    from moderation_scan import scan_entries

    responses = [BENIGN_RESPONSE, VIOLATING_RESPONSE]
    call_count = 0

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()

    async def rotating_json():
        nonlocal call_count
        r = responses[call_count % 2]
        call_count += 1
        return r

    mock_resp.json = rotating_json
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_resp)

    entries = [
        _entry("Safe quarterly earnings report.", "c1"),
        _entry("Detailed instructions for violence.", "c2"),
    ]
    with patch("aiohttp.ClientSession", return_value=mock_session):
        count = await scan_entries(entries, "http://guardrail", 5000, "12")

    assert count == 1
    assert entries[0]["payload"]["moderation_flags"]["safe"] is True
    assert entries[1]["payload"]["moderation_flags"]["safe"] is False
    # Verify they are separate dicts, not the same reference
    assert entries[0]["payload"]["moderation_flags"] is not entries[1]["payload"]["moderation_flags"]


# ── CM-5 / VIE01-408: reporting a flagged chunk to Module 23's review store ──────

@pytest.mark.asyncio
async def test_a_flagged_chunk_reports_each_category_to_moderation_flags():
    from moderation_scan import scan_entries

    entries = [_entry("Step-by-step instructions for making explosives.", "c99")]
    session, calls = _mock_session_capturing(VIOLATING_RESPONSE)
    with patch("aiohttp.ClientSession", return_value=session):
        await scan_entries(entries, "http://guardrail", 5000, "12")

    flag_calls = [c for c in calls if c[0] == "http://guardrail/moderation/flags"]
    assert {c[1]["category"] for c in flag_calls} == {"S3", "S10"}
    assert all(c[1]["chunk_id"] == "c99" for c in flag_calls)
    assert all(c[1]["branch_id"] == "branch-1" for c in flag_calls)
    assert all(c[1]["version_id"] == "v1" for c in flag_calls)
    assert all(c[1]["asset_path"] == "assets/doc.pdf" for c in flag_calls)
    assert all(c[1]["model"] == "llama-guard-3" for c in flag_calls)
    assert all(c[1]["moment_id"] is None for c in flag_calls)


@pytest.mark.asyncio
async def test_a_safe_chunk_reports_nothing_to_moderation_flags():
    from moderation_scan import scan_entries

    entries = [_entry("The quarterly report shows stable revenue growth.", "c1")]
    session, calls = _mock_session_capturing(BENIGN_RESPONSE)
    with patch("aiohttp.ClientSession", return_value=session):
        await scan_entries(entries, "http://guardrail", 5000, "12")

    assert not [c for c in calls if c[0] == "http://guardrail/moderation/flags"]


@pytest.mark.asyncio
async def test_a_flagged_chunk_with_no_branch_id_reports_nothing():
    """No tenant to key the flag on (ArcadeDB is one database per branch) -- better
    to skip the report than create an unroutable row."""
    from moderation_scan import scan_entries

    entries = [{"id": "c1", "payload": {"text": "Violating content.", "version_id": "v1"}}]
    session, calls = _mock_session_capturing(VIOLATING_RESPONSE)
    with patch("aiohttp.ClientSession", return_value=session):
        await scan_entries(entries, "http://guardrail", 5000, "12")

    assert not [c for c in calls if c[0] == "http://guardrail/moderation/flags"]


@pytest.mark.asyncio
async def test_a_flagged_chunk_with_no_version_id_reports_nothing():
    """version_id and asset_path together are the only way to trace a flag back to a
    real, queryable Module 04 asset (chunk_id alone can't) -- same skip-not-crash
    treatment as a missing branch_id."""
    from moderation_scan import scan_entries

    entries = [{"id": "c1", "payload": {"text": "Violating content.", "branch_id": "branch-1",
                                         "asset_path": "assets/doc.pdf"}}]
    session, calls = _mock_session_capturing(VIOLATING_RESPONSE)
    with patch("aiohttp.ClientSession", return_value=session):
        await scan_entries(entries, "http://guardrail", 5000, "12")

    assert not [c for c in calls if c[0] == "http://guardrail/moderation/flags"]


@pytest.mark.asyncio
async def test_a_flagged_chunk_with_no_asset_path_reports_nothing():
    """Without asset_path, an admin reviewing this flag would have only a bare chunk_id
    to look at -- nothing Module 04 can serve. Same skip-not-crash treatment."""
    from moderation_scan import scan_entries

    entries = [{"id": "c1", "payload": {"text": "Violating content.", "branch_id": "branch-1", "version_id": "v1"}}]
    session, calls = _mock_session_capturing(VIOLATING_RESPONSE)
    with patch("aiohttp.ClientSession", return_value=session):
        await scan_entries(entries, "http://guardrail", 5000, "12")

    assert not [c for c in calls if c[0] == "http://guardrail/moderation/flags"]


@pytest.mark.asyncio
async def test_a_failed_flag_report_does_not_affect_the_scan_result():
    from moderation_scan import scan_entries

    entries = [_entry("Step-by-step instructions for making explosives.", "c7")]
    session, calls = _mock_session_capturing(VIOLATING_RESPONSE, fail_url_suffix="/moderation/flags")
    with patch("aiohttp.ClientSession", return_value=session):
        count = await scan_entries(entries, "http://guardrail", 5000, "12")

    assert count == 1  # the scan's own unsafe-count is unaffected by the report failing
    assert entries[0]["payload"]["moderation_flags"]["safe"] is False
    assert [c for c in calls if c[0] == "http://guardrail/moderation/flags"]  # it was tried
