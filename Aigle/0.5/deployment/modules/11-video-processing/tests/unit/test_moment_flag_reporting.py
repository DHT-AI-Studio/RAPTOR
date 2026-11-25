"""CM-5 / VIE01-408: kafka_handler._report_moment_flags() -- reports each flagged
category of each flagged moment to Module 23's review store, alongside (not instead
of) the moderation_flags already written to Module 04/25 elsewhere in this file.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

pytestmark = pytest.mark.asyncio


def _mock_session_capturing(fail: bool = False):
    """An aiohttp.ClientSession mock that records every POST call's (url, json
    kwarg); can make every call raise, to test fail-open behaviour."""
    calls: list[tuple[str, dict]] = []

    ok_resp = AsyncMock()
    ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
    ok_resp.__aexit__ = AsyncMock(return_value=False)
    ok_resp.status = 201

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        if fail:
            raise aiohttp.ClientError("boom")
        return ok_resp

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(side_effect=_post)
    return mock_session, calls


def _moment(index: int, flagged: list[str], scores: dict | None = None) -> dict:
    return {"moment_index": index, "moderation_flags": {"flagged": flagged, "scores": scores or {}}}


async def test_one_row_per_flagged_category_per_flagged_moment(monkeypatch):
    from kafka_handler import _report_moment_flags

    monkeypatch.setenv("GUARDRAIL_URL", "http://guardrail")
    moments = [
        _moment(0, []),                                             # safe -- nothing reported
        _moment(1, ["violence", "nsfw"], {"violence": 0.9, "nsfw": 0.6}),
    ]
    rollup = {"model": "shieldgemma-2", "policy": {"policy_version": "2.0"}}
    session, calls = _mock_session_capturing()
    with patch("aiohttp.ClientSession", return_value=session):
        await _report_moment_flags("branch-1", "v1", "assets/video.mp4", moments, rollup)

    flag_calls = [c for c in calls if c[0] == "http://guardrail/moderation/flags"]
    assert {c[1]["category"] for c in flag_calls} == {"violence", "nsfw"}
    assert all(c[1]["branch_id"] == "branch-1" for c in flag_calls)
    assert all(c[1]["version_id"] == "v1" for c in flag_calls)
    assert all(c[1]["model"] == "shieldgemma-2" for c in flag_calls)
    assert all(c[1]["policy_version"] == "2.0" for c in flag_calls)
    assert all(c[1]["asset_path"] == "assets/video.mp4" and c[1]["chunk_id"] is None for c in flag_calls)
    # Every flagged moment reports under the same moment_id, every safe moment reports none
    moment_ids = {c[1]["moment_id"] for c in flag_calls}
    assert len(moment_ids) == 1 and all(moment_ids)


async def test_moment_id_matches_module_25s_stable_chunk_id_derivation():
    import hashlib

    from kafka_handler import _report_moment_flags

    moments = [_moment(3, ["hate"])]
    session, calls = _mock_session_capturing()
    with patch("aiohttp.ClientSession", return_value=session):
        await _report_moment_flags("branch-1", "v-abc", "assets/video.mp4", moments, {"model": "m"})

    expected = hashlib.sha256(b"v-abc\x1fmoment\x1f3").hexdigest()
    assert calls[0][1]["moment_id"] == expected


async def test_no_flagged_moments_reports_nothing():
    from kafka_handler import _report_moment_flags

    moments = [_moment(0, []), _moment(1, [])]
    session, calls = _mock_session_capturing()
    with patch("aiohttp.ClientSession", return_value=session):
        await _report_moment_flags("branch-1", "v1", "assets/video.mp4", moments, {"model": "m"})

    assert calls == []


async def test_a_moment_with_no_index_is_skipped_not_crashed():
    from kafka_handler import _report_moment_flags

    moments = [{"moderation_flags": {"flagged": ["violence"]}}]  # no moment_index
    session, calls = _mock_session_capturing()
    with patch("aiohttp.ClientSession", return_value=session):
        await _report_moment_flags("branch-1", "v1", "assets/video.mp4", moments, {"model": "m"})  # must not raise

    assert calls == []


async def test_report_failure_does_not_raise():
    from kafka_handler import _report_moment_flags

    moments = [_moment(0, ["violence"])]
    session, _calls = _mock_session_capturing(fail=True)
    with patch("aiohttp.ClientSession", return_value=session):
        await _report_moment_flags("branch-1", "v1", "assets/video.mp4", moments, {"model": "m"})  # must not raise
