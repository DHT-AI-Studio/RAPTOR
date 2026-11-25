"""CM-5 / VIE01-408: unit tests for the shared common/moderation_flag_reporter.py,
exercised from this module's suite (same precedent as 23-guardrail-service's own
test_raptor_audit.py testing the shared raptor_audit package).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

pytestmark = pytest.mark.asyncio


def _mock_session(status: int = 201, raise_on_post: Exception | None = None):
    calls: list[tuple[str, dict]] = []

    resp = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    resp.status = status

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        if raise_on_post is not None:
            raise raise_on_post
        return resp

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    return session, calls


async def test_one_post_per_category_deduplicated_order_preserved():
    from moderation_flag_reporter import report_moderation_flags

    session, calls = _mock_session()
    await report_moderation_flags(
        session, "http://guardrail/", branch_id="b1", version_id="v1", asset_path="a1",
        categories=["nsfw", "violence", "nsfw"],
        model="m", scores={"nsfw": 0.9, "violence": 0.7}, chunk_id="c1",
    )

    assert [c[0] for c in calls] == ["http://guardrail/moderation/flags"] * 2
    assert [c[1]["category"] for c in calls] == ["nsfw", "violence"]
    assert [c[1]["score"] for c in calls] == [0.9, 0.7]
    assert all(c[1]["chunk_id"] == "c1" and c[1]["model"] == "m" and c[1]["branch_id"] == "b1" for c in calls)
    assert all(c[1]["version_id"] == "v1" and c[1]["asset_path"] == "a1" for c in calls)
    assert all(c[1]["moment_id"] is None for c in calls)


async def test_trailing_slash_in_guardrail_url_does_not_double_up():
    from moderation_flag_reporter import report_moderation_flags

    session, calls = _mock_session()
    await report_moderation_flags(
        session, "http://guardrail/", branch_id="b1", version_id="v1", categories=["s"], model="m", asset_path="a1",
    )

    assert calls[0][0] == "http://guardrail/moderation/flags"


async def test_empty_categories_is_a_no_op():
    from moderation_flag_reporter import report_moderation_flags

    session, calls = _mock_session()
    await report_moderation_flags(
        session, "http://guardrail", branch_id="b1", version_id="v1", categories=[], model="m", asset_path="a1",
    )

    session.post.assert_not_called()
    assert calls == []


@pytest.mark.parametrize("status", [400, 500])
async def test_a_non_201_status_is_logged_not_raised(status):
    from moderation_flag_reporter import report_moderation_flags

    session, calls = _mock_session(status=status)
    await report_moderation_flags(
        session, "http://guardrail", branch_id="b1", version_id="v1", asset_path="a1",
        categories=["s"], model="m", chunk_id="c1",
    )

    assert len(calls) == 1  # it was tried, even though the status wasn't 201


async def test_a_client_error_on_one_category_does_not_stop_the_rest():
    from moderation_flag_reporter import report_moderation_flags

    session, calls = _mock_session(raise_on_post=aiohttp.ClientError("boom"))
    await report_moderation_flags(
        session, "http://guardrail", branch_id="b1", version_id="v1", asset_path="a1",
        categories=["a", "b"], model="m", chunk_id="c1",
    )

    assert len(calls) == 2  # both attempted despite each one failing
