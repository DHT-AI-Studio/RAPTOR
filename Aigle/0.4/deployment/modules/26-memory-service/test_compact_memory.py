"""
Unit tests for CompactMemoryService (MV-12).

Run from the 26-memory-service/ directory:
    pip install memvid-sdk pydantic-settings fakeredis pytest pytest-asyncio
    python -m pytest test_compact_memory.py -v
"""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio

from services.compact_memory import (
    CompactMemoryService,
    rough_token_count,
    estimate_turn_tokens,
    _format_turn_for_summary,
    _llm_summarize,
    is_session_memory_empty,
    truncate_session_memory_for_compact,
    fit_text_to_token_budget,
)
from prometheus_client import REGISTRY
from services.long_term_memory import FactAddRequest, LongTermMemoryService
from services.memvid_store import sync_append, sync_timeline
from services.multimedia_memory import MultimediaMemoryService, VideoIndexRequest
from services.session_memory import SessionMemoryService, SessionSearchRequest, TurnAppendRequest
from fakeredis import FakeAsyncRedis
from core.config import settings as cfg


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def svc(tmp_path):
    """CompactMemoryService backed by a real (temp) MemVID store."""
    yield CompactMemoryService(storage_root=str(tmp_path))


@pytest_asyncio.fixture
async def session_svc(tmp_path):
    redis = FakeAsyncRedis(decode_responses=True)
    yield SessionMemoryService(redis=redis, storage_root=str(tmp_path)), str(tmp_path)
    await redis.aclose()


@pytest.fixture
def small_tail_budget():
    """
    Patch compact_max_tail_tokens to 1000 so that small test messages (400 tokens/turn)
    still produce turns_to_summarize.  Without this, 20 × 400 = 8000 tokens would all
    fit within the default max_tail_tokens=40000, leaving nothing to compact.
    """
    original = cfg.compact_max_tail_tokens
    cfg.compact_max_tail_tokens = 1000
    yield
    cfg.compact_max_tail_tokens = original


async def _populate_session(
    session_svc_tuple, user_id: str, session_id: str, n: int = 20
) -> list[dict]:
    """
    Append n turns (~400 tokens each) without triggering the extractor.

    _trigger_extraction fires a background task that calls the LLM — which fails
    with DNS errors in unit tests and adds ~10s of retry delay per test.
    Patching it here keeps tests fast and quiet.
    """
    svc, storage_root = session_svc_tuple
    base_ts = time.time()
    turns = []
    with patch.object(svc, "_trigger_extraction", new=AsyncMock(return_value=None)):
        for i in range(n):
            req = TurnAppendRequest(
                user_message="x" * 800,        # ~200 tokens
                assistant_response="y" * 800,  # ~200 tokens
                timestamp=base_ts + i,
            )
            resp = await svc.append_turn(user_id, session_id, req)
            turns.append({"turn_index": resp.turn_index, "frame_id": resp.frame_id})
    return turns


# ── rough_token_count ─────────────────────────────────────────────────────────

def test_rough_token_count_string():
    assert rough_token_count("abcd") == 1         # 4 chars → 1 token
    assert rough_token_count("a" * 400) == 100    # 400 chars → 100 tokens
    assert rough_token_count("") == 0
    assert rough_token_count(None) == 0


def test_rough_token_count_dict():
    d = {"key": "a" * 40, "value": "b" * 40}
    # "key"=0, "value"=1, "a"*40=10, "b"*40=10 → ≥ 10
    assert rough_token_count(d) >= 10


def test_rough_token_count_list():
    lst = ["a" * 400, "b" * 400]
    assert rough_token_count(lst) == 200


def test_estimate_turn_tokens():
    turn = {"user_message": "a" * 400, "assistant_response": "b" * 400}
    assert estimate_turn_tokens(turn) == 200


def test_estimate_turn_tokens_includes_tool_calls_and_search_results():
    """Regression guard: tool_calls/search_results must count toward the
    turn's budget, or a turn with a large tool payload silently overflows
    max_tail_tokens without the retention math noticing."""
    bare = {"user_message": "a" * 400, "assistant_response": "b" * 400}
    with_tools = {
        **bare,
        "tool_calls": [{"name": "search", "result": "c" * 400}],
        "search_results": [{"text": "d" * 400}],
    }
    assert estimate_turn_tokens(with_tools) > estimate_turn_tokens(bare)


# ── CompactMemoryService.evaluate ─────────────────────────────────────────────

def test_evaluate_under_threshold(svc):
    messages = [{"role": "user", "content": "hi"}]
    result = svc.evaluate(messages, context_window=128000)
    assert result.should_compact is False
    assert result.auto_compact_threshold == 95000  # 128000 - 20000 - 13000
    assert result.tokens_over_threshold == 0


def test_evaluate_over_threshold(svc):
    # Create messages that sum to > 95000 tokens (each char = 0.25 tokens)
    big_msg = {"role": "user", "content": "x" * (95000 * 4 + 100)}
    result = svc.evaluate([big_msg], context_window=128000)
    assert result.should_compact is True
    assert result.tokens_over_threshold > 0


def test_evaluate_small_context_window(svc):
    result = svc.evaluate([], context_window=4096)
    # 4096 - 20000 - 13000 → clamped to 0
    assert result.auto_compact_threshold == 0
    assert result.should_compact is True  # 0 >= 0


def test_evaluate_extra_tokens(svc):
    result = svc.evaluate([], context_window=128000, extra_tokens=96000)
    assert result.should_compact is True
    assert result.token_count == 96000


def test_evaluate_max_tokens_below_default_uses_default(svc):
    # max_tokens (5000) < default reserved (20000) → reserved stays 20000
    result = svc.evaluate([], context_window=128000, max_tokens=5000)
    assert result.auto_compact_threshold == 95000  # 128000 - 20000 - 13000


def test_evaluate_max_tokens_above_default_widens_reserved(svc):
    # max_tokens (64000) > default reserved (20000) → reserved becomes 64000
    result = svc.evaluate([], context_window=128000, max_tokens=64000)
    assert result.auto_compact_threshold == 51000  # 128000 - 64000 - 13000


# ── CompactMemoryService.evaluate_session ─────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_session_no_session_no_facts_no_media(svc):
    result = await svc.evaluate_session("u_empty", "sess_empty", context_window=128000)
    assert result.token_count == 0
    assert result.should_compact is False


@pytest.mark.asyncio
async def test_evaluate_session_aggregates_session_facts_and_media(tmp_path):
    user_id = "u_agg"
    session_id = "sess_agg"
    storage_root = str(tmp_path)

    session_svc = SessionMemoryService(redis=FakeAsyncRedis(decode_responses=True), storage_root=storage_root)
    with patch.object(session_svc, "_trigger_extraction", new=AsyncMock(return_value=None)):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message="x" * 400, assistant_response="y" * 400,
        ))

    long_term_svc = LongTermMemoryService(storage_root=storage_root)
    await long_term_svc.add_fact(user_id, FactAddRequest(
        text="z" * 400, frame_type="preference",
    ))

    multimedia_svc = MultimediaMemoryService(storage_root=storage_root)
    await multimedia_svc.index_video(user_id, VideoIndexRequest(
        asset_path="videos/a.mp4", version_id="v1",
        start_sec=0.0, end_sec=5.0, transcription="w" * 400,
    ))

    compact_svc = CompactMemoryService(storage_root=storage_root)
    result = await compact_svc.evaluate_session(user_id, session_id, context_window=128000)

    # Each ~400-char field is ~100 tokens; three sources → token_count should
    # reflect all of them, not just the session turn.
    assert result.token_count >= 300
    assert result.should_compact is False


@pytest.mark.asyncio
async def test_evaluate_session_extra_tokens_pushes_over_threshold(svc):
    result = await svc.evaluate_session(
        "u_extra", "sess_extra", context_window=128000, extra_tokens=96000,
    )
    assert result.should_compact is True
    assert result.token_count == 96000


@pytest.mark.asyncio
async def test_evaluate_session_ignores_other_users_data(tmp_path):
    storage_root = str(tmp_path)
    long_term_svc = LongTermMemoryService(storage_root=storage_root)
    await long_term_svc.add_fact("owner", FactAddRequest(
        text="x" * 4000, frame_type="fact",
    ))

    compact_svc = CompactMemoryService(storage_root=storage_root)
    result = await compact_svc.evaluate_session("intruder", "sess_x", context_window=128000)
    assert result.token_count == 0


# ── _calculate_tail ───────────────────────────────────────────────────────────

def test_calculate_tail_basic(svc):
    turns = [
        {"turn_index": i, "user_message": "a" * 400, "assistant_response": "b" * 400}
        for i in range(10)
    ]
    tail = svc._calculate_tail(turns, boundary_turn_index=4)
    # 5 turns after boundary × 400 tokens = 2000 < min_tail_tokens=10000, so the
    # algorithm expands backward past the boundary to meet the minimum.  The tail
    # will therefore contain turns from both sides of the boundary.
    assert len(tail) > 0
    assert any(int(t["turn_index"]) == 9 for t in tail)   # newest always included
    assert all(t in turns for t in tail)                   # only real turns returned


def test_calculate_tail_expands_when_min_not_met(svc):
    # Only 1 turn after boundary — below min_text_messages=5
    turns = [
        {"turn_index": i, "user_message": "a" * 400, "assistant_response": "b" * 400}
        for i in range(10)
    ]
    tail = svc._calculate_tail(turns, boundary_turn_index=8)
    # Should expand backward to reach >= 5 turns
    assert len(tail) >= 5


def test_calculate_tail_empty_when_no_turns_after_boundary(svc):
    turns = [
        {"turn_index": i, "user_message": "x", "assistant_response": "y"}
        for i in range(5)
    ]
    tail = svc._calculate_tail(turns, boundary_turn_index=99)
    assert tail == []


def test_calculate_tail_keeps_tool_calls_intact_on_kept_turns(svc):
    """Turn atomicity invariant: a turn selected for the tail must keep its
    tool_calls/search_results verbatim — retention operates on whole turns,
    never splits a turn's internal fields."""
    turns = [
        {
            "turn_index": i,
            "user_message": "a" * 400,
            "assistant_response": "b" * 400,
            "tool_calls": [{"name": "lookup", "args": {"id": i}, "result": "ok"}],
        }
        for i in range(10)
    ]
    tail = svc._calculate_tail(turns, boundary_turn_index=4)
    assert len(tail) > 0
    for t in tail:
        original = turns[int(t["turn_index"])]
        assert t["tool_calls"] == original["tool_calls"]


def test_calculate_tail_preserves_streamed_assistant_group(svc):
    """A streamed assistant reply split across two append_turn calls shares one
    provider_message_id. If the boundary would land between them, both must
    end up in the tail together — never split across the summarize/keep line."""
    turns = [
        {
            "turn_index": i,
            "user_message": "a" * 3000,       # ~750 tokens
            "assistant_response": "b" * 9000,  # ~2250 tokens => ~3000 tokens/turn
        }
        for i in range(10)
    ]
    turns[4]["provider_message_id"] = "pm-A"
    turns[5]["provider_message_id"] = "pm-A"

    # after=[6,7,8,9] (4 turns, 12000 tokens) meets min_tail_tokens(10000) but
    # not min_text_messages(5) => expands backward, absorbing turn 5 only
    # (5 turns, 15000 tokens) — turn 4 (same group as 5) would be left behind
    # without the closure step.
    tail = svc._calculate_tail(turns, boundary_turn_index=5)
    tail_indices = {int(t["turn_index"]) for t in tail}
    assert 5 in tail_indices
    assert 4 in tail_indices, "streamed-assistant group must not be split across the tail boundary"


def test_calculate_tail_does_not_over_absorb_without_matching_group(svc):
    """Group-closure must not fire when provider_message_id doesn't match —
    only genuinely grouped turns get pulled across the boundary."""
    turns = [
        {
            "turn_index": i,
            "user_message": "a" * 3000,
            "assistant_response": "b" * 9000,
        }
        for i in range(10)
    ]
    turns[4]["provider_message_id"] = "pm-A"
    turns[5]["provider_message_id"] = "pm-B"  # different group — no bridge

    tail = svc._calculate_tail(turns, boundary_turn_index=5)
    tail_indices = {int(t["turn_index"]) for t in tail}
    assert 5 in tail_indices
    assert 4 not in tail_indices


# ── _calculate_tail: mandatory floors (compact_keep_turns / 24h tool-result) ───

def _override_tail_budget(min_tokens=1, min_msgs=1, keep_turns=1, max_tokens=250):
    """max_tail_tokens=250 caps the initial budget loop to ~1 turn (200
    tokens/turn here) so the mandatory floors' effect isn't masked by the
    default 40000 cap swallowing every turn regardless of the floors."""
    original = (
        cfg.compact_min_tail_tokens, cfg.compact_min_text_messages,
        cfg.compact_keep_turns, cfg.compact_max_tail_tokens,
    )
    cfg.compact_min_tail_tokens = min_tokens
    cfg.compact_min_text_messages = min_msgs
    cfg.compact_keep_turns = keep_turns
    cfg.compact_max_tail_tokens = max_tokens
    return original


def _restore_tail_budget(original):
    (
        cfg.compact_min_tail_tokens, cfg.compact_min_text_messages,
        cfg.compact_keep_turns, cfg.compact_max_tail_tokens,
    ) = original


def test_calculate_tail_keep_turns_floor_beyond_budget(svc):
    """compact_keep_turns must force in the N most recent turns even when the
    token budget alone would have stopped earlier."""
    turns = [
        {"turn_index": i, "user_message": "a" * 400, "assistant_response": "b" * 400}
        for i in range(10)
    ]
    original = _override_tail_budget(keep_turns=5)
    try:
        tail = svc._calculate_tail(turns, boundary_turn_index=-1)
    finally:
        _restore_tail_budget(original)

    tail_indices = {int(t["turn_index"]) for t in tail}
    assert {5, 6, 7, 8, 9}.issubset(tail_indices)


def test_calculate_tail_keeps_recent_tool_result_within_24h(svc):
    """A tool/result turn timestamped within the last 24h is force-kept even
    if the token budget and compact_keep_turns floor would both skip it."""
    now = time.time()
    turns = [
        {
            "turn_index": i, "user_message": "a" * 400, "assistant_response": "b" * 400,
            "timestamp": now - (10 - i) * 3600,  # turn 0 ~10h old ... turn 9 ~1h old
        }
        for i in range(10)
    ]
    turns[2]["tool_calls"] = [{"name": "lookup", "result": "ok"}]  # ~8h old — within 24h

    original = _override_tail_budget(keep_turns=1)
    try:
        tail = svc._calculate_tail(turns, boundary_turn_index=-1)
    finally:
        _restore_tail_budget(original)

    assert 2 in {int(t["turn_index"]) for t in tail}


def test_calculate_tail_does_not_keep_stale_tool_result_beyond_24h(svc):
    """The 24h floor is time-bound — a tool/result turn older than the
    retention window is not force-kept by it."""
    now = time.time()
    turns = [
        {
            "turn_index": i, "user_message": "a" * 400, "assistant_response": "b" * 400,
            "timestamp": now - (40 - i) * 3600,  # spans ~40h..31h ago
        }
        for i in range(10)
    ]
    turns[2]["tool_calls"] = [{"name": "lookup", "result": "ok"}]  # ~38h old — outside 24h

    original = _override_tail_budget(keep_turns=1)
    try:
        tail = svc._calculate_tail(turns, boundary_turn_index=-1)
    finally:
        _restore_tail_budget(original)

    assert 2 not in {int(t["turn_index"]) for t in tail}


# ── _format_turn_for_summary / LLM prompt content ─────────────────────────────

def test_format_turn_for_summary_includes_tool_calls_and_search_results():
    turn = {
        "turn_index": 3,
        "user_message": "查一下天氣",
        "assistant_response": "台北目前多雲",
        "tool_calls": [{"name": "get_weather", "result": {"temp": 28}}],
        "search_results": [{"text": "weather.gov snippet"}],
    }
    text = _format_turn_for_summary(turn, 0)
    assert "get_weather" in text
    assert "weather.gov snippet" in text


def test_format_turn_for_summary_omits_empty_tool_fields():
    turn = {"turn_index": 1, "user_message": "hi", "assistant_response": "hello"}
    text = _format_turn_for_summary(turn, 0)
    assert "Tool calls:" not in text
    assert "Search results:" not in text


@pytest.mark.asyncio
async def test_llm_summarize_prompt_carries_tool_call_content():
    """A summarized-away turn's tool_calls must reach the LLM prompt — otherwise
    that information (e.g. a tool's result) is lost with no other record of it."""
    turns = [{
        "turn_index": 0,
        "user_message": "訂 7/20 台北到東京的機票",
        "assistant_response": "已幫你查詢航班",
        "tool_calls": [{"name": "search_flights", "result": {"flight": "JL803"}}],
    }]
    with patch(
        "services.module07_client.summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nBooked flight.",
    ) as mock_call:
        await _llm_summarize(turns)

    sent_prompt = mock_call.call_args[0][0]
    assert "search_flights" in sent_prompt
    assert "JL803" in sent_prompt


# ── is_session_memory_empty ────────────────────────────────────────────────────

def test_is_session_memory_empty_blank_string():
    assert is_session_memory_empty("") is True
    assert is_session_memory_empty("   \n  \n") is True


def test_is_session_memory_empty_headings_only():
    # Only section skeleton, no body text under any heading — degenerate LLM output
    content = "# Current State\n\n# User Intent\n\n# Key Facts\n"
    assert is_session_memory_empty(content) is True


def test_is_session_memory_empty_with_real_content():
    content = "# Current State\nUser is booking a flight to Tokyo.\n\n# Key Facts\nDeparture 7/20."
    assert is_session_memory_empty(content) is False


# ── truncate_session_memory_for_compact ────────────────────────────────────────

def test_truncate_session_memory_leaves_small_content_untouched():
    content = "# Current State\nShort summary.\n\n# Key Facts\nA few facts."
    truncated, was_truncated = truncate_session_memory_for_compact(content, max_section_tokens=2000)
    assert was_truncated is False
    assert truncated == content


def test_truncate_session_memory_truncates_oversized_section():
    big_section = "\n".join(f"fact line {i}" for i in range(2000))  # far over budget
    content = f"# Current State\nfine.\n\n# Key Facts\n{big_section}"
    truncated, was_truncated = truncate_session_memory_for_compact(content, max_section_tokens=100)
    assert was_truncated is True
    assert "[... section truncated for length ...]" in truncated
    assert "# Current State" in truncated  # untouched small section preserved
    assert "fine." in truncated
    # Truncation happens on line boundaries — no line should be cut mid-way
    for line in truncated.splitlines():
        assert line == "" or line in content.splitlines() or line == "[... section truncated for length ...]"


def test_truncate_session_memory_truncates_each_section_independently():
    big = "\n".join(f"line {i}" for i in range(1000))
    content = f"# Section A\n{big}\n\n# Section B\nshort."
    truncated, was_truncated = truncate_session_memory_for_compact(content, max_section_tokens=50)
    assert was_truncated is True
    assert "# Section A" in truncated
    assert "# Section B" in truncated
    assert "short." in truncated  # small section B survives intact despite A being truncated


# ── fit_text_to_token_budget ────────────────────────────────────────────────────

def test_fit_text_to_token_budget_no_op_when_under_budget():
    text = "short text"
    trimmed, was_trimmed = fit_text_to_token_budget(text, max_tokens=1000)
    assert was_trimmed is False
    assert trimmed == text


def test_fit_text_to_token_budget_trims_on_line_boundary():
    text = "\n".join(f"line {i}" for i in range(2000))
    trimmed, was_trimmed = fit_text_to_token_budget(text, max_tokens=100)
    assert was_trimmed is True
    assert "[... summary truncated to fit budget ...]" in trimmed
    assert rough_token_count(trimmed) <= 100
    # every non-marker line must be a real line from the source (no mid-line cuts)
    for line in trimmed.splitlines():
        assert line == "" or line in text.splitlines() or line == "[... summary truncated to fit budget ...]"


def test_fit_text_to_token_budget_zero_budget_empties_text():
    trimmed, was_trimmed = fit_text_to_token_budget("some content", max_tokens=0)
    assert trimmed == ""
    assert was_trimmed is True


def test_fit_text_to_token_budget_zero_budget_noop_on_empty_text():
    trimmed, was_trimmed = fit_text_to_token_budget("", max_tokens=0)
    assert trimmed == ""
    assert was_trimmed is False


# ── compact_session — threshold-exceeded escalation ────────────────────────────

@pytest.mark.asyncio
async def test_compact_session_trims_summary_to_fit_after_threshold_exceeded(
    session_svc, small_tail_budget
):
    """A summary that alone blows the budget must get trimmed so the total
    (summary + tail) fits within threshold — the mathematically guaranteed
    outcome of fit_text_to_token_budget, regardless of how oversized the
    LLM's raw output was."""
    user_id, session_id = "u13", "sess_threshold_trim"
    svc2, storage_root = session_svc
    # 50 turns (~400 tokens each) so pre_tokens (~20000) comfortably clears a
    # context_window chosen to yield threshold=20000 — nonzero, so the "reduce"
    # step actually has budget to trim into (unlike the context_window=4096
    # trick used elsewhere, which forces threshold=0).
    await _populate_session(session_svc, user_id, session_id, n=50)

    compact = CompactMemoryService(storage_root=storage_root)
    huge_summary = "\n".join(f"fact {i}: " + "x" * 40 for i in range(3000))  # far over budget

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value=huge_summary,
    ):
        result = await compact.compact_session(
            user_id, session_id, context_window=53000  # threshold = 53000-20000-13000 = 20000
        )

    assert result.compacted is True
    assert result.post_compact_tokens <= 20000  # guaranteed by fit_text_to_token_budget's contract
    assert result.post_compact_tokens < rough_token_count(huge_summary)  # meaningfully reduced

    summaries = await compact.get_session_summaries(user_id, session_id)
    assert "[... summary truncated to fit budget ...]" in summaries[0].text


@pytest.mark.asyncio
async def test_compact_session_threshold_exceeded_when_tail_alone_too_big(
    session_svc, small_tail_budget
):
    """When the tail alone already consumes the whole threshold (context_window
    too small relative to reserved/buffer), trimming the summary can't help —
    must fail open with threshold_exceeded=True and leave the summary intact
    (emptying it would destroy content for zero benefit)."""
    user_id, session_id = "u14", "sess_threshold_stuck"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nUnmodified summary.",
    ):
        result = await compact.compact_session(
            user_id, session_id, context_window=4096  # threshold clamps to 0
        )

    assert result.compacted is True
    assert result.threshold_exceeded is True

    summaries = await compact.get_session_summaries(user_id, session_id)
    assert "Unmodified summary" in summaries[0].text
    assert "[... summary truncated to fit budget ...]" not in summaries[0].text


# ── compact_session — session memory quality gate ─────────────────────────────

@pytest.mark.asyncio
async def test_compact_session_rejects_template_only_existing_summary(session_svc, small_tail_budget):
    """An existing summary that's just section headings (no body) must not be
    reused via the incremental path — should fall straight to a fresh LLM summary."""
    user_id, session_id = "u11", "sess_template_summary"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)
    path = compact._session_path(user_id, session_id)
    await asyncio.to_thread(sync_append, path, {
        "frame_type": "summary",
        "label": "summary",
        "summary_id": "seed-template",
        "user_id": user_id,
        "session_id": session_id,
        "source": "llm_compact",
        "source_frame_ids": [],
        "source_turn_range": {"from": 0, "to": 3},
        "token_count_before": 0,
        "token_count_after": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "text": "# Current State\n\n# User Intent\n\n# Key Facts\n",
        "title": "seed template-only summary",
        "timestamp": time.time() - 10,
    })

    with patch(
        "services.compact_memory._incremental_summarize", new_callable=AsyncMock
    ) as mock_incr, patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nFresh summary.",
    ) as mock_fresh:
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    mock_incr.assert_not_called()
    mock_fresh.assert_called_once()
    assert result.compacted is True
    assert result.source == "llm_compact"


@pytest.mark.asyncio
async def test_compact_session_truncates_oversized_existing_summary(session_svc, small_tail_budget):
    """An existing summary with a bloated section must be truncated before being
    fed into the incremental-summarize prompt (bounds prompt growth over many passes)."""
    user_id, session_id = "u12", "sess_big_summary"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)
    path = compact._session_path(user_id, session_id)
    huge_section = "\n".join(f"fact {i}" for i in range(5000))
    with patch.object(cfg, "compact_max_section_tokens", 50):
        await asyncio.to_thread(sync_append, path, {
            "frame_type": "summary",
            "label": "summary",
            "summary_id": "seed-huge",
            "user_id": user_id,
            "session_id": session_id,
            "source": "llm_compact",
            "source_frame_ids": [],
            "source_turn_range": {"from": 0, "to": 3},
            "token_count_before": 0,
            "token_count_after": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "text": f"# Key Facts\n{huge_section}",
            "title": "seed huge summary",
            "timestamp": time.time() - 10,
        })

        with patch(
            "services.compact_memory._incremental_summarize",
            new_callable=AsyncMock,
            return_value="# Current State\nMerged.",
        ) as mock_incr:
            result = await compact.compact_session(user_id, session_id, context_window=4096)

        mock_incr.assert_called_once()
        sent_existing_summary = mock_incr.call_args[0][0]
        assert "[... section truncated for length ...]" in sent_existing_summary
        assert len(sent_existing_summary) < len(huge_section)
        assert result.source == "session_memory"


# ── compact_session — dry_run (no Module 07 needed) ──────────────────────────

@pytest.mark.asyncio
async def test_compact_session_under_budget(svc, session_svc):
    user_id, session_id = "u1", "sess_budget"
    # Append only 2 small turns — well under 95K threshold
    svc2, _ = session_svc
    await svc2.append_turn(user_id, session_id, TurnAppendRequest(
        user_message="hello", assistant_response="hi",
    ))
    # Point compact svc at same storage root
    compact = CompactMemoryService(storage_root=session_svc[1])
    result = await compact.compact_session(user_id, session_id, context_window=128000, dry_run=True)
    assert result.compacted is False
    assert result.source == "under_budget"


@pytest.mark.asyncio
async def test_compact_session_dry_run_no_frame_written(session_svc, small_tail_budget):
    """With enough tokens to trigger compaction, dry_run=True should not write a frame.

    small_tail_budget patches max_tail_tokens=1000 so that 20 × 400-token turns
    overflow the tail, leaving 18 turns to summarize.
    """
    user_id, session_id = "u2", "sess_dry"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nMocked summary.",
    ):
        result = await compact.compact_session(
            user_id, session_id, context_window=4096, dry_run=True
        )

    assert result.compacted is True
    assert result.summary_frame_written is False  # dry_run → no write
    assert result.turns_compacted > 0
    assert result.turns_kept >= 0


@pytest.mark.asyncio
async def test_compact_session_writes_frame(session_svc, small_tail_budget):
    """Without dry_run, a summary frame should be written to the .mv2."""
    user_id, session_id = "u3", "sess_write"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nTest summary.",
    ):
        result = await compact.compact_session(
            user_id, session_id, context_window=4096, dry_run=False
        )

    assert result.compacted is True
    assert result.summary_frame_written is True
    assert result.summary_id is not None

    # Verify it's retrievable
    summaries = await compact.get_session_summaries(user_id, session_id)
    assert len(summaries) == 1
    assert summaries[0].summary_id == result.summary_id
    assert "Test summary" in summaries[0].text


# ── compact_session — compact_boundary frame ──────────────────────────────────

@pytest.mark.asyncio
async def test_compact_session_writes_boundary_frame(session_svc, small_tail_budget):
    """A compact_boundary frame should be written alongside the summary frame,
    with preserved_segment anchored to the summary and spanning the kept tail."""
    user_id, session_id = "u7", "sess_boundary"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nBoundary test summary.",
    ):
        result = await compact.compact_session(
            user_id, session_id, context_window=4096, dry_run=False
        )

    assert result.compacted is True
    assert result.boundary_id is not None

    path = compact._session_path(user_id, session_id)
    frames = await asyncio.to_thread(sync_timeline, path)
    boundaries = [f for f in frames if f.get("frame_type") == "compact_boundary"]
    assert len(boundaries) == 1

    boundary = boundaries[0]
    assert boundary["boundary_id"] == result.boundary_id
    assert boundary["source"] == result.source

    segment = boundary["preserved_segment"]
    summaries = [f for f in frames if f.get("frame_type") == "summary"]
    assert segment["anchor_frame_id"] == summaries[0]["frame_id"]
    if result.turns_kept > 0:
        assert segment["head_frame_id"] is not None
        assert segment["tail_frame_id"] is not None
    else:
        assert segment["head_frame_id"] is None
        assert segment["tail_frame_id"] is None


@pytest.mark.asyncio
async def test_compact_session_dry_run_no_boundary_written(session_svc, small_tail_budget):
    """dry_run=True must not write a compact_boundary frame either."""
    user_id, session_id = "u8", "sess_boundary_dry"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nDry run summary.",
    ):
        result = await compact.compact_session(
            user_id, session_id, context_window=4096, dry_run=True
        )

    assert result.compacted is True
    assert result.boundary_id is None

    path = compact._session_path(user_id, session_id)
    frames = await asyncio.to_thread(sync_timeline, path)
    assert not any(f.get("frame_type") == "compact_boundary" for f in frames)


@pytest.mark.asyncio
async def test_compact_boundary_excluded_from_next_pass_turns(session_svc, small_tail_budget):
    """The compact_boundary frame from pass 1 must not be miscounted as a
    conversation turn (and re-summarized) in pass 2."""
    user_id, session_id = "u9", "sess_boundary_reuse"
    svc2, storage_root = session_svc
    compact = CompactMemoryService(storage_root=storage_root)

    await _populate_session(session_svc, user_id, session_id, n=20)
    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nFirst summary.",
    ):
        r1 = await compact.compact_session(user_id, session_id, context_window=4096)
    assert r1.compacted is True

    path = compact._session_path(user_id, session_id)
    frames_before = await asyncio.to_thread(sync_timeline, path)
    boundary_count = sum(1 for f in frames_before if f.get("frame_type") == "compact_boundary")
    assert boundary_count == 1

    await _populate_session(session_svc, user_id, session_id, n=5)
    with patch(
        "services.compact_memory._incremental_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nMerged summary.",
    ):
        r2 = await compact.compact_session(user_id, session_id, context_window=4096)

    # turns_compacted/turns_kept must only reflect real session_turn frames,
    # never the compact_boundary bookkeeping frame from the prior pass.
    if r2.compacted:
        assert r2.turns_compacted + r2.turns_kept <= 25  # 20 + 5 real turns, max


@pytest.mark.asyncio
async def test_compact_session_incremental_on_second_pass(session_svc, small_tail_budget):
    """Second compaction uses incremental path (session_memory) instead of LLM fallback."""
    user_id, session_id = "u4", "sess_incr"
    svc2, storage_root = session_svc
    compact = CompactMemoryService(storage_root=storage_root)

    # First compaction
    await _populate_session(session_svc, user_id, session_id, n=20)
    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nFirst summary.",
    ):
        r1 = await compact.compact_session(user_id, session_id, context_window=4096)
    assert r1.compacted is True

    # Append more turns to trigger second compaction
    await _populate_session(session_svc, user_id, session_id, n=5)

    # Second compaction should use incremental path
    with patch(
        "services.compact_memory._incremental_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nMerged summary.",
    ) as mock_incr:
        r2 = await compact.compact_session(user_id, session_id, context_window=4096)

    if r2.compacted:
        assert r2.source == "session_memory"
        mock_incr.assert_called_once()


@pytest.mark.asyncio
async def test_compact_session_fail_open_on_llm_error(session_svc, small_tail_budget):
    """LLM failure must return compacted=False (fail open), not raise."""
    user_id, session_id = "u5", "sess_fail"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Module 07 unreachable"),
    ):
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    assert result.compacted is False
    assert result.source == "llm_failed"
    assert result.summary_frame_written is False


@pytest.mark.asyncio
async def test_compact_session_no_session_returns_false(svc):
    result = await svc.compact_session("nobody", "no_sess", context_window=128000)
    assert result.compacted is False
    assert result.source == "no_session"


# ── delete_summary ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_summary_removes_frame(session_svc, small_tail_budget):
    user_id, session_id = "u6", "sess_del"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="Summary to delete.",
    ):
        r = await compact.compact_session(user_id, session_id, context_window=4096)

    assert r.summary_id is not None
    deleted = await compact.delete_summary(user_id, session_id, r.summary_id)
    assert deleted is True

    summaries = await compact.get_session_summaries(user_id, session_id)
    assert all(s.summary_id != r.summary_id for s in summaries)


@pytest.mark.asyncio
async def test_delete_summary_cascades_to_boundary_frame(session_svc, small_tail_budget):
    """Deleting a summary must also remove the compact_boundary frame anchored
    to it, so a future compaction pass never resolves a dangling anchor."""
    user_id, session_id = "u10", "sess_del_boundary"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="Summary with boundary to delete.",
    ):
        r = await compact.compact_session(user_id, session_id, context_window=4096)

    assert r.summary_id is not None
    assert r.boundary_id is not None

    deleted = await compact.delete_summary(user_id, session_id, r.summary_id)
    assert deleted is True

    path = compact._session_path(user_id, session_id)
    frames = await asyncio.to_thread(sync_timeline, path)
    assert not any(f.get("frame_type") == "compact_boundary" for f in frames)


# ── Metrics (compactdesign.md §11) ──────────────────────────────────────────────

def _sample_value(name: str, labels: dict | None = None) -> float:
    """Read one Prometheus sample from the default registry (0.0 if absent).

    Metrics are process-global singletons shared across every test in this
    file, so tests must compare a before/after delta rather than an absolute
    value — other tests running earlier in the same session already bumped
    these counters.
    """
    labels = labels or {}
    for metric_family in REGISTRY.collect():
        for sample in metric_family.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0.0


@pytest.mark.asyncio
async def test_compact_session_records_business_metrics(session_svc, small_tail_budget):
    user_id, session_id = "u15", "sess_metrics"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    before = {
        name: _sample_value(name)
        for name in (
            "memory_compact_pre_tokens_count",
            "memory_compact_post_tokens_count",
            "memory_compact_turns_compacted_count",
            "memory_compact_turns_kept_count",
            "memory_compact_summary_write_seconds_count",
            "memory_compact_evaluate_total",
            "memory_compact_llm_fallback_used_total",
            "memory_compact_summary_written_total",
        )
    }

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nMetrics test summary.",
    ):
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    assert result.compacted is True
    for name, prior in before.items():
        assert _sample_value(name) == prior + 1, name


@pytest.mark.asyncio
async def test_compact_session_records_session_memory_used_metric(session_svc, small_tail_budget):
    """Second pass over the same session reuses the existing summary — the
    session_memory_used counter (not llm_fallback_used) should tick."""
    user_id, session_id = "u16", "sess_metrics_sm"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nFirst pass summary.",
    ):
        first = await compact.compact_session(user_id, session_id, context_window=4096)
    assert first.compacted is True

    await _populate_session(session_svc, user_id, session_id, n=20)

    before = _sample_value("memory_compact_session_memory_used_total")
    with patch(
        "services.compact_memory._incremental_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nUpdated summary.",
    ):
        second = await compact.compact_session(user_id, session_id, context_window=4096)

    assert second.compacted is True
    assert second.source == "session_memory"
    assert _sample_value("memory_compact_session_memory_used_total") == before + 1


@pytest.mark.asyncio
async def test_compact_session_records_failed_open_metric(session_svc, small_tail_budget):
    user_id, session_id = "u17", "sess_metrics_fail"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)

    before = _sample_value("memory_compact_failed_open_total")
    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        side_effect=RuntimeError("module07 unavailable"),
    ):
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    assert result.compacted is False
    assert result.source == "llm_failed"
    assert _sample_value("memory_compact_failed_open_total") == before + 1


def test_evaluate_endpoint_increments_evaluate_metric(svc):
    before = _sample_value("memory_compact_evaluate_total")
    svc.evaluate(messages=[{"role": "user", "content": "hi"}], context_window=4096)
    assert _sample_value("memory_compact_evaluate_total") == before + 1


@pytest.mark.asyncio
async def test_llm_summarize_records_llm_call_metrics():
    labels = {"path": "llm_compact"}
    latency_before = _sample_value("memory_compact_llm_call_seconds_count", labels)
    tokens_before = _sample_value("memory_compact_llm_output_tokens_count", labels)

    with patch(
        "services.module07_client.summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nSome summary.",
    ):
        await _llm_summarize(
            [{"turn_index": 0, "user_message": "hi", "assistant_response": "hello"}]
        )

    assert _sample_value("memory_compact_llm_call_seconds_count", labels) == latency_before + 1
    assert _sample_value("memory_compact_llm_output_tokens_count", labels) == tokens_before + 1


@pytest.mark.asyncio
async def test_compact_session_threshold_exceeded_increments_counter(
    session_svc, small_tail_budget
):
    user_id, session_id = "u17", "sess_metrics_threshold"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)
    before = _sample_value("memory_compact_threshold_exceeded_total")

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nUnmodified.",
    ):
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    assert result.threshold_exceeded is True
    assert _sample_value("memory_compact_threshold_exceeded_total") == before + 1


# ── Integration: end-to-end compaction over a real (temp) MemVID store ────────
# These exercise the full append_turn -> compact_session -> search/read path
# against real storage (not just the pure functions above). The LLM call
# itself is still mocked — no network dependency on Ollama/Module 07.

@pytest.mark.asyncio
async def test_long_chat_session_compacts_and_stays_under_budget(session_svc):
    """A long session that genuinely exceeds a 128K context window's
    auto-compact threshold should compact down to something safely under
    that threshold — not just get flagged, but actually shrink."""
    user_id, session_id = "u_long", "sess_long_chat"
    svc2, storage_root = session_svc

    # 40 turns * ~3000 tokens/turn (6000 chars each field) comfortably exceeds
    # the default 128K window's ~95K auto-compact threshold. _trigger_compact
    # is also patched out here — its fire-and-forget background pass would
    # otherwise race the explicit compact_session() call below for the same
    # .mv2 file handle (MemVID only allows one writer at a time).
    with (
        patch.object(svc2, "_trigger_extraction", new=AsyncMock(return_value=None)),
        patch.object(svc2, "_trigger_compact", new=AsyncMock(return_value=None)),
    ):
        base_ts = time.time()
        for i in range(40):
            await svc2.append_turn(user_id, session_id, TurnAppendRequest(
                user_message="q" * 6000,
                assistant_response="a" * 6000,
                timestamp=base_ts + i,
            ))

    compact = CompactMemoryService(storage_root=storage_root)
    threshold = compact._threshold(context_window=128000)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value=(
            "# Current State\nLong session summarised.\n"
            "# User Intent\nExercise the compaction budget.\n"
        ),
    ):
        result = await compact.compact_session(user_id, session_id, context_window=128000)

    assert result.compacted is True
    assert result.pre_compact_tokens >= threshold  # actually was over budget
    assert result.post_compact_tokens < result.pre_compact_tokens
    assert result.post_compact_tokens < threshold  # and now comfortably under it


@pytest.mark.asyncio
async def test_200_turn_session_compacts_and_precompaction_context_still_retrievable(session_svc):
    """Literal MV-12 acceptance scenario: create a 200-turn session, trigger
    compaction, verify the context turn count actually shrinks, and verify a
    follow-up retrieval still finds content from the turns that got
    summarized away (they're archived, not deleted, so search must still
    reach them)."""
    user_id, session_id = "u_200", "sess_200_turns"
    svc2, storage_root = session_svc

    # ~1/5 the message size of the 40-turn test above (1100-1200 vs 6000
    # chars), scaled up to 200 turns — comfortably clears the 128K window's
    # ~95K auto-compact threshold (200 * ~575 tokens/turn ≈ 115K). Also
    # exercises session_shards.py rollover: a single shard can't hold all
    # 200 turns under the memvid-sdk free-tier's 50MB-per-file cap.
    with (
        patch.object(svc2, "_trigger_extraction", new=AsyncMock(return_value=None)),
        patch.object(svc2, "_trigger_compact", new=AsyncMock(return_value=None)),
    ):
        base_ts = time.time()
        for i in range(200):
            user_message = "q" * 1100
            if i == 5:
                user_message = "上次討論的冷卻系統維修計畫是什麼？" + user_message
            await svc2.append_turn(user_id, session_id, TurnAppendRequest(
                user_message=user_message,
                assistant_response="a" * 1200,
                timestamp=base_ts + i,
            ))

    compact = CompactMemoryService(storage_root=storage_root)
    threshold = compact._threshold(context_window=128000)

    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\n200-turn session summarised.\n",
    ):
        result = await compact.compact_session(user_id, session_id, context_window=128000)

    assert result.compacted is True
    assert result.pre_compact_tokens >= threshold
    assert result.turns_compacted > 0
    assert result.turns_kept < 200  # context turn count genuinely reduced
    assert result.post_compact_tokens < result.pre_compact_tokens

    hits = await svc2.search(
        user_id, session_id, SessionSearchRequest(query="冷卻系統維修計畫", top_k=5)
    )
    assert any("冷卻系統" in h.text for h in hits.hits)
    assert result.threshold_exceeded is False
    assert result.summary_frame_written is True


@pytest.mark.asyncio
async def test_compacted_summary_is_searchable_via_memvid(session_svc, small_tail_budget):
    """A summary frame written by compaction lives in the same .mv2 as the
    turns it replaced, so it must be reachable through normal session search —
    not just through get_session_summaries()."""
    user_id, session_id = "u_search", "sess_search_summary"
    svc2, storage_root = session_svc
    await _populate_session(session_svc, user_id, session_id, n=20)

    compact = CompactMemoryService(storage_root=storage_root)
    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nThe team discussed the unicorn migration rollout plan.",
    ):
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    assert result.compacted is True
    assert result.summary_frame_written is True

    hits = await svc2.search(
        user_id, session_id, SessionSearchRequest(query="unicorn migration rollout", top_k=5)
    )
    assert any("unicorn" in h.text.lower() for h in hits.hits)


@pytest.mark.asyncio
async def test_compact_user_isolation_cannot_read_or_compact_other_users_session(session_svc):
    """Storage is partitioned per user_id — user B must not be able to compact
    or list summaries for a session that only exists under user A."""
    owner_id, other_id, session_id = "u_owner", "u_intruder", "sess_isolated"
    svc2, storage_root = session_svc

    with patch.object(svc2, "_trigger_extraction", new=AsyncMock(return_value=None)):
        await svc2.append_turn(owner_id, session_id, TurnAppendRequest(
            user_message="secret question", assistant_response="secret answer",
        ))

    compact = CompactMemoryService(storage_root=storage_root)

    # Same session_id, wrong user_id => no session found under that user's
    # storage prefix, regardless of how much the owner's session has grown.
    result = await compact.compact_session(other_id, session_id, context_window=128000)
    assert result.compacted is False
    assert result.source == "no_session"

    summaries = await compact.get_session_summaries(other_id, session_id)
    assert summaries == []

    # The owner's own view is unaffected.
    owner_summaries = await compact.get_session_summaries(owner_id, session_id)
    assert owner_summaries == []  # nothing compacted yet, but path resolves fine


def test_metrics_endpoint_exposes_compact_metrics():
    """The generic Instrumentator (HTTP-level) and our custom business metrics
    share the default Prometheus registry, so both must show up on /metrics."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "memory_compact_pre_tokens" in r.text
    assert "memory_compact_threshold_exceeded_total" in r.text


# ── CompactResponse: turns_before/turns_after/summary_tokens/status aliases ────

@pytest.mark.asyncio
async def test_compact_response_exposes_ok_status_and_aliases(session_svc, small_tail_budget):
    user_id, session_id = "u_alias", "sess_alias"
    await _populate_session(session_svc, user_id, session_id, n=20)
    _svc2, storage_root = session_svc

    compact = CompactMemoryService(storage_root=storage_root)
    with patch(
        "services.compact_memory._llm_summarize",
        new_callable=AsyncMock,
        return_value="# Current State\nSummary text.",
    ):
        result = await compact.compact_session(user_id, session_id, context_window=4096)

    assert result.compacted is True
    assert result.status == "ok"
    assert result.turns_before == result.turns_compacted + result.turns_kept
    assert result.turns_after == result.turns_kept
    assert result.summary_tokens > 0
    assert result.model_dump()["status"] == "ok"


@pytest.mark.asyncio
async def test_compact_response_status_fallback_when_not_compacted(svc):
    result = await svc.compact_session("ghost_user", "ghost_session", context_window=128000)
    assert result.compacted is False
    assert result.status == "fallback"
    assert result.turns_before == 0
    assert result.turns_after == 0
