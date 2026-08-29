"""
MV-12 Memory Compaction Service.

Async service wrapper for Module 26.  Core token-counting and tail-retention
logic is adapted from compactmemory.py (project root reference implementation).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from core.config import settings
from models.memory import (
    CompactEvaluateRequest,
    CompactEvaluation,
    CompactRequest,
    CompactResponse,
    SummaryFrame,
)
from services import meta_index, module07_client
from services.long_term_memory import LongTermMemoryService
from services.memvid_store import sync_append, sync_timeline
from services.multimedia_memory import MultimediaMemoryService
from services.session_shards import get_active_shard_path, shard0_path, shard_paths

logger = logging.getLogger(__name__)

# Frame types that mark compaction bookkeeping rather than actual conversation
# content — excluded when a session's turns are recollected for a future pass.
# Also used by app/routers/management.py to keep GDPR stats/export counts
# from conflating summary/boundary bookkeeping frames with real turns.
NON_TURN_FRAME_TYPES = {"summary", "compact_boundary"}


# ── Metrics (compactdesign.md §11) ─────────────────────────────────────────────
# HTTP-level request count/latency (incl. P95 compact endpoint latency) comes
# for free from prometheus_fastapi_instrumentator wired up in main.py. These
# cover the compaction-specific business metrics the design calls out, which
# aren't derivable from generic per-route HTTP timing.

_TOKEN_BUCKETS = (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, float("inf"))
_TURN_COUNT_BUCKETS = (1, 5, 10, 20, 50, 100, 200, float("inf"))
_LATENCY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 20, 60, 120, float("inf"))

compact_pre_tokens_histogram = Histogram(
    "memory_compact_pre_tokens",
    "Estimated token count of a session's turns before compaction",
    buckets=_TOKEN_BUCKETS,
)
compact_post_tokens_histogram = Histogram(
    "memory_compact_post_tokens",
    "Estimated token count (summary + tail) after compaction",
    buckets=_TOKEN_BUCKETS,
)
compact_turns_compacted_histogram = Histogram(
    "memory_compact_turns_compacted",
    "Number of turns folded into a summary per compaction",
    buckets=_TURN_COUNT_BUCKETS,
)
compact_turns_kept_histogram = Histogram(
    "memory_compact_turns_kept",
    "Number of turns preserved verbatim (tail) per compaction",
    buckets=_TURN_COUNT_BUCKETS,
)
compact_summary_write_seconds_histogram = Histogram(
    "memory_compact_summary_write_seconds",
    "Latency of writing a summary frame to MemVID",
    buckets=_LATENCY_BUCKETS,
)
compact_llm_call_seconds_histogram = Histogram(
    "memory_compact_llm_call_seconds",
    "Latency of a Module 07 LLM summarization call",
    labelnames=("path",),  # "llm_compact" (fresh) | "session_memory" (incremental)
    buckets=_LATENCY_BUCKETS,
)
compact_llm_output_tokens_histogram = Histogram(
    "memory_compact_llm_output_tokens",
    "Estimated output token count of an LLM summarization call",
    labelnames=("path",),
    buckets=_TOKEN_BUCKETS,
)
compact_threshold_exceeded_counter = Counter(
    "memory_compact_threshold_exceeded_total",
    "Compactions where post-compact tokens still met/exceeded the auto-compact "
    "threshold after the trim-to-budget mitigation (fail-open)",
)
compact_evaluate_counter = Counter(
    "memory_compact_evaluate_total",
    "Budget evaluations completed, via POST /memory/compact/evaluate or the "
    "pre-compact check at the start of compact_session",
)
compact_session_memory_used_counter = Counter(
    "memory_compact_session_memory_used_total",
    "Compactions that reused the existing session-memory summary via an "
    "incremental update instead of a fresh LLM summarization",
)
compact_llm_fallback_used_counter = Counter(
    "memory_compact_llm_fallback_used_total",
    "Compactions that required a fresh full LLM summarization pass",
)
compact_summary_written_counter = Counter(
    "memory_compact_summary_written_total",
    "Summary frames successfully persisted to MemVID",
)
compact_failed_open_counter = Counter(
    "memory_compact_failed_open_total",
    "Compactions that failed (e.g. LLM error) and returned uncompacted, "
    "letting the caller continue with full context",
)


# ── Token estimation (chars/4 convention, matches compactmemory.py) ───────────

def rough_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, len(value) // 4) if value else 0
    if isinstance(value, dict):
        return sum(rough_token_count(k) + rough_token_count(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(rough_token_count(item) for item in value)
    return rough_token_count(str(value))


def estimate_turn_tokens(turn: dict) -> int:
    return (
        rough_token_count(turn.get("user_message", ""))
        + rough_token_count(turn.get("assistant_response", ""))
        + rough_token_count(turn.get("tool_calls") or [])
        + rough_token_count(turn.get("search_results") or [])
    )


# ── LLM summarizer (Module 07) ────────────────────────────────────────────────

_SUMMARIZE_SYSTEM = """\
Summarise the provided conversation turns into a compact memory record.
Structure your response with these exact headings:

# Current State
What the session is about and what was accomplished.

# User Intent
The user's goals and objectives expressed in this conversation.

# Key Decisions
Important decisions made or confirmed.

# Errors and Issues
Any errors, failures, or unresolved problems.

# Open Actions
Next steps, pending tasks, or follow-up items.

# Key Facts
Important entities, preferences, numbers, dates, or facts mentioned.

# Source References
Significant topics, documents, or resources referenced.

Be concise. Output plain markdown only — no surrounding code blocks.\
"""


# ── Session memory quality gate (compactdesign.md §8.2 steps 2-3) ────────────

def is_session_memory_empty(content: str) -> bool:
    """True if an existing summary has no reusable content.

    Catches the empty string and the degenerate case where a summary is
    nothing but "# heading" lines with no body text under them (e.g. an LLM
    response that got cut off before writing anything, or echoed the prompt's
    section skeleton back verbatim). Reusing either as incremental-summary
    context would feed the LLM zero signal while still counting as the
    "preferred path" — worse than just running a fresh summarization.
    """
    stripped = content.strip()
    if not stripped:
        return True
    semantic_lines = [
        line for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return not semantic_lines


def truncate_session_memory_for_compact(
    content: str, max_section_tokens: int = 2000
) -> tuple[str, bool]:
    """Truncate oversized "# heading" sections at line boundaries.

    An incrementally-updated summary can grow section-by-section across many
    compaction passes (each pass merges new turns into the existing text).
    This caps each section independently before it's fed back into the next
    incremental-summarize prompt, so one bloated section can't silently eat
    the whole summary budget. Truncates on line boundaries only, never
    mid-line, so the markdown stays valid.
    """
    max_chars_per_section = max_section_tokens * 4
    output_lines: list[str] = []
    current_header = ""
    current_lines: list[str] = []
    was_truncated = False

    def flush() -> None:
        nonlocal was_truncated
        if current_header:
            output_lines.append(current_header)
        section_text = "\n".join(current_lines)
        if len(section_text) <= max_chars_per_section:
            output_lines.extend(current_lines)
            return
        used_chars = 0
        for line in current_lines:
            next_size = used_chars + len(line) + 1
            if next_size > max_chars_per_section:
                break
            output_lines.append(line)
            used_chars = next_size
        output_lines.append("[... section truncated for length ...]")
        was_truncated = True

    for line in content.splitlines():
        if line.startswith("# "):
            flush()
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    return "\n".join(output_lines).strip(), was_truncated


def fit_text_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    """Hard-trim text to at most max_tokens (chars/4 convention), on line
    boundaries, appending a marker if anything was cut.

    Used as the "reduce" step in the threshold-exceeded escalation
    (compactdesign.md §13): when a freshly-generated summary plus the kept
    tail still exceeds the compact threshold, this is the one lever Module 26
    controls (it doesn't assemble the final LLM prompt, so it can't "reduce
    retrieved snippets" — that happens in Module 15/21).
    """
    if max_tokens <= 0:
        return "", bool(text.strip())
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text, False
    marker = "\n[... summary truncated to fit budget ...]"
    budget = max(0, max_chars - len(marker))
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        next_size = used + len(line) + 1
        if next_size > budget:
            break
        kept.append(line)
        used = next_size
    return "\n".join(kept).rstrip() + marker, True


def _format_turn_for_summary(t: dict, index: int) -> str:
    """Render one turn as prompt text, including tool_calls/search_results.

    These two fields carry the actual work product of a turn (what a tool
    returned, what was retrieved) — dropping them here means that information
    is permanently lost once the turn is summarized away, since nothing else
    references it. Keeping them in the prompt lets the LLM's "Key Facts" /
    "Source References" sections actually capture it.
    """
    lines = [
        f"Turn {t.get('turn_index', index)}",
        f"User: {t.get('user_message', '')}",
        f"Assistant: {t.get('assistant_response', '')}",
    ]
    tool_calls = t.get("tool_calls") or []
    if tool_calls:
        lines.append(f"Tool calls: {json.dumps(tool_calls, ensure_ascii=False, default=str)}")
    search_results = t.get("search_results") or []
    if search_results:
        lines.append(
            f"Search results: {json.dumps(search_results, ensure_ascii=False, default=str)}"
        )
    return "\n".join(lines)


async def _llm_summarize(turns: list[dict], custom_instructions: str | None = None) -> str:
    turns_text = "\n\n".join(
        _format_turn_for_summary(t, i) for i, t in enumerate(turns)
    )
    prompt = _SUMMARIZE_SYSTEM
    if custom_instructions:
        prompt += f"\n\nAdditional instructions: {custom_instructions}"
    prompt += f"\n\nConversation turns to summarise:\n{turns_text}"
    started = time.perf_counter()
    result = await module07_client.summarize(prompt)
    compact_llm_call_seconds_histogram.labels(path="llm_compact").observe(
        time.perf_counter() - started
    )
    compact_llm_output_tokens_histogram.labels(path="llm_compact").observe(
        rough_token_count(result)
    )
    return result


async def _incremental_summarize(
    existing_summary: str,
    new_turns: list[dict],
    custom_instructions: str | None = None,
) -> str:
    """Merge new turns into an existing summary; falls back to existing on LLM failure."""
    if not new_turns:
        return existing_summary
    turns_text = "\n\n".join(
        _format_turn_for_summary(t, i) for i, t in enumerate(new_turns)
    )
    prompt = (
        f"{_SUMMARIZE_SYSTEM}\n\n"
        f"Existing memory summary:\n{existing_summary}\n\n"
        f"New turns to incorporate:\n{turns_text}\n\n"
        "Update the memory summary to incorporate the new turns. "
        "Preserve the section structure and merge new information."
    )
    if custom_instructions:
        prompt += f"\n\nAdditional instructions: {custom_instructions}"
    started = time.perf_counter()
    try:
        result = await module07_client.summarize(prompt)
    except Exception as exc:
        logger.warning("Incremental summary failed, keeping existing summary: %s", exc)
        return existing_summary
    compact_llm_call_seconds_histogram.labels(path="session_memory").observe(
        time.perf_counter() - started
    )
    compact_llm_output_tokens_histogram.labels(path="session_memory").observe(
        rough_token_count(result)
    )
    return result


# ── Sidecar write helper ──────────────────────────────────────────────────────

def _sync_write_summary_frame(path: str, frame: dict) -> str:
    """Write a summary frame into a session .mv2.  Returns frame_id."""
    store_frame = {
        "title": f"summary [{frame.get('source', '')}] {frame['session_id']}",
        "label": "summary",
        "text": frame["text"],
        # All extra fields land in .meta.json via sync_append sidecar logic
        "frame_type": "summary",
        "summary_id": frame["summary_id"],
        "source": frame.get("source", ""),
        "source_frame_ids": frame.get("source_frame_ids", []),
        "source_turn_range": frame.get("source_turn_range", {}),
        "preserved_tail_from_turn": frame.get("preserved_tail_from_turn"),
        "token_count_before": frame.get("token_count_before", 0),
        "token_count_after": frame.get("token_count_after", 0),
        "timestamp": time.time(),
        "session_id": frame["session_id"],
        "user_id": frame["user_id"],
        "created_at": frame.get("created_at", ""),
    }
    return sync_append(path, store_frame)


def _sync_write_boundary_frame(path: str, frame: dict) -> str:
    """Write a compact_boundary frame into a session .mv2.  Returns frame_id.

    Marks the point where older turns were replaced by a summary (design
    compactdesign.md §7.2).  preserved_segment lets a future reader locate the
    summary frame (anchor) and the first/last turns kept verbatim (head/tail),
    mirroring the uuid-chain approach in compactmemory.py but addressed by
    MemVID frame_id since turns here don't carry provider message uuids.
    """
    store_frame = {
        "title": f"compact_boundary [{frame.get('source', '')}] {frame['session_id']}",
        "label": "compact_boundary",
        "text": f"Compact boundary ({frame.get('trigger', '')}/{frame.get('source', '')})",
        "frame_type": "compact_boundary",
        "boundary_id": frame["boundary_id"],
        "session_id": frame["session_id"],
        "user_id": frame["user_id"],
        "trigger": frame.get("trigger", ""),
        "source": frame.get("source", ""),
        "pre_compact_tokens": frame.get("pre_compact_tokens", 0),
        "post_compact_tokens": frame.get("post_compact_tokens", 0),
        "preserved_segment": frame.get("preserved_segment", {}),
        "timestamp": frame.get("timestamp", time.time()),
        "created_at": frame.get("created_at", ""),
    }
    return sync_append(path, store_frame)


def _make_boundary_frame(
    *,
    user_id: str,
    session_id: str,
    trigger: str,
    source: str,
    pre_compact_tokens: int,
    post_compact_tokens: int,
    anchor_frame_id: str,
    turns_to_keep: list[dict],
) -> dict:
    return {
        "boundary_id": str(uuid4()),
        "user_id": user_id,
        "session_id": session_id,
        "trigger": trigger,
        "source": source,
        "pre_compact_tokens": pre_compact_tokens,
        "post_compact_tokens": post_compact_tokens,
        "preserved_segment": {
            "head_frame_id": str(turns_to_keep[0].get("frame_id", "")) if turns_to_keep else None,
            "anchor_frame_id": anchor_frame_id,
            "tail_frame_id": str(turns_to_keep[-1].get("frame_id", "")) if turns_to_keep else None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": time.time(),
    }


# ── Service ───────────────────────────────────────────────────────────────────

class CompactMemoryService:
    def __init__(
        self,
        storage_root: str | None = None,
        long_term_svc: LongTermMemoryService | None = None,
        multimedia_svc: MultimediaMemoryService | None = None,
    ) -> None:
        self._root = Path(storage_root or settings.storage_root)
        self._long_term = long_term_svc or LongTermMemoryService(storage_root)
        self._multimedia = multimedia_svc or MultimediaMemoryService(storage_root)

    def _session_path(self, user_id: str, session_id: str) -> str:
        """Shard-0 path. Kept for unit-test direct seeding/inspection
        compatibility — reads below fan out across _session_shard_paths
        since a session can span more than one shard (see session_shards.py)."""
        return shard0_path(self._root, user_id, session_id)

    def _session_shard_paths(self, user_id: str, session_id: str) -> list[str]:
        return shard_paths(self._root, user_id, session_id)

    async def _all_session_frames(self, user_id: str, session_id: str) -> list[dict]:
        shards = self._session_shard_paths(user_id, session_id)
        if not shards:
            return []
        per_shard = await asyncio.gather(*(
            asyncio.to_thread(sync_timeline, p) for p in shards
        ))
        frames = [f for shard_frames in per_shard for f in shard_frames]
        frames.sort(key=lambda f: float(f.get("timestamp", 0)))
        return frames

    # ── Budget helpers ────────────────────────────────────────────────────────

    def _threshold(self, context_window: int, max_tokens: int | None = None) -> int:
        reserved = max(max_tokens or 0, settings.compact_reserved_output_tokens)
        effective = max(0, context_window - reserved)
        return max(0, effective - settings.compact_buffer_tokens)

    def evaluate(
        self,
        messages: list[dict],
        context_window: int,
        extra_tokens: int = 0,
        max_tokens: int | None = None,
    ) -> CompactEvaluation:
        token_count = sum(rough_token_count(m) for m in messages) + extra_tokens
        threshold = self._threshold(context_window, max_tokens)
        compact_evaluate_counter.inc()
        return CompactEvaluation(
            token_count=token_count,
            context_window=context_window,
            auto_compact_threshold=threshold,
            should_compact=token_count >= threshold,
            tokens_over_threshold=max(0, token_count - threshold),
        )

    async def _combined_context_tokens(self, user_id: str, session_turns: list[dict]) -> int:
        """Session turns + the user's long-term facts + multimedia snippets —
        the actual sources compactdesign.md §7/§8.1 describe as making up
        prompt context. Shared by evaluate_session() and compact_session()'s
        own under-budget gate so both agree on the same combined total."""
        session_tokens = sum(estimate_turn_tokens(t) for t in session_turns)
        facts = await self._long_term.get_facts(user_id)
        fact_tokens = sum(rough_token_count(f.get("text", "")) for f in facts)
        media = await self._multimedia.list_all(user_id)
        media_tokens = sum(rough_token_count(f.get("text", "")) for f in media)
        return session_tokens + fact_tokens + media_tokens

    async def evaluate_session(
        self,
        user_id: str,
        session_id: str,
        context_window: int,
        extra_tokens: int = 0,
        max_tokens: int | None = None,
    ) -> CompactEvaluation:
        """Estimate the real combined-context token budget for one session.

        Unlike `evaluate()` (which only sizes a client-supplied message list),
        this aggregates the session's own archived turns plus the user's
        long-term facts and multimedia snippets. `extra_tokens` covers the one
        piece with no store to read from yet: the current, not-yet-archived turn.
        """
        all_frames = await self._all_session_frames(user_id, session_id)
        session_turns = [
            f for f in all_frames
            if f.get("frame_type") not in NON_TURN_FRAME_TYPES
            and f.get("label") not in NON_TURN_FRAME_TYPES
        ]

        token_count = await self._combined_context_tokens(user_id, session_turns) + extra_tokens
        threshold = self._threshold(context_window, max_tokens)
        compact_evaluate_counter.inc()

        logger.info(
            "memory_compact_evaluate session=%s pre_tokens=%d threshold=%d",
            session_id, token_count, threshold,
        )

        return CompactEvaluation(
            token_count=token_count,
            context_window=context_window,
            auto_compact_threshold=threshold,
            should_compact=token_count >= threshold,
            tokens_over_threshold=max(0, token_count - threshold),
        )

    # ── Tail calculation ──────────────────────────────────────────────────────

    @staticmethod
    def _close_provider_message_groups(
        session_turns: list[dict], capped: list[dict]
    ) -> list[dict]:
        """Pull in preceding turns so the tail boundary never splits a
        streamed-assistant group.

        A single logical assistant reply can arrive as several separate
        `append_turn` calls (progressive/streamed writes) that share one
        `provider_message_id`. If the oldest kept turn belongs to such a
        group, walk backward and absorb every turn sharing that id — the
        group must be entirely inside the tail or entirely summarised,
        never split across the boundary.
        """
        if not capped:
            return capped
        by_index = {int(t.get("turn_index", 0)): t for t in session_turns}
        kept_indices = {int(t.get("turn_index", 0)) for t in capped}
        capped = list(capped)
        while True:
            head_pmid = capped[0].get("provider_message_id")
            if not head_pmid:
                break
            prev_index = int(capped[0].get("turn_index", 0)) - 1
            if prev_index in kept_indices:
                break
            prev_turn = by_index.get(prev_index)
            if prev_turn is None or prev_turn.get("provider_message_id") != head_pmid:
                break
            capped.insert(0, prev_turn)
            kept_indices.add(prev_index)
        return capped

    def _calculate_tail(
        self, session_turns: list[dict], boundary_turn_index: int
    ) -> list[dict]:
        """Return the turns to preserve as the recent full-fidelity tail.

        Starts with all turns after boundary_turn_index, capped at
        max_tail_tokens (most-recent wins).  If the result is below the
        min requirements the window expands backward into summarised turns.
        Finally, the boundary is closed around any streamed-assistant
        provider-message-id groups (see _close_provider_message_groups) —
        this last step is not re-capped against max_tail_tokens, since
        atomicity of a streamed group takes priority over the soft token cap.

        Invariant: selection operates on whole turns only — a turn's
        tool_calls/search_results always travel with it (compactdesign.md
        §8.5's tool_use/tool_result pairing concern doesn't apply here,
        since each turn already bundles a resolved call+result; there is no
        sub-turn message array to split).

        _apply_mandatory_floors adds two more turns on top: the most recent
        compact_keep_turns turns, and any tool/result turn within
        compact_tool_result_retention_hours — both regardless of token cost.
        """
        after = [
            t for t in session_turns
            if int(t.get("turn_index", 0)) > boundary_turn_index
        ]
        if not after:
            return []

        # Cap at max_tail_tokens, keeping most-recent turns
        capped: list[dict] = []
        total = 0
        for t in reversed(after):
            tokens = estimate_turn_tokens(t)
            if total + tokens > settings.compact_max_tail_tokens and capped:
                break
            total += tokens
            capped.insert(0, t)

        # If already meets minimums we're done
        if (
            total >= settings.compact_min_tail_tokens
            and len(capped) >= settings.compact_min_text_messages
        ):
            return self._apply_mandatory_floors(session_turns, after, capped)

        # Expand backward past the boundary to fill minimum requirements
        before = [
            t for t in session_turns
            if int(t.get("turn_index", 0)) <= boundary_turn_index
        ]
        for t in reversed(before):
            tokens = estimate_turn_tokens(t)
            if total + tokens > settings.compact_max_tail_tokens:
                break
            total += tokens
            capped.insert(0, t)
            if (
                total >= settings.compact_min_tail_tokens
                and len(capped) >= settings.compact_min_text_messages
            ):
                break

        return self._apply_mandatory_floors(session_turns, after, capped)

    def _apply_mandatory_floors(
        self, session_turns: list[dict], after: list[dict], capped: list[dict]
    ) -> list[dict]:
        kept_ids = {id(t) for t in capped}
        merged = list(capped)

        for t in after[-settings.compact_keep_turns:]:
            if id(t) not in kept_ids:
                kept_ids.add(id(t))
                merged.append(t)

        cutoff_ts = time.time() - settings.compact_tool_result_retention_hours * 3600
        for t in session_turns:
            if id(t) in kept_ids:
                continue
            has_tool_result = bool(t.get("tool_calls") or t.get("search_results"))
            if has_tool_result and float(t.get("timestamp", 0)) >= cutoff_ts:
                kept_ids.add(id(t))
                merged.append(t)

        merged.sort(key=lambda t: int(t.get("turn_index", 0)))
        return self._close_provider_message_groups(session_turns, merged)

    # ── Main compact ──────────────────────────────────────────────────────────

    async def compact_session(
        self,
        user_id: str,
        session_id: str,
        context_window: int,
        trigger: str = "auto",
        max_tokens: int | None = None,
        last_summarized_frame_id: str | None = None,
        custom_instructions: str | None = None,
        dry_run: bool = False,
    ) -> CompactResponse:
        correlation_id = str(uuid4())
        all_frames = await self._all_session_frames(user_id, session_id)
        if not all_frames:
            return CompactResponse(
                compacted=False, source="no_session",
                pre_compact_tokens=0, post_compact_tokens=0,
                turns_compacted=0, turns_kept=0,
                summary_frame_written=False, fallback_used=False,
            )

        session_turns = [
            f for f in all_frames
            if f.get("frame_type") not in NON_TURN_FRAME_TYPES
            and f.get("label") not in NON_TURN_FRAME_TYPES
        ]
        existing_summaries = [
            f for f in all_frames
            if f.get("frame_type") == "summary" or f.get("label") == "summary"
        ]

        if not session_turns:
            return CompactResponse(
                compacted=False, source="no_turns",
                pre_compact_tokens=0, post_compact_tokens=0,
                turns_compacted=0, turns_kept=0,
                summary_frame_written=False, fallback_used=False,
            )

        pre_tokens = sum(estimate_turn_tokens(t) for t in session_turns)
        combined_tokens = await self._combined_context_tokens(user_id, session_turns)
        threshold = self._threshold(context_window, max_tokens)
        compact_pre_tokens_histogram.observe(pre_tokens)
        compact_evaluate_counter.inc()

        logger.info(
            "memory_compact_evaluate session=%s pre_tokens=%d combined_tokens=%d threshold=%d",
            session_id, pre_tokens, combined_tokens, threshold,
        )

        if combined_tokens < threshold:
            return CompactResponse(
                compacted=False, source="under_budget",
                pre_compact_tokens=pre_tokens, post_compact_tokens=pre_tokens,
                turns_compacted=0, turns_kept=len(session_turns),
                summary_frame_written=False, fallback_used=False,
            )

        # Determine the compact boundary (turn_index of the last summarised turn)
        boundary_turn_index = -1
        latest_summary: dict | None = None

        if last_summarized_frame_id:
            for turn in session_turns:
                if str(turn.get("frame_id", "")) == last_summarized_frame_id:
                    boundary_turn_index = int(turn.get("turn_index", 0))
                    break
        elif existing_summaries:
            latest_summary = max(
                existing_summaries, key=lambda s: float(s.get("timestamp", 0))
            )
            raw_range = latest_summary.get("source_turn_range", {})
            if isinstance(raw_range, str):
                try:
                    raw_range = json.loads(raw_range)
                except json.JSONDecodeError:
                    raw_range = {}
            boundary_turn_index = int(raw_range.get("to", -1))

        turns_to_keep = self._calculate_tail(session_turns, boundary_turn_index)
        keep_ids = {str(t.get("frame_id", "")) for t in turns_to_keep}
        turns_to_summarize = [
            t for t in session_turns if str(t.get("frame_id", "")) not in keep_ids
        ]

        if not turns_to_summarize:
            return CompactResponse(
                compacted=False, source="nothing_to_compact",
                pre_compact_tokens=pre_tokens, post_compact_tokens=pre_tokens,
                turns_compacted=0, turns_kept=len(session_turns),
                summary_frame_written=False, fallback_used=False,
            )

        # Preferred path: incremental update on top of existing summary text
        source = "llm_compact"
        fallback_used = True
        summary_text: str | None = None

        existing_summary_text = (latest_summary or {}).get("text", "")
        if latest_summary and not is_session_memory_empty(existing_summary_text):
            truncated_summary, was_truncated = truncate_session_memory_for_compact(
                existing_summary_text, settings.compact_max_section_tokens
            )
            if was_truncated:
                logger.info(
                    "memory_compact_session_memory_truncated session=%s", session_id
                )
            try:
                summary_text = await _incremental_summarize(
                    truncated_summary, turns_to_summarize, custom_instructions
                )
                source = "session_memory"
                fallback_used = False
                compact_session_memory_used_counter.inc()
                logger.info("memory_compact_session_memory_used session=%s", session_id)
            except Exception as exc:
                logger.warning(
                    "Session memory path failed for %s: %s — falling back to LLM",
                    session_id, exc,
                )
        elif latest_summary:
            logger.info(
                "memory_compact_session_memory_rejected session=%s reason=empty_or_template_only",
                session_id,
            )

        # Fallback: full LLM summarisation
        if summary_text is None:
            try:
                summary_text = await _llm_summarize(turns_to_summarize, custom_instructions)
                compact_llm_fallback_used_counter.inc()
                logger.info(
                    "memory_compact_llm_fallback_used session=%s turns=%d",
                    session_id, len(turns_to_summarize),
                )
            except Exception as exc:
                compact_failed_open_counter.inc()
                logger.warning(
                    "memory_compact_failed_open session=%s correlation_id=%s reason=%s",
                    session_id, correlation_id, exc,
                )
                return CompactResponse(
                    compacted=False, source="llm_failed",
                    pre_compact_tokens=pre_tokens, post_compact_tokens=pre_tokens,
                    turns_compacted=0, turns_kept=len(session_turns),
                    summary_frame_written=False, fallback_used=True,
                )

        tail_tokens = sum(estimate_turn_tokens(t) for t in turns_to_keep)
        post_tokens = rough_token_count(summary_text) + tail_tokens
        # Keep the same combined (session + facts + media) accounting as the pre-check above.
        non_session_tokens = max(0, combined_tokens - pre_tokens)
        threshold_exceeded = (post_tokens + non_session_tokens) >= threshold

        # Escalation (compactdesign.md §13): reduce the one thing Module 26
        # controls — the freshly-generated summary — to fit whatever budget
        # remains after the (non-negotiable) tail. If the tail alone already
        # meets or exceeds the threshold, no amount of summary trimming can
        # fix it — fail open and let the caller see threshold_exceeded=true.
        if threshold_exceeded:
            summary_budget = max(0, threshold - tail_tokens - non_session_tokens)
            # summary_budget == 0 means the tail alone already consumes the
            # whole threshold — trimming the summary to nothing would destroy
            # its content for no gain (post_tokens would still be >= threshold
            # once tail_tokens alone is counted). Only trim when there's a
            # genuinely usable budget to trim down to.
            if summary_budget > 0 and rough_token_count(summary_text) > summary_budget:
                summary_text, was_trimmed = fit_text_to_token_budget(summary_text, summary_budget)
                if was_trimmed:
                    logger.info(
                        "memory_compact_summary_trimmed_to_budget session=%s summary_budget=%d",
                        session_id, summary_budget,
                    )
                post_tokens = rough_token_count(summary_text) + tail_tokens
                threshold_exceeded = (post_tokens + non_session_tokens) >= threshold

            if threshold_exceeded:
                compact_threshold_exceeded_counter.inc()
                logger.warning(
                    "memory_compact_threshold_exceeded session=%s post_tokens=%d threshold=%d "
                    "tail_tokens=%d — failing open",
                    session_id, post_tokens, threshold, tail_tokens,
                )

        summary_id = str(uuid4())
        source_frame_ids = [str(t.get("frame_id", "")) for t in turns_to_summarize]
        source_turn_range = {
            "from": int(session_turns[0].get("turn_index", 0)),
            "to": int(turns_to_summarize[-1].get("turn_index", 0)),
        }
        preserved_tail_from_turn = (
            int(turns_to_keep[0].get("turn_index", 0)) if turns_to_keep else None
        )

        summary_frame_written = False
        boundary_id: str | None = None
        if not dry_run:
            # Summary/boundary bookkeeping frames always land on the current
            # active (possibly freshly-rolled-over) shard, regardless of
            # which shard(s) the turns being summarized actually live on.
            path = await get_active_shard_path(self._root, user_id, session_id)
            frame_data = {
                "frame_type": "summary",
                "summary_id": summary_id,
                "user_id": user_id,
                "session_id": session_id,
                "source": source,
                "source_frame_ids": source_frame_ids,
                "source_turn_range": source_turn_range,
                "preserved_tail_from_turn": preserved_tail_from_turn,
                "token_count_before": pre_tokens,
                "token_count_after": post_tokens,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "text": summary_text,
            }
            write_started = time.perf_counter()
            try:
                summary_frame_id = await asyncio.to_thread(
                    _sync_write_summary_frame, path, frame_data
                )
                summary_frame_written = True
                compact_summary_write_seconds_histogram.observe(
                    time.perf_counter() - write_started
                )
                compact_summary_written_counter.inc()
                logger.info(
                    "memory_compact_summary_written session=%s summary_id=%s",
                    session_id, summary_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to write summary frame for session %s: %s — result still returned",
                    session_id, exc,
                )

            if summary_frame_written:
                boundary_data = _make_boundary_frame(
                    user_id=user_id,
                    session_id=session_id,
                    trigger=trigger,
                    source=source,
                    pre_compact_tokens=pre_tokens,
                    post_compact_tokens=post_tokens,
                    anchor_frame_id=summary_frame_id,
                    turns_to_keep=turns_to_keep,
                )
                try:
                    await asyncio.to_thread(
                        _sync_write_boundary_frame, path, boundary_data
                    )
                    boundary_id = boundary_data["boundary_id"]
                    logger.info(
                        "memory_compact_boundary_written session=%s boundary_id=%s",
                        session_id, boundary_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to write compact_boundary frame for session %s: %s — result still returned",
                        session_id, exc,
                    )

        compact_post_tokens_histogram.observe(post_tokens)
        compact_turns_compacted_histogram.observe(len(turns_to_summarize))
        compact_turns_kept_histogram.observe(len(turns_to_keep))

        return CompactResponse(
            compacted=True,
            source=source,
            summary_id=summary_id,
            boundary_id=boundary_id,
            pre_compact_tokens=pre_tokens,
            post_compact_tokens=post_tokens,
            turns_compacted=len(turns_to_summarize),
            turns_kept=len(turns_to_keep),
            summary_frame_written=summary_frame_written,
            fallback_used=fallback_used,
            threshold_exceeded=threshold_exceeded,
            summary_tokens=post_tokens - tail_tokens,
        )

    # ── Summary management ────────────────────────────────────────────────────

    async def get_session_summaries(
        self, user_id: str, session_id: str
    ) -> list[SummaryFrame]:
        all_frames = await self._all_session_frames(user_id, session_id)
        result: list[SummaryFrame] = []
        for f in all_frames:
            if f.get("frame_type") != "summary" and f.get("label") != "summary":
                continue
            raw_range = f.get("source_turn_range", {})
            if isinstance(raw_range, str):
                try:
                    raw_range = json.loads(raw_range)
                except json.JSONDecodeError:
                    raw_range = {}
            try:
                result.append(SummaryFrame(
                    frame_id=str(f.get("frame_id", "")),
                    summary_id=f.get("summary_id", ""),
                    source=f.get("source", ""),
                    source_turn_range=raw_range,
                    preserved_tail_from_turn=f.get("preserved_tail_from_turn"),
                    token_count_before=int(f.get("token_count_before", 0)),
                    token_count_after=int(f.get("token_count_after", 0)),
                    created_at=f.get("created_at", ""),
                    text=f.get("text", ""),
                ))
            except Exception:
                continue
        return result

    async def delete_summary(
        self, user_id: str, session_id: str, summary_id: str
    ) -> bool:
        shards = self._session_shard_paths(user_id, session_id)
        if not shards:
            return False

        # Each frame needs deleting from the specific shard it lives on, so
        # keep (frame, shard_path) pairs rather than a flat merged list.
        per_shard = await asyncio.gather(*(
            asyncio.to_thread(sync_timeline, p) for p in shards
        ))
        all_frames: list[tuple[dict, str]] = [
            (f, p) for p, frames in zip(shards, per_shard) for f in frames
        ]

        summary_frame_id: str | None = None
        for f, shard_path in all_frames:
            if (
                (f.get("frame_type") == "summary" or f.get("label") == "summary")
                and f.get("summary_id") == summary_id
            ):
                summary_frame_id = str(f.get("frame_id", ""))
                await asyncio.to_thread(meta_index.remove_entry, shard_path, summary_frame_id)
                break

        if summary_frame_id is None:
            return False

        # Cascade: drop the compact_boundary frame(s) anchored to this summary so
        # a future compaction pass doesn't try to resolve a dangling anchor_frame_id.
        for f, shard_path in all_frames:
            if f.get("frame_type") != "compact_boundary" and f.get("label") != "compact_boundary":
                continue
            segment = f.get("preserved_segment", {})
            if isinstance(segment, str):
                try:
                    segment = json.loads(segment)
                except json.JSONDecodeError:
                    segment = {}
            if str(segment.get("anchor_frame_id", "")) == summary_frame_id:
                await asyncio.to_thread(
                    meta_index.remove_entry, shard_path, str(f.get("frame_id", ""))
                )

        return True
