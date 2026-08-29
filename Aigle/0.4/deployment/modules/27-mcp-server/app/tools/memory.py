from __future__ import annotations

import json
import logging
from typing import Annotated, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.tools import get_client

logger = logging.getLogger(__name__)

_RETRIEVE_DESC = """\
Semantic + BM25 hybrid search across the caller's memory: session history,
long-term facts/preferences, and multimedia memory.

Maps to: GET /api/0.4/memory/retrieve (Module 13 → Module 26)
"""

_STORE_DESC = """\
Write a standalone memory node (fact, preference, or entity) not tied to any
session. To record a full conversation turn, use raptor_memory_archive instead.

Maps to: POST /api/0.4/memory/store (Module 13 → Module 26)
"""

_TIMELINE_DESC = """\
Paginated, time-ascending timeline of the caller's conversation turns across
all sessions, interleaved. For a single session's timeline, narrow the
results by session_id via raptor_memory_session_summaries instead.

Maps to: GET /api/0.4/memory/timeline (Module 13 → Module 26)
"""

_MULTIMEDIA_SEARCH_DESC = """\
Semantic + BM25 hybrid search across the caller's indexed video/audio/image
memory only (not session or long-term text memory — use raptor_memory_retrieve
for a combined search across everything).

Maps to: POST /api/0.4/memory/multimedia/search (Module 13 → Module 26)
"""

_SESSION_SUMMARIES_DESC = """\
List the summary frames produced by compacting a specific session (see
raptor_memory_compact).

Maps to: GET /api/0.4/memory/sessions/{session_id}/summaries (Module 13 → Module 26)
"""

_COMPACT_DESC = """\
Compact a session: summarise its older turns into a summary frame to free up
context window space. Defaults to the caller's "default" session.

Maps to: POST /api/0.4/memory/compact (Module 13 → Module 26)
"""

_COMPACT_EVALUATE_DESC = """\
Estimate the token budget a compaction would need, without actually running
it. Pass session_id to estimate against that session's archived turns +
long-term facts + multimedia snippets combined.

Maps to: POST /api/0.4/memory/compact/evaluate (Module 13 → Module 26)
"""

_ARCHIVE_DESC = """\
Archive one conversation turn (user_message + assistant_response) into a
session. Defaults to the caller's "default" session.

Maps to: POST /api/0.4/memory/archive (Module 13 → Module 26)
"""


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_RETRIEVE_DESC.strip())
    async def raptor_memory_retrieve(
        query: Annotated[str, "Search query text"],
        top_k: Annotated[int, "Maximum results to return (1-50)"] = 5,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_retrieve: query={query!r} top_k={top_k}")

        params = {"query": query, "top_k": top_k}

        try:
            client = await get_client(ctx)
            data = await client.get_json(
                "/memory/retrieve", params=params, tool_name="raptor_memory_retrieve")
        except Exception as exc:
            await ctx.error(f"raptor_memory_retrieve failed: {exc}")
            raise

        n = len(data.get("results", [])) if isinstance(data, dict) else 0
        await ctx.info(f"raptor_memory_retrieve: {n} results")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_STORE_DESC.strip())
    async def raptor_memory_store(
        text: Annotated[str, "The fact / preference / entity description to remember"],
        frame_type: Annotated[
            Literal["conversation", "preference", "entity", "fact"],
            "conversation (important dialogue summary) | preference | entity | fact (default)",
        ] = "fact",
        session_id: Annotated[Optional[str], "Originating session ID, for traceability only"] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_store: frame_type={frame_type!r}")

        body = {"text": text, "frame_type": frame_type}
        if session_id is not None:
            body["session_id"] = session_id

        try:
            client = await get_client(ctx)
            data = await client.post_json("/memory/store", body, tool_name="raptor_memory_store")
        except Exception as exc:
            await ctx.error(f"raptor_memory_store failed: {exc}")
            raise

        await ctx.info("raptor_memory_store: stored")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TIMELINE_DESC.strip())
    async def raptor_memory_timeline(
        page: Annotated[int, "Page number (1-based)"] = 1,
        page_size: Annotated[int, "Results per page (1-100)"] = 20,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_timeline: page={page} page_size={page_size}")

        params = {"page": page, "page_size": page_size}

        try:
            client = await get_client(ctx)
            data = await client.get_json(
                "/memory/timeline", params=params, tool_name="raptor_memory_timeline")
        except Exception as exc:
            await ctx.error(f"raptor_memory_timeline failed: {exc}")
            raise

        await ctx.info("raptor_memory_timeline: done")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_MULTIMEDIA_SEARCH_DESC.strip())
    async def raptor_memory_multimedia_search(
        query: Annotated[str, "Search query text"],
        top_k: Annotated[int, "Maximum results to return (1-50)"] = 5,
        media_type: Annotated[
            Optional[Literal["video", "audio", "image"]],
            "Restrict to a single media type. Omit to search across all types.",
        ] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_multimedia_search: query={query!r} media_type={media_type!r}")

        body = {"query": query, "top_k": top_k}
        if media_type is not None:
            body["media_type"] = media_type

        try:
            client = await get_client(ctx)
            data = await client.post_json(
                "/memory/multimedia/search", body, tool_name="raptor_memory_multimedia_search")
        except Exception as exc:
            await ctx.error(f"raptor_memory_multimedia_search failed: {exc}")
            raise

        await ctx.info("raptor_memory_multimedia_search: done")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_SESSION_SUMMARIES_DESC.strip())
    async def raptor_memory_session_summaries(
        session_id: Annotated[str, "Session ID to list summary frames for"],
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_session_summaries: session_id={session_id!r}")

        try:
            client = await get_client(ctx)
            data = await client.get_json(
                f"/memory/sessions/{session_id}/summaries",
                tool_name="raptor_memory_session_summaries",
            )
        except Exception as exc:
            await ctx.error(f"raptor_memory_session_summaries failed: {exc}")
            raise

        await ctx.info("raptor_memory_session_summaries: done")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_COMPACT_DESC.strip())
    async def raptor_memory_compact(
        session_id: Annotated[Optional[str], "Session to compact. Omit to use the 'default' session."] = None,
        trigger: Annotated[str, "Trigger source: auto | manual | reactive (for logging only)"] = "manual",
        context_window: Annotated[int, "Target LLM context window size, tokens"] = 128000,
        custom_instructions: Annotated[
            Optional[str], "Extra instructions for the LLM performing the summarisation"
        ] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_compact: session_id={session_id!r} trigger={trigger!r}")

        body = {"trigger": trigger, "context_window": context_window}
        if custom_instructions is not None:
            body["custom_instructions"] = custom_instructions
        params = {"session_id": session_id} if session_id is not None else None

        try:
            client = await get_client(ctx)
            data = await client.post_json(
                "/memory/compact", body, tool_name="raptor_memory_compact", params=params)
        except Exception as exc:
            await ctx.error(f"raptor_memory_compact failed: {exc}")
            raise

        await ctx.info("raptor_memory_compact: done")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_COMPACT_EVALUATE_DESC.strip())
    async def raptor_memory_compact_evaluate(
        session_id: Annotated[
            Optional[str],
            "Estimate against this session's archived turns + long-term facts + "
            "multimedia snippets. Omit to estimate only the messages you pass in.",
        ] = None,
        context_window: Annotated[int, "Target LLM context window size, tokens"] = 128000,
        extra_tokens: Annotated[int, "Estimated token count of the current, not-yet-archived turn"] = 0,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_compact_evaluate: session_id={session_id!r}")

        body = {"context_window": context_window, "extra_tokens": extra_tokens, "messages": []}
        if session_id is not None:
            body["session_id"] = session_id

        try:
            client = await get_client(ctx)
            data = await client.post_json(
                "/memory/compact/evaluate", body, tool_name="raptor_memory_compact_evaluate")
        except Exception as exc:
            await ctx.error(f"raptor_memory_compact_evaluate failed: {exc}")
            raise

        await ctx.info("raptor_memory_compact_evaluate: done")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_ARCHIVE_DESC.strip())
    async def raptor_memory_archive(
        user_message: Annotated[str, "The user's message for this turn"],
        assistant_response: Annotated[str, "The assistant's response for this turn"],
        session_id: Annotated[Optional[str], "Session to archive into. Omit to use the 'default' session."] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_memory_archive: session_id={session_id!r}")

        body = {"user_message": user_message, "assistant_response": assistant_response}
        params = {"session_id": session_id} if session_id is not None else None

        try:
            client = await get_client(ctx)
            data = await client.post_json(
                "/memory/archive", body, tool_name="raptor_memory_archive", params=params)
        except Exception as exc:
            await ctx.error(f"raptor_memory_archive failed: {exc}")
            raise

        await ctx.info("raptor_memory_archive: done")
        return json.dumps(data, ensure_ascii=False, indent=2)
