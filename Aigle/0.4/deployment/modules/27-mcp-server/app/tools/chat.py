from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.tools import get_client

logger = logging.getLogger(__name__)

_CHAT_DESC = """\
Send a message to Raptor's conversational RAG system and receive a grounded answer.

Pipeline: intent classification → hybrid search → context window → LLM generation.

Pass the same session_id across turns to maintain conversation history (Redis-backed).
Omit session_id to start a new session — the server assigns a new ID.

Response fields: response, session_id, search_triggered, search_results.
Typical latency: 10–60 seconds (LLM generation).
"""


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_CHAT_DESC.strip())
    async def raptor_chat(
        message: Annotated[str, "User message to send to the RAG chat system"],
        session_id: Annotated[Optional[str], "Session ID for conversation continuity. Omit to start a new session."] = None,
        history: Annotated[
            Optional[List[Dict[str, str]]],
            "Explicit conversation history as [{role, content}] dicts. "
            "Use to inject context when server-side session memory is unavailable.",
        ] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(
            f"raptor_chat: session={session_id!r} "
            f"message_len={len(message)} history_turns={len(history) if history else 0}"
        )

        body: dict = {"message": message}
        if session_id:
            body["session_id"] = session_id
        if history:
            valid_history: List[Dict[str, str]] = []
            for entry in history:
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    valid_history.append({
                        "role": str(entry["role"]),
                        "content": str(entry["content"]),
                    })
                else:
                    logger.warning("raptor_chat: skipping invalid history entry: %r", entry)
            if valid_history:
                body["history"] = valid_history

        try:
            client = await get_client(ctx)
            data = await client.post_json("/chat/chat", body, tool_name="raptor_chat")
        except Exception as exc:
            await ctx.error(f"raptor_chat failed: {exc}")
            raise

        returned_session = data.get("session_id") or session_id
        search_triggered = data.get("search_triggered", False)
        n_results = len(data.get("search_results") or [])
        await ctx.info(
            f"raptor_chat: complete session={returned_session!r} "
            f"search_triggered={search_triggered} results={n_results}"
        )

        result: Dict[str, Any] = {
            "response": data.get("response", ""),
            "session_id": returned_session,
            "search_triggered": search_triggered,
            "search_results": data.get("search_results") or [],
        }
        if data.get("tool_calls"):
            result["tool_calls"] = data["tool_calls"]

        return json.dumps(result, ensure_ascii=False, indent=2)
