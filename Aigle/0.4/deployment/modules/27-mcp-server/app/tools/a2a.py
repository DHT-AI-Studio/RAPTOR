from __future__ import annotations

import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.core.config import get_settings
from app.tools import get_client

logger = logging.getLogger(__name__)

_A2A_TIMEOUT = get_settings().timeout_a2a

_DIRECT_DESC = """\
Deterministic RAG pipeline: intent classification → multi-path search → rerank → LLM answer.

Returns: answer (LLM-generated), sources (retrieved chunks), graph_context.
Latency: 15–60 seconds. Use when you need a grounded, reproducible answer.
"""

_AGENT_DESC = """\
Agentic RAG: smolagents CodeAgent autonomously selects and calls Raptor tools.

The agent plans, searches, and synthesises an answer with a tool-call trace.
Returns: answer, sources, agent_trace (agent reasoning steps).
Latency: 30–120 seconds. Use for complex multi-hop questions.
"""


async def _a2a_query(ctx: Context, question: str, top_k: int, mode: str, tool_name: str) -> dict:
    client = await get_client(ctx, timeout=_A2A_TIMEOUT)
    body = {"question": question, "top_k": min(max(top_k, 1), 50), "mode": mode}
    return await client.post_json(
        "/a2a/query", body, tool_name=tool_name, timeout=_A2A_TIMEOUT)


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_DIRECT_DESC.strip())
    async def raptor_a2a_direct(
        question: Annotated[str, "Question to answer (at least 1 character)."],
        top_k: Annotated[int, "Number of chunks to retrieve (1–50)."] = 5,
        session_id: Annotated[Optional[str], "Optional session id (reserved; not yet used by the pipeline)."] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_a2a_direct: question={question!r} top_k={top_k}")
        try:
            data = await _a2a_query(ctx, question, top_k, "direct", "raptor_a2a_direct")
        except Exception as exc:
            await ctx.error(f"raptor_a2a_direct failed: {exc}")
            raise

        result = {
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
            "graph_context": data.get("graph_context"),
        }
        await ctx.info("raptor_a2a_direct: complete")
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(description=_AGENT_DESC.strip())
    async def raptor_a2a_agent(
        question: Annotated[str, "Question for the agent to reason about."],
        top_k: Annotated[int, "Number of chunks per retrieval step (1–50)."] = 5,
        session_id: Annotated[Optional[str], "Optional session id (reserved; not yet used by the pipeline)."] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_a2a_agent: question={question!r} top_k={top_k}")
        try:
            data = await _a2a_query(ctx, question, top_k, "agent", "raptor_a2a_agent")
        except Exception as exc:
            await ctx.error(f"raptor_a2a_agent failed: {exc}")
            raise

        result = {
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
            # The backend may return the agent's reasoning under one of these keys.
            "agent_trace": data.get("agent_trace") or data.get("tool_calls") or [],
        }
        await ctx.info("raptor_a2a_agent: complete")
        return json.dumps(result, ensure_ascii=False, indent=2)
