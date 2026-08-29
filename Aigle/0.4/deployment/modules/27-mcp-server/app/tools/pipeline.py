from __future__ import annotations

import json
import logging
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP

from app.core.config import get_settings
from app.tools import get_client

logger = logging.getLogger(__name__)

# Same call path as tools/a2a.py (both drive the a2a `direct` pipeline), so
# they share the one timeout_a2a setting.
_ORCHESTRATE_TIMEOUT = get_settings().timeout_a2a

_ORCHESTRATE_DESC = """\
Auto-routed query — sends the question through Raptor's full RAG pipeline, which
internally classifies intent (Module 18) and picks the best retrieval path
(VideoRAG / DocumentRAG / GraphRAG / TKG / RDBMS) before generating an answer.

Use when you don't want to choose a search tool yourself. Returns a grounded
answer with sources.
"""


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_ORCHESTRATE_DESC.strip())
    async def raptor_query_orchestrate(
        query: Annotated[str, "Natural-language query to auto-route and answer."],
        top_k: Annotated[int, "Number of chunks to retrieve (1–50)."] = 10,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_query_orchestrate: query={query!r} top_k={top_k}")

        # No gateway route exposes Module 18 /classify directly, so we drive the
        # a2a `direct` pipeline, which runs the same classify → route → answer flow
        # internally. pipeline_used / confidence are not returned by that endpoint
        # yet — left null pending a gateway /classify passthrough.
        body = {"question": query, "top_k": min(max(top_k, 1), 50), "mode": "direct"}

        try:
            client = await get_client(ctx, timeout=_ORCHESTRATE_TIMEOUT)
            data = await client.post_json(
                "/a2a/query", body,
                tool_name="raptor_query_orchestrate",
                timeout=_ORCHESTRATE_TIMEOUT,
            )
        except Exception as exc:
            await ctx.error(f"raptor_query_orchestrate failed: {exc}")
            raise

        result = {
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
            # Auto-populated once Module 21 /query surfaces the Module 18
            # classification (pipeline + confidence); null until the backend adds them.
            "pipeline_used": data.get("pipeline_used"),
            "confidence": data.get("confidence"),
            "graph_context": data.get("graph_context"),
        }
        await ctx.info("raptor_query_orchestrate: complete")
        return json.dumps(result, ensure_ascii=False, indent=2)
