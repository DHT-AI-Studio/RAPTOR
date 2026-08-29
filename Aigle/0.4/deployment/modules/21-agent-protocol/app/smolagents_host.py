"""
app/smolagents_host.py
smolagents integration for Module 21 orchestrator.

Two agentic modes beyond the default "direct" deterministic pipeline:
  agent — ToolCallingAgent dynamically discovers all A2A sub-agents and routes searches
  tool  — ToolCallingAgent with RAGSearchTool + WebSearchTool (DuckDuckGo fallback)

Ported from 22-agent-protocol/host/orchestrator.py; adapted for Module 21.
"""
from __future__ import annotations
import asyncio
import os
import uuid

import httpx
from loguru import logger

from a2a.client import A2AClient
from a2a.types import AgentCard, SendMessageRequest, MessageSendParams
from a2a.client.helpers import create_text_message_object

from smolagents import ToolCallingAgent, OpenAIModel, Tool
# from smolagents import LiteLLMModel  -- superseded by OpenAIModel -> module 07, see get_ollama_model() below

from a2a_helpers import extract_text_from_response
from pipeline import run_rag_pipeline, get_agent_urls


# ── LLM wrapper ───────────────────────────────────────────────────────────────

def get_ollama_model() -> OpenAIModel:
    """OpenAIModel -> module 07's /v1/chat/completions, not the Ollama daemon
    directly. Was LiteLLMModel hitting OLLAMA_HOST's native API -- that path
    has no think control and qwen3.x-family models think by default, the
    same latency problem already fixed for pipeline.py's _llm_answer() and
    graphrag_agent.py's/tkg_agent.py's _summarise() elsewhere in this module
    (this one was missed the first time round). OpenAIModel wraps the real
    openai SDK, and any constructor kwarg beyond the named ones (extra_body
    here) flows straight into the request body -- same mechanism as
    ChatOpenAI(extra_body=...) in module 15's chat_service.py.
    """
    model_name    = os.environ.get("OLLAMA_MODEL", "qwen2.5")
    inference_url = os.environ.get("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
    think         = os.environ.get("LLM_THINK", "false").lower() == "true"
    return OpenAIModel(
        model_id=model_name,
        api_base=f"{inference_url}/v1",
        api_key="not-needed",
        extra_body={"engine": "ollama", "think": think},  # model_name isn't MLflow-registered; see /v1's engine fallback
    )


# Old LiteLLMModel implementation -- commented out, not deleted, for rollback.
# Talked to the Ollama daemon's native API directly via LiteLLM's ollama/
# provider routing; superseded once module 07 gained tool-calling support.
# def get_ollama_model() -> LiteLLMModel:
#     model_name  = os.environ.get("OLLAMA_MODEL", "qwen2.5")
#     ollama_host = os.environ.get("OLLAMA_HOST",  "http://<host_ip>:11434")
#     return LiteLLMModel(model_id=f"ollama/{model_name}", api_base=ollama_host)


# ── A2A Tool wrapper (agent mode) ─────────────────────────────────────────────

class A2AAgentTool(Tool):
    """Wraps a remote A2A sub-agent as a smolagents Tool (JSON tool-calling)."""
    output_type = "string"

    def __init__(self, tool_name: str, description: str, card: AgentCard, branch_id: str = ""):
        self.name        = tool_name
        self.description = description
        self.inputs      = {
            "question": {
                "type":        "string",
                "description": "Search query or question to send to this specialized agent",
            },
        }
        super().__init__()
        self.card      = card
        self.branch_id = branch_id
        self.last_hits: list[dict] = []

    def forward(self, question: str) -> str:
        # asyncio.run is safe: called inside asyncio.to_thread (no running event loop in thread)
        result = asyncio.run(self._call_agent(question))
        import json as _json
        try:
            data = _json.loads(result)
            self.last_hits = data.get("hits", []) if isinstance(data, dict) else []
        except Exception:
            self.last_hits = []
        return result

    async def _call_agent(self, question: str) -> str:
        import json as _json
        # Sub-agents expect JSON-encoded request body, not plain text.
        # All search agents share the `query` field; top_k/extra fields are ignored by pydantic.
        body: dict = {"query": question, "top_k": 5}
        if self.branch_id:
            body["branch_id"] = self.branch_id
        payload = _json.dumps(body)
        async with httpx.AsyncClient(timeout=60) as client:
            a2a = A2AClient(httpx_client=client, agent_card=self.card)
            req = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(message=create_text_message_object(content=payload)),
            )
            resp = await a2a.send_message(req)
            return extract_text_from_response(resp)


class A2AToolBootstrap:
    """Discovers A2A sub-agents and wraps them as smolagents Tools."""

    def __init__(self, agent_urls: dict[str, str], branch_id: str = ""):
        self.agent_urls = agent_urls  # logical_name → url
        self.branch_id  = branch_id

    async def get_tools(self) -> list[A2AAgentTool]:
        tools: list[A2AAgentTool] = []
        async with httpx.AsyncClient(timeout=10) as client:
            for logical_name, url in self.agent_urls.items():
                try:
                    resp = await client.get(f"{url}/.well-known/agent-card.json")
                    resp.raise_for_status()
                    card = AgentCard(**resp.json())
                    # Trust the URL we successfully reached over the agent's
                    # self-declared url — they diverge when discovery goes
                    # through a different network path than the card's own
                    # (e.g. host-mapped port vs. docker-internal hostname).
                    card.url = url
                    tools.append(A2AAgentTool(
                        tool_name=logical_name,
                        description=card.description or f"{logical_name} search agent",
                        card=card,
                        branch_id=self.branch_id,
                    ))
                    logger.success("Discovered A2A tool: {} @ {}", logical_name, url)
                except Exception as exc:
                    logger.warning("Could not discover A2A tool at {}: {}", url, exc)
        return tools


# ── smolagents tools (tool mode) ──────────────────────────────────────────────

class RAGSearchTool(Tool):
    """Searches the internal RAG knowledge base (vector + keyword + graph)."""
    name        = "rag_search"
    description = (
        "Search the internal knowledge base using the full RAG pipeline. "
        "Returns a grounded answer plus source chunks. "
        "Use this before falling back to web search."
    )
    inputs = {
        "question": {"type": "string",  "description": "Natural language question to answer"},
        "top_k":    {"type": "integer", "description": "Max chunks to retrieve", "nullable": True},
    }
    output_type = "string"

    def __init__(self, user_id: str = "", branch_id: str = "", session_id: str = ""):
        super().__init__()
        self.user_id             = user_id
        self.branch_id           = branch_id
        self.session_id          = session_id
        self.last_answer:        str        = ""
        self.last_sources:       list[dict] = []
        self.last_chunks_used:   int        = 0
        self.last_graph_context: str        = ""

    def forward(self, question: str, top_k: int | None = None) -> str:
        import json as _json
        # asyncio.run is safe here: runs inside asyncio.to_thread (no event
        # loop in thread). get_memory_client() caches per-event-loop, so each
        # fresh loop here gets its own client without disturbing others.
        result = asyncio.run(run_rag_pipeline(
            question, top_k=top_k or 5,
            user_id=self.user_id, branch_id=self.branch_id, session_id=self.session_id,
            mode="tool",
        ))
        self.last_answer        = result.answer
        self.last_sources       = result.sources
        self.last_chunks_used   = result.chunks_used
        self.last_graph_context = result.graph_context or ""
        # Trimmed for the LLM (full chunk dicts run 1-2k tokens each and blow
        # past Ollama's default 2048-token context); self.last_sources above
        # still keeps the full dicts for the HTTP response.
        brief_sources = [
            {
                "doc_id":   s.get("doc_id"),
                "score":    s.get("score"),
                "filename": (s.get("metadata") or {}).get("filename"),
                "excerpt":  (s.get("content") or "")[:200],
            }
            for s in result.sources[:3]
        ]
        return _json.dumps({
            "answer":      result.answer,
            "chunks_used": result.chunks_used,
            "sources":     brief_sources,
        }, ensure_ascii=False)


class WebSearchTool(Tool):
    """Falls back to DuckDuckGo web search when the knowledge base has no answer."""
    name        = "web_search"
    description = (
        "Search the public web via DuckDuckGo. "
        "Use only when the internal RAG search returns no useful answer."
    )
    inputs = {
        "query":       {"type": "string",  "description": "Search query"},
        "max_results": {"type": "integer", "description": "Number of results", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: str, max_results: int | None = None) -> str:
        import json as _json
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return _json.dumps({"error": "duckduckgo_search not installed — run: pip install duckduckgo-search"})
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results or 5))
            return _json.dumps([
                {"title": h.get("title"), "href": h.get("href"), "body": h.get("body")}
                for h in hits
            ], ensure_ascii=False)
        except Exception as exc:
            logger.warning("[WebSearchTool] DuckDuckGo error: {}", exc)
            return _json.dumps({"error": str(exc)})


# ── Mode builders ─────────────────────────────────────────────────────────────

async def build_host_agent(
    branch_id: str = "", memory_context: str = "",
) -> tuple[ToolCallingAgent, list[A2AAgentTool]]:
    """Build a ToolCallingAgent backed by A2A sub-agents wrapped as tools.

    memory_context (long-term facts + MV-12 compact summary, see
    pipeline.gather_memory_context) is injected via the `instructions` param,
    which smolagents renders into the system prompt as `custom_instructions`
    — kept separate from the user's task so the agent doesn't mistake
    remembered context for this turn's question.
    """
    urls  = get_agent_urls()
    tools = await A2AToolBootstrap(urls, branch_id=branch_id).get_tools()
    if not tools:
        logger.warning("No A2A tools discovered — agent mode will answer without search")
    instructions = f"USER MEMORY (from prior sessions):\n{memory_context}" if memory_context else None
    host = ToolCallingAgent(
        tools=tools,
        model=get_ollama_model(),
        max_steps=10,
        instructions=instructions,
    )
    logger.success(
        "Host ToolCallingAgent ready with {} A2A tools: {}",
        len(tools), [t.name for t in tools],
    )
    return host, tools


def build_tool_agent(user_id: str = "", branch_id: str = "", session_id: str = "") -> tuple[ToolCallingAgent, RAGSearchTool]:
    """Build a ToolCallingAgent with RAGSearchTool + WebSearchTool."""
    rag_tool = RAGSearchTool(user_id=user_id, branch_id=branch_id, session_id=session_id)
    agent = ToolCallingAgent(
        tools=[rag_tool, WebSearchTool()],
        model=get_ollama_model(),
        max_steps=8,
    )
    logger.success("ToolCallingAgent ready (rag_search + web_search)")
    return agent, rag_tool
