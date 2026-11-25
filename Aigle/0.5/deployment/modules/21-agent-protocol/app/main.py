# app/main.py
# Agent Protocol Service — port 8030
# REST entry point: POST /query (mode=direct|agent|tool), GET /health

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx
from discovery import DiscoveryService
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from jsonrpc_handler import JSONRPCHandler
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Agent Card ────────────────────────────────────────────────────────────────

class AgentCardService:
    def build(self) -> dict[str, Any]:
        return {
            "id":          os.getenv("AGENT_ID",       "raptor-agent"),
            "name":        os.getenv("AGENT_NAME",     "Raptor Intelligence Agent"),
            "version":     "0.3.0",
            "description": "Multimodal RAG + Knowledge Graph agent",
            "endpoint":    os.getenv("AGENT_ENDPOINT", "http://raptor-agent-protocol:8030"),
            "capabilities": [
                "document_rag",
                "video_rag",
                "graph_rag",
                "tkg_query",
            ],
            "protocols":       ["a2a/1.0", "jsonrpc/2.0"],
            "authentication":  {"type": "bearer"},
        }


discovery       = DiscoveryService()
agent_card_svc  = AgentCardService()
rpc_handler     = JSONRPCHandler(agent_card_svc, discovery)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await discovery.start()
    await discovery.register_self(agent_card_svc.build())
    from core.config import get_settings as _get_settings
    _s = _get_settings()
    if _s.guardrail_active:
        try:
            await discovery._redis.set("guardrail:integration:a2a_host_orchestrator", "1")
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.warning(
                "[GB-22] guardrail startup registration failed (screened=false): %s", exc
            )
    logger.info("Agent Protocol Service started")
    yield
    await discovery.stop()
    logger.info("Agent Protocol Service stopped")


_TAGS = [
    {"name": "RAG Query",       "description": "直接送問題進 RAG pipeline，支援 direct / agent / tool 三種模式"},
    {"name": "A2A Protocol",    "description": "JSON-RPC 2.0 機器對機器協議 (agent.discover / agent.query / agent.delegate) 及訊息轉發"},
    {"name": "Peer Discovery",  "description": "外部 A2A agent 動態登記、心跳續約、下線（Redis TTL-based）"},
    {"name": "Sub-agents",      "description": "查詢內部 A2A sub-agent 清單、AgentCard 及訊息代理"},
    {"name": "Health",          "description": "服務與下游健康狀態"},
]

app = FastAPI(title="Agent Protocol Service", version="0.3.0", lifespan=lifespan, openapi_tags=_TAGS)
Instrumentator(
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
).instrument(app).expose(app)


# ── A2A endpoints ─────────────────────────────────────────────────────────────

@app.get("/.well-known/agent-card", include_in_schema=False)
async def agent_card_legacy():
    return agent_card_svc.build()


@app.get("/.well-known/agent.json", include_in_schema=False)
async def agent_card_json():
    """Standard A2A spec AgentCard endpoint (used by sub-agent discovery)."""
    return {
        "name":               "raptor_agent_protocol",
        "description":        "Multimodal RAG + Knowledge Graph orchestrator. Coordinates vector search, keyword search, Graph RAG, TKG, People Search, and reranking via A2A agents.",
        "version":            "0.3.0",
        "url":                os.getenv("AGENT_ENDPOINT", "http://raptor-agent-protocol:8030"),
        "defaultInputModes":  ["text"],
        "defaultOutputModes": ["text"],
        "capabilities":       {},
        "skills": [{
            "id":          "rag_query",
            "name":        "RAG Query",
            "description": "Full RAG pipeline: intent classification → fan-out → RRF fusion → rerank → LLM answer.",
            "tags":        ["rag", "orchestrator", "multimodal", "graph"],
            "examples":    ['{"question": "What is the cooling system?", "top_k": 5}'],
            "inputModes":  ["text"],
            "outputModes": ["text"],
        }],
    }


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id:      Any            = None
    method:  str
    params:  dict[str, Any] = {}


_JSONRPC_EXAMPLES = {
    "discover": {
        "summary": "agent.discover — 查詢本 agent 能力",
        "value":   {"jsonrpc": "2.0", "id": 1, "method": "agent.discover", "params": {}},
    },
    "query": {
        "summary": "agent.query — 送問題進 RAG pipeline",
        "value":   {"jsonrpc": "2.0", "id": 2, "method": "agent.query",
                    "params": {"query": "冷卻系統的工作原理是什麼？", "top_k": 5}},
    },
    "delegate": {
        "summary": "agent.delegate — 委派任務給指定 agent",
        "value":   {"jsonrpc": "2.0", "id": 3, "method": "agent.delegate",
                    "params": {"task": {"query": "find Peter Parker in the video", "top_k": 5},
                               "target_agent_id": "people_search_agent"}},
    },
}


@app.post("/a2a/jsonrpc", summary="JSON-RPC 2.0 A2A endpoint", tags=["A2A Protocol"])
async def a2a_jsonrpc(
    body: Annotated[JSONRPCRequest, Body(openapi_examples=_JSONRPC_EXAMPLES)],
    request: Request,
):
    """
    JSON-RPC 2.0 dispatcher.
    Supported methods: agent.discover, agent.delegate, agent.query
    GB-21: TextPart content is screened by the guardrail service before routing.
    """
    from handlers.task_handler import guardrail_precheck
    from memory_client import set_auth_token

    set_auth_token(request.headers.get("Authorization", ""))

    body_dict = body.model_dump()

    # ── GB-21 Guardrail Pre-check ─────────────────────────────────────────────
    # Only screen tasks/send style calls that carry message.parts content.
    # agent.discover has no content; agent.query/delegate carry params.task.
    method = body_dict.get("method", "")
    if method in ("agent.query", "agent.delegate", "message/send"):
        # Normalise: tasks/send uses params.message; agent.query uses params directly
        raw_task = body_dict.get("params", {})
        request_id = str(body_dict.get("id") or "")
        headers = dict(request.headers)
        source_agent_id = request.headers.get("x-a2a-source-agent")

        screened_task, error_resp = await guardrail_precheck(
            task=raw_task,
            request_id=request_id,
            headers=headers,
            source_agent_id=source_agent_id,
        )
        if error_resp is not None:
            return JSONResponse(content=error_resp)

        # Patch the screened (possibly redacted) params back
        if screened_task is not None:
            body_dict["params"] = screened_task

    result = await rpc_handler.handle(body_dict)
    return JSONResponse(content=result)


@app.get("/a2a/agents", summary="List registered peer agents", tags=["Peer Discovery"])
async def list_agents():
    agents = await discovery.list_agents()
    return {"agents": agents, "count": len(agents)}


class PeerAgentCard(BaseModel):
    id:          str
    name:        str
    version:     str             = "1.0.0"
    description: str             = ""
    endpoint:    str
    capabilities: list[str]     = []
    protocols:   list[str]      = ["a2a/1.0"]


@app.post("/a2a/agents/register", summary="Register an external peer agent", status_code=201, tags=["Peer Discovery"])
async def register_peer_agent(card: PeerAgentCard) -> dict[str, Any]:
    """外部 A2A agent 呼叫此端點登記自己，TTL 由 AGENT_TTL 環境變數控制。"""
    await discovery.register_peer(card.model_dump())
    return {"status": "registered", "id": card.id, "ttl": discovery.AGENT_TTL}


@app.delete("/a2a/agents/{agent_id}", summary="Deregister an external peer agent", tags=["Peer Discovery"])
async def deregister_peer_agent(agent_id: str) -> dict[str, Any]:
    """外部 A2A agent 下線時呼叫，立即從 Redis 移除。"""
    found = await discovery.deregister_peer(agent_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {"status": "deregistered", "id": agent_id}


@app.post("/a2a/agents/{agent_id}/heartbeat", summary="Renew peer agent TTL", tags=["Peer Discovery"])
async def peer_agent_heartbeat(agent_id: str) -> dict[str, Any]:
    """外部 A2A agent 定期呼叫以續約，建議間隔為 AGENT_TTL × 0.8 秒。"""
    found = await discovery.heartbeat_peer(agent_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or expired, please re-register")
    return {"status": "ok", "id": agent_id, "ttl_renewed": discovery.AGENT_TTL}


# ── Direct REST query endpoint ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k:    int = Field(5, ge=1, le=50)
    mode:     Literal["direct", "agent", "tool"] = Field(
        "direct",
        description=(
            "direct — deterministic pipeline (default); "
            "agent  — smolagents ToolCallingAgent discovers A2A sub-agents as tools and routes dynamically; "
            "tool   — smolagents ToolCallingAgent with RAGSearchTool + WebSearchTool fallback"
        ),
    )
    session_id: str = Field(
        "",
        description=(
            "Optional session ID. When set on an authenticated request, turns from "
            "all three modes are archived via Module 26 (through Module 13's gateway) "
            "for cross-session memory; empty keeps the request fully stateless."
        ),
    )
    source_version_ids: list[str] | None = Field(
        None, max_length=100,
        description=(
            "P21: answer from these sources only (their version_ids) -- vector, keyword and TKG "
            "search are scoped to them; GraphRAG and the user's memory are left out. mode=direct only."
        ),
    )


@app.post("/query", summary="RAG query — mode=direct|agent|tool", tags=["RAG Query"])
async def query_endpoint(body: QueryRequest, request: Request):
    from memory_client import set_auth_token
    from pipeline import _s2tw

    user_id   = request.headers.get("X-User-ID", "")
    branch_id = request.headers.get("X-Branch-ID", "")
    set_auth_token(request.headers.get("Authorization", ""))

    if body.source_version_ids and body.mode != "direct":
        raise HTTPException(status_code=400, detail={"error": "SCOPE_NEEDS_DIRECT_MODE",
                                                     "reason": "source_version_ids is served by mode=direct only"})

    if body.mode == "agent":
        from memory_client import fire_and_forget
        from pipeline import _fetch_presigned_url, archive_turn, gather_memory_context
        from smolagents.memory import ActionStep
        from smolagents_host import build_host_agent
        memory_context = await gather_memory_context(user_id, body.session_id, body.question)
        agent, tools = await build_host_agent(branch_id=branch_id, memory_context=memory_context)
        answer = _s2tw(str(await asyncio.to_thread(agent.run, body.question)))
        fire_and_forget(archive_turn(user_id, body.session_id, body.question, answer))
        tool_calls = [
            tc.dict()["function"]
            for step in agent.memory.steps
            if isinstance(step, ActionStep) and step.tool_calls
            for tc in step.tool_calls
        ]
        # Aggregate hits from every A2A tool that was called, dedup by doc_id, sort by score
        seen: set[str] = set()
        sources: list[dict] = []
        for t in tools:
            for hit in t.last_hits:
                key = hit.get("doc_id") or hit.get("id", "")
                if key and key not in seen:
                    seen.add(key)
                    sources.append(hit)
        sources.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        # Resolve presigned URLs (same auth headers as direct mode)
        auth_headers: dict = {}
        if user_id:
            auth_headers["X-User-ID"]   = user_id
            auth_headers["X-Branch-ID"] = branch_id or user_id
        if auth_headers:
            async def _resolve(src: dict) -> None:
                ap = src.get("asset_path")
                vid = src.get("version_id")
                if ap and vid:
                    filename = (src.get("metadata") or {}).get("filename")
                    url = await _fetch_presigned_url(ap, vid, filename, auth_headers)
                    if url:
                        src["storage_uri"] = url
            await asyncio.gather(*[_resolve(s) for s in sources])
        return {
            "answer":        answer,
            "sources":       sources,
            "graph_context": "",
            "chunks_used":   len(sources),
            "tool_calls":    tool_calls,
        }

    if body.mode == "tool":
        from smolagents_host import build_tool_agent
        agent, rag_tool = build_tool_agent(user_id=user_id, branch_id=branch_id, session_id=body.session_id)
        agent_answer = await asyncio.to_thread(agent.run, body.question)
        # Prefer rag_search's own grounded answer over the outer agent's
        # free-form restatement of it — smolagents sometimes paraphrases
        # specific facts (names, details) incorrectly even when the correct
        # answer is right there in its own tool-call history.
        final_answer = rag_tool.last_answer or str(agent_answer)
        return {
            "answer":        _s2tw(final_answer),
            "sources":       rag_tool.last_sources,
            "graph_context": rag_tool.last_graph_context,
            "chunks_used":   rag_tool.last_chunks_used,
        }

    # mode == "direct"
    from pipeline import run_rag_pipeline
    result = await run_rag_pipeline(
        body.question, top_k=body.top_k, user_id=user_id, branch_id=branch_id,
        session_id=body.session_id, mode="direct", source_version_ids=body.source_version_ids,
    )
    return result.model_dump()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health() -> dict[str, Any]:
    """Check liveness of all A2A sub-agents and downstream services."""
    from pipeline import get_agent_urls
    agent_urls = get_agent_urls()

    agent_statuses: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in agent_urls.items():
            try:
                resp = await client.get(f"{url}/.well-known/agent-card.json")
                agent_statuses[name] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
            except httpx.HTTPError as e:
                logger.debug(f"Agent {name} health check failed: {type(e).__name__}: {e}")
                agent_statuses[name] = "unreachable"

    # downstream service liveness (non-blocking, best-effort)
    downstream_health_paths = {
        "hybrid_search":    (os.getenv("HYBRID_SEARCH_URL",    "http://raptor-hybridsearch-api:8000"),  "/api/v1/health/live"),
        "graph_service":    (os.getenv("GRAPH_SERVICE_URL",    "http://raptor-graph-service:8843"),      "/health"),
        "inference":        (os.getenv("INFERENCE_URL",        "http://raptor-ai-lifecycle-api:8010"),   "/health"),
    }
    downstream_statuses: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=3) as client:
        for name, (url, path) in downstream_health_paths.items():
            try:
                resp = await client.get(f"{url}{path}")
                downstream_statuses[name] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
            except httpx.HTTPError as e:
                logger.debug(f"Downstream {name} health check failed: {type(e).__name__}: {e}")
                downstream_statuses[name] = "unreachable"

    agents_ok     = all(v == "ok" for v in agent_statuses.values())
    downstream_ok = all(v == "ok" for v in downstream_statuses.values())
    return {
        "status":     "ok" if (agents_ok and downstream_ok) else "degraded",
        "service":    "agent-protocol",
        "agents":     agent_statuses,
        "downstream": downstream_statuses,
    }


# ── A2A agent card / pass-through endpoints ───────────────────────────────────

def _get_agent_urls() -> dict[str, str]:
    """Named agent URLs (sub-agents + self)."""
    from pipeline import get_agent_urls
    urls = get_agent_urls()
    # expose self under "orchestrator" key for symmetry with 22-agent-protocol
    urls["orchestrator"] = "http://localhost:8030"
    return urls


async def _proxy_to_agent(agent_url: str, path: str, request: Request) -> Response:
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body    = await request.body()
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.request(
                method=request.method, url=f"{agent_url.rstrip('/')}{path}",
                headers=headers, params=request.query_params, content=body,
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    drop = {"content-length", "transfer-encoding", "connection", "keep-alive", "date", "server"}
    return Response(
        content=resp.content, status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() not in drop},
    )


@app.get("/agents", summary="List registered A2A sub-agents", tags=["Sub-agents"])
async def list_registered_agents() -> dict[str, Any]:
    return {"agents": _get_agent_urls()}


@app.get("/agents/cards", summary="Fetch AgentCard from every registered sub-agent", tags=["Sub-agents"])
async def all_agent_cards() -> dict[str, Any]:
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in _get_agent_urls().items():
            try:
                resp = await client.get(f"{url.rstrip('/')}/.well-known/agent-card.json")
                resp.raise_for_status()
                results[name] = {"status": "ok", "card": resp.json()}
            except (httpx.HTTPError, ValueError) as exc:
                results[name] = {"status": "error", "error": str(exc)}
    return results


@app.get("/agents/{agent_name}/card", summary="Fetch AgentCard for a single sub-agent", tags=["Sub-agents"])
async def single_agent_card(agent_name: str) -> dict[str, Any]:
    urls = _get_agent_urls()
    if agent_name not in urls:
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'. Valid: {list(urls)}")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{urls[agent_name].rstrip('/')}/.well-known/agent-card.json")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc))


class A2AMessageRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id:      Any            = None
    method:  str            = "message/send"
    params:  dict[str, Any] = {}

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": "msg-001",
                        "parts": [{"type": "text", "text": "{\"query\": \"冷卻系統\", \"top_k\": 3, \"type\": \"documents\"}"}],
                    }
                },
            }]
        }
    }


@app.post("/agents/{agent_name}/message", summary="Forward A2A message to a specific sub-agent", tags=["Sub-agents"])
async def forward_agent_message(agent_name: str, body: A2AMessageRequest, request: Request) -> Response:
    urls = _get_agent_urls()
    if agent_name not in urls:
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'. Valid: {list(urls)}")
    return await _proxy_to_agent(urls[agent_name], "/", request)


@app.post("/a2a/message", summary="Forward A2A message (default: self/orchestrator; override with ?target=<name>)", tags=["A2A Protocol"])
async def a2a_message(body: A2AMessageRequest, request: Request, target: str | None = None) -> Response:
    name = target or request.headers.get("x-a2a-target-agent", "orchestrator")
    urls = _get_agent_urls()
    if name not in urls:
        raise HTTPException(status_code=404, detail=f"Unknown agent '{name}'. Valid: {list(urls)}")
    return await _proxy_to_agent(urls[name], "/", request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8030, reload=False)
