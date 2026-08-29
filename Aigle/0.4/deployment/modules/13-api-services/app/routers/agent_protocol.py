"""A2A protocol proxy router — Module 13 forwards all A2A traffic to Module 21."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.services.agent_card_service import AgentCardService

_logger   = logging.getLogger(__name__)
_settings = get_settings()
_card_svc = AgentCardService()

# Two routers: root-level for /.well-known, versioned for everything else
well_known_router = APIRouter()
router            = APIRouter()


# ── Request models (mirrors Module 21, used by Swagger UI) ───────────────────

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id:      Any            = None
    method:  str
    params:  Dict[str, Any] = {}

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"jsonrpc": "2.0", "id": 1, "method": "agent.discover", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "agent.query",
                 "params": {"query": "冷卻系統的工作原理？", "top_k": 5}},
            ]
        }
    }


class PeerAgentCard(BaseModel):
    id:           str
    name:         str
    version:      str        = "1.0.0"
    description:  str        = ""
    endpoint:     str
    capabilities: List[str]  = []
    protocols:    List[str]  = ["a2a/1.0"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["冷卻系統的工作原理是什麼？"])
    top_k:    int = Field(5, ge=1, le=50)
    mode:     Literal["direct", "agent", "tool"] = Field(
        "direct",
        description="direct — deterministic pipeline; agent — smolagents CodeAgent; tool — ToolCallingAgent",
    )


class A2AMessageRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id:      Any            = None
    method:  str            = "message/send"
    params:  Dict[str, Any] = {}

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
                        "parts": [{"type": "text", "text": "{\"query\": \"冷卻系統\", \"top_k\": 3}"}],
                    }
                },
            }]
        }
    }


# ── Internal proxy helper ─────────────────────────────────────────────────────

async def _proxy(request: Request, upstream_path: str, timeout: float = 60.0) -> Response:
    """Forward the request to Module 21 at the given upstream path."""
    base = _settings.agent_protocol_url.rstrip("/")
    url  = f"{base}{upstream_path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body    = await request.body()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=body,
            )
    except httpx.RequestError as exc:
        _logger.error("A2A proxy error → %s: %s", url, exc)
        raise HTTPException(status_code=502, detail=str(exc))

    drop = {"content-length", "transfer-encoding", "connection", "keep-alive", "date", "server"}
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() not in drop},
    )


# ── /.well-known — root-level, A2A spec requires these paths ─────────────────

@well_known_router.get("/.well-known/agent-card", include_in_schema=False)
async def agent_card_legacy():
    return _card_svc.build()


@well_known_router.get("/.well-known/agent.json", include_in_schema=False)
async def agent_card_json():
    s = _settings
    return {
        "name":               s.agent_name,
        "description":        "Multimodal RAG + Knowledge Graph platform gateway",
        "version":            "0.3.0",
        "url":                s.agent_endpoint,
        "defaultInputModes":  ["text"],
        "defaultOutputModes": ["text"],
        "capabilities":       {},
        "skills": [{
            "id":          "rag_query",
            "name":        "RAG Query",
            "description": "Full RAG pipeline via agent-protocol backend.",
            "tags":        ["rag", "orchestrator", "multimodal", "graph"],
            "inputModes":  ["text"],
            "outputModes": ["text"],
        }],
    }


# ── Versioned A2A proxy endpoints ─────────────────────────────────────────────
# All routes below are included with prefix /api/{v}/a2a (canonical: /api/0.4/a2a/*).
# Also reachable at /api/0.3/a2a/* via LegacyApiAliasMiddleware, which rewrites
# to the canonical path and marks the response Deprecated.

@router.post("/jsonrpc", summary="JSON-RPC 2.0 A2A endpoint", tags=["A2A Protocol"], include_in_schema=False)
async def a2a_jsonrpc(body: JSONRPCRequest, request: Request) -> Response:
    return await _proxy(request, "/a2a/jsonrpc")


@router.get("/agents", summary="List registered peer agents", tags=["A2A Protocol"], include_in_schema=False)
async def list_agents(request: Request) -> Response:
    return await _proxy(request, "/a2a/agents")


@router.post("/agents/register", summary="Register an external peer agent", status_code=201, tags=["A2A Protocol"], include_in_schema=False)
async def register_peer(body: PeerAgentCard, request: Request) -> Response:
    return await _proxy(request, "/a2a/agents/register")


@router.delete("/agents/{agent_id}", summary="Deregister a peer agent", tags=["A2A Protocol"], include_in_schema=False)
async def deregister_peer(agent_id: str, request: Request) -> Response:
    return await _proxy(request, f"/a2a/agents/{agent_id}")


@router.post("/agents/{agent_id}/heartbeat", summary="Renew peer agent TTL", tags=["A2A Protocol"], include_in_schema=False)
async def peer_heartbeat(agent_id: str, request: Request) -> Response:
    return await _proxy(request, f"/a2a/agents/{agent_id}/heartbeat")


@router.get("/agents/cards", summary="Fetch AgentCard from every sub-agent", tags=["A2A Protocol"], include_in_schema=False)
async def all_agent_cards(request: Request) -> Response:
    return await _proxy(request, "/agents/cards")


@router.get("/agents/{agent_name}/card", summary="Fetch AgentCard for a single sub-agent", tags=["A2A Protocol"], include_in_schema=False)
async def single_agent_card(agent_name: str, request: Request) -> Response:
    return await _proxy(request, f"/agents/{agent_name}/card")


@router.post("/agents/{agent_name}/message", summary="Forward A2A message to a specific sub-agent", tags=["A2A Protocol"], include_in_schema=False)
async def forward_agent_message(agent_name: str, body: A2AMessageRequest, request: Request) -> Response:
    return await _proxy(request, f"/agents/{agent_name}/message")


@router.post("/query", summary="RAG query — direct / agent / tool mode", tags=["A2A Protocol"])
async def rag_query(
    body: QueryRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    # Inject X-User-ID / X-Branch-ID so Module 21 can forward them to Module 04
    base = _settings.agent_protocol_url.rstrip("/")
    url  = f"{base}/query"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    user_sub = current_user.get("sub", "")
    headers["X-User-ID"]   = user_sub
    headers["X-Branch-ID"] = user_sub
    body_bytes = await request.body()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request("POST", url, headers=headers, content=body_bytes)
    except httpx.RequestError as exc:
        _logger.error("A2A proxy error → %s: %s", url, exc)
        raise HTTPException(status_code=502, detail=str(exc))
    drop = {"content-length", "transfer-encoding", "connection", "keep-alive", "date", "server"}
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() not in drop},
    )


@router.post("/message", summary="Forward A2A message (default: orchestrator; ?target=<name>)", tags=["A2A Protocol"], include_in_schema=False)
async def a2a_message(body: A2AMessageRequest, request: Request, target: Optional[str] = None) -> Response:
    return await _proxy(request, "/a2a/message")
