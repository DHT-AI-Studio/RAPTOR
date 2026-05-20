"""Chat endpoints — no auth required (14 is an internal service called from 12)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from ..core.config import settings
from ..services.chat_service import UnifiedChatService, strip_think_tags
from ..services.search_service import HybridSearchService

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Models ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[ChatMessage]] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: Optional[str] = None
    search_triggered: bool = False
    search_results: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_redis_client(request: Request) -> Redis:
    redis = getattr(request.app.state, "redis_client", None)
    if not redis:
        raise RuntimeError("Redis client not initialised")
    return redis


def get_http_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "http_client", None)
    if not client:
        raise RuntimeError("HTTP client not initialised")
    return client


async def get_chat_service(request: Request) -> UnifiedChatService:
    service = getattr(request.app.state, "_chat_service", None)
    if service is None:
        http_client = get_http_client(request)
        redis_client = get_redis_client(request)
        service = UnifiedChatService(
            http_client=http_client,
            redis_client=redis_client,
            search_service=HybridSearchService(settings.HYBRID_SEARCH_URL, http_client),
            model_name=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            llm_base_url=settings.LLM_BASE_URL,
            memory_context_window=settings.MEMORY_CONTEXT_WINDOW,
            memory_ttl=settings.MEMORY_TTL,
        )
        request.app.state._chat_service = service
    return service


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: UnifiedChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        history = None
        if payload.history:
            history = [{"role": m.role, "content": m.content} for m in payload.history]

        result = await chat_service.chat(
            user_id=payload.user_id,
            message=payload.message,
            history=history,
            session_id=payload.session_id,
        )

        return ChatResponse(
            response=strip_think_tags(result.get("response", "")),
            user_id=payload.user_id,
            session_id=payload.session_id,
            search_triggered=result.get("search_triggered", False),
            search_results=result.get("search_results"),
            tool_calls=result.get("tool_calls"),
        )
    except Exception as e:
        _logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/memory/{user_id}")
async def get_memory(
    user_id: str,
    session_id: Optional[str] = None,
    redis_client: Redis = Depends(get_redis_client),
):
    import json
    memory_key = f"chat_memory:{user_id}:{session_id}" if session_id else f"chat_memory:{user_id}"
    try:
        data = await redis_client.get(memory_key)
        memories = json.loads(data) if data else []
        return {"user_id": user_id, "session_id": session_id, "memory_count": len(memories), "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(
    user_id: str,
    session_id: Optional[str] = None,
    redis_client: Redis = Depends(get_redis_client),
):
    memory_key = f"chat_memory:{user_id}:{session_id}" if session_id else f"chat_memory:{user_id}"
    try:
        await redis_client.delete(memory_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
