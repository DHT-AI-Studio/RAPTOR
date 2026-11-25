"""Chat JSON and SSE endpoints for the internal chat service."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import redis
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from redis.asyncio import Redis

from ..core.config import settings
from ..services.chat_service import UnifiedChatService, strip_think_tags
from ..services.memory_client import MemoryClient
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
    history: list[ChatMessage] | None = None
    session_id: str | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str | None = None
    request_id: str | None = None
    search_triggered: bool = False
    search_results: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None


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
            search_service=HybridSearchService(settings.PERSONAL_DB_SERVICE_URL, http_client),
            model_name=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            llm_base_url=settings.LLM_BASE_URL,
            llm_fallback_base_url=settings.LLM_FALLBACK_BASE_URL or None,
            guardrail_fallback=settings.GUARDRAIL_FALLBACK_PASSTHROUGH,
            num_ctx=settings.LLM_NUM_CTX,
            think=settings.LLM_THINK,
            memory_context_window=settings.MEMORY_CONTEXT_WINDOW,
            memory_ttl=settings.MEMORY_TTL,
            memory_client=MemoryClient(
                settings.CHAT_MEMORY_SERVICE_URL,
                http_client,
                timeout=settings.MEMORY_REQUEST_TIMEOUT,
                compact_timeout=settings.COMPACT_REQUEST_TIMEOUT,
            ),
        )
        request.app.state._chat_service = service
    return service


# ── Endpoints ─────────────────────────────────────────────────────────────────

_SENTENCE = re.compile(r"[^。！？!?\n]*(?:[。！？!?]+|\n+|$)")
SSE_PING_S = 10.0


def _sentences(text: str) -> list[str]:
    return [m.group(0) for m in _SENTENCE.finditer(text) if m.group(0).strip()]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_chat(chat_service: UnifiedChatService, payload: ChatRequest) -> dict[str, Any]:
    history = None
    if payload.history:
        history = [{"role": m.role, "content": m.content} for m in payload.history]
    return await chat_service.chat(
        user_id=payload.user_id,
        message=payload.message,
        history=history,
        session_id=payload.session_id,
    )


async def _event_stream(chat_service: UnifiedChatService, payload: ChatRequest,
                        request_id: str) -> AsyncIterator[str]:
    yield _sse("start", {"request_id": request_id, "session_id": payload.session_id})
    task = asyncio.create_task(_run_chat(chat_service, payload))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=SSE_PING_S)
            if done:
                break
            yield ": ping\n\n"
        result = task.result()
    except Exception as e:
        _logger.error("Chat error (request %s): %s", request_id, e, exc_info=True)
        yield _sse("error", {"request_id": request_id, "message": str(e)})
        return
    finally:
        task.cancel()            # the client went away: stop the graph
    response = strip_think_tags(result.get("response", ""))
    for sentence in _sentences(response):
        yield _sse("delta", {"text": sentence})
    yield _sse("end", {"request_id": request_id, "response": response,
                       "search_triggered": result.get("search_triggered", False),
                       "tool_calls": len(result.get("tool_calls") or [])})


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: UnifiedChatService = Depends(get_chat_service),
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
):
    request_id = x_request_id or uuid.uuid4().hex
    _logger.info("chat request %s (user %s, session %s, stream=%s)", request_id, payload.user_id,
                 payload.session_id, payload.stream)
    if payload.stream:
        return StreamingResponse(_event_stream(chat_service, payload, request_id), media_type="text/event-stream",
                                 headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"})
    try:
        result = await _run_chat(chat_service, payload)
        return ChatResponse(
            response=strip_think_tags(result.get("response", "")),
            user_id=payload.user_id,
            session_id=payload.session_id,
            request_id=request_id,
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
    session_id: str | None = None,
    redis_client: Redis = Depends(get_redis_client),
):
    import json
    memory_key = f"chat_memory:{user_id}:{session_id}" if session_id else f"chat_memory:{user_id}"
    try:
        data = await redis_client.get(memory_key)
        memories = json.loads(data) if data else []
        return {"user_id": user_id, "session_id": session_id, "memory_count": len(memories), "memories": memories}
    except HTTPException:
        raise
    except (redis.RedisError, json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(
    user_id: str,
    session_id: str | None = None,
    redis_client: Redis = Depends(get_redis_client),
):
    memory_key = f"chat_memory:{user_id}:{session_id}" if session_id else f"chat_memory:{user_id}"
    try:
        await redis_client.delete(memory_key)
    except HTTPException:
        raise
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=str(e))
