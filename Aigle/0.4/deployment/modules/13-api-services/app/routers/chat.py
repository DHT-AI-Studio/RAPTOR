"""Chat router."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: `user` or `assistant`")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message for this turn")
    history: Optional[List[ChatMessage]] = Field(
        None,
        description=(
            "Conversation history before this turn, ordered oldest to newest. "
            "Format: `[{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]`. "
            "When omitted, only the memory stored in Redis is used."
        ),
    )
    session_id: Optional[str] = Field(
        None,
        description=(
            "Session ID for isolating conversation memory across different contexts. "
            "Multiple sessions can coexist under the same user; omit to use the default session."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "那這份文件裡有提到訓練資料的來源嘛？",
                "history": [
                    {"role": "user", "content": "幫我找一下關於 AI 訓練的文件"},
                    {"role": "assistant", "content": "找到 3 份相關文件，主要討論模型訓練流程與資料前處理。"},
                ],
                "session_id": "project-alpha",
            }
        }
    }


class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM reply (Traditional Chinese; `<think>` tags stripped)")
    user_id: str = Field(..., description="User UUID extracted from the JWT token")
    session_id: Optional[str] = Field(None, description="Session ID used for this conversation turn")
    search_triggered: bool = Field(
        False,
        description="Whether a RAG search was triggered. Automatically triggered when the message contains search-related keywords (search, find, pdf, video, etc.)",
    )
    search_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="RAG search results containing relevant segments from document / image / video / audio assets with similarity scores",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Tool execution log for this turn, covering `unified_search_tool`, `calculate`, `get_current_time`, and similar tools",
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat (RAG)",
    description="""
Send a message to the LLM with automatic RAG search and multi-turn conversation memory.

**Processing pipeline (LangGraph):**
1. **load_memory** — load the last N turns from Redis
2. **prepare_context** — combine system prompt, memory, and the current message
3. **call_model** — call the LLM (with tool use enabled)
4. **execute_tools** *(conditional)* — run search, calculation, or time tools
5. **save_memory** — persist this turn back to Redis

**Auto-search trigger:**
A hybrid search is automatically called when the message contains any of the following keywords:
`search / find / pdf / document / image / video / audio`, or ends with `?`.

**Notes:**
- `user_id` is extracted automatically from the Bearer token; do not pass it manually.
- The response is converted to **Traditional Chinese**.
""",
)
async def chat(
    payload: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage_svc: StorageService = Depends(get_storage_service),
) -> ChatResponse:
    user_id = current_user["sub"]

    try:
        chat_payload = {
            "user_id": user_id,
            "message": payload.message,
            "history": [m.model_dump() for m in payload.history] if payload.history else None,
            "session_id": payload.session_id,
        }

        resp = await http_client.post(
            f"{settings.chat_service_url}/api/v1/chat",
            json=chat_payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        chat_data = resp.json()

        search_results = chat_data.get("search_results") or []
        if search_results:
            svc = HybridSearchService(
                hybrid_search_url=settings.hybrid_search_url,
                http_client=http_client,
                storage_service=storage_svc,
            )
            user_dict = {"user_id": user_id, "branch_id": user_id}
            chat_data["search_results"] = await svc._resolve_asset_urls(search_results, user_dict)

        chat_data["session_id"] = payload.session_id or chat_data.get("session_id")
        return ChatResponse(**chat_data)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _logger.error(f"Chat proxy error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/memory",
    tags=["Chat"],
    summary="Get conversation memory",
    description="""
Retrieve the current user's conversation memory stored in Redis.

Each memory entry contains:
- `timestamp` — Unix timestamp of the recorded turn
- `user_message` — the user's message for that turn
- `assistant_response` — the LLM's reply
- `search_results` — RAG search results, if a search was triggered

**Notes:**
- Memory is scoped to the current user (extracted from the JWT token); users can only access their own memory.
- Omit `session_id` to retrieve the default session memory.
- This is Module 15's own short-term Redis cache (`chat_memory:*`) —
  the last few turns only, capped by `MEMORY_TTL`. It is NOT Module 26
  (Memory Service)'s long-term facts/preferences, searchable session
  history, or multimedia index, and does not reflect anything Module 26
  has compacted/summarized. For that data use `GET /api/{version}/memory/retrieve`
  or `/timeline` instead — see 26-memory-service/README.md.
""",
)
async def get_own_memory(
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    user_id = current_user["sub"]
    params = {"session_id": session_id} if session_id else {}
    try:
        resp = await http_client.get(
            f"{settings.chat_service_url}/api/v1/chat/memory/{user_id}",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    "/memory",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    tags=["Chat"],
    summary="Clear short-term conversation context (not a full memory wipe)",
    description="""
Clear the current user's short-term conversation context from Redis
(Module 15's own `chat_memory:*` cache) — use this for a "start a new
conversation" action.

- With `?session_id=xxx` — clears only the specified session
- Without the parameter — clears the default session memory

**This does NOT delete anything in Module 26 (Memory Service):**
long-term facts/preferences, this session's archived `.mv2` history
(still searchable via `/memory/retrieve`), and multimedia memory are
all untouched — this endpoint only clears the last few turns cached
here for fast lookup, which would otherwise also just expire on its
own after `MEMORY_TTL`. For an actual GDPR-style erasure of everything
Module 26 has stored for this user, use `DELETE /api/{version}/memory`
instead (`26-memory-service`'s `delete_all_memory`), which also cleans
up this same Redis cache as part of that erasure.
""",
)
async def clear_own_memory(
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> None:
    user_id = current_user["sub"]
    params = {"session_id": session_id} if session_id else {}
    try:
        resp = await http_client.delete(
            f"{settings.chat_service_url}/api/v1/chat/memory/{user_id}",
            params=params,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/completions",
    tags=["Chat"],
    summary="Chat completions (OpenAI-compatible)",
    description="""
Gateway-level OpenAI-compatible `chat/completions` proxy to Module 07's `/v1/chat/completions`
(`settings.aiml_lifecycle_api_url`).

**Request:** standard OpenAI `chat/completions` shape — at minimum `model` and
`messages`. `model` must already be registered in Module 07's MLflow registry
(`GET /api/{version}/aiml/models/local` lists what's available); an
unregistered name fails with `404` (`model_not_found`). To call a model that
isn't registered, add `engine` (e.g. `"engine": "ollama"`) — an
OpenAI-extension field Module 07 uses as a fallback to hit that runtime
directly by tag name, bypassing the registry lookup.

**Streaming is not supported here** — `stream` is always forwarded as `false` regardless
of what the caller sends. `GuardrailMiddleware` (`GR_ENABLED=true`) needs the complete
response body to run its post-generation policy check before releasing it to the
caller, so a partial SSE stream can never be released early. Module 07's endpoint
already defaults to `stream=false` and returns a fully spec-compliant non-streaming
`chat.completion` object regardless of the request, so this does not change the
response shape — only whether it can arrive incrementally.

Guardrail checks (when enabled) happen in `GuardrailMiddleware`, not here — this
endpoint is a plain proxy.
""",
)
async def chat_completions(
    body: Dict[str, Any] = Body(
        ...,
        example={
            "model": "qwen2.5:7b",
            "engine": "ollama",
            "messages": [{"role": "user", "content": "你好，用一句話自我介紹"}],
        },
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    body["stream"] = False
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.aiml_lifecycle_api_url}/v1/chat/completions",
                json=body,
            )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _logger.error(f"Chat completions proxy error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
