"""Chat router."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
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
    summary="Clear conversation memory",
    description="""
Clear the current user's conversation memory from Redis.

- With `?session_id=xxx` — clears only the specified session
- Without the parameter — clears the default session memory

**Note:** This operation is irreversible; cleared memory cannot be recovered.
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
