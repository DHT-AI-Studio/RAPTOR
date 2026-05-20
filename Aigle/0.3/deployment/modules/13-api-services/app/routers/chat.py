"""Chat router — proxies authenticated requests to module 14 (chat-service)."""
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
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request without user_id - will be auto-populated from JWT token."""
    message: str
    history: Optional[List[ChatMessage]] = None
    session_id: Optional[str] = Field(None, description="Optional session ID for isolated conversation memory")


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: Optional[str] = None
    search_triggered: bool = False
    search_results: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"], include_in_schema=False)
async def chat(
    payload: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage_svc: StorageService = Depends(get_storage_service),
) -> ChatResponse:
    """Chat endpoint - user_id is automatically extracted from JWT token."""
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

        # Resolve asset URLs in search_results (same enrichment as search endpoints)
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


@router.get("/memory", tags=["Chat"], include_in_schema=False)
async def get_own_memory(
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Get own chat memory. Pass ?session_id=xxx to retrieve a specific session."""
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


@router.delete("/memory", status_code=status.HTTP_204_NO_CONTENT, response_model=None, tags=["Chat"], include_in_schema=False)
async def clear_own_memory(
    session_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> None:
    """Clear own chat memory. Pass ?session_id=xxx to clear a specific session only."""
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
