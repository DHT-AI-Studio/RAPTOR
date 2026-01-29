"""
API router for chat endpoints integrating UnifiedChatService.
需要認證才能使用的對話端點
"""
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.api.dependencies import get_current_user
from app.core.config import Settings, get_settings
from app.services.chat_service import UnifiedChatService, strip_think_tags
from app.services.search_service import SearchService

_logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 請求/回應模型 ====================

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="消息角色: 'user' 或 'assistant'")
    content: str = Field(..., description="消息內容")


class ChatRequest(BaseModel):
    """聊天請求"""
    user_id: str = Field(..., description="用戶 ID")
    message: str = Field(..., description="用戶消息")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="對話歷史（可選）"
    )


class SearchResult(BaseModel):
    """搜尋結果"""
    collection: str
    items: List[Dict[str, Any]]
    stats: Optional[Dict[str, Any]] = None
    query: Optional[str] = None


class ToolCall(BaseModel):
    """工具調用記錄"""
    tool: str
    query: Optional[str] = None
    results_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    api_endpoint: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天回應"""
    response: str = Field(..., description="助手回應")
    user_id: str = Field(..., description="用戶 ID")
    search_triggered: bool = Field(default=False, description="是否觸發搜尋")
    search_results: Optional[List[SearchResult]] = Field(
        default=None,
        description="搜尋結果（如果有搜尋）"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="工具調用記錄"
    )


# ==================== 依賴注入 ====================

def get_http_client(request: Request) -> httpx.AsyncClient:
    """取得 HTTP 客戶端"""
    client = getattr(request.app.state, "http_client", None)
    if not client:
        raise RuntimeError("HTTP client is not initialised")
    return client


def get_redis_client(request: Request) -> Redis:
    """取得 Redis 客戶端"""
    redis = getattr(request.app.state, "redis_client", None)
    if not redis:
        raise RuntimeError("Redis client is not initialised")
    return redis


async def get_chat_service(
    request: Request,
    http_client: httpx.AsyncClient = Depends(get_http_client),
    redis_client: Redis = Depends(get_redis_client),
    settings: Settings = Depends(get_settings),
) -> UnifiedChatService:
    """取得或創建聊天服務實例"""
    service_key = "_chat_service"
    
    # 從應用狀態中取得已存在的服務（如果有）
    service = getattr(request.app.state, service_key, None)
    
    if service is None:
        # 創建新服務
        service = UnifiedChatService(
            http_client=http_client,
            redis_client=redis_client,
            search_service=SearchService(settings.qdrant_host, http_client),
            api_base_url=settings.api_base_url,
            model_name=settings.chat_model_name,
            temperature=settings.chat_temperature,
            api_key=settings.openai_api_key,
            llm_base_url=settings.llm_base_url,
            memory_context_window=settings.chat_memory_context_window,
            memory_ttl=settings.chat_memory_ttl,
        )
        
        # 保存到應用狀態
        setattr(request.app.state, service_key, service)
    
    return service


# ==================== 端點 ====================

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Send a message to the unified chat service. Requires authentication."
)
async def chat(
    request: Request,
    payload: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: UnifiedChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    發送聊天消息
    
    需要通過 Bearer token 認證。
    
    **Parameters:**
    - `user_id`: 用戶 ID（必須與認證用戶相符或為管理員）
    - `message`: 用戶消息
    - `history`: 對話歷史（可選）
    
    **Returns:**
    - 聊天回應、搜尋結果和工具調用日誌
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8012/api/v1/chat/chat" \\
      -H "Authorization: Bearer YOUR_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d {
        "user_id": "dht_admin",
        "message": "什麼是機器學習？",
        "history": []
      }
    ```

        **Usage:**
        ```python
        url = f"{base_url}/api/v1/chat/chat"
        payload = {
                "user_id": user_id,
                "message": message
        }
        if search_results:
                payload["search_results"] = search_results

        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()
        ```
    """
    # 驗證用戶權限
    # 用戶只能查詢自己的數據，或者是管理員
    authenticated_user_id = current_user.get("sub")
    
    if payload.user_id != authenticated_user_id:
        _logger.warning(
            f"Unauthorized user access attempt: {authenticated_user_id} trying to access {payload.user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own chat data"
        )
    
    try:
        _logger.info(
            f"Processing chat request",
            extra={
                "user_id": payload.user_id,
                "message_length": len(payload.message),
                "has_history": payload.history is not None
            }
        )
        
        # 轉換歷史格式
        history = None
        if payload.history:
            history = [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in payload.history
            ]
        
        # 調用聊天服務
        result = await chat_service.chat(
            user_id=payload.user_id,
            message=payload.message,
            history=history
        )

        # 過濾掉 <think>...</think> 內容
        response_content = strip_think_tags(result.get("response", ""))
        
        # 構建回應
        return ChatResponse(
            response=response_content,
            user_id=payload.user_id,
            search_triggered=result.get("search_triggered", False),
            search_results=result.get("search_results"),
            tool_calls=result.get("tool_calls")
        )
    
    except Exception as e:
        _logger.error(
            f"Chat service error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat service error: {str(e)}"
        )


@router.get(
    "/memory/{user_id}",
    tags=["chat"],
    status_code=status.HTTP_200_OK,
    summary="Get user chat memory",
    description="Retrieve the short-term memory for a user. Requires authentication."
)
async def get_memory(
    user_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis_client),
) -> Dict[str, Any]:
    """
    取得用戶的短期記憶
    
    需要通過 Bearer token 認證。
    
    **Parameters:**
    - `user_id`: 用戶 ID（必須與認證用戶相符）
    
    **Returns:**
    - 用戶的短期對話記憶列表
    
    **Example:**
    ```bash
    curl -X GET "http://localhost:8012/api/v1/chat/memory/dht_admin" \\
      -H "Authorization: Bearer YOUR_TOKEN"
    ```

        **Usage:**
        ```python
        url = f"{base_url}/api/v1/chat/memory/{user_id}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()
        ```
    """
    # 驗證用戶權限
    authenticated_user_id = current_user.get("sub")
    
    if user_id != authenticated_user_id:
        _logger.warning(
            f"Unauthorized memory access attempt: {authenticated_user_id} trying to access {user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own memory"
        )
    
    try:
        memory_key = f"chat_memory:{user_id}"
        memory_data = await redis_client.get(memory_key)
        
        if memory_data:
            import json
            memories = json.loads(memory_data)
            return {
                "user_id": user_id,
                "memory_count": len(memories),
                "memories": memories
            }
        else:
            return {
                "user_id": user_id,
                "memory_count": 0,
                "memories": []
            }
    
    except Exception as e:
        _logger.error(
            f"Error retrieving memory: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving memory: {str(e)}"
        )


@router.delete(
    "/memory/{user_id}",
    tags=["chat"],
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear user chat memory",
    description="Clear the short-term memory for a user. Requires authentication."
)
async def clear_memory(
    user_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis_client),
) -> None:
    """
    清除用戶的短期記憶
    
    需要通過 Bearer token 認證。
    
    **Parameters:**
    - `user_id`: 用戶 ID（必須與認證用戶相符）
    
    **Example:**
    ```bash
    curl -X DELETE "http://localhost:8012/api/v1/chat/memory/dht_admin" \\
      -H "Authorization: Bearer YOUR_TOKEN"
    ```

        **Usage:**
        ```python
        url = f"{base_url}/api/v1/chat/memory/{user_id}"
        response = requests.delete(url, headers=self._get_headers())
        response.raise_for_status()
        ```
    """
    # 驗證用戶權限
    authenticated_user_id = current_user.get("sub")
    
    if user_id != authenticated_user_id:
        _logger.warning(
            f"Unauthorized memory clear attempt: {authenticated_user_id} trying to clear {user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only clear your own memory"
        )
    
    try:
        memory_key = f"chat_memory:{user_id}"
        await redis_client.delete(memory_key)
        
        _logger.info(f"Cleared memory for user {user_id}")
    
    except Exception as e:
        _logger.error(
            f"Error clearing memory: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing memory: {str(e)}"
        )
