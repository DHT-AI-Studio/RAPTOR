from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from typing import Optional
from redis.asyncio import Redis

from core.dependencies import get_current_user, get_redis
from services.session_memory import (
    SessionMemoryService,
    SessionSearchRequest,
    SessionSearchResponse,
    SessionInfo,
    TimelineResponse,
    TurnAppendRequest,
    TurnAppendResponse,
)

router = APIRouter(prefix="/memory/sessions", tags=["Session Memory"])


def get_session_service(redis: Redis = Depends(get_redis)) -> SessionMemoryService:
    return SessionMemoryService(redis=redis)


_SESSION_ID = Path(..., description="Session ID；同一使用者可有多個 session，彼此獨立")


@router.post(
    "/{session_id}/turns",
    response_model=TurnAppendResponse,
    status_code=201,
    summary="寫入一輪對話",
    description="將一輪 user_message + assistant_response 寫入該 session 的 .mv2。"
    "寫入後會在背景 fire-and-forget 觸發 long-term 記憶提取，不阻塞本次回應。",
)
async def append_turn(
    session_id: str = _SESSION_ID,
    body: TurnAppendRequest = ...,
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service),
) -> TurnAppendResponse:
    return await svc.append_turn(user_id, session_id, body)


@router.post(
    "/{session_id}/search",
    response_model=SessionSearchResponse,
    summary="Session 內混合搜尋",
    description="在單一 session 內做 BM25 + 語意向量的混合搜尋，可選日期區間過濾。",
)
async def search_session(
    session_id: str = _SESSION_ID,
    body: SessionSearchRequest = ...,
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service),
) -> SessionSearchResponse:
    return await svc.search(user_id, session_id, body)


@router.get(
    "/{session_id}/timeline",
    response_model=TimelineResponse,
    summary="分頁瀏覽對話歷史（支援 time-travel）",
    description="依時間升序分頁回傳該 session 的對話輪次；帶 `at` 參數時只回傳該時間點之前的內容，"
    "可用來重現歷史某一刻的對話狀態。",
)
async def get_timeline(
    session_id: str = _SESSION_ID,
    page: int = Query(1, ge=1, description="頁碼（1-based）"),
    page_size: int = Query(20, ge=1, le=100, description="每頁筆數（最大 100）"),
    at: Optional[str] = Query(
        None, description="ISO 8601 timestamp；指定後只回傳該時間點之前（不含）的 frames（time-travel）"
    ),
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service),
) -> TimelineResponse:
    return await svc.get_timeline(user_id, session_id, page, page_size, at)


@router.get(
    "/{session_id}/recent",
    summary="取最近 N 輪對話",
    description="時間升序回傳最近 N 輪對話（讀 .mv2 timeline，已過濾摘要／壓縮邊界等非對話 frame）。",
)
async def get_recent_turns(
    session_id: str = _SESSION_ID,
    n: int = Query(10, ge=1, le=100, alias="n", description="回傳最近幾輪對話"),
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service),
) -> list[dict]:
    return await svc.get_recent(user_id, session_id, n)


@router.get(
    "",
    response_model=list[SessionInfo],
    summary="列出目前使用者的所有 session",
)
async def list_sessions(
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service),
) -> list[SessionInfo]:
    return await svc.list_sessions(user_id)


@router.delete(
    "/{session_id}",
    status_code=204,
    response_class=Response,
    summary="刪除 session",
    description="永久刪除該 session 的 .mv2 檔案與 Redis 索引；不存在則回 404。",
)
async def delete_session(
    session_id: str = _SESSION_ID,
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service),
) -> Response:
    found = await svc.delete_session(user_id, session_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return Response(status_code=204)
