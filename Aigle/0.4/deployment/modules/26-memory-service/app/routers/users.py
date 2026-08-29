from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from pydantic import BaseModel, Field

from core.dependencies import get_current_user
from services.long_term_memory import (
    FactAddRequest,
    FactAddResponse,
    LongTermMemoryService,
    SearchHit,
    SearchRequest,
)

router = APIRouter(prefix="/memory/longterm", tags=["Long-term Memory"])


def get_lt_service() -> LongTermMemoryService:
    return LongTermMemoryService()


@router.post(
    "/search",
    response_model=list[SearchHit],
    summary="跨 session 語意搜尋 long-term 記憶",
    description="對目前使用者的所有事實/偏好/實體做 BM25 + bge-m3 語意混合搜尋，可選 Unix timestamp 日期範圍過濾。",
)
async def search_longterm(
    body: SearchRequest,
    user_id: str = Depends(get_current_user),
    svc: LongTermMemoryService = Depends(get_lt_service),
) -> list[SearchHit]:
    return await svc.search(user_id, body)


@router.post(
    "/facts",
    response_model=FactAddResponse,
    status_code=201,
    summary="寫入一筆事實 / 偏好 / 實體",
    description="直接新增一筆 long-term 記憶，不經過 LLM 提取。",
)
async def add_fact(
    body: FactAddRequest,
    user_id: str = Depends(get_current_user),
    svc: LongTermMemoryService = Depends(get_lt_service),
) -> FactAddResponse:
    return await svc.add_fact(user_id, body)


@router.get(
    "/facts",
    summary="列出所有 long-term 記憶",
    description="回傳目前使用者所有 preference + fact + entity（時間降序），不含已刪除的 frame。",
)
async def get_facts(
    user_id: str = Depends(get_current_user),
    svc: LongTermMemoryService = Depends(get_lt_service),
) -> list[dict]:
    return await svc.get_facts(user_id)


@router.delete(
    "/facts/{frame_id}",
    status_code=204,
    response_class=Response,
    summary="刪除一筆 long-term 記憶",
    description="依 frame_id 刪除單筆事實/偏好/實體；不存在則回 404。",
)
async def delete_fact(
    frame_id: str = Path(..., description="要刪除的 frame ID（由 GET /facts 或提取結果取得）"),
    user_id: str = Depends(get_current_user),
    svc: LongTermMemoryService = Depends(get_lt_service),
) -> Response:
    found = await svc.delete_fact(user_id, frame_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fact not found")
    return Response(status_code=204)


class ExtractRequest(BaseModel):
    session_id: str = Field(..., description="這一輪對話所屬的 session ID（僅作追溯用途）")
    turn: dict = Field(
        ...,
        description="對話輪內容，至少需含 user_message 與 assistant_response 兩個欄位",
    )


@router.post(
    "/extract",
    summary="手動觸發 long-term 記憶提取（內部端點）",
    description="Internal endpoint — not exposed via Module 13 gateway.\n\n"
    "呼叫 Module 07 LLM 對這一輪對話做 ADD/DELETE/UPDATE 提取並直接執行，回傳實際執行的操作數。"
    "通常由 append_turn 背景自動觸發，此端點供測試 / 手動觸發使用。"
    "`user_id` 一律取自 JWT `sub`，body 不接受 user_id 欄位。",
)
async def extract_longterm(
    body: ExtractRequest,
    user_id: str = Depends(get_current_user),
    svc: LongTermMemoryService = Depends(get_lt_service),
) -> dict:
    from services.extractor import extract_and_store
    stored = await extract_and_store(user_id, body.session_id, body.turn, svc)
    return {"stored": stored}


@router.delete(
    "",
    status_code=204,
    response_class=Response,
    summary="清空整個 long-term 記憶",
    description="永久刪除目前使用者所有 long_term_N.mv2 shard；不存在則回 404。",
)
async def delete_longterm(
    user_id: str = Depends(get_current_user),
    svc: LongTermMemoryService = Depends(get_lt_service),
) -> Response:
    found = await svc.delete(user_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Long-term memory not found")
    return Response(status_code=204)
