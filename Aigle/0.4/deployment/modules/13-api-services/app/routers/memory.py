"""Memory Service (Module 26) proxy router.

Proxies a subset of Module 26's endpoints under /api/{version}/memory with
JWT enforcement (compactdesign.md §10.3): every route requires a valid
bearer token, and X-User-ID is always derived from the verified JWT `sub`.
A caller-supplied X-User-ID that disagrees with that sub is rejected with
403 rather than silently overwritten — see _reject_cross_user_spoofing.

Gateway surface intentionally scoped down (per Fung's review) to just 6
flat routes, all tagged "Memory": POST /store, GET /retrieve, GET
/timeline, POST /compact, DELETE /, POST /archive. Every other Module 26
route this file used to proxy (session-scoped memory, long-term facts,
multimedia indexing, global search, stats/export) is commented out in
place rather than deleted, in case a richer surface is needed later.

Body/Path/Query parameters below are declared with Pydantic models and
`example=` purely for Swagger UI documentation (same pattern as
app/routers/training.py) — the actual bytes forwarded upstream always come
from `request.body()` / `request.query_params`, not from the parsed values,
so upstream stays the single source of truth for validation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response
# StreamingResponse: only used by the disabled export_memory route below;
# re-add to the import above if that route is restored.
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user
from app.core.config import get_settings

_logger   = logging.getLogger(__name__)
_settings = get_settings()

router = APIRouter()


def _reject_cross_user_spoofing(request: Request, current_user: Dict[str, Any]) -> None:
    """A caller-supplied X-User-ID that disagrees with the verified JWT sub is
    a cross-user access attempt, not a value to silently correct — reject it
    outright so spoofing attempts are visible (403 + logged) rather than
    quietly overwritten and let through under the caller's own identity."""
    spoofed = request.headers.get("x-user-id")
    sub = current_user.get("sub", "")
    if spoofed and spoofed != sub:
        _logger.warning(
            "Rejected cross-user access attempt: caller supplied X-User-ID=%r, JWT sub=%r",
            spoofed, sub,
        )
        raise HTTPException(status_code=403, detail="Cross-user access denied")


async def _proxy(
    request: Request,
    upstream_path: str,
    current_user: Dict[str, Any],
    timeout: float = 90.0,
    method: Optional[str] = None,
    json_body: Optional[dict] = None,
) -> Response:
    """Forward the request to Module 26 under the verified JWT sub.

    `method`/`json_body` let a route reshape the call for upstream (e.g. the
    GET /retrieve alias turning query params into the POST body /memory/search
    expects) instead of straight passthrough of the incoming request.
    """
    _reject_cross_user_spoofing(request, current_user)
    base = _settings.memory_service_url.rstrip("/")
    url  = f"{base}{upstream_path}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "x-user-id"}
    }
    headers["X-User-ID"] = current_user.get("sub", "")
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers["content-type"] = "application/json"
        params = None
    else:
        body = await request.body()
        params = request.query_params
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method=method or request.method,
                url=url,
                headers=headers,
                params=params,
                content=body,
            )
    except httpx.RequestError as exc:
        _logger.error("Memory proxy error → %s: %s", url, exc)
        raise HTTPException(status_code=502, detail=str(exc))

    drop = {"content-length", "transfer-encoding", "connection", "keep-alive", "date", "server"}
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() not in drop},
    )


# ── Path/Query param helpers ──────────────────────────────────────────────────

_SESSION_ID  = Path(..., description="Session ID；同一使用者可有多個 session，彼此獨立", example="sess_test_001")
_FRAME_ID    = Path(..., description="要刪除的 frame ID（由 GET /longterm/facts 或提取結果取得）", example="frame_a1b2c3")
_SUMMARY_ID  = Path(..., description="要刪除的 summary frame ID（由 GET .../summaries 取得）", example="sum_9f8e7d")


# ── Body models (docs/validation only — see module docstring) ────────────────

class CompactEvaluateBody(BaseModel):
    messages: List[dict] = Field(
        default_factory=list,
        description="要估算 token 用量的訊息列表（不限格式，僅用於估算大小）；有帶 session_id 時忽略此欄位",
        example=[{"role": "user", "content": "上季營收多少？"}],
    )
    session_id: Optional[str] = Field(
        None,
        description="省略時只估算 messages 大小；有帶時改用合併算法，"
        "彙總該 session 的歸檔對話 + long-term facts + multimedia snippets",
        example="sess_test_001",
    )
    context_window: int = Field(128000, ge=4096, description="目標 LLM 的 context window 大小（tokens）", example=128000)
    extra_tokens: int = Field(0, ge=0, description="尚未寫入 session 的當前這輪 turn 的估算 token 數", example=0)
    max_tokens: Optional[int] = Field(None, ge=1, description="本次請求實際的輸出 max_tokens", example=4096)


class SessionCompactEvaluateBody(BaseModel):
    context_window: int = Field(128000, ge=4096, description="目標 LLM 的 context window 大小（tokens）", example=128000)
    extra_tokens: int = Field(0, ge=0, description="尚未寫入 session 的當前這輪 turn 的估算 token 數", example=0)
    max_tokens: Optional[int] = Field(None, ge=1, description="本次請求實際的輸出 max_tokens", example=4096)


class CompactBody(BaseModel):
    trigger: str = Field("auto", description="觸發來源：auto | manual | reactive，僅作記錄用途", example="manual")
    context_window: int = Field(128000, ge=4096, description="目標 LLM 的 context window 大小（tokens）", example=128000)
    max_tokens: Optional[int] = Field(None, ge=1, description="本次請求實際的輸出 max_tokens", example=4096)
    last_summarized_frame_id: Optional[str] = Field(
        None,
        description="上次壓縮邊界的 frame_id；省略時自動尋找該 session 最新的 summary frame",
        example="frame_9f8e7d",
    )
    custom_instructions: Optional[str] = Field(
        None, description="給 LLM 摘要時的額外指示（可省略）", example="請特別保留與合約條款相關的細節"
    )


class TurnAppendBody(BaseModel):
    user_message: str = Field(..., description="使用者這一輪輸入的訊息原文", example="幫我查一下上季的營收報告")
    assistant_response: str = Field(..., description="助手對這一輪訊息的回覆原文", example="已為您找到上季營收報告，重點如下…")
    search_results: List[dict] = Field(default_factory=list, description="這一輪回覆中引用的檢索結果（RAG hits）")
    tool_calls: List[dict] = Field(default_factory=list, description="這一輪回覆中觸發的工具呼叫紀錄")
    timestamp: Optional[float] = Field(None, description="Unix timestamp（秒）；省略則使用伺服器目前時間")
    provider_message_id: Optional[str] = Field(None, description="上游 LLM provider 的 message ID（可省略）")


class SessionSearchBody(BaseModel):
    query: str = Field(..., description="查詢文字，會同時做 BM25 關鍵字比對與 bge-m3 語意向量比對", example="營收報告")
    top_k: int = Field(5, ge=1, le=50, description="回傳筆數上限")
    from_date: Optional[str] = Field(None, description="ISO 8601 字串；只回傳此時間（含）之後的對話輪")
    to_date: Optional[str] = Field(None, description="ISO 8601 字串；只回傳此時間（含）之前的對話輪")


class LongtermSearchBody(BaseModel):
    query: str = Field(..., description="查詢文字，做 BM25 + bge-m3 語意向量混合搜尋", example="使用者語言偏好")
    top_k: int = Field(5, ge=1, le=50, description="回傳筆數上限")
    from_date: Optional[float] = Field(None, description="Unix timestamp；只回傳此時間（含）之後寫入的記憶")
    to_date: Optional[float] = Field(None, description="Unix timestamp；只回傳此時間（含）之前寫入的記憶")


class FactAddBody(BaseModel):
    text: str = Field(..., description="要記住的事實 / 偏好 / 實體描述文字", example="使用者偏好繁體中文回覆")
    frame_type: Literal["conversation", "preference", "entity", "fact"] = Field(
        "fact",
        description="conversation（重要對話摘要）| preference（使用者偏好）| entity（實體）| fact（一般事實）",
    )
    session_id: Optional[str] = Field(None, description="來源 session ID（可省略，僅作追溯用途）")


class VideoIndexBody(BaseModel):
    asset_path: str = Field(..., description="影片檔在儲存系統中的路徑（如 SeaweedFS/S3 key）", example="videos/2026/report.mp4")
    version_id: str = Field(..., description="該資產的版本 ID，用於區分同一路徑的不同版本", example="v1")
    start_sec: float = Field(..., description="片段在影片中的起始秒數")
    end_sec: float = Field(..., description="片段在影片中的結束秒數")
    transcription: str = Field(..., description="該片段的逐字稿文字，用於 embedding 及全文搜尋")
    session_id: str = Field("", description="觸發索引的 session ID（可省略，僅作追溯用途）")
    context_query: str = Field("", description="使用者當時查詢的問題（可省略）")


class AudioIndexBody(BaseModel):
    asset_path: str = Field(..., description="音訊檔在儲存系統中的路徑", example="audio/2026/meeting.wav")
    version_id: str = Field(..., description="該資產的版本 ID", example="v1")
    start_sec: float = Field(..., description="片段起始秒數")
    end_sec: float = Field(..., description="片段結束秒數")
    transcription: Optional[str] = Field(None, description="逐字稿文字；省略時會自動呼叫 Module 07 ASR 產生")
    session_id: str = Field("", description="觸發索引的 session ID（可省略）")
    context_query: str = Field("", description="使用者當時查詢的問題（可省略）")


class ImageIndexBody(BaseModel):
    asset_path: str = Field(..., description="圖片檔在儲存系統中的路徑", example="images/2026/whiteboard.png")
    version_id: str = Field(..., description="該資產的版本 ID", example="v1")
    ocr_text: str = Field("", description="圖片 OCR 辨識出的文字（可省略）")
    description: str = Field("", description="圖片的文字描述（可省略），與 ocr_text 一併做 embedding")
    session_id: str = Field("", description="觸發索引的 session ID（可省略）")


class MediaSearchBody(BaseModel):
    query: str = Field(..., description="查詢文字，做 BM25 + bge-m3 語意向量混合搜尋", example="上季會議白板照片")
    top_k: int = Field(5, ge=1, le=50, description="回傳筆數上限")
    media_type: Optional[Literal["video", "audio", "image"]] = Field(
        None, description="限定搜尋的媒體類型；省略則搜尋全部類型"
    )


class GlobalSearchBody(BaseModel):
    query: str = Field(..., description="查詢文字，做 BM25 + bge-m3 語意向量混合搜尋", example="上季營收與相關會議紀錄")
    top_k: int = Field(5, ge=1, le=50, description="每個 scope 各自的回傳筆數上限")
    scope: List[Literal["sessions", "longterm", "multimedia"]] = Field(
        default=["sessions", "longterm", "multimedia"],
        description="要搜尋的記憶範圍，可縮小至任意子集；預設全選",
    )


# ── Compact ────────────────────────────────────────────────────────────────
# evaluate_compact and get_session_summaries (below) are kept live even
# though they're outside Fung's approved flat surface — 21-agent-protocol's
# memory_client.py calls both through this gateway by design (JWT
# propagation; see its own comment), so removing them would break Module 21
# at runtime, not just trim unused user-facing surface.

@router.post(
    "/compact/evaluate",
    summary="估算壓縮預算（可選 session_id 合併算法）",
    description="省略 `session_id` 時只估算 `messages` 大小（舊行為）；帶 `session_id` 時改用合併算法，"
    "彙總該 session 的歸檔對話 + long-term facts + multimedia snippets 一併估算。",
    tags=["Memory"],
)
async def evaluate_compact(
    request: Request,
    body: CompactEvaluateBody = Body(
        ...,
        example={"session_id": "sess_test_001", "context_window": 128000},
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, "/memory/compact/evaluate", current_user, timeout=10.0)


# @router.post("/sessions/{session_id}/compact/evaluate", summary="估算單一 session 的合併壓縮預算", tags=["Memory Compact"])
# async def evaluate_session_compact(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     body: SessionCompactEvaluateBody = Body(..., example={"context_window": 128000}),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}/compact/evaluate", current_user, timeout=10.0)
#
#
# @router.post("/sessions/{session_id}/compact", summary="壓縮一個 session", tags=["Memory Compact"])
# async def compact_session(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     body: CompactBody = Body(..., example={"trigger": "manual", "context_window": 128000}),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}/compact", current_user)
#
#
@router.get("/sessions/{session_id}/summaries", summary="列出 session 的 summary frames", tags=["Memory"])
async def get_session_summaries(
    request: Request,
    session_id: str = _SESSION_ID,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, f"/memory/sessions/{session_id}/summaries", current_user, timeout=10.0)


#
# @router.delete("/sessions/{session_id}/summaries/{summary_id}", summary="刪除一個 summary frame", tags=["Memory Compact"])
# async def delete_session_summary(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     summary_id: str = _SUMMARY_ID,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(
#         request, f"/memory/sessions/{session_id}/summaries/{summary_id}", current_user, timeout=10.0
#     )


# ── Session Memory (disabled — see note near "Flat top-level aliases" below) ─
#
# @router.post(
#     "/sessions/{session_id}/turns",
#     summary="寫入一輪對話",
#     description="將一輪 user_message + assistant_response 寫入該 session。"
#     "寫入後會在背景 fire-and-forget 觸發 long-term 記憶提取，不阻塞本次回應。",
#     tags=["Session Memory"],
# )
# async def append_turn(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     body: TurnAppendBody = Body(...),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}/turns", current_user)
#
#
# @router.post(
#     "/sessions/{session_id}/search",
#     summary="Session 內混合搜尋",
#     description="在單一 session 內做 BM25 + 語意向量的混合搜尋，可選日期區間過濾。",
#     tags=["Session Memory"],
# )
# async def search_session(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     body: SessionSearchBody = Body(..., example={"query": "營收報告", "top_k": 5}),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}/search", current_user, timeout=30.0)
#
#
# @router.get(
#     "/sessions/{session_id}/timeline",
#     summary="分頁瀏覽對話歷史（支援 time-travel）",
#     description="依時間升序分頁回傳該 session 的對話輪次；帶 `at` 參數時只回傳該時間點之前的內容。",
#     tags=["Session Memory"],
# )
# async def get_timeline(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     page: int = Query(1, ge=1, description="頁碼（1-based）"),
#     page_size: int = Query(20, ge=1, le=100, description="每頁筆數（最大 100）"),
#     at: Optional[str] = Query(
#         None, description="ISO 8601 timestamp；指定後只回傳該時間點之前（不含）的內容（time-travel）",
#         example="2026-07-29T08:00:00Z",
#     ),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}/timeline", current_user, timeout=15.0)
#
#
# @router.get(
#     "/sessions/{session_id}/recent",
#     summary="取最近 N 輪對話",
#     description="時間升序回傳最近 N 輪對話，已過濾摘要／壓縮邊界等非對話 frame。",
#     tags=["Session Memory"],
# )
# async def get_recent_turns(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     n: int = Query(10, ge=1, le=100, description="回傳最近幾輪對話"),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}/recent", current_user, timeout=15.0)
#
#
# @router.get("/sessions", summary="列出目前使用者的所有 session", tags=["Session Memory"])
# async def list_sessions(
#     request: Request,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/sessions", current_user, timeout=15.0)
#
#
# @router.delete(
#     "/sessions/{session_id}",
#     summary="刪除 session",
#     description="永久刪除該 session 的所有記憶；不存在則回 404。此操作不可復原。",
#     tags=["Session Memory"],
# )
# async def delete_session(
#     request: Request,
#     session_id: str = _SESSION_ID,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/sessions/{session_id}", current_user, timeout=15.0)


# ── Long-term Memory (disabled — see note near "Flat top-level aliases" below) ─
#
# @router.post(
#     "/longterm/search",
#     summary="跨 session 語意搜尋 long-term 記憶",
#     description="對目前使用者的所有事實/偏好/實體做 BM25 + bge-m3 語意混合搜尋，可選日期範圍過濾。",
#     tags=["Long-term Memory"],
# )
# async def search_longterm(
#     request: Request,
#     body: LongtermSearchBody = Body(..., example={"query": "使用者語言偏好", "top_k": 5}),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/longterm/search", current_user, timeout=30.0)
#
#
# @router.post(
#     "/longterm/facts",
#     summary="寫入一筆事實 / 偏好 / 實體",
#     description="直接新增一筆 long-term 記憶，不經過 LLM 提取。",
#     tags=["Long-term Memory"],
# )
# async def add_fact(
#     request: Request,
#     body: FactAddBody = Body(..., example={"text": "使用者偏好繁體中文回覆", "frame_type": "preference"}),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/longterm/facts", current_user)
#
#
# @router.get(
#     "/longterm/facts",
#     summary="列出所有 long-term 記憶",
#     description="回傳目前使用者所有 preference + fact + entity（時間降序），不含已刪除的 frame。",
#     tags=["Long-term Memory"],
# )
# async def get_facts(
#     request: Request,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/longterm/facts", current_user, timeout=15.0)
#
#
# @router.delete(
#     "/longterm/facts/{frame_id}",
#     summary="刪除一筆 long-term 記憶",
#     description="依 frame_id 刪除單筆事實/偏好/實體；不存在則回 404。",
#     tags=["Long-term Memory"],
# )
# async def delete_fact(
#     request: Request,
#     frame_id: str = _FRAME_ID,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, f"/memory/longterm/facts/{frame_id}", current_user, timeout=15.0)
#
#
# @router.delete(
#     "/longterm",
#     summary="清空整個 long-term 記憶",
#     description="永久刪除目前使用者所有 long-term 記憶；不存在則回 404。此操作不可復原。",
#     tags=["Long-term Memory"],
# )
# async def delete_longterm(
#     request: Request,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/longterm", current_user, timeout=15.0)
#
#
# # ── Multimedia Memory ─────────────────────────────────────────────────────────
#
# @router.post(
#     "/multimedia/video",
#     summary="索引一段影片片段",
#     description="將影片片段的逐字稿以 bge-m3 embedding 後寫入使用者的 multimedia 記憶，供之後語意搜尋回放。",
#     tags=["Multimedia Memory"],
# )
# async def index_video(
#     request: Request,
#     body: VideoIndexBody = Body(...),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/multimedia/video", current_user)
#
#
# @router.post(
#     "/multimedia/audio",
#     summary="索引一段音訊片段",
#     description="省略 `transcription` 時會自動呼叫 Module 07 ASR 產生逐字稿，再以 bge-m3 embedding 寫入索引。",
#     tags=["Multimedia Memory"],
# )
# async def index_audio(
#     request: Request,
#     body: AudioIndexBody = Body(...),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/multimedia/audio", current_user)
#
#
# @router.post(
#     "/multimedia/image",
#     summary="索引一張圖片",
#     description="將圖片的描述文字 + OCR 文字以 bge-m3 embedding 寫入使用者的 multimedia 記憶。",
#     tags=["Multimedia Memory"],
# )
# async def index_image(
#     request: Request,
#     body: ImageIndexBody = Body(...),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/multimedia/image", current_user)
#
#
# Kept live — 21-agent-protocol's memory_client.py calls this through the
# gateway by design (same reasoning as evaluate_compact/get_session_summaries
# above); everything else in this Multimedia Memory section stays disabled.
@router.post(
    "/multimedia/search",
    summary="跨媒體語意搜尋",
    description="在目前使用者的所有 video/audio/image 索引中做混合搜尋，可用 `media_type` 限定單一媒體類型。",
    tags=["Memory"],
)
async def search_multimedia(
    request: Request,
    body: MediaSearchBody = Body(..., example={"query": "上季會議白板照片", "top_k": 5}),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, "/memory/multimedia/search", current_user, timeout=30.0)
#
#
# # ── Global Search ──────────────────────────────────────────────────────────────
#
# @router.post(
#     "/search",
#     summary="全域搜尋（sessions + long-term + multimedia）",
#     description="跨所有 session、long-term 記憶與多媒體記憶做混合搜尋，各 scope 平行查詢並各自回傳 top_k 筆。",
#     tags=["Global Search"],
# )
# async def global_search(
#     request: Request,
#     body: GlobalSearchBody = Body(..., example={"query": "上季營收與相關會議紀錄", "top_k": 5}),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/search", current_user, timeout=30.0)
#
#
# # ── Management (stats/export disabled; delete-all kept below) ──────────────────
#
# @router.get(
#     "/stats",
#     summary="記憶統計",
#     description="回傳目前使用者的 session 數、對話輪數、多媒體項目數、long-term frame 數與儲存用量。",
#     tags=["Management"],
# )
# async def get_stats(
#     request: Request,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> Response:
#     return await _proxy(request, "/memory/stats", current_user, timeout=15.0)
#
#
# @router.get(
#     "/export",
#     summary="匯出完整記憶（GDPR / 資料可攜）",
#     description="以串流方式回傳 JSON，內容含全部 sessions、long-term 記憶與多媒體 metadata（不含原始媒體檔案）。"
#     "回應帶 `Content-Disposition: attachment`，可直接下載。",
#     tags=["Management"],
# )
# async def export_memory(
#     request: Request,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> StreamingResponse:
#     """Streams the Module 26 export response through instead of buffering it —
#     exports can be large, so this bypasses `_proxy()`'s in-memory Response."""
#     _reject_cross_user_spoofing(request, current_user)
#     base = _settings.memory_service_url.rstrip("/")
#     url = f"{base}/memory/export"
#     headers = {
#         k: v for k, v in request.headers.items()
#         if k.lower() not in {"host", "x-user-id"}
#     }
#     headers["X-User-ID"] = current_user.get("sub", "")
#
#     client = httpx.AsyncClient(timeout=120.0)
#     try:
#         upstream = client.stream("GET", url, headers=headers, params=request.query_params)
#         resp = await upstream.__aenter__()
#     except httpx.RequestError as exc:
#         await client.aclose()
#         _logger.error("Memory export proxy error → %s: %s", url, exc)
#         raise HTTPException(status_code=502, detail=str(exc))
#
#     drop = {"content-length", "transfer-encoding", "connection", "keep-alive", "date", "server"}
#     passthrough_headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
#
#     async def _body() -> Any:
#         try:
#             async for chunk in resp.aiter_bytes():
#                 yield chunk
#         finally:
#             await upstream.__aexit__(None, None, None)
#             await client.aclose()
#
#     return StreamingResponse(
#         _body(),
#         status_code=resp.status_code,
#         media_type=resp.headers.get("content-type", "application/json"),
#         headers=passthrough_headers,
#     )


# ── Management (kept) ───────────────────────────────────────────────────────

@router.delete(
    "",
    summary="GDPR 抹除：刪除當前使用者所有記憶",
    description="永久刪除該使用者所有記憶（sessions、long-term、multimedia）；找不到任何記憶則回 404。此操作不可復原。",
    tags=["Memory"],
)
async def delete_all_memory(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, "/memory", current_user, timeout=30.0)


# ── Flat top-level aliases (the only endpoints kept — see Fung's scope-down) ──
# Fung reviewed the memory surface exposed on the gateway and asked to trim it
# down to just these 6 flat routes (store/retrieve/timeline/compact/archive +
# delete-all above); everything else in this file is commented out rather
# than deleted, in case a richer session-scoped surface is needed later.
# All 6 share a single "Memory" tag so they group together in Swagger UI.

@router.post(
    "/store",
    summary="寫入一則記憶節點（/longterm/facts 別名）",
    description="不綁定任何 session 的獨立記憶節點——事實 / 偏好 / 實體。"
    "要記錄的是完整一輪對話，請改用 POST /sessions/{session_id}/turns。",
    tags=["Memory"],
)
async def store_memory(
    request: Request,
    body: FactAddBody = Body(..., example={"text": "使用者偏好繁體中文回覆", "frame_type": "preference"}),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, "/memory/longterm/facts", current_user)


@router.get(
    "/retrieve",
    summary="語意 + BM25 混合搜尋（/search 別名）",
    description="跨所有 session、long-term 記憶與多媒體記憶做混合搜尋。等同 POST /search，"
    "改成 GET + query string 以符合部分客戶端預期的介面。",
    tags=["Memory"],
)
async def retrieve_memory(
    request: Request,
    query: str = Query(..., description="查詢文字", example="上季營收與相關會議紀錄"),
    top_k: int = Query(5, ge=1, le=50, description="回傳筆數上限"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(
        request, "/memory/search", current_user, timeout=30.0,
        method="POST", json_body={"query": query, "top_k": top_k},
    )


@router.get(
    "/timeline",
    summary="跨 session 時間軸（分頁，/memory/timeline 別名）",
    description="依時間升序分頁回傳使用者所有 session 的對話輪次，彼此穿插排列。"
    "只想看單一 session 請用 GET /sessions/{session_id}/timeline。",
    tags=["Memory"],
)
async def timeline_memory(
    request: Request,
    page: int = Query(1, ge=1, description="頁碼（1-based）"),
    page_size: int = Query(20, ge=1, le=100, description="每頁筆數（最大 100）"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, "/memory/timeline", current_user)


@router.post(
    "/compact",
    summary="壓縮一個 session（/sessions/{session_id}/compact 別名，預設 default）",
    description="觸發指定 session 的壓縮；省略 `session_id` 時壓縮使用者的 `default` session。"
    "跟 POST /sessions/{session_id}/compact 是同一個底層操作，只是路徑扁平化。",
    tags=["Memory"],
)
async def compact_memory(
    request: Request,
    session_id: Optional[str] = Query(None, description="要壓縮哪個 session；省略則用 `default`"),
    body: CompactBody = Body(..., example={"trigger": "manual", "context_window": 128000}),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, f"/memory/sessions/{session_id or 'default'}/compact", current_user)


@router.post(
    "/archive",
    summary="歸檔一輪對話（/sessions/{session_id}/turns 別名，預設 default）",
    description="將一輪 user_message + assistant_response 寫入指定 session；省略 `session_id` 時"
    "寫入使用者的 `default` session。跟 POST /sessions/{session_id}/turns 是同一個底層操作，"
    "只是路徑扁平化。",
    tags=["Memory"],
)
async def archive_memory(
    request: Request,
    session_id: Optional[str] = Query(None, description="要歸檔到哪個 session；省略則用 `default`"),
    body: TurnAppendBody = Body(
        ..., example={"user_message": "上次討論的冷卻系統維修計畫是什麼？", "assistant_response": "..."}
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    return await _proxy(request, f"/memory/sessions/{session_id or 'default'}/turns", current_user)
