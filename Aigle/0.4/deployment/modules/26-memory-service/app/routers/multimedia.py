from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from core.dependencies import get_current_user
from services import embedder
from services.multimedia_memory import (
    AudioIndexRequest,
    ImageIndexRequest,
    IndexResponse,
    MediaSearchHit,
    MediaSearchRequest,
    MultimediaMemoryService,
    VideoIndexRequest,
)

router = APIRouter(prefix="/memory/multimedia", tags=["Multimedia Memory"])


def get_multimedia_service() -> MultimediaMemoryService:
    return MultimediaMemoryService()


@router.post(
    "/video",
    response_model=IndexResponse,
    status_code=201,
    summary="索引一段影片片段",
    description="將影片片段的逐字稿以 bge-m3 embedding 後寫入使用者的 multimedia 記憶，供之後語意搜尋回放。",
)
async def index_video(
    body: VideoIndexRequest,
    user_id: str = Depends(get_current_user),
    svc: MultimediaMemoryService = Depends(get_multimedia_service),
) -> IndexResponse:
    return await svc.index_video(user_id, body)


@router.post(
    "/audio",
    response_model=IndexResponse,
    status_code=201,
    summary="索引一段音訊片段",
    description="省略 `transcription` 時會自動呼叫 Module 07 ASR 產生逐字稿，再以 bge-m3 embedding 寫入索引。",
)
async def index_audio(
    body: AudioIndexRequest,
    user_id: str = Depends(get_current_user),
    svc: MultimediaMemoryService = Depends(get_multimedia_service),
) -> IndexResponse:
    transcription = body.transcription
    if not transcription:
        transcription = await embedder.transcribe(body.asset_path)
    return await svc.index_audio(user_id, body, transcription)


@router.post(
    "/image",
    response_model=IndexResponse,
    status_code=201,
    summary="索引一張圖片",
    description="將圖片的描述文字 + OCR 文字以 bge-m3 embedding 寫入使用者的 multimedia 記憶。",
)
async def index_image(
    body: ImageIndexRequest,
    user_id: str = Depends(get_current_user),
    svc: MultimediaMemoryService = Depends(get_multimedia_service),
) -> IndexResponse:
    return await svc.index_image(user_id, body)


@router.post(
    "/search",
    response_model=list[MediaSearchHit],
    summary="跨媒體語意搜尋",
    description="在目前使用者的所有 video/audio/image 索引中做混合搜尋，可用 `media_type` 限定單一媒體類型。",
)
async def search_multimedia(
    body: MediaSearchRequest,
    user_id: str = Depends(get_current_user),
    svc: MultimediaMemoryService = Depends(get_multimedia_service),
) -> list[MediaSearchHit]:
    return await svc.search(user_id, body)
