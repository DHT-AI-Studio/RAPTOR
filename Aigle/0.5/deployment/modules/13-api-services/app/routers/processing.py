"""
API router for triggering Kafka processing requests.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from aiokafka import AIOKafkaProducer  # type: ignore[import]
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis  # type: ignore[import]
from redis.exceptions import RedisError

from app.api.dependencies import get_current_user, get_storage_service
from app.core.config import Settings, get_settings
from app.services.kafka_service import KafkaService
from app.services.storage_service import StorageService, User

logger = logging.getLogger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    asset_path: str = Field(..., description="The asset_path returned by POST /fileupload.")
    version_id: str = Field(..., description="The version_id returned by POST /fileupload.")
    processing_mode: str | None = Field(None, description="Optional processing mode (e.g., 'ocr' for PDFs).")


def get_kafka_producer(request: Request) -> AIOKafkaProducer:
    """Dependency to get the application's Kafka producer."""
    producer = getattr(request.app.state, "kafka_producer", None)
    if not producer:
        raise RuntimeError("Kafka producer is not initialised")
    return producer


def get_redis_client(request: Request) -> Redis:
    """Dependency to get the application's Redis client."""
    client = getattr(request.app.state, "redis_client", None)
    if not client:
        raise RuntimeError("Redis client is not initialised")
    return client

def combine_key_(key:str, m_type:str)->str:
    return f"{m_type}_orchestrator:{key}"

@router.post(
    "/process-file",
    tags=["Processing"],
    status_code=status.HTTP_200_OK,
    summary="Register an uploaded/failed asset for AI analysis",
    description="""
**VIE01-215 — Register API.** Explicitly triggers AI analysis for an asset previously committed via `POST /fileupload` (status `uploaded`), decoupling upload from analysis.

**Behavior depends on the asset's current status:**
- Not found → `404`
- `unregistered` (content already indexed via deduplication) → flipped to `active` synchronously, no Kafka message published
- `processing` or `active` → `409` (already being analyzed, or already live)
- `uploaded` or `failed` → a Kafka analysis request is published; **status is left unchanged** until the orchestrator that picks up the job calls the internal claim-processing step

Re-registering a `failed` asset re-publishes Kafka; the asset stays `failed` until claimed.
""",
)
async def process_file(
    proc_request: RegisterRequest,
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    current_user: dict[str, Any] = Depends(get_current_user),
    svc: StorageService = Depends(get_storage_service),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    user_id = current_user["sub"]
    user = User(user_id=user_id, branch_id=user_id)

    # The four-branch guard (not_found/conflict/registered/needs_processing)
    # lives in Module 04 -- it owns the status row and the Module 25 call
    # for the "unregistered" fast path. This raises 404/409 straight
    # through when Module 04 returns them.
    register_result = await svc.register_asset(proc_request.asset_path, proc_request.version_id, user)

    if register_result["outcome"] == "registered":
        # "unregistered" -> "active": Module 04 already did the synchronous
        # flip (and the Module 25 call behind it). No Kafka needed.
        return register_result

    # "needs_processing": status is "uploaded" or "failed", left untouched.
    # Publish to Kafka same as before; "processing" is only ever written by
    # the orchestrator that actually claims the job (POST /claim-processing
    # on Module 04), not here.
    kafka_service = KafkaService(producer, settings)
    kafka_result = await kafka_service.send_processing_request(
        upload_result=register_result["metadata"],
        user_id=user_id,
        branch_id=user_id,
        processing_mode=proc_request.processing_mode,
    )
    return kafka_result


@router.get("/cache/{m_type}/{key}", tags=["Processing"], status_code=status.HTTP_200_OK,
            summary="查詢單一檔案的處理進度")
async def get_cached_value(
    key: str,
    m_type: Literal["document", "video", "image", "audio"],
    redis_client: Redis = Depends(get_redis_client),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    查詢指定檔案的處理進度（來自 Redis 快取）。

    - `m_type`：媒體類型，固定為 `document` / `video` / `image` / `audio` 之一
    - `key`：上傳時回傳的 `correlation_id`（即處理任務 ID）

    回傳該任務目前的處理狀態與結果（JSON 格式）；若任務尚未存在或已過期則回傳 404。
    """
    # 使用 combine_key_ 函數組合完整的 key
    full_key = combine_key_(key, m_type)

    value = await redis_client.get(full_key)
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{full_key}' not found in cache"
        )

    # 嘗試將 value 解析為 JSON，如果解析成功則返回解析後的對象
    try:
        parsed_value = json.loads(value)
        return {
            "key": full_key,
            "original_key": key,
            "m_type": m_type,
            "value": parsed_value  # 返回解析後的 JSON 對象
        }
    except json.JSONDecodeError:
        # 如果不是有效的 JSON，返回原始字串
        return {
            "key": full_key,
            "original_key": key,
            "m_type": m_type,
            "value": value
        }


_ORCHESTRATOR_PATTERNS = [
    "audio_orchestrator:*",
    "video_orchestrator:*",
    "image_orchestrator:*",
    "document_orchestrator:*",
]


@router.get("/cache/all", tags=["Processing"], status_code=status.HTTP_200_OK,
            summary="查詢目前使用者所有檔案的處理進度")
async def get_all_redis_cache(
    redis_client: Redis = Depends(get_redis_client),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    查詢目前登入使用者所有上傳任務的處理進度。

    只掃描 `audio / video / image / document` 四種 orchestrator 快取 key，
    並過濾出 `branch_id` 與當前使用者相符的條目，不會回傳其他使用者的資料。
    """
    user_id = current_user["sub"]
    cache_data = {}

    try:
        for pattern in _ORCHESTRATOR_PATTERNS:
            keys = await redis_client.keys(pattern)
            for key in keys:
                try:
                    raw = await redis_client.get(key)
                    if raw is None:
                        continue
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if parsed.get("branch_id") == user_id:
                        cache_data[key] = parsed
                except RedisError as e:
                    logger.debug(f"Skipping cache key {key}: {e}")
                    continue

        return {"count": len(cache_data), "data": cache_data}

    except RedisError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve processing cache: {e!s}",
        )
