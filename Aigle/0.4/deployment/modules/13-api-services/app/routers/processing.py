"""
API router for triggering Kafka processing requests.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Literal

from aiokafka import AIOKafkaProducer  # type: ignore[import]
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis  # type: ignore[import]

from app.api.dependencies import get_current_user
from app.services.kafka_service import KafkaService


router = APIRouter()


class ProcessingRequest(BaseModel):
    upload_result: Dict[str, Any] = Field(..., description="The result from the file upload endpoint.")
    processing_mode: Optional[str] = Field(None, description="Optional processing mode (e.g., 'ocr' for PDFs).")


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

@router.post("/process-file", tags=["Processing"], status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
async def process_file(
    request: Request,
    proc_request: ProcessingRequest,
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Sends a request to a Kafka topic to process a previously uploaded file.

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/processing/process-file"
    payload = {"upload_result": upload_result}
    response = requests.post(url, json=payload, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    user_id = current_user["sub"]

    kafka_service = KafkaService(producer, request.app.state.settings)

    result = await kafka_service.send_processing_request(
        upload_result=proc_request.upload_result,
        user_id=user_id,
        branch_id=user_id,
        processing_mode=proc_request.processing_mode,
    )
    
    return result


@router.get("/cache/{m_type}/{key}", tags=["Processing"], status_code=status.HTTP_200_OK,
            summary="查詢單一檔案的處理進度")
async def get_cached_value(
    key: str,
    m_type: Literal["document", "video", "image", "audio"],
    redis_client: Redis = Depends(get_redis_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
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
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
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
                except Exception:
                    continue

        return {"count": len(cache_data), "data": cache_data}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve processing cache: {str(e)}",
        )
