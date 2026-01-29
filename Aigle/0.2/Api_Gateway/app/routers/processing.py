"""
API router for triggering Kafka processing requests.
"""
from __future__ import annotations

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

@router.post("/process-file", tags=["processing"], status_code=status.HTTP_202_ACCEPTED)
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
    
    # Extract the raw token for forwarding
    auth_header = request.headers.get("Authorization", "")
    _, _, access_token = auth_header.partition(" ")

    kafka_service = KafkaService(producer, request.app.state.settings)
    
    result = await kafka_service.send_processing_request(
        upload_result=proc_request.upload_result,
        user_id=user_id,
        access_token=access_token,
        processing_mode=proc_request.processing_mode,
    )
    
    return result


@router.get("/processing/cache/{m_type}/{key}", tags=["processing"], status_code=status.HTTP_200_OK)
async def get_cached_value(
    key: str,
    m_type: Literal["document", "video", "image", "audio"],
    redis_client: Redis = Depends(get_redis_client),
) -> Dict[str, Any]:
    """
    Retrieve a cached value from Redis by key and media type.
    
    **Parameters:**
    - `m_type`: Media type - one of: document, video, image, audio
    - `key`: The cache key to retrieve
    
    **Returns:**
    - The cached value associated with the combined key

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/processing/processing/cache/{m_type}/{key}"
    response = requests.get(url)
    response.raise_for_status()
    result = response.json()
    ```
    """
    # 使用 combine_key_ 函數組合完整的 key
    full_key = combine_key_(key, m_type)
    
    value = await redis_client.get(full_key)
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{full_key}' not found in cache"
        )
    return {"key": full_key, "original_key": key, "m_type": m_type, "value": value}


@router.get("/cache/all", tags=["processing"], status_code=status.HTTP_200_OK)
async def get_all_redis_cache(
    redis_client: Redis = Depends(get_redis_client),
) -> Dict[str, Any]:
    """
    Retrieve all key-value pairs from Redis cache.
    
    **Returns:**
    - A dictionary containing all keys and their corresponding values from Redis

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/processing/cache/all"
    response = requests.get(url)
    response.raise_for_status()
    result = response.json()
    ```
    """
    try:
        # 獲取所有的 keys
        all_keys = await redis_client.keys("*")
        
        if not all_keys:
            return {"count": 0, "data": {}}
        
        # 逐個獲取所有 keys 的值
        cache_data = {}
        for key in all_keys:
            try:
                value = await redis_client.get(key)
                cache_data[key] = value
            except Exception as e:
                # 如果某個 key 讀取失敗，記錄錯誤但繼續
                cache_data[key] = f"Error reading value: {str(e)}"
        
        return {
            "count": len(cache_data),
            "data": cache_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Redis cache: {str(e)}"
        )