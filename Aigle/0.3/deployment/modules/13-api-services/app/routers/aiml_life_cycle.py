"""
Proxy router for Model & Dataset APIs
Forward requests to: http://192.168.157.167:8010
"""

from enum import Enum
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter(tags=["Model & Dataset Proxy"])


# --- Request Models ---

class DownloadModelRequest(BaseModel):
    model_config = {'protected_namespaces': ()}

    model_source: str = Field(..., example="huggingface")
    model_name: str = Field(..., example="google/gemma-3-270m-it")


class DownloadDatasetRequest(BaseModel):
    dataset_name: str = Field(..., example="cifar10")
    dataset_source: str = Field(..., example="huggingface", description="數據集來源，例如 huggingface")
    extract_multimedia: bool = Field(True, example=True, description="是否提取多媒體文件")
    max_samples: int = Field(1000, example=1000, description="最大下載樣本數量")
    split: str = Field("train", example="train", description="數據集分割，例如 'train', 'test', 'validation'")


class ModelSourceEnum(str, Enum):
    huggingface = "huggingface"
    ollama = "ollama"


class LocalModelSourceEnum(str, Enum):
    huggingface = "huggingface"
    ollama = "ollama"


# --- Core Proxy Function ---

async def forward_request(request: Request, path: str):
    settings = get_settings()
    url = f"{settings.aiml_lifecycle_api_url}{path}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            body = await request.body()
            response = await client.request(
                method=request.method,
                url=url,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() != "host"
                },
                params=request.query_params,
                content=body
            )

        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if response.content else None
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to backend service: {str(e)}"
        )


# --- Proxy Endpoints ---

@router.post(
    "/models/download",
    summary="從來源下載模型",
    description="""
從指定的來源下載模型到本地暫存區

使用範例：
{
  "model_source": "huggingface",
  "model_name": "google/gemma-3-270m-it"
}
"""
)
async def download_model_proxy(
    model_source: ModelSourceEnum = Query(
        None,
        description="模型來源",
        example=None
    ),
    model_name: str = Query(
        ...,
        description="模型名稱，例如 'google/gemma-3-270m-it' 或 'llama3:latest'",
    ),
    request: Request = None
):
    return await forward_request(request, "/models/download")


@router.post(
    "/datasets/download_from_network",
    summary="從網路下載數據集",
    description="""
從 HuggingFace Hub 等網路來源下載數據集到本地，支援多媒體數據集。

可下載：
- 文本數據集：JSON, CSV, Parquet
- 圖像數據集：PNG/JPG
- 音頻數據集：WAV/NPY
- 視頻數據集：MP4
- 多模態數據集：同時處理多種媒體類型
"""
)
async def download_dataset_proxy(
    payload: DownloadDatasetRequest = Body(
        ...,
        example={
            "dataset_name": "cifar10",
            "dataset_source": "huggingface",
            "extract_multimedia": True,
            "max_samples": 1000,
            "split": "train"
        }
    ),
    request: Request = None
):
    return await forward_request(request, "/datasets/download_from_network")


@router.get(
    "/models/local",
    summary="列出本地模型",
    description="""
列出本地暫存區中已下載的模型。

查詢參數：
- model_source (可選): 指定模型來源類型
  - "huggingface": 列出 HuggingFace 模型
  - "ollama": 列出 Ollama 模型
  - 未指定：列出所有本地模型
"""
)
async def list_models_proxy(
    model_source: LocalModelSourceEnum = Query(
        default=None,
        description="指定模型來源類型",
    ),
    request: Request = None
):
    return await forward_request(request, "/models/local")


@router.get(
    "/datasets/local",
    summary="List local datasets"
)
async def list_datasets_proxy(request: Request):
    return await forward_request(request, "/datasets/local")
