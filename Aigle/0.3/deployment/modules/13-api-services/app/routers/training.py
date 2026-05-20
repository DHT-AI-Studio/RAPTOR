"""
Proxy router that forwards training API requests
to remote service: 192.168.157.163:8009
"""

from typing import Optional, Dict, Any, List

import httpx
from fastapi import APIRouter, Request, HTTPException, Body, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter(tags=["Training Proxy"])


# --- Models ---

class DatasetConfig(BaseModel):
    cache_dir: Optional[str] = "tmp/datasets/rajpurkar_squad_v2"
    column_mapping: Dict[str, str]
    dataset_name_or_path: str
    max_length: int = 2048
    train_size: int = 100
    train_split_name: str = "train"
    val_size: int = 50


class TrainingConfig(BaseModel):
    model_config = {'protected_namespaces': ()}
    experiment_name: str
    model_name_or_path: str
    batch_size: int = 1
    learning_rate: float = 2e-5
    max_epochs: int = 3
    gradient_accumulation_steps: int = 4
    logging_steps: int = 5
    warmup_steps: int = 100
    weight_decay: float = 0.01

    deepspeed_config: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    use_bfloat16: Optional[bool] = True
    use_flash_attn: Optional[bool] = False
    use_8bit_adamw: Optional[bool] = False

    adam_beta1: Optional[float] = 0.9
    adam_beta2: Optional[float] = 0.999
    adam_epsilon: Optional[float] = 1e-8

    val_check_interval: Optional[float] = 0.5


class TrainingJobSubmitRequest(BaseModel):
    dataset_config: DatasetConfig
    training_config: TrainingConfig

    task_type: str = Field(..., example="instruction")
    select_multiple_gpus: bool = False
    vram_budget_gb: int = 5


class JobStatusResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    job_id: str
    gpu_id: Optional[int | List[int]] = None
    status: str
    config: Optional[dict] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    metrics: Optional[dict] = None
    model_path: Optional[str] = None


# --- Core Proxy Function ---

async def forward_request(request: Request, path: str):
    settings = get_settings()
    url = f"{settings.training_service_url}{path}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            body = await request.body()
            response = await client.request(
                method=request.method,
                url=url,
                headers={
                    key: value for key, value in request.headers.items()
                    if key.lower() != "host"
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
            detail=f"Failed to connect to training service: {str(e)}"
        )


# --- Proxy Endpoints ---

@router.post(
    "/submit",
    response_model=JobStatusResponse,
    summary="Submit a new training job",
    description="Proxy endpoint for training job submission"
)
async def submit_proxy(
    payload: TrainingJobSubmitRequest = Body(
        ...,
        example={
            "dataset_config": {
                "cache_dir": "tmp/datasets/rajpurkar_squad_v2",
                "column_mapping": {
                    "context": "context",
                    "input": "instruction",
                    "messages": "messages",
                    "output": "output",
                    "text": "text"
                },
                "dataset_name_or_path": "rajpurkar/squad_v2",
                "max_length": 2048,
                "train_size": 100,
                "train_split_name": "train",
                "val_size": 50
            },
            "select_multiple_gpus": False,
            "task_type": "instruction",
            "training_config": {
                "experiment_name": "squad_finetune",
                "model_name_or_path": "tmp/models/google_gemma-3-270m-it",
                "batch_size": 1,
                "learning_rate": 2e-5,
                "max_epochs": 3,
                "gradient_accumulation_steps": 4,
                "logging_steps": 5,
                "warmup_steps": 100,
                "weight_decay": 0.01
            },
            "vram_budget_gb": 5
        }
    ),
    request: Request = None
):
    """Submit a new AI training job (Proxy)."""
    return await forward_request(request, "/submit")


@router.api_route("/status/{job_id}", methods=["GET"])
async def status_proxy(job_id: str, request: Request):
    return await forward_request(request, f"/status/{job_id}")


@router.get(
    "/list",
    response_model=List[JobStatusResponse],
    summary="List all training jobs",
    description="List all submitted training jobs, optionally filtered by status."
)
async def list_proxy(
    status: Optional[str] = Query(
        None,
        description="Filter by status (queued, running, completed, failed, cancelled)",
        example="running"
    ),
    request: Request = None
):
    """Proxy endpoint for listing training jobs."""
    return await forward_request(request, "/list")


@router.api_route("/cancel/{job_id}", methods=["POST"])
async def cancel_proxy(job_id: str, request: Request):
    return await forward_request(request, f"/cancel/{job_id}")


@router.delete(
    "/delete/{job_id}",
    summary="Delete job records",
    description="""
Delete job records from Redis and clean up associated files.

- If the job is still running or queued, you must set `force=true` to delete.
- This will remove records from Redis and delete logs and checkpoints from disk.

Warning: This operation cannot be undone.
"""
)
async def delete_proxy(
    job_id: str = Path(
        ...,
        description="Job ID to delete",
    ),
    force: bool = Query(
        False,
        description="Force cancel and delete if job is still running or queued",
        example=True
    ),
    request: Request = None
):
    """Proxy endpoint for deleting training jobs."""
    return await forward_request(request, f"/delete/{job_id}")
