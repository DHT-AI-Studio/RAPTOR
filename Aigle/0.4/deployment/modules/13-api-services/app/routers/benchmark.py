"""
Proxy router that forwards benchmark service API requests
to remote service: raptor-benchmark-service:8000 (module 22)
"""

from enum import Enum
from typing import Optional, Dict, Any, List

import httpx
from fastapi import APIRouter, Depends, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user
from app.core.config import get_settings

router = APIRouter(tags=["Benchmark Proxy"])


# --- Models (mirror module 22's app/models/*.py, for Swagger typing only) ---

class ScoringDimension(BaseModel):
    name: str = Field(..., description="Dimension name, e.g. 'relevance'")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in [0, 1]; all weights sum to 1.0")
    method: str = Field(..., description="Registered scoring method name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")
    rubric: Optional[str] = Field(None, description="(legacy) -> params.rubric")
    max_ms: Optional[float] = Field(None, description="(legacy) -> params.max_ms")
    pattern: Optional[str] = Field(None, description="(legacy) -> params.pattern")


class ScoringSchema(BaseModel):
    dimensions: List[ScoringDimension] = Field(..., min_length=1)
    aggregate: str = Field("weighted_sum", description="Aggregation strategy")
    score_range: List[float] = Field(default=[1, 5], description="Raw score range for llm_judge normalization")


class TestCase(BaseModel):
    id: str = Field(..., description="Unique test-case id within the schema")
    input: Dict[str, Any] = Field(..., description="Pipeline input payload")
    expected_keywords: Optional[List[str]] = Field(None, description="Keywords expected in output")
    expected_answer: Optional[str] = Field(None, description="Reference answer for similarity scoring")


class TargetPipeline(str, Enum):
    chat = "chat"
    search = "search"
    rag = "rag"
    classify = "classify"
    local_infer = "local_infer"
    lifecycle_infer = "lifecycle_infer"


class BenchmarkSchema(BaseModel):
    name: str = Field(..., description="Human-readable schema name")
    version: str = Field("1.0", description="Schema version")
    target_pipeline: TargetPipeline = Field(..., description="Pipeline under test")
    target_url: Optional[str] = Field(None, description="Optional URL override for the pipeline")
    test_cases: List[TestCase] = Field(..., min_length=1)
    scoring_schema: ScoringSchema
    judge_model: Optional[str] = Field(None, description="LLM-as-judge model override")


class RunSubmitRequest(BaseModel):
    schema_id: str = Field(..., description="Id of a previously uploaded schema")
    config_override: Optional[Dict[str, Any]] = Field(
        None, description="Optional pipeline configuration override recorded with the run"
    )


class DimensionDelta(BaseModel):
    name: str
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    delta: Optional[float] = None


class CaseDelta(BaseModel):
    case_id: str
    aggregate_a: Optional[float] = None
    aggregate_b: Optional[float] = None
    delta: Optional[float] = None


class CompareResult(BaseModel):
    delta_aggregate: float
    dimensions: List[DimensionDelta] = Field(default_factory=list)
    cases: List[CaseDelta] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pairwise: Optional[Dict[str, Any]] = None


class Budget(BaseModel):
    max_experiments: int = Field(10, ge=1, le=200, description="Max training+eval iterations")
    minutes_per_experiment: float = Field(
        30.0, gt=0, le=720, description="Per-iteration training timeout (job is cancelled past this)"
    )
    early_stop_patience: int = Field(
        3, ge=1, le=100, description="Stop after this many consecutive non-improving iterations"
    )


class SearchDimension(BaseModel):
    type: str = Field(..., description="float | int | categorical")
    min: Optional[float] = None
    max: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None


class Plan(BaseModel):
    task_type: str = Field("instruction", description="instruction | text")
    select_multiple_gpus: bool = False
    vram_budget_gb: float = Field(12.0, gt=0)
    base_training_config: Dict[str, Any] = Field(..., description="Fixed TrainerConfig fields")
    dataset_config: Dict[str, Any] = Field(..., description="Module 16 DatasetConfig")
    search_space: Dict[str, SearchDimension] = Field(..., description="Tunable knobs and their ranges")
    eval_schema_id: Optional[str] = Field(None, description="Benchmark schema used to score every candidate")
    holdout_schema_id: Optional[str] = Field(None, description="Independent held-out schema")
    budget: Optional[Budget] = Field(None, description="Budget the planner derived from the goal")
    needs_download: List[Dict[str, Any]] = Field(
        default_factory=list, description="Resources the plan references that are NOT present locally"
    )


class OptimizeRequest(BaseModel):
    goal: str = Field(..., description="Natural-language optimization goal")
    budget: Optional[Budget] = Field(None, description="Explicit hard budget; wins over a planner-derived one")
    plan: Optional[Plan] = Field(None, description="Explicit plan (required until the planner is implemented)")


# --- Core Proxy Function ---

async def forward_request(request: Request, path: str, base: Optional[str] = None,
                          json_override: Optional[Dict[str, Any]] = None):
    """Proxy `request` to Module 22, byte-for-byte by default.

    `json_override` replaces the forwarded body with this JSON payload instead
    of passing `request.body()` through untouched -- used by submit_run_proxy
    to inject the caller's branch_id (see its docstring) without every other
    route needing to know about it.
    """
    settings = get_settings()
    url = f"{base or settings.benchmark_service_url}{path}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                key: value for key, value in request.headers.items()
                if key.lower() != "host"
            }
            if json_override is not None:
                # The replacement body has a different length/type than the
                # original request -- drop the stale Content-Length/Type so
                # httpx computes fresh ones for the JSON it's actually sending.
                headers = {k: v for k, v in headers.items()
                          if k.lower() not in ("content-length", "content-type")}
                response = await client.request(
                    method=request.method, url=url, headers=headers,
                    params=request.query_params, json=json_override,
                )
            else:
                body = await request.body()
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
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
            detail=f"Failed to connect to benchmark service: {str(e)}"
        )


# --- Proxy Endpoints: Schemas (module 22 app/routers/schemas.py) ---

@router.post(
    "/schemas",
    summary="Upload a marking schema (JSON body)",
    description="Upload a user-defined marking schema (test cases + scoring rubric)."
)
async def upload_schema_proxy(
    payload: BenchmarkSchema = Body(...),
    request: Request = None
):
    """Upload a marking schema (Proxy)."""
    return await forward_request(request, "/benchmark/schemas")


@router.post(
    "/schemas/upload",
    summary="Upload a marking schema from a YAML or JSON file",
    description="Multipart file upload — the raw request body/headers are forwarded as-is, "
                "so FastAPI's own multipart parsing is intentionally not used here."
)
async def upload_schema_file_proxy(request: Request):
    return await forward_request(request, "/benchmark/schemas/upload")


@router.get(
    "/schemas",
    summary="List marking schemas (paginated)"
)
async def list_schemas_proxy(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    request: Request = None
):
    return await forward_request(request, "/benchmark/schemas")


@router.get(
    "/schemas/{schema_id}",
    summary="Get full schema definition"
)
async def get_schema_proxy(schema_id: str, request: Request):
    return await forward_request(request, f"/benchmark/schemas/{schema_id}")


@router.delete(
    "/schemas/{schema_id}",
    summary="Delete a schema"
)
async def delete_schema_proxy(schema_id: str, request: Request):
    return await forward_request(request, f"/benchmark/schemas/{schema_id}")


# --- Proxy Endpoints: Runs (module 22 app/routers/runs.py) ---

@router.post(
    "/runs",
    summary="Start a benchmark run (async)"
)
async def submit_run_proxy(
    payload: RunSubmitRequest = Body(...),
    request: Request = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Submit a new benchmark run (Proxy).

    Injects the caller's own sub as branch_id -- same pattern asset.py uses to
    bake user_id/branch_id into an upload's Kafka job at submission time
    (`User(user_id=sub, branch_id=sub)`), not something Module 22 re-derives
    later. Used as the default X-Branch-ID for target_pipeline=search test
    cases that don't set their own input.branch_id/input.user_id.
    """
    body = payload.model_dump()
    body["branch_id"] = current_user["sub"]
    return await forward_request(request, "/benchmark/runs", json_override=body)


@router.get(
    "/runs/{run_id}",
    summary="Get run status + scores"
)
async def get_run_proxy(run_id: str, request: Request):
    return await forward_request(request, f"/benchmark/runs/{run_id}")


@router.get(
    "/schemas/{schema_id}/runs",
    summary="Run history for a schema"
)
async def list_runs_proxy(
    schema_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    request: Request = None
):
    return await forward_request(request, f"/benchmark/schemas/{schema_id}/runs")


@router.get(
    "/schemas/{schema_id}/leaderboard",
    summary="Top-N runs by aggregate score, with their config_override"
)
async def leaderboard_proxy(
    schema_id: str,
    limit: int = Query(5, ge=1, le=100),
    request: Request = None
):
    return await forward_request(request, f"/benchmark/schemas/{schema_id}/leaderboard")


@router.get(
    "/runs/{run_id_a}/compare/{run_id_b}",
    response_model=CompareResult,
    summary="Delta report between two completed runs"
)
async def compare_runs_proxy(
    run_id_a: str,
    run_id_b: str,
    pairwise: bool = Query(False, description="Also run per-case head-to-head LLM comparison"),
    request: Request = None
):
    return await forward_request(request, f"/benchmark/runs/{run_id_a}/compare/{run_id_b}")


# --- Proxy Endpoints: Auto-Tune (module 22 app/routers/optimize.py) ---

@router.post(
    "/optimize",
    summary="Create an auto-tuning experiment from a goal + plan"
)
async def create_optimize_proxy(
    payload: OptimizeRequest = Body(...),
    auto_confirm: bool = Query(False, description="Skip the confirm gate and start immediately"),
    request: Request = None
):
    return await forward_request(request, "/optimize")


@router.get(
    "/optimize",
    summary="List experiments (newest first)"
)
async def list_optimize_proxy(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    request: Request = None
):
    return await forward_request(request, "/optimize")


@router.get(
    "/optimize/{experiment_id}",
    summary="Experiment status, best result, and history"
)
async def get_optimize_proxy(experiment_id: str, request: Request):
    return await forward_request(request, f"/optimize/{experiment_id}")


@router.get(
    "/optimize/{experiment_id}/plan",
    summary="Plan preview + eval preview (customer confirms before GPU burns)"
)
async def get_optimize_plan_proxy(experiment_id: str, request: Request):
    return await forward_request(request, f"/optimize/{experiment_id}/plan")


@router.post(
    "/optimize/{experiment_id}/confirm",
    summary="Confirm the plan and start the optimization loop"
)
async def confirm_optimize_proxy(experiment_id: str, request: Request):
    return await forward_request(request, f"/optimize/{experiment_id}/confirm")


@router.post(
    "/optimize/{experiment_id}/stop",
    summary="Stop the experiment after the current iteration"
)
async def stop_optimize_proxy(experiment_id: str, request: Request):
    return await forward_request(request, f"/optimize/{experiment_id}/stop")


@router.delete(
    "/optimize/{experiment_id}",
    summary="Delete an experiment and its auto-generated dev/held-out schemas"
)
async def delete_optimize_proxy(experiment_id: str, request: Request):
    return await forward_request(request, f"/optimize/{experiment_id}")


# --- Proxy Endpoint: Health ---

@router.get(
    "/health",
    summary="Benchmark service health check"
)
async def health_proxy(request: Request):
    # Unlike every other route, module 22 mounts /health at the service root,
    # not under /api/v1 — benchmark_service_url has /api/v1 baked in (see
    # config.py), so strip it back off just for this one call.
    settings = get_settings()
    root = settings.benchmark_service_url.rsplit("/api/v1", 1)[0]
    return await forward_request(request, "/health", base=root)
