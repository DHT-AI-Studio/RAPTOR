"""Run lifecycle + compare REST API (BM-7 / BM-8)."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.models.run import CompareResult, RunSubmitRequest
from app.services import run_manager
from app.services.run_manager import CompareError

router = APIRouter(prefix="/benchmark", tags=["Runs"])


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED,
             summary="Start a benchmark run (async)")
async def submit_run(payload: RunSubmitRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    try:
        result = await run_manager.create_run(payload.schema_id, payload.config_override,
                                              branch_id=payload.branch_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema not found")
    background_tasks.add_task(run_manager.execute_run, result["run_id"])
    return result


@router.get("/runs/{run_id}", summary="Get run status + scores")
async def get_run(run_id: str) -> Dict[str, Any]:
    record = await run_manager.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return record


@router.get("/schemas/{schema_id}/runs", summary="Run history for a schema")
async def list_runs(
    schema_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    return await run_manager.list_runs_for_schema(schema_id, limit=limit, offset=offset)


@router.get("/schemas/{schema_id}/leaderboard",
            summary="Top-N runs by aggregate score, with their config_override")
async def leaderboard(
    schema_id: str,
    limit: int = Query(5, ge=1, le=100),
) -> List[Dict[str, Any]]:
    return await run_manager.leaderboard(schema_id, limit=limit)


@router.get("/runs/{run_id_a}/compare/{run_id_b}", response_model=CompareResult,
            summary="Delta report between two completed runs")
async def compare_runs(
    run_id_a: str,
    run_id_b: str,
    pairwise: bool = Query(False, description="Also run per-case head-to-head LLM comparison"),
) -> CompareResult:
    try:
        return await run_manager.compare_runs(run_id_a, run_id_b, pairwise=pairwise)
    except CompareError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
