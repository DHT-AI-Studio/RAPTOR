"""Async run manager (BM-7) + run comparison (BM-8).

Owns the benchmark run state machine (queued → running → completed/failed),
persisting live progress to Redis and final results to PostgreSQL.

JSONB data contract written to ``benchmark_runs`` (shared with BM-8 compare):
  scores_per_dimension : {dimension_name: mean_score_across_cases}
  scores_per_case      : [{case_id, output, latency_ms, per_dimension, aggregate, error}]
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.db import db
from app.models.run import CaseDelta, CompareResult, DimensionDelta
from app.models.schema import BenchmarkSchema
from app.services import executor, judge, mlflow_logger, scorer

logger = logging.getLogger(__name__)


def _redis_key(run_id: str) -> str:
    return f"benchmark:run:{run_id}"


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


async def _write_redis(run_id: str, data: Dict[str, Any]) -> None:
    settings = get_settings()
    await db.redis.set(_redis_key(run_id), json.dumps(data), ex=settings.redis_run_ttl_seconds)


async def _write_progress(run_id: str, status: str, progress_pct: float,
                          schema_id: Optional[str] = None, error: Optional[str] = None) -> None:
    """Persist the live run-state snapshot (read back by get_run for queued/running)."""
    await _write_redis(run_id, {"run_id": run_id, "schema_id": schema_id,
                                "status": status, "progress_pct": progress_pct, "error": error})


async def create_run(schema_id: str, config_override: Optional[Dict[str, Any]] = None,
                     branch_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate the schema, create a queued run in PostgreSQL + Redis."""
    if not _is_uuid(schema_id):
        raise ValueError("schema not found")
    exists = await db.pool.fetchrow("SELECT 1 FROM benchmark_schemas WHERE id = $1", schema_id)
    if exists is None:
        raise ValueError("schema not found")

    row = await db.pool.fetchrow(
        """
        INSERT INTO benchmark_runs (schema_id, status, config_override, branch_id)
        VALUES ($1, 'queued', $2, $3)
        RETURNING id, created_at
        """,
        schema_id,
        json.dumps(config_override) if config_override is not None else None,
        branch_id,
    )
    run_id = str(row["id"])
    await _write_progress(run_id, "queued", 0.0, schema_id)
    return {"run_id": run_id, "status": "queued"}


async def _load_schema(schema_id: str) -> BenchmarkSchema:
    row = await db.pool.fetchrow(
        "SELECT definition FROM benchmark_schemas WHERE id = $1", schema_id
    )
    definition = row["definition"]
    if isinstance(definition, str):
        definition = json.loads(definition)
    return BenchmarkSchema.model_validate(definition)


async def execute_run(run_id: str) -> None:
    """Background state machine: run every test case, score, persist results."""
    # Tracked in the outer scope so the finally can release the served model's
    # VRAM whether the run succeeds OR fails.
    pipeline: Optional[str] = None
    target_url: Optional[str] = None
    config_override: Dict[str, Any] = {}
    try:
        run_row = await db.pool.fetchrow(
            "SELECT schema_id, config_override, branch_id FROM benchmark_runs WHERE id = $1", run_id
        )
        if run_row is None:
            logger.error("execute_run: run %s not found", run_id)
            return
        schema_id = str(run_row["schema_id"])
        config_override = _maybe_json(run_row["config_override"]) or {}
        run_branch_id = run_row["branch_id"]
        schema = await _load_schema(schema_id)

        started = datetime.now(timezone.utc)
        await db.pool.execute(
            "UPDATE benchmark_runs SET status = 'running', started_at = $2 WHERE id = $1",
            run_id, started,
        )
        await _write_progress(run_id, "running", 0.0, schema_id)

        pipeline = schema.target_pipeline.value
        target_url = schema.target_url
        judge_model = schema.judge_model
        total = len(schema.test_cases)
        case_results: List[Dict[str, Any]] = []

        # Fast path: score a local_infer schema with ONE batched inference call
        # (all cases at once) instead of one round-trip per case. Falls back to
        # per-case if the batch fails or returns the wrong number of outputs.
        batch_outputs: Optional[List[str]] = None
        if pipeline == "local_infer" and get_settings().local_infer_batch:
            br = await executor.infer_batch_local(
                [tc.input for tc in schema.test_cases], target_url, config_override)
            outs = br.get("outputs") or []
            if len(outs) == total:
                batch_outputs = outs
            elif outs or br.get("error"):
                logger.warning("batch local_infer gave %d/%d outputs — per-case fallback",
                               len(outs), total)

        for idx, tc in enumerate(schema.test_cases, start=1):
            if batch_outputs is not None:
                output, latency_ms, case_error = batch_outputs[idx - 1], 0.0, None
            else:
                exec_result = await executor.call_pipeline(pipeline, tc.input, target_url,
                                                           config_override=config_override,
                                                           run_branch_id=run_branch_id)
                output = exec_result.get("output", "")
                latency_ms = exec_result.get("latency_ms", 0.0)
                case_error = exec_result.get("error")
            scored = await scorer.score(tc, output, latency_ms, schema.scoring_schema, judge_model)
            case_results.append({
                "case_id": tc.id,
                "output": output,
                "latency_ms": latency_ms,
                "per_dimension": scored["per_dimension"],
                "aggregate": scored["aggregate"],
                "error": case_error,
            })
            await _write_progress(run_id, "running", round(idx / total * 100, 1), schema_id)

        scores_per_dimension = _mean_per_dimension(schema, case_results)
        run_aggregate = mean(c["aggregate"] for c in case_results) if case_results else 0.0
        completed = datetime.now(timezone.utc)

        await db.pool.execute(
            """
            UPDATE benchmark_runs
            SET status = 'completed', aggregate_score = $2,
                scores_per_dimension = $3, scores_per_case = $4, completed_at = $5
            WHERE id = $1
            """,
            run_id, run_aggregate,
            json.dumps(scores_per_dimension), json.dumps(case_results), completed,
        )
        await _write_progress(run_id, "completed", 100.0, schema_id)
        logger.info("Run %s completed: aggregate=%.4f", run_id, run_aggregate)

        # MLflow run history — after the run is already persisted as completed,
        # so a slow/unreachable tracking server never delays or fails the run.
        mlflow_run_id = await mlflow_logger.log_run({
            "run_id": run_id,
            "schema_id": schema_id,
            "schema_name": schema.name,
            "schema_version": schema.version,
            "target_pipeline": pipeline,
            "test_case_count": total,
            "aggregate_score": run_aggregate,
            "scores_per_dimension": scores_per_dimension,
        })
        if mlflow_run_id:
            try:
                await db.pool.execute(
                    "UPDATE benchmark_runs SET mlflow_run_id = $2 WHERE id = $1",
                    run_id, mlflow_run_id,
                )
            except Exception:  # noqa: BLE001 — the run itself already completed
                logger.warning("Run %s: could not persist mlflow_run_id %s",
                               run_id, mlflow_run_id)

    except Exception as exc:  # noqa: BLE001 — a failed run must never crash the worker
        logger.exception("Run %s failed", run_id)
        try:
            await db.pool.execute(
                "UPDATE benchmark_runs SET status = 'failed', scores_per_case = $2 WHERE id = $1",
                run_id, json.dumps({"error": str(exc)}),
            )
            await _write_progress(run_id, "failed", 0.0, error=str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("Run %s: failed to persist failure state", run_id)

    finally:
        # Always release the served model's VRAM — on success AND failure — so a
        # local_infer checkpoint never lingers on the GPU between iterations.
        # (Best-effort: unload_local_infer never raises.)
        if pipeline == "local_infer" and get_settings().unload_after_run:
            await executor.unload_local_infer(target_url, config_override.get("model_path"))


def _mean_per_dimension(schema: BenchmarkSchema, case_results: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for dim in schema.scoring_schema.dimensions:
        vals = [c["per_dimension"].get(dim.name, 0.0) for c in case_results]
        out[dim.name] = mean(vals) if vals else 0.0
    return out


async def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a run from PostgreSQL (if persisted), else Redis live state."""
    if not _is_uuid(run_id):
        return None
    row = await db.pool.fetchrow(
        """
        SELECT id, schema_id, status, aggregate_score, scores_per_dimension,
               scores_per_case, config_override, started_at, completed_at, created_at,
               mlflow_run_id
        FROM benchmark_runs WHERE id = $1
        """,
        run_id,
    )
    if row is None:
        return None

    record = {
        "run_id": str(row["id"]),
        "schema_id": str(row["schema_id"]),
        "status": row["status"],
        "aggregate_score": row["aggregate_score"],
        "scores_per_dimension": _maybe_json(row["scores_per_dimension"]),
        "scores_per_case": _maybe_json(row["scores_per_case"]),
        "config_override": _maybe_json(row["config_override"]),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "created_at": row["created_at"],
        "mlflow_run_id": row["mlflow_run_id"],
        "progress_pct": 100.0 if row["status"] == "completed" else 0.0,
    }

    # For queued/running rows, overlay live progress from Redis.
    if row["status"] in ("queued", "running"):
        live = await db.redis.get(_redis_key(run_id))
        if live:
            live_data = json.loads(live)
            record["progress_pct"] = live_data.get("progress_pct", record["progress_pct"])
            record["status"] = live_data.get("status", record["status"])
            record["error"] = live_data.get("error")
    return record


async def list_runs_for_schema(schema_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    if not _is_uuid(schema_id):
        return []
    rows = await db.pool.fetch(
        """
        SELECT id, status, aggregate_score, created_at, mlflow_run_id
        FROM benchmark_runs
        WHERE schema_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        """,
        schema_id, limit, offset,
    )
    return [
        {
            "run_id": str(r["id"]),
            "status": r["status"],
            "aggregate_score": r["aggregate_score"],
            "created_at": r["created_at"],
            "mlflow_run_id": r["mlflow_run_id"],
        }
        for r in rows
    ]


async def leaderboard(schema_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Top-N completed runs for a schema by aggregate_score, with their config.

    This is the history an optimizer/agent reads to see the best configurations
    tried so far and propose the next one.
    """
    if not _is_uuid(schema_id):
        return []
    rows = await db.pool.fetch(
        """
        SELECT id, aggregate_score, config_override, scores_per_dimension, created_at
        FROM benchmark_runs
        WHERE schema_id = $1 AND status = 'completed' AND aggregate_score IS NOT NULL
        ORDER BY aggregate_score DESC, created_at DESC
        LIMIT $2
        """,
        schema_id, limit,
    )
    return [
        {
            "rank": i,
            "run_id": str(r["id"]),
            "aggregate_score": r["aggregate_score"],
            "config_override": _maybe_json(r["config_override"]),
            "scores_per_dimension": _maybe_json(r["scores_per_dimension"]),
            "created_at": r["created_at"],
        }
        for i, r in enumerate(rows, start=1)
    ]


# ── BM-8 compare ─────────────────────────────────────────────────────

class CompareError(Exception):
    """Raised by compare_runs with an HTTP-ish status hint."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def _fetch_full_run(run_id: str):
    if not _is_uuid(run_id):
        return None
    return await db.pool.fetchrow(
        """
        SELECT id, schema_id, status, aggregate_score, scores_per_dimension,
               scores_per_case, created_at
        FROM benchmark_runs WHERE id = $1
        """,
        run_id,
    )


async def compare_runs(run_id_a: str, run_id_b: str, pairwise: bool = False) -> CompareResult:
    row_a = await _fetch_full_run(run_id_a)
    row_b = await _fetch_full_run(run_id_b)
    if row_a is None:
        raise CompareError(404, f"Run {run_id_a} not found")
    if row_b is None:
        raise CompareError(404, f"Run {run_id_b} not found")
    if str(row_a["schema_id"]) != str(row_b["schema_id"]):
        raise CompareError(422, "Cannot compare runs from different schemas")
    for row in (row_a, row_b):
        if row["status"] != "completed":
            raise CompareError(409, f"Run {row['id']} is not completed")

    dims_a = _maybe_json(row_a["scores_per_dimension"]) or {}
    dims_b = _maybe_json(row_b["scores_per_dimension"]) or {}
    dimensions = [
        DimensionDelta(name=name, score_a=dims_a.get(name), score_b=dims_b.get(name),
                       delta=_delta(dims_b.get(name), dims_a.get(name)))
        for name in sorted(set(dims_a) | set(dims_b))
    ]

    cases_a = _case_map(_maybe_json(row_a["scores_per_case"]))
    cases_b = _case_map(_maybe_json(row_b["scores_per_case"]))
    cases = [
        CaseDelta(case_id=cid, aggregate_a=cases_a.get(cid), aggregate_b=cases_b.get(cid),
                  delta=_delta(cases_b.get(cid), cases_a.get(cid)))
        for cid in sorted(set(cases_a) | set(cases_b))
    ]

    schema_row = await db.pool.fetchrow(
        "SELECT name FROM benchmark_schemas WHERE id = $1", row_a["schema_id"]
    )
    delta_aggregate = _delta(row_b["aggregate_score"], row_a["aggregate_score"]) or 0.0

    pairwise_summary = None
    if pairwise:
        pairwise_summary = await _pairwise_summary(row_a, row_b, str(row_a["schema_id"]))

    return CompareResult(
        delta_aggregate=delta_aggregate,
        dimensions=dimensions,
        cases=cases,
        pairwise=pairwise_summary,
        metadata={
            "schema_id": str(row_a["schema_id"]),
            "schema_name": schema_row["name"] if schema_row else None,
            "run_a": str(row_a["id"]),
            "run_b": str(row_b["id"]),
            "run_a_created_at": row_a["created_at"].isoformat() if row_a["created_at"] else None,
            "run_b_created_at": row_b["created_at"].isoformat() if row_b["created_at"] else None,
        },
    )


# ── Pairwise (LLM head-to-head) compare ──────────────────────────────

async def _pairwise_summary(row_a: Any, row_b: Any, schema_id: str) -> Dict[str, Any]:
    """Head-to-head LLM verdict per shared test case, plus a win-rate summary.

    Uses the outputs stored in each run's scores_per_case, so no pipeline is
    re-run. Position bias is removed by asking twice with A/B swapped and only
    counting a win when both orderings agree.
    """
    outputs_a = _output_map(_maybe_json(row_a["scores_per_case"]))
    outputs_b = _output_map(_maybe_json(row_b["scores_per_case"]))
    inputs = await _case_input_map(schema_id)

    case_verdicts: List[Dict[str, Any]] = []
    a_wins = b_wins = ties = 0
    for cid in sorted(set(outputs_a) & set(outputs_b)):
        verdict = await _debiased_pairwise(inputs.get(cid, ""), outputs_a[cid], outputs_b[cid])
        case_verdicts.append({"case_id": cid, "winner": verdict})
        if verdict == "A":
            a_wins += 1
        elif verdict == "B":
            b_wins += 1
        else:
            ties += 1

    total = a_wins + b_wins + ties
    return {
        "cases": case_verdicts,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "total": total,
        "b_win_rate": (b_wins / total) if total else 0.0,
    }


async def _debiased_pairwise(task_input: str, out_a: str, out_b: str) -> str:
    """Return 'A'/'B'/'TIE', guarding against LLM position bias."""
    v1 = await judge.pairwise(task_input, out_a, out_b)          # A=out_a, B=out_b
    v2 = await judge.pairwise(task_input, out_b, out_a)          # swapped
    v2_normalized = {"A": "B", "B": "A", "TIE": "TIE"}[v2]       # undo the swap
    return v1 if v1 == v2_normalized else "TIE"


async def _case_input_map(schema_id: str) -> Dict[str, str]:
    """Map case_id → a display string of the test-case input (for the judge)."""
    if not _is_uuid(schema_id):
        return {}
    row = await db.pool.fetchrow(
        "SELECT definition FROM benchmark_schemas WHERE id = $1", schema_id
    )
    if row is None:
        return {}
    definition = _maybe_json(row["definition"]) or {}
    result: Dict[str, str] = {}
    for tc in definition.get("test_cases", []):
        inp = tc.get("input", {}) or {}
        text = inp.get("message") or inp.get("question") or inp.get("query") or inp.get("inputs")
        if text is None:
            text = json.dumps(inp, ensure_ascii=False)
        result[tc.get("id")] = str(text)
    return result


def _delta(b: Optional[float], a: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def _case_map(cases: Any) -> Dict[str, float]:
    if not isinstance(cases, list):
        return {}
    return {c["case_id"]: c.get("aggregate") for c in cases if isinstance(c, dict) and "case_id" in c}


def _output_map(cases: Any) -> Dict[str, str]:
    if not isinstance(cases, list):
        return {}
    return {c["case_id"]: (c.get("output") or "")
            for c in cases if isinstance(c, dict) and "case_id" in c}


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value
