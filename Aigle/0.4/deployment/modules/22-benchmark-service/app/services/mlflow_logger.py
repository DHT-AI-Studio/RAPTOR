"""Best-effort MLflow run-history logging (MLflow Run History).

After a benchmark run completes, its aggregate + per-dimension scores are
logged to the Module 07 MLflow tracking server (``BM_MLFLOW_URL``) so score
improvement curves per schema are visible in the MLflow UI — the Raptor
equivalent of AutoResearch's nightly experiment trend.

Logging is strictly best effort: any MLflow failure (server down, package
missing, bad payload) is a warning, never a benchmark failure.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from app.core.config import get_settings

try:  # mlflow is optional — its absence must not break the service
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


async def log_run(run: Dict[str, Any]) -> Optional[str]:
    """Log a completed run to MLflow; return the MLflow run id, or None.

    ``run`` carries the fields collected by run_manager at completion time:
    run_id, schema_id, schema_name, schema_version, target_pipeline,
    test_case_count, aggregate_score, scores_per_dimension.
    """
    if mlflow is None:
        logger.warning("mlflow package not installed — skipping run-history logging")
        return None
    try:
        # The MLflow SDK is synchronous; keep the event loop free.
        return await asyncio.to_thread(_log_run_sync, run)
    except Exception as exc:  # noqa: BLE001 — best effort, never fail the run
        logger.warning("MLflow logging failed for run %s: %s", run.get("run_id"), exc)
        return None


def _log_run_sync(run: Dict[str, Any]) -> str:
    settings = get_settings()
    # Fail fast when the tracking server is unreachable instead of the SDK's
    # multi-minute default retry/backoff (the caller treats failure as a warning).
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(settings.mlflow_timeout))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

    mlflow.set_tracking_uri(settings.mlflow_url)
    mlflow.set_experiment(f"benchmark_{run['schema_name']}")
    with mlflow.start_run(run_name=str(run["run_id"])) as active:
        mlflow.set_tags({
            "schema_id": str(run.get("schema_id")),
            "schema_version": str(run.get("schema_version")),
            "target_pipeline": str(run.get("target_pipeline")),
            "test_case_count": str(run.get("test_case_count")),
            "benchmark_run_id": str(run["run_id"]),
        })
        mlflow.log_metric("aggregate_score", float(run.get("aggregate_score") or 0.0))
        for dim, score in (run.get("scores_per_dimension") or {}).items():
            mlflow.log_metric(f"score_{dim}", float(score))
        return active.info.run_id
