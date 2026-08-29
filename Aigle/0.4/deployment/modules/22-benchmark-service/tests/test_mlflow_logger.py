"""Tests for best-effort MLflow run-history logging (MLflow Run History)."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.services import mlflow_logger


def _run_payload() -> dict:
    return {
        "run_id": "run-1",
        "schema_id": "schema-1",
        "schema_name": "RAG Quality v1",
        "schema_version": "1.0",
        "target_pipeline": "chat",
        "test_case_count": 3,
        "aggregate_score": 0.82,
        "scores_per_dimension": {"relevance": 0.9, "latency": 0.7},
    }


async def test_log_run_logs_metrics_and_tags(monkeypatch):
    fake = MagicMock()
    fake.start_run.return_value.__enter__.return_value.info.run_id = "mlrun-123"
    monkeypatch.setattr(mlflow_logger, "mlflow", fake)

    out = await mlflow_logger.log_run(_run_payload())

    assert out == "mlrun-123"
    fake.set_experiment.assert_called_once_with("benchmark_RAG Quality v1")
    fake.log_metric.assert_any_call("aggregate_score", 0.82)
    fake.log_metric.assert_any_call("score_relevance", 0.9)
    fake.log_metric.assert_any_call("score_latency", 0.7)

    tags = fake.set_tags.call_args[0][0]
    assert tags["schema_id"] == "schema-1"
    assert tags["schema_version"] == "1.0"
    assert tags["target_pipeline"] == "chat"
    assert tags["test_case_count"] == "3"


async def test_log_run_unreachable_server_is_best_effort(monkeypatch):
    fake = MagicMock()
    fake.set_experiment.side_effect = ConnectionError("mlflow down")
    monkeypatch.setattr(mlflow_logger, "mlflow", fake)

    # Must swallow the failure and report "not logged" instead of raising.
    assert await mlflow_logger.log_run(_run_payload()) is None


async def test_log_run_without_mlflow_package(monkeypatch):
    monkeypatch.setattr(mlflow_logger, "mlflow", None)
    assert await mlflow_logger.log_run(_run_payload()) is None
