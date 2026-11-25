"""Run-lifecycle Pydantic models (BM-2 / BM-7 / BM-8).

Covers run submission, live/completed run records, per-case and per-dimension
scores, and the compare-two-runs delta report.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class RunSubmitRequest(BaseModel):
    """POST /runs body."""

    schema_id: str = Field(..., description="Id of a previously uploaded schema")
    config_override: dict[str, Any] | None = Field(
        None, description="Optional pipeline configuration override recorded with the run"
    )
    branch_id: str | None = Field(
        None,
        description="Submitter's branch_id (their own JWT sub), injected by Module 13's "
                    "gateway proxy from the caller's Authorization header -- not meant to be "
                    "set by a client directly. Default X-Branch-ID for target_pipeline=search "
                    "test cases that don't set their own input.branch_id/input.user_id.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"schema_id": "0b3c...uuid", "config_override": {"temperature": 0.2}}
        }
    }


class DimensionScore(BaseModel):
    """A single dimension's normalized [0, 1] score for one test case."""

    name: str
    score: float = Field(..., ge=0.0, le=1.0)


class CaseResult(BaseModel):
    """Scoring outcome for a single test case within a run."""

    case_id: str
    output: str = ""
    latency_ms: float = 0.0
    per_dimension: dict[str, float] = Field(default_factory=dict)
    aggregate: float = 0.0
    error: str | None = None


class RunResult(BaseModel):
    """Aggregated scoring output produced by run_manager for a whole run."""

    aggregate_score: float = 0.0
    scores_per_dimension: dict[str, float] = Field(default_factory=dict)
    scores_per_case: list[CaseResult] = Field(default_factory=list)


class BenchmarkRun(BaseModel):
    """A benchmark run record (Redis while live, PostgreSQL once persisted)."""

    id: str
    schema_id: str
    status: RunStatus = RunStatus.queued
    progress_pct: float = 0.0
    aggregate_score: float | None = None
    scores_per_dimension: dict[str, float] | None = None
    scores_per_case: Any | None = None
    config_override: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    mlflow_run_id: str | None = None


class RunSummary(BaseModel):
    """Lightweight row for run-history listings."""

    run_id: str
    status: RunStatus
    aggregate_score: float | None = None
    created_at: datetime | None = None
    mlflow_run_id: str | None = None


# ── Compare (BM-8) ───────────────────────────────────────────────────

class DimensionDelta(BaseModel):
    name: str
    score_a: float | None = None
    score_b: float | None = None
    delta: float | None = None


class CaseDelta(BaseModel):
    case_id: str
    aggregate_a: float | None = None
    aggregate_b: float | None = None
    delta: float | None = None


class CompareResult(BaseModel):
    delta_aggregate: float
    dimensions: list[DimensionDelta] = Field(default_factory=list)
    cases: list[CaseDelta] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Populated only when compare is requested with pairwise=true: per-case
    # head-to-head LLM verdicts + win-rate summary (see run_manager).
    pairwise: dict[str, Any] | None = None
