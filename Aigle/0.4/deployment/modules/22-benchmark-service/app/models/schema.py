"""Marking-schema Pydantic models (BM-2).

A marking schema is the benchmark equivalent of AutoResearch's ``program.md``:
it declares the test cases to run and the scoring rubric to grade them with.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ScoringMethod(str, Enum):
    """Canonical names of the shipped built-in scorers.

    NOTE: this enum is a convenience/reference only — ``ScoringDimension.method``
    is an open string validated against the pluggable scorer registry, so custom
    or newly added strategies work without extending this enum.
    """

    llm_judge = "llm_judge"
    keyword_match = "keyword_match"
    latency_threshold = "latency_threshold"
    cosine_similarity = "cosine_similarity"
    regex_match = "regex_match"
    exact_match = "exact_match"
    contains_all = "contains_all"
    contains_any = "contains_any"
    numeric_tolerance = "numeric_tolerance"


class TargetPipeline(str, Enum):
    chat = "chat"
    search = "search"
    rag = "rag"
    classify = "classify"
    local_infer = "local_infer"  # serve a local/fine-tuned checkpoint via Module 16
    lifecycle_infer = "lifecycle_infer"  # benchmark a model registered in Module 07 (AI Lifecycle API)


# Legacy flat params kept for backward compatibility; folded into `params`.
_LEGACY_PARAM_FIELDS = ("rubric", "max_ms", "pattern")


class ScoringDimension(BaseModel):
    """A single scored dimension inside a scoring schema.

    ``method`` is any name registered in the scorer registry. Method-specific
    configuration goes in ``params`` (e.g. ``{"max_ms": 5000}``,
    ``{"pattern": "..."}``, ``{"rubric": "..."}``, ``{"expected": 42, "tolerance": 1}``).
    """

    name: str = Field(..., description="Dimension name, e.g. 'relevance'")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in [0, 1]; all weights sum to 1.0")
    method: str = Field(..., description="Registered scoring method name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")

    # Legacy flat params — still accepted, folded into `params` below.
    rubric: Optional[str] = Field(None, description="(legacy) → params.rubric")
    max_ms: Optional[float] = Field(None, description="(legacy) → params.max_ms")
    pattern: Optional[str] = Field(None, description="(legacy) → params.pattern")

    @field_validator("method", mode="before")
    @classmethod
    def _coerce_method(cls, v: Any) -> Any:
        return v.value if isinstance(v, Enum) else v

    @field_validator("method")
    @classmethod
    def _method_is_registered(cls, v: str) -> str:
        # Lazy import avoids a models → services import cycle at module load.
        from app.services.scoring import is_registered, list_scorers

        if not is_registered(v):
            raise ValueError(f"unknown scoring method '{v}'; available: {list_scorers()}")
        return v

    @model_validator(mode="after")
    def _fold_legacy_params(self) -> "ScoringDimension":
        for key in _LEGACY_PARAM_FIELDS:
            value = getattr(self, key)
            if value is not None and key not in self.params:
                self.params[key] = value
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "relevance",
                "weight": 0.4,
                "method": "llm_judge",
                "params": {"rubric": "Score 1-5: Does the answer address the question without hallucination?"},
            }
        }
    }


class ScoringSchema(BaseModel):
    """The scoring rubric: a set of weighted dimensions + aggregate rule."""

    dimensions: List[ScoringDimension] = Field(..., min_length=1)
    aggregate: str = Field("weighted_sum", description="Aggregation strategy")
    score_range: List[float] = Field(
        default=[1, 5], description="Raw score range for llm_judge normalization to 0-1"
    )

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "ScoringSchema":
        total = sum(d.weight for d in self.dimensions)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"dimension weights must sum to 1.0 (±0.01), got {total:.3f}")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "dimensions": [
                    {"name": "relevance", "weight": 0.6, "method": "llm_judge",
                     "rubric": "Score 1-5: is the answer relevant?"},
                    {"name": "keywords", "weight": 0.2, "method": "keyword_match"},
                    {"name": "latency", "weight": 0.2, "method": "latency_threshold", "max_ms": 5000},
                ],
                "aggregate": "weighted_sum",
                "score_range": [1, 5],
            }
        }
    }


class TestCase(BaseModel):
    """A single benchmark test case."""

    __test__ = False  # prevent pytest from collecting this as a test class

    id: str = Field(..., description="Unique test-case id within the schema")
    input: Dict[str, Any] = Field(..., description="Pipeline input payload")
    expected_keywords: Optional[List[str]] = Field(None, description="Keywords expected in output")
    expected_answer: Optional[str] = Field(None, description="Reference answer for similarity scoring")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "tc001",
                "input": {"message": "What are the main findings of the Q3 report?",
                          "user_id": "benchmark-user"},
                "expected_keywords": ["revenue", "growth"],
                "expected_answer": "Revenue grew 12% in Q3...",
            }
        }
    }


class BenchmarkSchema(BaseModel):
    """Top-level marking schema uploaded by a user."""

    name: str = Field(..., description="Human-readable schema name")
    version: str = Field("1.0", description="Schema version")
    target_pipeline: TargetPipeline = Field(..., description="Pipeline under test")
    target_url: Optional[str] = Field(None, description="Optional URL override for the pipeline")
    test_cases: List[TestCase] = Field(..., min_length=1)
    scoring_schema: ScoringSchema
    judge_model: Optional[str] = Field(None, description="LLM-as-judge model override")

    @field_validator("test_cases")
    @classmethod
    def _unique_case_ids(cls, v: List[TestCase]) -> List[TestCase]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("test_case ids must be unique")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "RAG Quality v1",
                "version": "1.0",
                "target_pipeline": "chat",
                "test_cases": [
                    {"id": "tc001",
                     "input": {"message": "What are the main findings?", "user_id": "benchmark-user"},
                     "expected_keywords": ["revenue", "growth"]},
                ],
                "scoring_schema": {
                    "dimensions": [
                        {"name": "relevance", "weight": 0.6, "method": "llm_judge",
                         "rubric": "Score 1-5: is the answer relevant?"},
                        {"name": "keywords", "weight": 0.2, "method": "keyword_match"},
                        {"name": "latency", "weight": 0.2, "method": "latency_threshold", "max_ms": 5000},
                    ],
                    "aggregate": "weighted_sum",
                    "score_range": [1, 5],
                },
                "judge_model": "qwen3.5:0.8b",
            }
        }
    }
