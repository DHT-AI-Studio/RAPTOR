"""Auto-tuning experiment models (AUTOTUNE Phase A/B).

An *experiment* is one natural-language optimization goal turned into a plan
(base training config + a search space + an eval schema) that the orchestrator
loop then optimizes against, one training+eval iteration at a time.

See AUTOTUNE_DESIGN.md §4 (API contract) and §7 (persistence).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ExperimentStatus(str, Enum):
    planning = "planning"                # planner is deriving a plan from the goal
    awaiting_confirm = "awaiting_confirm"  # plan ready, waiting for the customer to confirm
    running = "running"                  # orchestrator loop is iterating
    completed = "completed"              # budget exhausted / converged
    failed = "failed"                    # unrecoverable error
    stopped = "stopped"                  # cancelled by the customer


class Budget(BaseModel):
    """Hard budget guardrails enforced by the orchestrator (never by the agent)."""

    max_experiments: int = Field(10, ge=1, le=200, description="Max training+eval iterations")
    minutes_per_experiment: float = Field(
        30.0, gt=0, le=720, description="Per-iteration training timeout (job is cancelled past this)"
    )
    early_stop_patience: int = Field(
        3, ge=1, le=100, description="Stop after this many consecutive non-improving iterations"
    )


class SearchDimension(BaseModel):
    """One tunable knob and its allowed range — the agent's action space.

    - float / int : bounded numeric range (``log`` samples on a log scale).
    - categorical : one of an explicit ``choices`` list.
    """

    type: str = Field(..., description="float | int | categorical")
    min: Optional[float] = None
    max: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None

    @model_validator(mode="after")
    def _check(self) -> "SearchDimension":
        if self.type in ("float", "int"):
            if self.min is None or self.max is None:
                raise ValueError(f"{self.type} dimension requires 'min' and 'max'")
            if self.min > self.max:
                raise ValueError("'min' must be <= 'max'")
            if self.log and self.min <= 0:
                raise ValueError("log-scale dimension requires min > 0")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical dimension requires a non-empty 'choices' list")
        else:
            raise ValueError(f"unknown dimension type: {self.type}")
        return self


class Plan(BaseModel):
    """Everything needed to launch a Module 16 training job, minus the searched knobs.

    The orchestrator merges ``base_training_config`` with one sampled point from
    ``search_space`` per iteration to form the full ``training_config``.
    """

    task_type: str = Field("instruction", description="instruction | text")
    select_multiple_gpus: bool = False
    vram_budget_gb: float = Field(12.0, gt=0)
    base_training_config: Dict[str, Any] = Field(
        ..., description="Fixed TrainerConfig fields (model, quantization, lora, epochs...)"
    )
    dataset_config: Dict[str, Any] = Field(..., description="Module 16 DatasetConfig")
    search_space: Dict[str, SearchDimension] = Field(
        ..., description="Tunable knobs and their ranges (the agent's action space)"
    )
    eval_schema_id: Optional[str] = Field(
        None,
        description="Benchmark schema used to score every candidate (dev set). None while the "
                    "dataset still needs downloading — the eval is built from real rows once it's local.",
    )
    holdout_schema_id: Optional[str] = Field(
        None,
        description="Independent held-out schema the planner pre-sampled (real-data path). When set, "
                    "the orchestrator skips the ratio split and validates the best config on this.",
    )
    budget: Optional[Budget] = Field(
        None,
        description="Budget the planner derived from the goal (e.g. '跑 5 次,每次最多 10 分鐘'). "
                    "An explicit budget on the request overrides this; if neither is set, defaults apply.",
    )
    needs_download: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Resources the plan references that are NOT present locally (grounded against "
                    "Module 07). Each: {kind: model|dataset, id, download_endpoint}. Surfaced in the "
                    "plan preview so the customer can approve/trigger a download before running.",
    )

    @model_validator(mode="after")
    def _non_empty_search_space(self) -> "Plan":
        if not self.search_space:
            raise ValueError("search_space must contain at least one dimension")
        return self


class OptimizeRequest(BaseModel):
    """POST /optimize body.

    Step 1 (no planner yet): supply an explicit ``plan``. Once the planner lands,
    ``plan`` becomes optional and is derived from ``goal`` + ``eval``.
    """

    goal: str = Field(..., description="Natural-language optimization goal (recorded; used by the planner)")
    budget: Optional[Budget] = Field(
        None,
        description="Explicit hard budget. Wins over any budget the planner derives from the goal; "
                    "omit it to let the planner read the goal (or fall back to system defaults).",
    )
    plan: Optional[Plan] = Field(None, description="Explicit plan (required until the planner is implemented)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "goal": "用 philschmid/dolly-15k-oai-style 微調 gemma-3-270m-it，讓它更會遵循指令",
                "budget": {"max_experiments": 4, "minutes_per_experiment": 20, "early_stop_patience": 3},
                "plan": {
                    "task_type": "instruction",
                    "select_multiple_gpus": False,
                    "vram_budget_gb": 12,
                    "base_training_config": {
                        "model_name_or_path": "/app/tmp/models/google_gemma-3-270m-it",
                        "use_bfloat16": True,
                        "use_flash_attn": False,
                        "max_epochs": 1,
                        "batch_size": 1,
                        "gradient_accumulation_steps": 2,
                        "gradient_checkpointing": True,
                        "use_8bit_adamw": True,
                        "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                                 "bnb_4bit_use_double_quant": True,
                                                 "bnb_4bit_compute_dtype": "bfloat16"},
                        "lora_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                                         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                                         "bias": "none", "modules_to_save": None},
                    },
                    "dataset_config": {
                        "dataset_name_or_path": "philschmid/dolly-15k-oai-style",
                        "train_size": 30, "val_size": 6, "max_length": 512,
                        "column_mapping": {"messages": "messages"},
                    },
                    "search_space": {
                        "learning_rate": {"type": "float", "min": 1e-5, "max": 5e-4, "log": True},
                        "lora_r": {"type": "categorical", "choices": [8, 16, 32]},
                    },
                    "eval_schema_id": "<a local_infer schema id>",
                },
            }
        }
    }


class ExperimentRecord(BaseModel):
    """Experiment row returned by GET /optimize/{id}."""

    experiment_id: str
    goal: str
    status: ExperimentStatus
    plan: Optional[Dict[str, Any]] = None
    eval_schema_id: Optional[str] = None
    budget: Dict[str, Any] = Field(default_factory=dict)
    iterations_done: int = 0
    best_run_id: Optional[str] = None
    best_score: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
