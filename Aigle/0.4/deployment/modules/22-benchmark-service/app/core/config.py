from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Benchmark Service 統一設定。

    Benchmark 專屬的環境變數加上 BM_ 前綴（例如 BM_LOG_LEVEL、BM_JUDGE_MODEL）。
    共用的基礎設施憑證（POSTGRES_*、REDIS_*）沿用整個 stack 的 ../.env，
    因此以 AliasChoices 同時接受「無前綴」與「BM_ 前綴」兩種來源。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BM_",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")

    # ── PostgreSQL (schema definitions + run history) ────────────────
    # Host/port/user/password come from the shared ../.env (POSTGRES_*).
    postgres_host: str = Field(
        "raptor-postgres",
        validation_alias=AliasChoices("BM_POSTGRES_HOST", "POSTGRES_HOST"),
    )
    postgres_port: int = Field(
        5432,
        validation_alias=AliasChoices("BM_POSTGRES_PORT", "POSTGRES_PORT"),
    )
    postgres_user: str = Field(
        "raptor",
        validation_alias=AliasChoices("BM_POSTGRES_USER", "POSTGRES_USER"),
    )
    postgres_password: str = Field(
        "raptor",
        validation_alias=AliasChoices("BM_POSTGRES_PASSWORD", "POSTGRES_PASSWORD"),
    )
    # Dedicated benchmark database (see 03-database/init 001_init.sql).
    postgres_db: str = Field("benchmark", description="Benchmark PostgreSQL database name")

    # ── Redis (live run state) ───────────────────────────────────────
    redis_host: str = Field(
        "raptor-redis-standalone",
        validation_alias=AliasChoices("BM_REDIS_HOST", "REDIS_HOST"),
    )
    redis_port: int = Field(
        6379,
        validation_alias=AliasChoices("BM_REDIS_PORT", "REDIS_PORT"),
    )
    redis_db: int = Field(
        0,
        validation_alias=AliasChoices("BM_REDIS_DB", "REDIS_DB"),
    )
    redis_password: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("BM_REDIS_PASSWORD", "REDIS_PASSWORD"),
    )
    redis_run_ttl_seconds: int = Field(
        604800, description="TTL for benchmark:run:{run_id} keys (default 7 days)"
    )

    # ── LLM-as-judge / embeddings (Module 07 AI Lifecycle API) ───────
    inference_url: str = Field(
        "http://raptor-ai-lifecycle-api:8010",
        description="AI Lifecycle inference API base URL",
    )
    judge_model: str = Field("qwen3.5:9b", description="Default LLM-as-judge model")
    judge_engine: str = Field("ollama", description="Inference engine for the judge model")
    judge_think: bool = Field(False, description="Ollama 'thinking' mode for judge/complete() calls")
    inference_timeout: int = Field(
        30,
        validation_alias=AliasChoices("BM_JUDGE_TIMEOUT_S", "BM_INFERENCE_TIMEOUT"),
        description="LLM judge / inference API timeout in seconds",
    )
    # Embeddings (cosine_similarity scorer) go through Module 07's `embedding`
    # task family; the model must be registered there (e.g. bge-m3 via
    # POST /models/register_ollama with task="embedding").
    embed_model: str = Field("bge-m3", description="Embedding model registered in Module 07 (multilingual, 1024-dim)")
    embed_timeout: int = Field(30, description="Timeout (s) for an embedding call")

    # ── Lifecycle inference target (benchmark a Module 07-registered model) ──
    # 當 target_pipeline=lifecycle_infer 時，受測模型由這個 Module 07 AI Lifecycle
    # inference API 提供（例如 aiml-v314 的 lifecycle-api）。刻意與 judge 用的
    # inference_url 分開：受測模型與評審模型可以是不同的服務/版本。
    lifecycle_infer_url: str = Field(
        "http://raptor-ai-lifecycle-api:8010",
        validation_alias=AliasChoices("BM_LIFECYCLE_INFER_URL", "BM_LIFECYCLE_URL"),
        description="Module 07 inference base URL serving the model under test (lifecycle_infer)",
    )
    lifecycle_infer_timeout: int = Field(
        180, description="Timeout (s) for a lifecycle_infer call (first call loads the model onto GPU)")
    lifecycle_infer_engine: str = Field(
        "transformers", description="Default inference engine for lifecycle_infer (transformers | ollama); "
                                    "config_override.engine wins per-run")
    lifecycle_infer_temperature: float = Field(
        0.7, description="Default sampling temperature for lifecycle_infer; the transformers engine "
                         "rejects 0.0 (needs strictly > 0), unlike ollama")
    lifecycle_infer_max_length: int = Field(
        256, description="Default max_length for a lifecycle_infer generation")
    lifecycle_infer_think: bool = Field(
        False, description="Default Ollama 'thinking' mode for lifecycle_infer; only applies when "
                           "engine=ollama, config_override.think wins per-run")

    # ── MLflow run history (Module 07 tracking server) ───────────────
    mlflow_url: str = Field(
        "http://raptor-mlflow:5555",
        description="MLflow tracking server URL (BM_MLFLOW_URL); runs are logged best-effort",
    )
    mlflow_timeout: int = Field(
        10, description="HTTP timeout (s) per MLflow tracking request"
    )

    # ── Pipeline executor ────────────────────────────────────────────
    pipeline_timeout: int = Field(30, description="Per-pipeline HTTP timeout in seconds")
    chat_url: str = Field("http://raptor-chat-service:8021")
    # Module 17 (hybrid-search) is retired; Module 25's per-user ArcadeDB search
    # replaced it (see root README.md's Deprecated Modules table). Module 25's
    # /personal/search/* routes are physically isolated per X-Branch-ID (a
    # separate ArcadeDB database per branch, one person's data per branch) --
    # there is no shared/global corpus to fall back to, so a search test case
    # must always name whose branch it's testing against (input.branch_id or
    # input.user_id) -- see executor.py's _build_headers, which errors rather
    # than silently defaulting to someone else's data.
    search_url: str = Field("http://raptor-personal-db-service:8000")
    rag_url: str = Field("http://raptor-agent-protocol:8030")
    classify_url: str = Field("http://raptor-query-orchestrator:8000")
    # Module 16 training service — serves local/fine-tuned checkpoints.
    local_infer_url: str = Field("http://raptor-training-service:8009")
    local_infer_timeout: int = Field(180, description="Timeout for local inference (first call loads the model)")
    local_infer_batch: bool = Field(
        True, description="Score a local_infer schema with ONE batched inference call (all test cases "
                         "at once) instead of one call per case. Falls back to per-case on any failure.")
    unload_after_run: bool = Field(
        True, description="After a local_infer run completes, tell Module 16 to unload the model "
                         "and free VRAM (so a following training job/iteration can use the GPU)"
    )

    # ── Auto-tuning orchestrator (AUTOTUNE) ──────────────────────────
    # Module 16 training API — submit / poll / cancel training jobs.
    training_url: str = Field("http://raptor-training-service:8009",
                              description="Module 16 training service base URL")
    training_submit_path: str = Field("/api/v1/training/submit")
    training_status_path: str = Field("/api/v1/training/status")   # + /{job_id}
    training_cancel_path: str = Field("/api/v1/training/cancel")   # + /{job_id}
    training_submit_timeout: int = Field(60, description="HTTP timeout for the submit call")
    training_poll_seconds: float = Field(15.0, description="Interval between training status polls")
    eval_poll_seconds: float = Field(3.0, description="Interval between eval status polls (in-process)")
    training_cancel_grace_seconds: int = Field(
        180, description="After cancelling an over-budget job, wait up to this long for it to reach "
                         "a terminal state (GPU teardown) before moving on")

    # ── Held-out eval (anti-overfit: optimize on dev, validate on held-out) ──
    holdout_enabled: bool = Field(
        True, description="Split the eval schema into dev/held-out so the optimizer can't overfit "
                         "the whole test set; the best config is re-scored on the untouched held-out")
    # ratio/min_cases apply ONLY to the LLM-fallback path (a scarce set that must be
    # partitioned). The real-data path samples dev and held-out as two independent,
    # disjoint draws sized directly (planner_dev_cases / planner_holdout_cases).
    holdout_ratio: float = Field(0.3, ge=0.05, le=0.5,
                                 description="LLM-fallback only: fraction of the LLM cases held out")
    holdout_min_cases: int = Field(
        4, description="LLM-fallback only: min cases needed to bother splitting")
    holdout_seed: int = Field(1234, description="Seed for the deterministic dev/held-out split")

    # ── Proposer (Phase 2: LLM proposes the next config from leaderboard) ──
    proposer: str = Field("llm", description="Optimization proposer: 'llm' | 'random'")
    proposer_model: Optional[str] = Field(
        None, description="Model for the LLM proposer (defaults to judge_model)")
    proposer_max_tokens: int = Field(384, description="Max tokens for a proposer decision (JSON + reason)")
    proposer_timeout: int = Field(60, description="Timeout (s) for a proposer LLM call")

    # ── Planner (Phase 3: NL goal → base_config + search_space + eval schema) ──
    planner_max_tokens: int = Field(2048, description="Max tokens for a planner output (config + eval)")
    planner_timeout: int = Field(120, description="Timeout (s) for a planner LLM call")
    # Real-data eval: sample rows straight from the local dataset (input=user turn,
    # expected_answer=gold) instead of trusting LLM-invented test cases. Scored by
    # cosine_similarity (bge-m3). Falls back to LLM-generated cases only when the
    # dataset isn't local yet. Because the dataset is abundant, dev and held-out are
    # sampled as two independent, disjoint sets sized directly (no ratio needed).
    planner_dev_cases: int = Field(30, ge=4, le=200,
                                   description="Real dataset rows sampled for the dev (tuning) set")
    planner_holdout_cases: int = Field(15, ge=1, le=100,
                                       description="Real dataset rows sampled for the held-out set")
    planner_eval_seed: int = Field(7, description="Seed for sampling eval rows (distinct from "
                                                  "training's seed to keep the split meaningful)")
    # Training subset size (Module 16 shuffles the dataset and takes this many rows).
    # Small by default for fast iterations; the goal can override ("用 200 筆訓練").
    planner_train_size: int = Field(30, ge=4, le=5000, description="Rows used to fine-tune each candidate")
    planner_val_size: int = Field(6, ge=1, le=1000, description="Rows held for training-time validation")
    planner_seed: int = Field(42, description="Training seed pinned into every candidate's config so "
                                              "identical hyperparameters reproduce (fair comparison, no "
                                              "re-run noise). Shared across candidates; NOT a search knob.")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance shared across the app lifecycle."""
    return Settings()
