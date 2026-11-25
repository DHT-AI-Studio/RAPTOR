from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ─ Shared ─────────────────────────────────────────────────────────────────
    ollama_url: str = Field("")
    request_timeout: float = Field(120.0)
    log_level: str = Field("INFO")
    # GB-17 / VIE01-245: IANA zone for log timestamps (violation logs included).
    # Applied to os.environ["TZ"] + time.tzset() at startup (app/main.py).
    # docker-compose.yml only passes TZ (from the root .env's TIMEZONE), so TZ
    # is read as a fallback name for the same setting.
    timezone: str = Field("Asia/Taipei", validation_alias=AliasChoices("TIMEZONE", "TZ"))

    # ─ Proxy + confidence-based classifier ─────────────────────────────────────
    proxy_model: str = Field("gemma4:12b")
    proxy_mode: str = Field("monitor")               # monitor | enforce
    proxy_confidence_threshold: float = Field(0.7)
    default_model: str = Field("gemma4:31b")

    # ─ OpenAI-compatible guardrail proxy (/v1/*, app/routers/proxy_openai.py) ──
    # Module 07 by default, not Ollama: callers send engine/num_ctx/think, which
    # only module 07's /v1/chat/completions honours.
    proxy_upstream_url: str = Field("http://raptor-ai-lifecycle-api:8010")
    proxy_upstream_timeout: float = Field(120.0)

    # ─ Guard models: Llama-Guard3 / Granite / GPT-OSS ────────────────────────
    guard_model: str = Field("llama-guard3:8b")
    guard_model_2: str = Field("")                   # 空字串 = 不啟用
    guard_model_3: str = Field("")                   # 空字串 = 不啟用
    guard_models: str = Field("")                    # optional open-ended comma-separated list;
                                                       # wins over guard_model/2/3 when set (see active_models)
    guard_conflict_strategy: Literal["strict", "majority", "permissive"] = Field(
        "strict",   # any model unsafe -> unsafe; matches the pre-existing hardcoded behavior
    )

    # ─ Policy storage: module 03's shared PostgreSQL, database "guardrails" ──
    postgres_host: str = Field("raptor-postgres")
    postgres_port: int = Field(5432)
    postgres_user: str = Field("raptor")
    postgres_password: str = Field("")
    postgres_db: str = Field("guardrails")

    # ─ Active-policy cache: module 02's shared Redis (DB 0, "guardrail:" key prefix) ──
    redis_host: str = Field("raptor-redis-standalone")
    redis_port: int = Field(6379)
    redis_password: str = Field("")
    redis_db: int = Field(0)

    # ─ Guardrail global switch (guardrail:enabled in Redis) ───────────────────
    gr_default_enabled: bool = Field(True)

    # ─ Audit (CF-4 audit_events, GRD-X4 / VIE01-354) ──────────────────────────
    # retention_days on every guardrail audit event: how long it is kept.
    guardrail_audit_retention_days: int = Field(90, ge=1)

    # ─ CM-5 (VIE01-408): automated flagging system ────────────────────────────
    kafka_bootstrap: str = Field("kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092")
    kafka_moderation_topic: str = Field("moderation.flagged")

    # ── Compat aliases ─────────────────────────────────────────────────────────
    # Backward-compatible property names kept so downstream routers/services
    # can keep calling settings.xxx without needing changes.

    @property
    def active_models(self) -> list[str]:
        if self.guard_models:
            return [m.strip() for m in self.guard_models.split(",") if m.strip()]
        models = [self.guard_model]
        if self.guard_model_2:
            models.append(self.guard_model_2)
        if self.guard_model_3:
            models.append(self.guard_model_3)
        return models

    @property
    def confidence_threshold(self) -> float:
        return self.proxy_confidence_threshold

    @property
    def real_ollama_url(self) -> str:
        return self.ollama_url

    @property
    def guardrails_model(self) -> str:
        return self.proxy_model

    @property
    def guardrails_mode(self) -> str:
        return self.proxy_mode


@lru_cache
def get_settings() -> Settings:
    return Settings()
