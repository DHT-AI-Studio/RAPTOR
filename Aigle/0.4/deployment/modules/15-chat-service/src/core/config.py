"""Chat Service configuration."""
from __future__ import annotations

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # Service
    API_PORT: int = 8021

    # LLM -- goes through module 07 (ai-lifecycle-api)'s /v1/chat/completions,
    # not the Ollama daemon directly (see chat_service.py's ChatOpenAI init).
    # Needs the /v1 suffix, same as any other OpenAI-compatible base_url.
    LLM_BASE_URL: str = "http://raptor-ai-lifecycle-api:8010/v1"
    LLM_MODEL: str = "qwen3.5:9b"
    TEMPERATURE: float = 0.7
    OPENAI_API_KEY: Optional[str] = None  # unused (07 needs no auth); kept for rollback
    # Module 07's /v1/chat/completions accepts num_ctx as an explicit field and
    # threads it into Ollama's native options -- unlike Ollama's own /v1
    # OpenAI-compat layer, which silently ignores it (confirmed live). Default
    # matches LLM_MODEL's own trained context length (check via Ollama's
    # /api/show, model_info.*.context_length) -- update this when LLM_MODEL
    # changes to a model with a different one. Not the same number as the
    # higher LLM_CONTEXT_WINDOW_TOKENS budget below (that's Module 26's own
    # compaction threshold, unrelated to what's actually settable here).
    LLM_NUM_CTX: int = 32768
    # qwen3.x-family models "think" by default unless this is explicitly
    # false -- see chat_service.py's ChatOpenAI(extra_body=...) init.
    LLM_THINK: bool = False

    # Hybrid Search (module 17) — superseded by Module 25, kept for rollback
    HYBRID_SEARCH_URL: str = "http://raptor-hybridsearch-api:8000"

    # Personal DB Service (module 25) — per-user isolated search (was module 17/20)
    PERSONAL_DB_SERVICE_URL: str = "http://raptor-personal-db-service:8000"

    # Memory Service (module 26) — cross-session memory archive
    CHAT_MEMORY_SERVICE_URL: str = "http://raptor-memory-service:8026"
    MEMORY_REQUEST_TIMEOUT: float = 3.0
    MEMORY_RETRIEVE_TIMEOUT: float = 0.5
    # MV-12 compact_session call: LLM summarization on the Module 26 side can take longer
    COMPACT_REQUEST_TIMEOUT: float = 90.0
    LLM_CONTEXT_WINDOW_TOKENS: int = 128000

    # Redis (conversation memory)
    REDIS_HOST: str = "raptor-redis-standalone"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Chat memory
    MEMORY_CONTEXT_WINDOW: int = 5
    MEMORY_TTL: int = 3600

    # Compact context budget: max tokens for context_window before trimming (chars/4)
    COMPACT_CONTEXT_BUDGET: int = 20000

    # HTTP client
    REQUEST_TIMEOUT: float = 60.0
    MAX_CONNECTIONS: int = 100
    MAX_KEEPALIVE_CONNECTIONS: int = 20


settings = Settings()
