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

    # LLM (Ollama OpenAI-compat endpoint)
    LLM_BASE_URL: str = "http://192.168.157.135:11434/v1"
    LLM_MODEL: str = "Qwen3-14B-AWQ"
    TEMPERATURE: float = 0.7
    OPENAI_API_KEY: Optional[str] = None

    # Hybrid Search (module 17)
    HYBRID_SEARCH_URL: str = "http://raptor-hybridsearch-api:8000"

    # Redis (conversation memory)
    REDIS_HOST: str = "raptor-redis-standalone"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Chat memory
    MEMORY_CONTEXT_WINDOW: int = 5
    MEMORY_TTL: int = 3600

    # HTTP client
    REQUEST_TIMEOUT: float = 60.0
    MAX_CONNECTIONS: int = 100
    MAX_KEEPALIVE_CONNECTIONS: int = 20


settings = Settings()
