"""DA-09a — DocAgent service configuration.

All settings are read from DA_-prefixed environment variables so they are
clearly namespaced and never collide with other modules' vars.  Pydantic
Settings handles .env loading, type coercion, and validation in one place.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DocAgentSettings(BaseSettings):
    """All DA_* environment variables consumed by this service."""

    model_config = SettingsConfigDict(
        env_prefix="DA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Downstream service URLs ────────────────────────────────────────────────
    module12_sync_url: str = Field(
        default="http://raptor-document-analysis:8020",
        description="Module 12 document-analysis sync API base URL",
    )
    module07_url: str = Field(
        default="http://raptor-ai-lifecycle-api:8010",
        description="Module 07 AI/ML inference API base URL",
    )

    # ── Qdrant / embedding (shared with DA-6 EmbeddingSearchTool) ─────────────
    qdrant_url: str = Field(
        default="http://raptor-qdrant:6333",
        description="Qdrant vector DB URL",
    )
    qdrant_doc_collection: str = Field(
        default="doc_agent_embeddings",
        description="Qdrant collection name for DocAgent embeddings",
    )
    qdrant_vector_size: int = Field(default=1024)
    embed_model: str = Field(
        default="bge-m3",
        description="Embedding model name registered in Module 07",
    )
    embed_timeout: float = Field(default=30.0)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    embed_batch_size: int = Field(default=64)

    # ── LLM (used by classify / extract / summarize / compare / qa / parse_table)
    llm_model: str = Field(
        default="gemma4-12b-ollama",
        description="LLM model name registered in Module 07",
    )
    llm_timeout: float = Field(
        default=60.0,
        description="Per-request LLM timeout in seconds",
    )
    llm_max_tokens: int = Field(default=2048)

    # ── Module 12 adapter ─────────────────────────────────────────────────────
    doc12_timeout: float = Field(
        default=120.0,
        description="Timeout for Module 12 /analyze calls (large PDFs are slow)",
    )

    # ── Service ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")   # nosec B104 — container-internal bind
    port: int = Field(default=8290)
    log_level: str = Field(default="INFO")
    max_concurrent_requests: int = Field(default=10)

    # ── AI_LIFECYCLE_URL alias (keeps EmbeddingSearchConfig.from_env() working)
    ai_lifecycle_url: str = Field(
        default="http://raptor-ai-lifecycle-api:8010",
        description="Alias for DA_MODULE07_URL used by EmbeddingSearchConfig",
    )


@lru_cache(maxsize=1)
def get_settings() -> DocAgentSettings:
    return DocAgentSettings()
