"""Application configuration utilities."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GATEWAY_",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    api_version: str = Field("0.3", description="API version prefix used in all route paths (e.g. '0.3' → /api/0.3/...).")
    log_level: str = Field("INFO", description="Logging level for the gateway.")
    request_timeout: float = Field(
        10.0,
        gt=0,
        description="Default timeout (in seconds) for proxy requests to downstream services.",
    )
    upload_timeout: float = Field(
        300.0,
        gt=0,
        description="Timeout (in seconds) for file upload requests to the asset management service.",
    )

    # ONLY for DEBUG and TESTING, if you want to DEPLOY PLEASE REMOVE "Optional"
    kafka_bootstrap_servers: str = Field(..., description="Comma-separated list of Kafka bootstrap servers.")
    kafka_topics: Dict[str, str] = Field(..., description="Mapping of file types to Kafka topics.")

    # Redis 快取配置
    redis_host: str = Field("raptor-redis-standalone", description="Hostname for the Redis cache.")
    redis_port: int = Field(6379, description="Port for the Redis cache.")
    redis_db: int = Field(0, description="Database index for the Redis cache.")
    redis_password: Optional[str] = Field(None, description="Password for the Redis cache.")

    # Hybrid Search 服務 (module 16)
    hybrid_search_url: str = Field(
        "http://raptor-hybridsearch-api:8000",
        description="Base URL of the hybrid search service.",
    )

    # Chat service (module 14)
    chat_service_url: str = Field(
        "http://raptor-chat-service:8021",
        description="Base URL of the chat service (module 14).",
    )

    # Document sync API (module 11)
    document_sync_url: str = Field(
        "http://raptor-document-analysis:8020",
        description="Base URL of the document sync analysis API (module 11).",
    )

    # Image sync API (module 09)
    image_sync_url: str = Field(
        "http://raptor-image-processing:8018",
        description="Base URL of the image sync analysis API (module 09).",
    )

    # Audio sync API (module 08)
    audio_sync_url: str = Field(
        "http://raptor-audio-recognizer:8019",
        description="Base URL of the audio sync transcription API (module 08).",
    )

    # Video frame sync API (module 11)
    video_frame_sync_url: str = Field(
        "http://raptor-video-frame-description:8031",
        description="Base URL of the video frame sync analysis API (module 11).",
    )

    # 效能配置
    max_connections: int = Field(
        100,
        ge=1,
        description="Maximum HTTP connections in httpx pool.",
    )
    max_keepalive_connections: int = Field(
        20,
        ge=1,
        description="Maximum keep-alive HTTP connections in httpx pool.",
    )
    batch_upload_concurrency: int = Field(
        4,
        ge=1,
        le=16,
        description="Max concurrent uploads for batch operations.",
    )
    rate_limit_per_user_per_minute: int = Field(
        1000,
        ge=1,
        description="Per-user rate limit (requests/minute) enforced via Redis.",
    )

    # AI Lifecycle API 服務
    aiml_lifecycle_api_url: str = Field(
        "http://raptor-ai-lifecycle-api:8010",
        description="Base URL of the AI Lifecycle API service."
    )

    # Training Service
    training_service_url: str = Field(
        "http://raptor-training-service:8009/api/v1/training",
        description="Base URL of the training service."
    )

    # Authentication Service (Keycloak wrapper, module 06)
    auth_service_url: str = Field(
        "http://keycloak-api:8800",
        description="Base URL of the auth service (module 06).",
    )

    # Internal Keycloak URL for JWKS fetch (must be reachable from inside Docker)
    keycloak_url: str = Field(
        "http://keycloak:8080",
        description="Internal Keycloak base URL used to fetch JWKS (token iss may use external IP).",
    )

    # Vision Analysis Service
    vision_service_url: str = Field(
        "http://raptor-temporal-model-service:8000/analyze/video",
        description="Base URL of the vision analysis service."
    )

    # Asset Management Service
    asset_management_url: str = Field(
        "http://raptor-asset-management:8000",
        description="Base URL of the asset management service."
    )

    # Agent Protocol Service
    agent_protocol_url: str = Field(
        "http://raptor-agent-protocol:8030",
        description="Base URL of the agent protocol service."
    )

    # OpenSearch Bridge Service
    opensearch_bridge_url: str = Field(
        "http://raptor-opensearch-bridge:8840",
        description="Base URL of the OpenSearch bridge service."
    )

    # Graph Service (module 20) — GraphRAG + TKG combined
    graph_service_url: str = Field(
        "http://raptor-graph-service:8843",
        description="Base URL of the Graph Service (module 20)."
    )

    # Query Orchestrator (module 18) — intent classification + signal extraction
    query_orchestrator_url: str = Field(
        "http://raptor-query-orchestrator:8000",
        description="Base URL of the Query Orchestrator service (module 18).",
    )

    # 模型名稱設定
    intent_model: str = Field(
        "qwen2.5:7b",
        description="Model name used for intent classification in query orchestrator.",
    )
    answer_model: str = Field(
        "qwen2.5:7b",
        description="Model name used for answer generation in query orchestrator.",
    )
    smolagents_model: str = Field(
        "ollama/qwen2.5:7b",
        description="LiteLLM model ID for smolagents (e.g. ollama/Qwen3-14B-AWQ).",
    )
    vibevoice_asr_model: str = Field(
        "microsoft/VibeVoice-ASR",
        description="Model name for VibeVoice ASR inference.",
    )
    agent_id: str = Field(
        "raptor-gateway",
        description="Agent Card unique identifier.",
    )
    agent_name: str = Field(
        "Raptor Gateway Agent",
        description="Agent Card display name.",
    )
    agent_endpoint: str = Field(
        "http://raptor-api-gateway:8012",
        description="Public URL of this gateway (used in AgentCard and A2A discovery).",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    logging.getLogger(__name__).debug("Application settings initialised")
    return settings
