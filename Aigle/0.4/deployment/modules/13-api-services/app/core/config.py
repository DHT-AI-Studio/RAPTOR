"""Application configuration utilities."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional

from pydantic import AliasChoices, Field
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

    api_version: str = Field(
        "0.4",
        validation_alias=AliasChoices("RAPTOR_API_VERSION", "GATEWAY_API_VERSION"),
        description="API version prefix used in all route paths (e.g. '0.4' → /api/0.4/...). Shared with Module 27 MCP via the root .env's API_VERSION.",
    )
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

    # Benchmark Service (module 22)
    benchmark_service_url: str = Field(
        "http://raptor-benchmark-service:8000/api/v1",
        description="Base URL of the benchmark service (module 22). Every route except "
                    "/health is mounted under /api/v1 (see app/main.py of module 22), so "
                    "that prefix is baked in here — mirrors training_service_url's pattern.",
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

    # Memory Service (Module 26)
    memory_service_url: str = Field(
        "http://raptor-memory:8026",
        description="Base URL of the memory service."
    )

    # Personal DB Service (module 25)
    personal_db_url: str = Field(
        "http://raptor-personal-db-service:8000",
        description="Base URL of the personal DB service (module 25). Internal-only — "
                    "it trusts the X-User-ID this gateway sends, so it must not be exposed."
    )

    # MCP Server (module 27) — MCP Streamable HTTP transport
    mcp_server_url: str = Field(
        "http://raptor-mcp-server:8027",
        description="Base URL of the MCP server (module 27).",
    )

    # Guardrail Service (module 23) — GB-4 policy checker used by GuardrailMiddleware
    # to intercept /chat/completions and /a2a/query (V04-10). GUARDRAIL_URL /
    # GUARDRAIL_TIMEOUT are the same names module 07's guardrail hook and the root
    # deployment/modules/.env.example already use — reused rather than duplicated.
    # gr_enabled below is THIS module's own on/off switch only — module 07's hook
    # has its own separate GUARDRAIL_ENABLED switch (src/api/guardrail_hook.py),
    # deliberately independent so GUARDRAIL_URL being populated here (which this
    # module needs regardless) can never silently turn 07's hook on too.
    guardrail_url: str = Field(
        "http://raptor-guardrail-service:8026",
        validation_alias=AliasChoices("GUARDRAIL_URL", "GATEWAY_GUARDRAIL_URL"),
        description="Base URL of the Guardrail Service (module 23). Container-to-container "
                    "address on the `raptor` network — port 8026 is what the container listens "
                    "on internally; 8023 (PORT_GUARDRAIL_SERVICE) is only the host-published port "
                    "for out-of-cluster access and must not be used here.",
    )
    gr_enabled: bool = Field(
        False,
        validation_alias=AliasChoices("GR_ENABLED", "GATEWAY_GR_ENABLED"),
        description="Master switch for GuardrailMiddleware. False = complete bypass, "
                    "no calls to the Guardrail Service.",
    )
    guardrail_timeout: float = Field(
        20.0,
        gt=0,
        validation_alias=AliasChoices("GUARDRAIL_TIMEOUT", "GATEWAY_GUARDRAIL_TIMEOUT"),
        description="Timeout (seconds) for calls to the Guardrail Service. On timeout the "
                    "check fails open (see app/middlewares/guardrail.py).",
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
