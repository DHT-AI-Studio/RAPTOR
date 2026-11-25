"""Application configuration utilities."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

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
        "0.5",
        validation_alias="GATEWAY_API_VERSION",
        description="API version prefix used in all route paths (e.g. '0.5' → /api/0.5/...). Shared with Module 27 MCP via the root .env's API_VERSION.",
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
    kafka_topics: dict[str, str] = Field(..., description="Mapping of file types to Kafka topics.")

    # Redis 快取配置
    redis_host: str = Field("raptor-redis-standalone", description="Hostname for the Redis cache.")
    redis_port: int = Field(6379, description="Port for the Redis cache.")
    redis_db: int = Field(0, description="Database index for the Redis cache.")
    redis_password: str | None = Field(None, description="Password for the Redis cache.")

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
    face_image_presign_concurrency: int = Field(
        16,
        ge=1,
        le=64,
        description="Max concurrent Module 04 presign calls when adding face image URLs to one response.",
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

    # Authentication Service (Keycloak wrapper, module 06). raptor-keycloak-api is the
    # network alias 06-authentication's docker-compose.yml gives that container on the
    # shared external `raptor` network -- the bare compose service name `keycloak-api`
    # only resolves inside module 06's own compose project, not across modules.
    auth_service_url: str = Field(
        "http://raptor-keycloak-api:8800",
        description="Base URL of the auth service (module 06).",
    )

    # Internal Keycloak URL for JWKS fetch (must be reachable from inside Docker).
    # raptor-keycloak is likewise the real cross-module network alias, not `keycloak`.
    keycloak_url: str = Field(
        "http://raptor-keycloak:8080",
        description="Internal Keycloak base URL used to fetch JWKS (token iss may use external IP).",
    )

    # Externally-reachable Keycloak URL — MUST be set correctly per deployment.
    # Used only for MCP OAuth discovery (see docs/MCP_OAUTH_REQUIREMENTS.md): an
    # MCP client (e.g. Claude Code, running on a user's own machine, outside
    # the raptor Docker network) needs a URL it can actually resolve, unlike
    # keycloak_url above. Matches the KC_HOSTNAME the Keycloak container
    # itself is configured with (see 06-authentication/docker-compose.yml).
    keycloak_public_url: str = Field(
        "http://localhost:8080",
        description="Externally-reachable Keycloak base URL, for MCP OAuth discovery metadata. "
                    "Must match this deployment's KC_HOSTNAME/HOST_IP, not the internal Docker hostname.",
    )
    keycloak_realm: str = Field(
        "dhtsolution",
        description="Keycloak realm name, for MCP OAuth discovery metadata.",
    )
    # The URL clients use to reach this gateway, when that is not what the gateway itself sees -- behind a
    # TLS-terminating proxy (CF-3a) the request arrives as plain http from the proxy, so request.base_url
    # would advertise http://<internal> as the MCP resource and the client would reject the metadata.
    # Set it instead of trusting X-Forwarded-* headers: the proxy's address is not fixed, and trusting
    # every sender would let anyone forge X-Forwarded-For, which feeds the audit log's IP hash.
    public_url: str = Field(
        "",
        description="Externally-reachable base URL of this gateway (e.g. https://raptor.example:8443), for MCP "
                    "OAuth discovery metadata. Empty: derived from the request, as before.",
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

    # Privacy Compliance Service (Module 33, GD-1 / VIE01-246)
    privacy_compliance_url: str = Field(
        "http://raptor-privacy-compliance-service:8033",
        description="Base URL of the GDPR/CCPA erasure orchestrator."
    )
    gdpr_internal_token: str = Field(
        "",
        description="Shared secret presented to Module 33 as X-Internal-Token. "
                     "Empty means /gdpr/* calls fail closed rather than silently unauthenticated."
    )
    # CM-7 / VIE01-410: GET /compliance/report waits for Module 33 to read five
    # tables for the whole period -- longer than an ordinary proxy call.
    compliance_report_timeout: float = Field(
        120.0,
        gt=0,
        description="Timeout (seconds) for GET /compliance/report's call to Module 33.",
    )

    # Write-freeze maintenance mode (VIE01-369): /admin/maintenance/* gated by
    # this token as X-Internal-Token, same pattern as Module 33's own admin
    # router. Called by host-level backup scripts (nfs-freeze-archive.sh),
    # not by any end user.
    internal_admin_token: str = Field(
        "",
        description="Shared secret required as X-Internal-Token on /admin/maintenance/*. "
                     "Empty means those routes fail closed rather than silently unauthenticated."
    )

    # Agent Protocol Service
    agent_protocol_url: str = Field(
        "http://raptor-agent-protocol:8030",
        description="Base URL of the agent protocol service."
    )

    # Memory Service (Module 26)
    memory_service_url: str = Field(
        "http://raptor-memory-service:8026",
        description="Base URL of the memory service."
    )

    # Personal DB Service (module 25)
    personal_db_url: str = Field(
        "http://raptor-personal-db-service:8000",
        description="Base URL of the personal DB service (module 25). Internal-only — "
                    "it trusts the X-User-ID this gateway sends, so it must not be exposed."
    )

    # Face embedding (module 11's video-face-detection sync API, dev plan v1.4 §4.5)
    face_embed_url: str = Field(
        "http://raptor-video-face-detection:8033",
        description="Base URL of module 11's face-detection sync API (POST /embed-face), "
                    "used to embed a person's seed photo."
    )

    # MCP Server (module 27) — MCP Streamable HTTP transport
    mcp_server_url: str = Field(
        "http://raptor-mcp-server:8027",
        description="Base URL of the MCP server (module 27).",
    )

    inference_service_url: str = Field(
        "http://raptor-inference-service:8035",
        description="Module 35 internal address, proxied at /api/0.5/inference.",
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
    # GB-19 / VIE01-379: more routes screened by GuardrailMiddleware (input only),
    # and how. Both only matter while GR_ENABLED is on.
    guardrail_mode: Literal["audit", "strict"] = Field(
        "audit",
        validation_alias=AliasChoices("GATEWAY_GUARDRAIL_MODE"),
        description="For the guardrail_routes below only. audit (default): check in a background "
                    "task, never wait or block, log unsafe content as WARNING. strict: wait for the "
                    "check and answer 422 content_policy_violation on a block. /chat/chat, "
                    "/chat/completions and /a2a/query always block, whatever this is.",
    )
    guardrail_routes: dict[str, str] = Field(
        {
            "/search/hybrid": "query",
            "/search/vector": "query",
            "/search/bm25": "query",
            "/docagent/process": "options.question",
        },
        validation_alias=AliasChoices("GATEWAY_GUARDRAIL_ROUTES"),
        description="{path_prefix: body_field} for GuardrailMiddleware's input-only check. Prefixes "
                    "are relative to /api/{api_version}; only POST requests are checked. The field is "
                    "a JSON-body or form field; 'options.question' reads `question` from the JSON "
                    "string in the `options` form field (DocAgent). /docagent/process is checked "
                    "only when task=qa. Set as JSON in GATEWAY_GUARDRAIL_ROUTES.",
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
        "gemma4:12b",
        description="Model name used for intent classification in query orchestrator.",
    )
    answer_model: str = Field(
        "gemma4:12b",
        description="Model name used for answer generation in query orchestrator.",
    )
    smolagents_model: str = Field(
        "ollama/gemma4:12b",
        description="LiteLLM model ID for smolagents (e.g. ollama/gemma4:26b).",
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
    doc_agent_url: str = Field(
        "http://raptor-doc-processing-agent:8290",
        description="Base URL of the DocAgent service (module 29).",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    logging.getLogger(__name__).debug("Application settings initialised")
    return settings
