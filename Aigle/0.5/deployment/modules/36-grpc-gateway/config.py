"""Application configuration for the gRPC gateway."""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global gRPC gateway settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GRPC_GATEWAY_",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = Field("INFO", description="Logging level for the gRPC gateway.")
    grpc_port: int = Field(8825, description="Port the gRPC server binds to.")
    metrics_port: int = Field(9825, description="Port the Prometheus /metrics HTTP endpoint binds to.")

    keycloak_url: str = Field(
        "http://raptor-keycloak:8080",
        description="Internal Keycloak base URL used to fetch JWKS for Bearer-token verification "
                    "(IdentityInterceptor). raptor-keycloak is the network alias 06-authentication's "
                    "docker-compose.yml gives the Keycloak container on the shared external `raptor` "
                    "network (hostname/aliases) -- the bare compose service name `keycloak` only "
                    "resolves inside module 06's own compose project, not across modules. A token's own "
                    "iss claim may also carry an externally-reachable host unreachable from inside "
                    "Docker (same issue module 13's app/core/security.py handles) -- JWKS is always "
                    "fetched from this URL instead, never from iss itself.",
    )
    jwt_expected_realm: str = Field(
        "dhtsolution",
        description="Keycloak realm a Bearer token's iss claim must name. Rejects a token issued by a "
                    "different realm on the same Keycloak instance (e.g. master) even though its "
                    "signature would otherwise verify.",
    )
    jwt_audience: str = Field(
        "account",
        description="Expected aud claim; empty = not checked. \"account\" is every client's aud "
                    "today (no audience-mapper configured in Keycloak), so this doesn't yet separate "
                    "a token meant for this service from one meant for another -- real per-service "
                    "separation needs an audience-mapper added first.",
    )
    allow_insecure_dev_fallback: bool = Field(
        False,
        description="DEV ONLY, NEVER true in production. When true, a Bearer token that fails "
                    "verification (bad signature, wrong realm/audience, expired, unreachable "
                    "Keycloak) falls back to the fixed dev-user identity instead of UNAUTHENTICATED "
                    "-- restores the pre-VIE01-383 behaviour, for local development without a real "
                    "Keycloak. True in any other environment defeats this ticket's entire fix.",
    )
    auth_service_url: str = Field(
        "http://raptor-keycloak-api:8800",
        description="Base URL of the auth service (module 06) -- same UMA permission check REST "
                    "already uses (see module 13's check_uma_permission); AuthorizationInterceptor "
                    "calls its /auth/permission with the RPC's full method name as request_path.",
    )
    api_keys: str = Field(
        "",
        description="Comma-separated list of accepted API keys for the x-api-key credential "
                    "(AuthenticationInterceptor._valid_api_key). Empty means no key is ever accepted.",
    )

    personal_db_url: str = Field(
        "http://raptor-personal-db-service:8000",
        description="Base URL of the personal DB service (module 25), backing SearchByPerson/StreamSearchByPerson.",
    )
    asset_management_url: str = Field(
        "http://raptor-asset-management:8000",
        description="Base URL of the asset management service (module 04) — presigned URLs for search results.",
    )
    hybrid_search_url: str = Field(
        "http://raptor-hybridsearch-api:8000",
        description="Base URL of the (deprecated) module 17 hybrid search API — HybridSearchService only uses this class for URL resolution here, never calls module 17 itself.",
    )
    request_timeout: float = Field(
        10.0, gt=0, description="Default timeout (seconds) for downstream requests — read by StorageService."
    )
    upload_timeout: float = Field(
        300.0,
        gt=0,
        description="Timeout (seconds) for the file-upload request to the asset management service — read by "
                    "StorageService.upload_file, so UploadAsset fails with AttributeError without it. Same default "
                    "as module 13.",
    )

    redis_host: str = Field("raptor-redis-standalone", description="Hostname for the Redis cache (upload idempotency/concurrency).")
    redis_port: int = Field(6379, description="Port for the Redis cache.")
    redis_db: int = Field(0, description="Database index for the Redis cache.")
    redis_password: str = Field("", description="Password for the Redis cache; empty means no auth.")

    upload_idle_timeout_s: float = Field(
        60.0, gt=0, description="Abort an in-progress UploadAsset stream if no chunk arrives within this many seconds."
    )
    max_streams_per_user: int = Field(
        4, gt=0,
        description="Per-user concurrent stream limit, shared across every streaming RPC "
                    "(UploadAsset/DownloadAsset/StreamSearchByPerson/WatchJob/...). Enforced by StreamLimitInterceptor.",
    )

    grpc_secure: bool = Field(
        False, description="Serve gRPC over TLS (grpc.ssl_server_credentials()) instead of plaintext."
    )
    grpc_tls_cert_path: str = Field("", description="Server certificate PEM path; required when grpc_secure=true.")
    grpc_tls_key_path: str = Field("", description="Server private key PEM path; required when grpc_secure=true.")
    grpc_mtls_ca_cert_path: str = Field(
        "",
        description="CA certificate PEM path for client-certificate validation (mTLS). "
                    "Empty -> server certificate only, no client cert required. Only meaningful when grpc_secure=true.",
    )

    kafka_bootstrap_servers: str = Field(
        "", description="Comma-separated Kafka brokers; empty disables event publishing (fails open)."
    )
    kafka_asset_finalized_topic: str = Field("asset-upload-finalized", description="Topic for UploadAsset completion events.")
    kafka_topics: dict[str, str] = Field(
        default_factory=lambda: {
            "document": "document-processing-requests",
            "image": "image-processing-requests",
            "video": "video-processing-requests",
            "audio": "audio-processing-requests",
        },
        description="File-type -> Kafka topic, read by KafkaService.send_processing_request (SubmitJob).",
    )
    watch_job_poll_interval_s: float = Field(
        1.0, gt=0, description="How often WatchJob polls the orchestrator Redis key for a state change."
    )

    ffmpeg_binary: str = Field(
        "ffmpeg", description="Path/name of the FFmpeg executable used by StreamVideoSegment's transcode formats."
    )
    ffprobe_binary: str = Field(
        "ffprobe", description="Path/name of the ffprobe executable used to fill DownloadAsset's duration_sec."
    )
    probe_duration: bool = Field(
        True,
        description="Fill DownloadAsset's MediaHeader.duration_sec for audio/video files by asking ffprobe. "
                    "Off -> duration_sec is always 0 (unknown).",
    )
    probe_duration_timeout_s: float = Field(
        3.0, gt=0,
        description="How long DownloadAsset waits for ffprobe before sending the header with duration_sec 0. "
                    "The probe runs while the storage stream is being opened, so this is the most it can delay the header.",
    )
    ffmpeg_terminate_timeout_s: float = Field(
        5.0, gt=0,
        description="How long to wait after SIGTERM before SIGKILL-ing an FFmpeg subprocess on stream cancellation.",
    )

    analyze_stream_enabled: bool = Field(
        False,
        description="Serve AnalyzeStream. Off by default: no service in Raptor consumes the topic it "
                    "forwards to, so an enabled stream accepts a client's media, publishes it where nothing "
                    "reads it, and ends UNIMPLEMENTED. Turn on once a consumer exists "
                    "(see docs/_internal/OPEN_QUESTION_analyze_stream_results.md).",
    )
    analyze_idle_timeout_s: float = Field(
        120.0, gt=0,
        description="Abort an in-progress AnalyzeStream if no chunk arrives within this many seconds.",
    )
    kafka_analysis_requests_topic: str = Field(
        "media-analysis-requests",
        description="Topic AnalyzeStream forwards each MediaChunk to. NOTE: no consumer anywhere in Raptor "
                    "reads this topic today -- see docs/_internal/OPEN_QUESTION_analyze_stream_results.md.",
    )
    guardrail_enabled: bool = Field(
        True,
        validation_alias=AliasChoices("GR_GRPC_GUARDRAIL_ENABLED", "GRPC_GATEWAY_GUARDRAIL_ENABLED"),
        description="Screen the user text of Search/SearchBm25/SearchVector/SearchByPerson/"
                    "StreamSearchByPerson/Chat with Module 23 before the handler runs "
                    "(GuardrailInterceptor, GB-20). Needs guardrail_url too.",
    )
    guardrail_url: str = Field(
        "",
        validation_alias=AliasChoices("GR_GRPC_GUARDRAIL_URL", "GRPC_GATEWAY_GUARDRAIL_URL", "GUARDRAIL_URL"),
        description="Module 23 base URL. With guardrail_enabled, GuardrailInterceptor calls its "
                    "/guardrail/policy/decide, and grpc_interceptor is registered as screened in the "
                    "GB-22 coverage matrix at startup.",
    )
    guardrail_timeout_ms: int = Field(
        20000, gt=0,
        validation_alias=AliasChoices("GR_GRPC_TIMEOUT_MS", "GRPC_GATEWAY_GUARDRAIL_TIMEOUT_MS"),
        description="How long GuardrailInterceptor waits for Module 23. On timeout (or any other "
                    "failure) the call proceeds unscreened, with a warning logged (fail-open).",
    )

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
