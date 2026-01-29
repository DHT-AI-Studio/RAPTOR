"""Application configuration and service registry utilities."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import yaml
from pydantic import BaseModel, Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseModel):
    """Configuration for a downstream service."""

    name: str = Field(..., description="Human-readable identifier for the service.")
    url: HttpUrl = Field(..., description="Base URL of the downstream service.")
    prefix: str = Field(..., description="URL path prefix routed to this service, e.g. /service-a.")
    auth_required: bool = Field(True, description="Whether JWT authentication is required for this service.")
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional override for request timeout in seconds for this service.",
    )

    def normalised_prefix(self) -> str:
        """Return the prefix with a leading slash and no trailing slash."""

        prefix = self.prefix.strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        if len(prefix) > 1 and prefix.endswith("/"):
            prefix = prefix[:-1]
        return prefix


class Settings(BaseSettings):
    """Global application settings loaded from the environment."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="GATEWAY_", case_sensitive=False)

    log_level: str = Field("INFO", description="Logging level for the gateway.")
    services_config_path: Path = Field(
        Path("services.yaml"),
        description="Filesystem path to the downstream service configuration file.",
    )
    # jwt_secret_key: str = Field(..., description="Secret key for signing and verifying JWTs (HS256).")
    # jwt_algorithm: str = Field("HS256", description="JWT signing algorithm (HS256 for symmetric key).")
    # jwt_audience: Optional[str] = Field(
    #     default=None,
    #     description="Optional expected audience claim in JWTs. If omitted the claim is not validated.",
    # )
    request_timeout: float = Field(
        10.0,
        gt=0,
        description="Default timeout (in seconds) for proxy requests to downstream services.",
    )
    object_storage_url: HttpUrl = Field(
        ...,
        description="Base URL of the downstream object storage service.",
    )

    # ONLY for DEBUG and TESTING, if you want to DEPLOY PLEASE REMOVE "Optional"
    kafka_bootstrap_servers: List[str] = Field(..., description="List of Kafka bootstrap servers.")
    kafka_topics: Dict[str, str] = Field(..., description="Mapping of file types to Kafka topics.")
    
    gateway_db_host: str = Field(
        "postgres",
        description="Hostname for the PostgreSQL database storing gateway users.",
    )
    gateway_db_port: int = Field(
        5432,
        gt=0,
        description="Port for the PostgreSQL database storing gateway users.",
    )
    gateway_db_user: str = Field(
        "admin",
        description="Database user for the PostgreSQL gateway user store.",
    )
    gateway_db_password: str = Field(
        "admin123",
        description="Database password for the PostgreSQL gateway user store.",
        repr=False,
    )
    gateway_db_name: str = Field(
        "mydb",
        description="Database name for the PostgreSQL gateway user store.",
    )
    gateway_db_pool_min_size: int = Field(
        1,
        ge=1,
        description="Minimum number of connections for the gateway PostgreSQL pool.",
    )
    gateway_db_pool_max_size: int = Field(
        5,
        ge=1,
        description="Maximum number of connections for the gateway PostgreSQL pool.",
    )
    redis_host: str = Field("192.168.157.165", description="Hostname for the Redis cache.")
    redis_port: int = Field(6391, description="Port for the Redis cache.")
    redis_db: int = Field(0, description="Database index for the Redis cache.")
    qdrant_host: str = Field(
        "192.168.157.165",
        description="Hostname for the Qdrant vector database.",
    )
    
    # 聊天服務配置
    api_base_url: str = Field(
        "http://localhost:8012",
        description="Base URL of the API gateway for chat service.",
    )
    chat_model_name: str = Field(
        "Qwen3-14B-AWQ",
        description="LLM model name for chat service.",
    )
    chat_temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Temperature parameter for LLM.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (optional).",
        repr=False,
    )
    llm_base_url: str = Field(
        "http://192.168.157.169:8000/v1",
        description="Base URL for LLM API (Ollama).",
    )
    chat_memory_context_window: int = Field(
        5,
        ge=1,
        description="Context window size for chat memory.",
    )
    chat_memory_ttl: int = Field(
        3600,
        gt=0,
        description="TTL (seconds) for chat memory in Redis.",
    )
    
    # 效能與流量治理配置
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
    storage_token_cache_ttl: int = Field(
        3600,
        ge=60,
        description="TTL (seconds) for cached object storage access tokens.",
    )
    batch_upload_concurrency: int = Field(
        4,
        ge=1,
        le=16,
        description="Max concurrent uploads for batch operations.",
    )
    batch_upload_chunk_size: int = Field(
        8192,
        ge=1024,
        description="Stream chunk size (bytes) for file uploads.",
    )
    rate_limit_per_second: float = Field(
        100.0,
        gt=0,
        description="Global rate limit per second.",
    )
    rate_limit_per_minute: float = Field(
        5000.0,
        gt=0,
        description="Global rate limit per minute.",
    )
    enable_tracing: bool = Field(
        False,
        description="Enable OpenTelemetry tracing.",
    )
    jaeger_host: str = Field(
        "localhost",
        description="Jaeger agent host for tracing.",
    )
    jaeger_port: int = Field(
        6831,
        gt=0,
        description="Jaeger agent port for tracing.",
    )


    @property
    def gateway_db_dsn(self) -> str:
        """Connection string for the gateway user PostgreSQL database."""

        user = quote_plus(self.gateway_db_user)
        password = quote_plus(self.gateway_db_password)
        host = self.gateway_db_host
        port = self.gateway_db_port
        database = self.gateway_db_name
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    @model_validator(mode="after")
    def _validate_db_pool(self) -> "Settings":
        if self.gateway_db_pool_max_size < self.gateway_db_pool_min_size:
            raise ValueError("gateway_db_pool_max_size must be >= gateway_db_pool_min_size")
        return self


class ServiceRegistry:
    """Registry that maps request prefixes to downstream service configurations."""

    def __init__(self, configuration_path: Path) -> None:
        self._configuration_path = configuration_path
        self._services: List[ServiceConfig] = []
        self._prefix_map: Dict[str, ServiceConfig] = {}
        self.reload()

    def reload(self) -> None:
        """Reload the service configuration from disk."""

        if not self._configuration_path.exists():
            raise FileNotFoundError(f"Service configuration file not found: {self._configuration_path}")

        try:
            raw = self._configuration_path.read_text(encoding="utf-8")
            payload = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - configuration errors should surface early
            raise ValueError("Invalid YAML in service configuration file") from exc

        services_raw: Iterable[dict] = payload.get("services", [])
        services: List[ServiceConfig] = []
        prefixes: Dict[str, ServiceConfig] = {}
        for service_dict in services_raw:
            service = ServiceConfig.model_validate(service_dict)
            prefix = service.normalised_prefix()
            if prefix in prefixes:
                raise ValueError(f"Duplicate prefix '{prefix}' defined in service configuration")
            prefixes[prefix] = service
            services.append(service)

        # Sort by descending prefix length to support nested prefixes
        services.sort(key=lambda svc: len(svc.normalised_prefix()), reverse=True)

        self._services = services
        self._prefix_map = prefixes
        logging.getLogger(__name__).info(
            "Loaded service configuration", extra={"service_count": len(services)}
        )

    def find_service(self, request_path: str) -> Optional[ServiceConfig]:
        """Find the best matching service for the given request path."""

        if not request_path.startswith("/"):
            request_path = f"/{request_path}"

        for service in self._services:
            prefix = service.normalised_prefix()
            if request_path == prefix or request_path.startswith(prefix + "/"):
                return service
        return None

    @property
    def services(self) -> List[ServiceConfig]:
        """Return the current list of registered services."""

        return list(self._services)

    @property
    def configuration_path(self) -> Path:
        return self._configuration_path

@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    logging.getLogger(__name__).debug(
        "Application settings initialised", extra={"services_config_path": str(settings.services_config_path)}
    )
    return settings
