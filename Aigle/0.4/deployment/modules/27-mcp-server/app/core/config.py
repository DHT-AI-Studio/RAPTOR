from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",          # copy .env.example → .env to activate
        env_prefix="MCP_",
        case_sensitive=False,
        extra="ignore",
    )

    # Server identity
    server_name: str = Field("raptor-mcp")
    server_version: str = Field("1.0.0")

    # Module 13 API Gateway. api_version is shared with Module 13's own
    # GATEWAY_API_VERSION via the root .env's API_VERSION — see
    # deployment/modules/.env.example — so the two never drift apart when
    # deployed together; api_gateway_url below is derived from these two,
    # not a separate literal, so every existing call site (which reads
    # settings.api_gateway_url) is unaffected. Tool/resource call sites pass
    # version-less paths (e.g. "/search/hybrid") appended onto this base
    # (see app/services/raptor_client.py).
    api_gateway_base_url: str = Field("http://raptor-api-gateway:8012")
    api_version: str = Field("0.4")

    @property
    def api_gateway_url(self) -> str:
        return f"{self.api_gateway_base_url.rstrip('/')}/api/{self.api_version}"

    # Module 06 Keycloak
    keycloak_url: str = Field("http://raptor-keycloak:8080")
    realm_name: str = Field("dhtsolution")
    client_id: str = Field("raptor")
    keycloak_username: str = Field("")
    keycloak_password: str = Field("")

    # Module 02 Redis — agent token cache
    redis_url: str = Field("redis://raptor-redis-standalone:6379/3")

    log_level: str = Field("INFO")

    # Transport: "http" (uvicorn) or "stdio" (MCP_TRANSPORT=stdio python -m app)
    transport: str = Field("http")

    # HTTP port (uvicorn, transport="http" only). Must match the Dockerfile's
    # EXPOSE/HEALTHCHECK port (8027) unless that's changed too.
    port: int = Field(8027)

    # RaptorClient (app/services/raptor_client.py) request timeouts, in seconds
    timeout_default: float = Field(30.0, description="Default timeout for calls to Module 13.")
    timeout_upload: float = Field(120.0, description="Timeout for file-upload calls to Module 13.")

    # RaptorClient retry behaviour for 5xx / network errors
    max_attempts: int = Field(3, description="Total attempts (1 initial + retries) before giving up.")
    retry_backoff_seconds: float = Field(1.0, description="Fixed delay between retry attempts.")

    # TokenManager (app/services/token_manager.py)
    token_ttl_seconds: int = Field(1800, description="How long the server's own service-account token is cached.")
    agent_token_ttl_seconds: int = Field(
        86400,
        description="Agent token lifetime / idle-refresh window (also returned as `expires_in` "
                    "by POST /mcp/auth/register — single source of truth, see app/routers/auth.py).",
    )
    refresh_margin_seconds: int = Field(60, description="Refresh a cached token when fewer than this many seconds remain.")

    # Tool-specific timeouts and limits
    timeout_a2a: float = Field(120.0, description="Timeout for A2A tool calls (tools/a2a.py, tools/pipeline.py).")
    timeout_resource_list: float = Field(15.0, description="Timeout for short resource reads (resources/raptor_resources.py).")
    cooccur_limit: int = Field(20, description="Max co-occurring entities returned by graph tools (tools/graph.py).")
    max_upload_bytes: int = Field(50 * 1024 * 1024, description="Max asset upload size accepted by tools/asset.py.")

    # stdio transport (app/stdio_main.py) — stdout is reserved for the MCP
    # protocol, so logs go to a file instead.
    stdio_log_path: str = Field("/tmp/raptor-mcp-stdio.log")

    # Encrypts agent client_secret before it's written to Redis (see
    # app/services/token_manager.py) — Redis must never hold it in
    # plaintext. Required for POST /mcp/auth/register to work; generate
    # with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    secret_encryption_key: str = Field("")


@lru_cache
def get_settings() -> Settings:
    return Settings()
