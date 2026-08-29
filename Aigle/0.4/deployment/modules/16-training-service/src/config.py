from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Redis settings (unified naming: REDIS_HOST / REDIS_PORT / REDIS_PASSWORD)
    redis_host: str = "raptor-redis-standalone"
    redis_port: int = 6379
    redis_password: str = "dht888888"

    @computed_field
    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"

    # MLflow settings (unified naming: MLFLOW_TRACKING_URI)
    mlflow_tracking_uri: str = "http://mlflow:5555"

    @computed_field
    @property
    def mlflow_uri(self) -> str:
        return self.mlflow_tracking_uri

    # GPU settings
    vram_overhead_gb: float = 1.0

    # FastAPI settings
    fastapi_port: int = 8009


# Global settings instance
settings = Settings()
