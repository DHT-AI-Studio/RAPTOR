from datetime import datetime
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


def parse_time_24h(time_str: str):
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        return dt.hour, dt.minute
    except ValueError:
        raise ValueError(f"Invalid 24-hour time format: {time_str}")


def _parse_duration_seconds(s: str) -> int:
    """Parse LakeFS duration string ('20m', '1h', '90s') to seconds."""
    s = s.strip()
    if s.endswith('h'):
        return int(s[:-1]) * 3600
    if s.endswith('m'):
        return int(s[:-1]) * 60
    if s.endswith('s'):
        return int(s[:-1])
    return int(s)


class Settings(BaseSettings):
    s3_endpoint: str
    s3_public_url: str
    aws_access_key: str
    aws_secret_key: str
    s3_bucket: str
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str = "raptor"
    opensearch_host: str
    opensearch_port: int = 9200
    opensearch_user: str
    opensearch_password: str
    opensearch_index: str = "hybrid_index"
    opensearch_verify_certs: bool = False
    lakefs_endpoint: str
    lakefs_access_key: str
    lakefs_secret_key: str
    lakefs_repository: str = "asset-management"
    lakefs_branch: str = "main"
    lakefs_pre_signed_expiry: str = "20m"
    # lakefs_default_retention_days: int = 60
    # lakefs_main_branch_retention_days: int = 90
    redis_host: str = "raptor-redis-standalone"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    presign_url_cache_enabled: bool = True
    timezone: str = "Asia/Taipei"
    auto_daily_archive_time: str = "00:00"
    auto_daily_destroy_time: str = "01:00"
    graph_service_url: str = ""
   
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def presign_url_ttl(self) -> int:
        """Redis cache TTL = 75% of LakeFS presigned URL expiry, so cached URLs never outlive LakeFS signatures."""
        return int(_parse_duration_seconds(self.lakefs_pre_signed_expiry) * 0.75)

    @property
    def auto_daily_archive_hour_minute(self):
        return parse_time_24h(self.auto_daily_archive_time)

    @property
    def auto_daily_destroy_hour_minute(self):
        return parse_time_24h(self.auto_daily_destroy_time)

settings = Settings()
