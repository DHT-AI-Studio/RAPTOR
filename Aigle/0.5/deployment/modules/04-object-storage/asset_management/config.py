from datetime import datetime

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


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
    # Module 17 retired -- search_sync.py's Qdrant/OpenSearch clients are
    # commented out, nothing reads these any more. Defaulted (not required)
    # so module 04 doesn't hard-depend on env vars that only existed for a
    # client that no longer runs, matching graph_service_url's own "" default
    # below for the same reason (module 20 retired).
    qdrant_host: str = ""
    qdrant_port: int = 0
    qdrant_collection: str = "raptor"
    opensearch_host: str = ""
    opensearch_port: int = 9200
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_index: str = "hybrid_index"
    opensearch_verify_certs: bool = False
    lakefs_endpoint: str
    lakefs_access_key: str
    lakefs_secret_key: str
    lakefs_repository: str = "asset-management"
    lakefs_branch: str = "main"
    lakefs_pre_signed_expiry: str = "20m"
    # batch_upload()/transact() tunables. lakefs_batch_group_max_* are not
    # caller-facing reject limits -- callers can send a batch of any size --
    # they set when batch_upload() splits internally into more than one
    # LakeFS transaction, so no single transaction's memory footprint grows
    # unboundedly.
    lakefs_tx_max_retries: int = 3
    lakefs_batch_group_max_files: int = 500
    lakefs_batch_group_max_bytes: int = 200 * 1024 * 1024
    # lakefs_default_retention_days: int = 60
    # lakefs_main_branch_retention_days: int = 90
    redis_host: str = "raptor-redis-standalone"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    presign_url_cache_enabled: bool = True
    timezone: str = "Asia/Taipei"
    auto_daily_archive_time: str = "00:00"
    auto_daily_destroy_time: str = "01:00"
    graph_service_url: str = ""
    personal_db_service_url: str = ""
    # MTR-7 / VIE01-306: GPU-isolated thumbnail-extraction sidecar (see
    # thumbnail_worker/) -- this process itself never runs ffmpeg/NVDEC.
    thumbnail_worker_url: str = "http://thumbnail-worker:8034"
    # MTR-8 / VIE01-307: concurrent DB lookups per /scan batch request.
    scan_concurrency: int = 10
   
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

    # GD-1 / VIE01-246: shared secret Module 33 (the GDPR erasure
    # orchestrator) presents as X-Internal-Token on DELETE /internal/gdpr/
    # user/{user_id}. Empty by default so that endpoint 401s until an
    # operator deliberately sets it, rather than silently accepting
    # requests with no token at all.
    gdpr_internal_token: str = ""

settings = Settings()
