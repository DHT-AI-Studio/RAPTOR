from datetime import datetime
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


def parse_time_24h(time_str: str):
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        return dt.hour, dt.minute
    except ValueError:
        raise ValueError(f"Invalid 24-hour time format: {time_str}")


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
    lakefs_endpoint: str
    lakefs_access_key: str
    lakefs_secret_key: str
    lakefs_repository: str = "asset-management"
    lakefs_branch: str = "main"
    # lakefs_default_retention_days: int = 60
    # lakefs_main_branch_retention_days: int = 90
    timezone: str = "Asia/Taipei"
    auto_daily_archive_time: str = "00:00"
    auto_daily_destroy_time: str = "01:00"
   
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def auto_daily_archive_hour_minute(self):
        return parse_time_24h(self.auto_daily_archive_time)

    @property
    def auto_daily_destroy_hour_minute(self):
        return parse_time_24h(self.auto_daily_destroy_time)

settings = Settings()
