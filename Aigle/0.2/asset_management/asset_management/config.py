from datetime import datetime
from pydantic_settings import BaseSettings


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
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_root_password: str
    qdrant_host: str
    qdrant_port: int
    jwt_secret_key: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    lakefs_endpoint: str
    lakefs_access_key: str
    lakefs_secret_key: str
    lakefs_repository: str = "asset-management"
    lakefs_branch: str = "main"
    lakefs_default_retention_days: int = 60
    lakefs_main_branch_retention_days: int = 90
    timezone: str = "Asia/Taipei"
    auto_daily_archive_time: str = "00:00"
    auto_daily_destroy_time: str = "01:00"

    @property
    def mysql_dsn(self) -> str:
        return f"mysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
    
    @property
    def auto_daily_archive_hour_minute(self):
        return parse_time_24h(self.auto_daily_archive_time)

    @property
    def auto_daily_destroy_hour_minute(self):
        return parse_time_24h(self.auto_daily_destroy_time)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
