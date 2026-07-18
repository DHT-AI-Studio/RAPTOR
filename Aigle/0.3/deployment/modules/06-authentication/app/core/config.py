from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    KEYCLOAK_URL: str = "http://keycloak:8080"
    KC_BOOTSTRAP_ADMIN_USERNAME: str = "admin"
    KC_BOOTSTRAP_ADMIN_PASSWORD: str = ""
    PERMANENT_MASTER_ADMIN_USER: str = ""
    PERMANENT_ADMIN_PASSWORD: str = ""
    KEYCLOAK_REALM: str = "dhtsolution"
    KEYCLOAK_CLIENT_ID: str = "raptor"
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""
    SMTP_TO_EMAIL: str = ""


settings = Settings()
