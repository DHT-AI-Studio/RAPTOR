# 21-agent-protocol/app/core/config.py
"""
Module 21 runtime configuration.
Guardrail settings added for GB-21.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Guardrail (GB-21) ─────────────────────────────────────────────────────
    guardrail_url: str = ""
    # Default: enabled whenever GUARDRAIL_URL is set; can be forced off with =false
    a2a_guardrail_enabled: bool = True

    @property
    def guardrail_active(self) -> bool:
        """True only when both the URL is configured AND the feature flag is on."""
        return bool(self.guardrail_url) and self.a2a_guardrail_enabled


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
