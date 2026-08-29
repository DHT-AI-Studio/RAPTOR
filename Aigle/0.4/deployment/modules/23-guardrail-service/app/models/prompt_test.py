"""Request model for the raw prompt-test endpoint (app/routers/guardrail.py)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PromptTestRequest(BaseModel):
    prompt: str = Field(..., description="要送給模型的原始 prompt，不做任何修改或包裝")
    model: str = Field(..., description="Ollama 模型名稱，例如 llama-guard3:8b、granite4.1-guardian:8b")
