"""Ollama guard API — safety classification only, no content generation."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.proxy_checker import CheckerUnavailableError
from app.services.proxy_shared import evaluate

_logger = logging.getLogger(__name__)


class OllamaGenerateRequest(BaseModel):
    prompt: str


class GuardResult(BaseModel):
    allowed: bool
    category: str
    confidence: float
    reason: str


router = APIRouter()


@router.post("/api/generate", response_model=GuardResult)
async def guard_generate(body: OllamaGenerateRequest):
    _logger.info("→ prompt | %s", body.prompt)
    try:
        result = await evaluate(body.prompt, "prompt")
    except CheckerUnavailableError:
        return JSONResponse({"error": "安全檢查服務暫時無法使用，請稍後再試。"}, status_code=503)
    return GuardResult(
        allowed=result.allowed,
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
    )
