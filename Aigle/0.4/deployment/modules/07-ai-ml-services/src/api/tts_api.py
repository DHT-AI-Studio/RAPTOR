# src/api/tts_api.py
"""
TTS endpoint — POST /inference/tts

LEGACY: This endpoint predates the v3 spec-driven InferenceService and still
calls VibeVoice directly via the deprecated `VibeVoiceEngine`. It is kept only
because module 09 (`audio-processing`) still consumes it. The proper migration
path is:
  1. Register VibeVoice-1.5B in MLflow with `custom_handler="...:VibeVoiceTTSHandler"`.
  2. Delete this file and `inference/engines/vibevoice_engine.py`.
  3. Update module 09 to call POST /inference/infer with task="tts".
"""

import logging
import threading
import time
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["TTS (Legacy)"])

TTS_MODEL_NAME = "microsoft/VibeVoice-1.5B"

# Module-level singleton with lazy init — keeps import cheap and avoids loading
# torch/transformers at FastAPI startup.
_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            from ..inference.engines.vibevoice_engine import VibeVoiceEngine
            _engine = VibeVoiceEngine()
    return _engine


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise")
    voice: str = Field("default", description="Voice identifier (model-dependent)")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech rate multiplier")
    output_format: Literal["wav", "mp3"] = Field("wav", description="Output audio format")


class TTSResponse(BaseModel):
    audio_base64: str
    duration_seconds: float
    format: str
    processing_time: float


@router.post("/tts", response_model=TTSResponse, summary="Text-to-Speech synthesis (legacy)")
def synthesize(request: TTSRequest):
    """
    Synthesise speech from text using microsoft/VibeVoice-1.5B.
    Returns base64-encoded audio.

    NOTE: This is a legacy endpoint kept for module 09 compatibility. It does
    not go through the v3 spec-driven inference pipeline.
    """
    start = time.time()
    try:
        engine = _get_engine()
        model = engine.load_model(TTS_MODEL_NAME, task="tts")
        result = engine.infer(
            model,
            inputs={
                "text":  request.text,
                "voice": request.voice,
                "speed": request.speed,
            },
            options={
                "task":          "tts",
                "output_format": request.output_format,
                "speed":         request.speed,
            },
        )
    except Exception as exc:
        logger.error(f"TTS synthesis failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return TTSResponse(
        audio_base64=result.get("audio_base64", ""),
        duration_seconds=result.get("duration_seconds", 0.0),
        format=result.get("format", request.output_format),
        processing_time=time.time() - start,
    )
