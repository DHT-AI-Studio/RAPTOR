# src/api/tts_api.py
"""
TTS endpoint — POST /inference/tts
Delegates to VibeVoice-1.5B via the inference_manager.
"""

import logging
import time
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..inference.manager import inference_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["TTS"])

TTS_ENGINE     = "vibevoice"
TTS_MODEL_NAME = "microsoft/VibeVoice-1.5B"


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


@router.post("/tts", response_model=TTSResponse, summary="Text-to-Speech synthesis")
def synthesize(request: TTSRequest):
    """
    Synthesise speech from text using microsoft/VibeVoice-1.5B.
    Returns base64-encoded WAV audio.
    """
    start = time.time()
    try:
        result = inference_manager.infer(
            task="tts",
            engine=TTS_ENGINE,
            model_name=TTS_MODEL_NAME,
            data={
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
