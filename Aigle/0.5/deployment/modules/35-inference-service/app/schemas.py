"""Request bodies for the text inference routes (INF-5..8) and model defaults (INF-3)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import DEFAULT_TASKS

# BCP-47 language tag, e.g. "en", "zh-TW", "zh-Hant-TW".
_BCP47 = r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranslateOptions(_Strict):
    target_language: str = Field(..., pattern=_BCP47, description="BCP-47, e.g. zh-TW")
    source_language: str | None = Field(None, pattern=_BCP47, description="Omit to auto-detect")
    formality: Literal["formal", "casual"] | None = None


class SummarizeOptions(_Strict):
    style: Literal["bullet", "paragraph", "headline"] = "bullet"
    max_sentences: int | None = Field(None, ge=1, description="Max bullets (bullet) or sentences (paragraph)")
    language: str | None = Field(None, pattern=_BCP47, description="Omit to keep the source language")


class CompleteOptions(_Strict):
    system_prompt: str | None = None
    max_tokens: int = Field(1024, ge=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stop_sequences: list[str] | None = None

    @field_validator("stop_sequences")
    @classmethod
    def _no_empty_stop(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and any(s == "" for s in v):
            raise ValueError("stop_sequences must not contain empty strings")
        return v


class TransformOptions(_Strict):
    instruction: str = Field(..., min_length=1, description="Free-text rewrite instruction")
    preserve_structure: bool = False

    @field_validator("instruction")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("instruction must not be blank")
        return v


class _TextRequest(_Strict):
    model: str | None = Field(None, description="Overrides the task default for this request only")
    input: str = Field(..., min_length=1)


class TranslateRequest(_TextRequest):
    options: TranslateOptions


class SummarizeRequest(_TextRequest):
    options: SummarizeOptions = Field(default_factory=SummarizeOptions)


class CompleteRequest(_TextRequest):
    options: CompleteOptions = Field(default_factory=CompleteOptions)


class TransformRequest(_TextRequest):
    options: TransformOptions


class SetDefaultRequest(_Strict):
    task: Literal[DEFAULT_TASKS]  # type: ignore[valid-type]
    model: str = Field(..., min_length=1)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TextResponse(BaseModel):
    """Shared response body for all text inference endpoints."""
    output: str
    latency_ms: float


class VoiceStatusDetail(BaseModel):
    diarize_loaded: bool
    clone_loaded: bool
    clone_configured_backends: list[str]


class HealthResponse(BaseModel):
    """GET /health response."""
    status: Literal["ok", "degraded"]
    module: int
    gpu_available: bool
    gpu_name: str | None
    module07_reachable: bool
    module34_reachable: bool
    loaded_voice_models: list[str]
    voice: VoiceStatusDetail
    warnings: list[str]


class DiarizeSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float


class DiarizeResponse(BaseModel):
    """POST /inference/voice/diarize response."""
    speakers: list[str]
    segments: list[DiarizeSegment]
    latency_ms: float


class SpeakerEntry(BaseModel):
    id: str
    ttl_remaining_s: int


class SpeakersResponse(BaseModel):
    """GET /inference/voice/clone/speakers response."""
    speakers: list[SpeakerEntry]


class GdprDeleteResponse(BaseModel):
    """DELETE /internal/gdpr/user/{user_id} response."""
    user_id: str
    speaker_embeddings_deleted: int


class GdprSummaryResponse(BaseModel):
    """GET /internal/gdpr/user/{user_id}/summary response."""
    user_id: str
    speaker_embeddings: int
