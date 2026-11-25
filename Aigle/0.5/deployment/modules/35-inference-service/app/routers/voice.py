"""Multipart voice APIs; deployment is private behind the JWT gateway."""
import time
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, Response, UploadFile

from app.errors import ServiceError
from app.metrics import (
    inf_gpu_memory_used_bytes,
    inf_request_duration_seconds,
    inf_request_errors_total,
)
from app.schemas import DiarizeResponse, SpeakersResponse
from app.services.voice.audio import read_audio
from app.services.voice.consent import require_voice_clone_consent
from app.services.voice.speaker_cache import SpeakerCache

router = APIRouter(prefix="/inference/voice", tags=["voice inference"])

def _update_gpu_gauge() -> None:
    """Update inf_gpu_memory_used_bytes from torch.cuda.memory_allocated().
    Silent no-op when torch is absent (text-only profile) or CUDA unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            inf_gpu_memory_used_bytes.set(torch.cuda.memory_allocated())
    except (ImportError, AttributeError, RuntimeError):
        pass

async def _run_voice_task(task: str, model: str, coro):
    """Execute a voice inference coroutine and record duration / error / GPU metrics."""
    t0 = time.perf_counter()
    try:
        result = await coro
        inf_request_duration_seconds.labels(task=task, model=model).observe(
            time.perf_counter() - t0
        )
        _update_gpu_gauge()
        return result
    except ServiceError as exc:
        inf_request_duration_seconds.labels(task=task, model=model).observe(
            time.perf_counter() - t0
        )
        inf_request_errors_total.labels(task=task, error_type=exc.error).inc()
        _update_gpu_gauge()
        raise
    except Exception:
        inf_request_duration_seconds.labels(task=task, model=model).observe(
            time.perf_counter() - t0
        )
        inf_request_errors_total.labels(task=task, error_type="unknown").inc()
        _update_gpu_gauge()
        raise

@router.post("/diarize", summary="Speaker diarization with WhisperX transcription (INF-9)", response_model=DiarizeResponse)
async def diarize(
    request: Request, audio: Annotated[UploadFile, File()], model: str = Form("pyannote-3.1"),
    num_speakers: int | None = Form(None), language: str | None = Form(None),
) -> dict:
    """Return pyannote speaker attribution with WhisperX aligned transcription."""
    decoded = await read_audio(audio)
    return await _run_voice_task(
        "diarize", model, request.app.state.diarization.diarize(decoded, model, num_speakers, language)
    )


def _require_user_id(request: Request) -> str:
    """Identity is supplied by Module 13 from the verified JWT, never a form field."""
    user = request.headers.get("x-user-id")
    if not user:
        raise ServiceError(401, "missing_user_identity")
    return user


@router.post("/stt", summary="Speech-to-text (INF-10) -- forwards to Module 34's loaded Whisper model")
async def stt(request: Request, audio: Annotated[UploadFile, File()], language: str | None = Form(None)) -> dict:
    """VIE01-316: no local model -- Module 34 already has one loaded for live audio sessions
    (see app/services/module34_client.py)."""
    user_id = _require_user_id(request)
    decoded = await read_audio(audio)
    return await _run_voice_task(
        "stt", "module34-whisper", request.app.state.module34.transcribe(decoded.wav(), language, user_id)
    )


@router.post(
    "/tts",
    summary="Text-to-speech (INF-11) -- forwards to Module 34's loaded TTS model",
    responses={200: {"content": {"audio/wav": {}}, "description": "Synthesized WAV audio"}},
)
async def tts(request: Request, text: str = Form(...), language: str | None = Form(None)) -> Response:
    """VIE01-317: no local model -- same reasoning as /stt above."""
    user_id = _require_user_id(request)
    data = await _run_voice_task("tts", "module34-tts", request.app.state.module34.synthesize(text, language, user_id))
    return Response(data, media_type="audio/wav")


def speaker_cache(request: Request) -> SpeakerCache:
    user = _require_user_id(request)
    settings = request.app.state.settings
    return SpeakerCache(request.app.state.voice_redis, user,
                        settings.speaker_cache_ttl_s, settings.speaker_cache_max)


@router.post(
    "/clone",
    summary="Voice clone synthesis — XTTS-v2 / Chatterbox Multilingual V3 (INF-19)",
    responses={200: {"content": {"audio/wav": {}}, "description": "Synthesized WAV audio; X-Speaker-Id header contains the stable speaker ID, X-Watermarked whether the backend embeds an audio watermark"}},
)
async def clone(
    request: Request, reference_audio: Annotated[UploadFile, File()], text: str = Form(...),
    language: str = Form(...), speaker_name: str | None = Form(None), model: str | None = Form(None),
) -> Response:
    """Synthesize a cloned voice with stable, user-scoped conditioning cache.

    GD-11: needs the caller's voice_clone consent -- checked before the
    reference audio is read, so nothing biometric is processed without it."""
    cloning = request.app.state.cloning
    if cloning.disabled_reason is not None:
        # Early, cheap, no I/O -- preserves 403-before-everything (even before
        # the resolver, which would otherwise 400/422 first once this is disabled,
        # since builtin_model_entries() returns [] while disabled).
        raise ServiceError(403, "clone_disabled", reason=cloning.disabled_reason)
    cache = speaker_cache(request)
    # speaker_cache() has already answered 401 when X-User-ID is missing.
    await require_voice_clone_consent(request.app.state.http, request.app.state.settings,
                                      request.headers["x-user-id"])
    audio = await read_audio(reference_audio, max_seconds=60)
    entry = await request.app.state.resolver.resolve("clone", model)
    data, speaker_id = await _run_voice_task(
        "clone", entry.name, cloning.clone(audio, text, language, cache, entry.name, speaker_name)
    )
    return Response(data, media_type="audio/wav",
                     headers={"X-Speaker-Id": speaker_id, "X-Watermarked": str(entry.watermarked).lower()})


@router.get("/clone/speakers", summary="List caller's cached speaker embeddings", response_model=SpeakersResponse)
async def speakers(request: Request) -> dict:
    """List the caller's cached speakers and remaining TTL."""
    return {"speakers": await speaker_cache(request).list()}


@router.delete("/clone/speakers/{speaker_id}", status_code=204, summary="Evict a caller-owned speaker embedding")
async def delete_speaker(request: Request, speaker_id: str) -> Response:
    """Evict a caller-owned speaker embedding."""
    await speaker_cache(request).delete(speaker_id)
    return Response(status_code=204)
