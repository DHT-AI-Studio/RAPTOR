"""Bounded audio upload and decode; accepts WAV, MP3 and FLAC."""
from __future__ import annotations

import asyncio
import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import UploadFile

from app.errors import ServiceError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")


def language_code(language: str | None) -> str | None:
    """Validate a supported BCP-47 shape and return the STT primary subtag."""
    if language is None:
        return None
    if not _LANGUAGE.fullmatch(language):
        raise ServiceError(422, "unsupported_language", language=language)
    return language.split("-")[0].lower()


@dataclass(frozen=True)
class Audio:
    """Mono 16 kHz float32 samples plus original encoded bytes for cache identity."""

    samples: np.ndarray
    original: bytes

    @property
    def duration(self) -> float:
        return len(self.samples) / 16000

    def wav(self) -> bytes:
        """Encode normalized audio for local model inputs."""
        buffer = io.BytesIO()
        sf.write(buffer, self.samples, 16000, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


def decode(data: bytes, max_seconds: float) -> Audio:
    """Use FFmpeg with a time/output bound; corrupt audio never reaches a model."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", "pipe:0", "-t", str(max_seconds + 1),
             "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
            input=data, capture_output=True, timeout=30, check=True,
        )
        samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    except FileNotFoundError as exc:
        raise ServiceError(503, "audio_decoder_unavailable") from exc
    except (subprocess.SubprocessError, ValueError) as exc:
        raise ServiceError(400, "invalid_audio") from exc
    audio = Audio(samples, data)
    if not len(samples) or not np.isfinite(samples).all():
        raise ServiceError(400, "invalid_audio")
    if audio.duration > max_seconds:
        raise ServiceError(400, "invalid_audio_duration", max_seconds=max_seconds)
    return audio


async def read_audio(upload: UploadFile, max_seconds: float = 600) -> Audio:
    """Bound the upload before decoding; extension is validated independently of MIME."""
    if Path(upload.filename or "").suffix.lower() not in {".wav", ".mp3", ".flac"}:
        raise ServiceError(400, "unsupported_audio_format")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ServiceError(413, "audio_too_large", max_bytes=MAX_UPLOAD_BYTES)
    return await asyncio.to_thread(decode, data, max_seconds)
