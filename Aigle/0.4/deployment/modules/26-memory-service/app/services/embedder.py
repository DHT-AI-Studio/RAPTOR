"""Module 07 integration: ASR transcription."""
from __future__ import annotations

import httpx
from fastapi import HTTPException, status

from core.config import settings


async def transcribe(asset_path: str, language: str = "zh") -> str:
    """Call Module 07 Whisper ASR to transcribe an audio/video file.

    asset_path is the file path accessible to Module 07 (e.g. NFS mount path).
    Returns the transcribed text string.
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.module07_url.rstrip('/')}/inference/infer",
                json={
                    "task": "asr",
                    "engine": "transformers",
                    "model_name": "whisper-large-v3",
                    "data": {"audio": asset_path},
                    "options": {"language": language},
                },
            )
            resp.raise_for_status()
    except httpx.RequestError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Cannot reach Module 07: {exc}",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Module 07 ASR error {exc.response.status_code}: {exc.response.text}",
        ) from exc

    result = resp.json()
    # ASR handler returns {"result": {"text": "...", "metadata": {}}} — "transcription"
    # and a top-level "text" are never set by Module 07, only kept here as a
    # defensive fallback in case that ever changes.
    text = result.get("result", {}).get("text") or result.get("transcription") or result.get("text")
    if not text:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Module 07 ASR returned no transcription text: {result}",
        )
    return text
