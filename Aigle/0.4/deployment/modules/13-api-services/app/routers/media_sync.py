"""Synchronous media analysis proxy — image, document, audio, video.

All four endpoints share the same pattern: receive a file upload, forward it
to the corresponding downstream sync API, and return the result directly.
No Kafka, no indexing — results are returned in the HTTP response.

Routes (all under /api/{version}/sync):
  POST /image    → module 09 image-processing   :8018/analyze
  POST /document → module 11 document-processing:8020/analyze
  POST /audio    → module 08 audio-processing   :8019/transcribe
  POST /video    → module 10 video-processing   :8021/analyze
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_http_client
from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)

_TAG = "Media Sync"

router = APIRouter()


@router.post("/image", tags=[_TAG], status_code=status.HTTP_200_OK, include_in_schema=False)
async def analyze_image(
    file: UploadFile = File(...),
    processing_mode: str = Form("default"),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Synchronous image analysis — returns VLM description + OCR text without indexing.

    Returns:
        - summary: VLM-generated description
        - text: OCR-extracted text
    """
    content = await file.read()
    try:
        resp = await client.post(
            f"{settings.image_sync_url}/analyze",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            data={"processing_mode": processing_mode},
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        _logger.error("Image sync API error: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _logger.error("Image sync API unreachable: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.post("/document", tags=[_TAG], status_code=status.HTTP_200_OK, include_in_schema=False)
async def analyze_document(
    file: UploadFile = File(...),
    processing_mode: str = Form("default"),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Synchronous document analysis — returns chunks + summary without indexing.
    """
    content = await file.read()
    try:
        resp = await client.post(
            f"{settings.document_sync_url}/analyze",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            data={"processing_mode": processing_mode},
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        _logger.error("Document sync API error: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _logger.error("Document sync API unreachable: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.post("/audio", tags=[_TAG], status_code=status.HTTP_200_OK, include_in_schema=False)
async def transcribe_audio(
    file: UploadFile = File(...),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Synchronous audio transcription — returns transcript text without indexing.

    Returns:
        - text: full transcript
        - language: detected language
        - duration: audio duration in seconds
        - segments_count: number of transcript segments
    """
    content = await file.read()
    try:
        resp = await client.post(
            f"{settings.audio_sync_url}/transcribe",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        _logger.error("Audio sync API error: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _logger.error("Audio sync API unreachable: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.post("/video", tags=[_TAG], status_code=status.HTTP_200_OK, include_in_schema=False)
async def analyze_video(
    file: UploadFile = File(...),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Synchronous video frame analysis — returns VLM event summary without indexing.

    Returns:
        - filename: original filename
        - summary: VLM-generated event description
    """
    content = await file.read()
    try:
        resp = await client.post(
            f"{settings.video_frame_sync_url}/analyze",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        _logger.error("Video frame sync API error: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _logger.error("Video frame sync API unreachable: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
