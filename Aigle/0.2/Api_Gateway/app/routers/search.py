"""
API router for handling Qdrant semantic search queries.
Supports searching across multiple data types: video, audio, document, and image.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user
from app.core.config import Settings, get_settings
from app.services.search_service import SearchService

_logger = logging.getLogger(__name__)

router = APIRouter()


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency to get the application's HTTP client."""
    client = getattr(request.app.state, "http_client", None)
    if not client:
        raise RuntimeError("HTTP client is not initialised")
    return client


# ============================================================================
# Request Models
# ============================================================================


class VideoSearchRequest(BaseModel):
    """Request model for video similarity search."""

    query_text: str = Field(..., description="Search query text")
    embedding_type: str = Field(
        default="text", description="Type of embedding: 'text' or 'summary'"
    )
    filename: List[str] = Field(
        default_factory=list, description="List of video filenames to search"
    )
    speaker: List[str] = Field(
        default_factory=list, description="List of speakers to filter by"
    )
    limit: int = Field(default=5, description="Maximum number of results")


class AudioSearchRequest(BaseModel):
    """Request model for audio similarity search."""

    query_text: str = Field(..., description="Search query text")
    embedding_type: str = Field(
        default="text", description="Type of embedding: 'text' or 'summary'"
    )
    filename: List[str] = Field(
        default_factory=list, description="List of audio filenames to search"
    )
    speaker: List[str] = Field(
        default_factory=list, description="List of speakers to filter by"
    )
    limit: int = Field(default=5, description="Maximum number of results")


class DocumentSearchRequest(BaseModel):
    """Request model for document similarity search."""

    query_text: str = Field(..., description="Search query text")
    embedding_type: str = Field(
        default="text", description="Type of embedding: 'text' or 'summary'"
    )
    filename: List[str] = Field(
        default_factory=list, description="List of document filenames to search"
    )
    source: str = Field(default="csv", description="File type (e.g., csv, pdf, docx)")
    limit: int = Field(default=5, description="Maximum number of results")


class ImageSearchRequest(BaseModel):
    """Request model for image similarity search."""

    query_text: str = Field(..., description="Search query text")
    embedding_type: str = Field(
        default="summary", description="Type of embedding: 'text' or 'summary'"
    )
    filename: List[str] = Field(
        default_factory=list, description="List of image filenames to search"
    )
    source: str = Field(default="jpg", description="File extension (e.g., jpg, png)")
    limit: int = Field(default=5, description="Maximum number of results")


class UnifiedSearchRequest(BaseModel):
    """Request model for unified cross-collection search."""

    query_text: str = Field(..., description="Search query text")
    embedding_type: str = Field(
        default="text", description="Type of embedding: 'text' or 'summary'"
    )
    filters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Per-collection filters",
        examples=[
            {
                "video": {"filename": ["lecture.mp4"], "speaker": ["SPEAKER_00"]},
                "document": {"source": "pdf"},
            }
        ],
    )
    limit_per_collection: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum results per collection",
    )
    global_limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum total results after aggregation",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold (0.0-1.0)",
    )


# ============================================================================
# Video Search Endpoints
# ============================================================================


@router.post("/video_search", tags=["search"], status_code=status.HTTP_200_OK)
async def search_video(
    request: Request,
    query: VideoSearchRequest,
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    # current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Search for video similarity using semantic embeddings.

    **Parameters:**
    - `query_text`: Search query (required)
    - `embedding_type`: 'text' or 'summary'
    - `filename`: List of video filenames to search
    - `speaker`: List of speakers to filter by
    - `limit`: Number of results to return (default: 5)

    **Example:**
    ```json
    {
      "query_text": "機器學習",
      "embedding_type": "text",
      "filename": ["MV.mp4"],
      "speaker": ["SPEAKER_00"],
      "limit": 5
    }
    ```

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/search/video_search"
    payload = {
        "query_text": query_text,
        "embedding_type": embedding_type,
        "limit": limit
    }
    if filename:
        payload["filename"] = filename
    if speaker:
        payload["speaker"] = speaker

    response = requests.post(url, json=payload, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    
    qdrant_host = getattr(settings, "qdrant_host", "192.168.157.165")    
    qdrant_port = getattr(settings, "qdrant_port", 8821)
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}/video_search"

    try:
        response = await client.post(
            qdrant_url,
            json=query.model_dump(exclude_unset=True),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        _logger.error(
            f"Video search failed with status {e.response.status_code}: {e.response.text}"
        )
        return {
            "error": "Video search failed",
            "detail": str(e),
            "status_code": e.response.status_code,
        }
    except Exception as e:
        _logger.error(f"Video search error: {str(e)}")
        return {"error": "Internal server error", "detail": str(e)}


# ============================================================================
# Audio Search Endpoints
# ============================================================================


@router.post("/audio_search", tags=["search"], status_code=status.HTTP_200_OK)
async def search_audio(
    request: Request,
    query: AudioSearchRequest,
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    # current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Search for audio similarity using semantic embeddings.

    **Parameters:**
    - `query_text`: Search query (required)
    - `embedding_type`: 'text' or 'summary'
    - `filename`: List of audio filenames to search
    - `speaker`: List of speakers to filter by
    - `limit`: Number of results to return (default: 5)

    **Example:**
    ```json
    {
      "query_text": "機器學習",
      "embedding_type": "text",
      "filename": ["MV.mp4"],
      "speaker": ["SPEAKER_00"],
      "limit": 5
    }
    ```

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/search/audio_search"
    payload = {
        "query_text": query_text,
        "embedding_type": embedding_type,
        "limit": limit
    }
    if filename:
        payload["filename"] = filename
    if speaker:
        payload["speaker"] = speaker

    response = requests.post(url, json=payload, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    qdrant_host = getattr(settings, "qdrant_host", "localhost")
    qdrant_port = getattr(settings, "qdrant_port", 8822)
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}/audio_search"

    try:
        response = await client.post(
            qdrant_url,
            json=query.model_dump(exclude_unset=True),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        _logger.error(
            f"Audio search failed with status {e.response.status_code}: {e.response.text}"
        )
        return {
            "error": "Audio search failed",
            "detail": str(e),
            "status_code": e.response.status_code,
        }
    except Exception as e:
        _logger.error(f"Audio search error: {str(e)}")
        return {"error": "Internal server error", "detail": str(e)}


# ============================================================================
# Document Search Endpoints
# ============================================================================


@router.post("/document_search", tags=["search"], status_code=status.HTTP_200_OK)
async def search_document(
    request: Request,
    query: DocumentSearchRequest,
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    # current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Search for document similarity using semantic embeddings.

    **Parameters:**
    - `query_text`: Search query (required)
    - `embedding_type`: 'text' or 'summary'
    - `filename`: List of document filenames to search
    - `source`: File type (e.g., csv, pdf, docx)
    - `limit`: Number of results to return (default: 5)

    **Example:**
    ```json
    {
      "query_text": "財務報告",
      "embedding_type": "text",
      "filename": ["EF25Y01.csv"],
      "source": "csv",
      "limit": 5
    }
    ```

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/search/document_search"
    payload = {
        "query_text": query_text,
        "embedding_type": embedding_type,
        "limit": limit
    }
    if filename:
        payload["filename"] = filename
    if source:
        payload["source"] = source

    response = requests.post(url, json=payload, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    qdrant_host = getattr(settings, "qdrant_host", "localhost")
    qdrant_port = getattr(settings, "qdrant_port", 8823)
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}/document_search"

    try:
        response = await client.post(
            qdrant_url,
            json=query.model_dump(exclude_unset=True),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        _logger.error(
            f"Document search failed with status {e.response.status_code}: {e.response.text}"
        )
        return {
            "error": "Document search failed",
            "detail": str(e),
            "status_code": e.response.status_code,
        }
    except Exception as e:
        _logger.error(f"Document search error: {str(e)}")
        return {"error": "Internal server error", "detail": str(e)}


# ============================================================================
# Image Search Endpoints
# ============================================================================


@router.post("/image_search", tags=["search"], status_code=status.HTTP_200_OK)
async def search_image(
    request: Request,
    query: ImageSearchRequest,
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    # current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Search for image similarity using semantic embeddings.

    **Parameters:**
    - `query_text`: Search query (required)
    - `embedding_type`: 'text' or 'summary'
    - `filename`: List of image filenames to search
    - `source`: File extension (e.g., jpg, png)
    - `limit`: Number of results to return (default: 5)

    **Example:**
    ```json
    {
      "query_text": "風景照片",
      "embedding_type": "summary",
      "filename": ["發票.jpg"],
      "source": "jpg",
      "limit": 5
    }
    ```

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/search/image_search"
    payload = {
        "query_text": query_text,
        "embedding_type": embedding_type,
        "limit": limit
    }
    if filename:
        payload["filename"] = filename
    if source:
        payload["source"] = source

    response = requests.post(url, json=payload, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    qdrant_host = getattr(settings, "qdrant_host", "localhost")
    qdrant_port = getattr(settings, "qdrant_port", 8824)
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}/image_search"

    try:
        response = await client.post(
            qdrant_url,
            json=query.model_dump(exclude_unset=True),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        _logger.error(
            f"Image search failed with status {e.response.status_code}: {e.response.text}"
        )
        return {
            "error": "Image search failed",
            "detail": str(e),
            "status_code": e.response.status_code,
        }
    except Exception as e:
        _logger.error(f"Image search error: {str(e)}")
        return {"error": "Internal server error", "detail": str(e)}


# ============================================================================
# Unified Search Endpoint
# ============================================================================


@router.post("/unified_search", tags=["search"], status_code=status.HTTP_200_OK)
async def unified_search(
    request: Request,
    query: UnifiedSearchRequest,
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    # current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Unified search across multiple collections (video, audio, document, image).
    
    Performs parallel searches across selected collections and aggregates results
    using the specified strategy.

    **Parameters:**
    - `query_text`: Search query (required)
    - `embedding_type`: 'text' or 'summary' (default: 'text')
    - `filters`: Per-collection filter dictionaries
    - `limit_per_collection`: Number of results per collection (default: 5, max: 50)
    - `global_limit`: Maximum total results after aggregation (max: 100)
    - `score_threshold`: Minimum score threshold (0.0-1.0)

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/search/unified_search"
    payload = {
        "query_text": query_text,
        "embedding_type": embedding_type,
        "limit_per_collection": limit_per_collection
    }
    if filters:
        payload["filters"] = filters
    if global_limit:
        payload["global_limit"] = global_limit
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold

    response = requests.post(url, json=payload, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    qdrant_host = getattr(settings, "qdrant_host", "localhost")

    # Initialize search service
    search_service = SearchService(
        qdrant_host=qdrant_host,
        http_client=client,
    )

    try:
        # Perform unified search
        results = await search_service.unified_search(
            query_text=query.query_text,
            embedding_type=query.embedding_type,
            filters=query.filters,
            limit_per_collection=query.limit_per_collection,
            global_limit=query.global_limit,
            score_threshold=query.score_threshold,
            search_timeout=30.0,
        )

        return results

    except ValueError as e:
        _logger.error(f"Unified search validation error: {str(e)}")
        return {
            "error": "Invalid search parameters",
            "detail": str(e),
            "query": query.query_text,
        }
    except Exception as e:
        _logger.error(f"Unified search error: {str(e)}")
        return {
            "error": "Unified search failed",
            "detail": str(e),
            "query": query.query_text,
        }

