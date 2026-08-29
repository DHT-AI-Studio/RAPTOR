"""Chunk index request model (PA-4, unified `Chunk`).

One `ChunkIndexRequest` covers all four media types (documents|videos|images|
audios). Media-specific fields are optional and only sent when applicable.
`sparse_*` is OPTIONAL — the 0.3 pipeline produces dense only.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.config import settings


class ChunkIndexRequest(BaseModel):
    # --- identity / common (required) ---
    chunk_id: str
    type: str                          # documents | videos | images | audios
    embedding_type: str                # text | summary
    # Optional: if omitted, the service embeds `text`/`summary` with its own BGE-M3
    # (production pipeline / Module 07 may still pass a pre-computed vector).
    embedding: Optional[List[float]] = Field(default=None,
                                             min_length=settings.vector_dim, max_length=settings.vector_dim)

    # --- common (optional) ---
    text: Optional[str] = None
    summary: Optional[str] = None
    filename: Optional[str] = None
    source: Optional[str] = None
    asset_path: Optional[str] = None
    version_id: Optional[str] = None
    upload_time: Optional[str] = None
    status: str = "active"

    # --- sparse (OPTIONAL; kept for future hybrid) ---
    sparse_indices: Optional[List[int]] = None
    sparse_weights: Optional[List[float]] = None

    # --- temporal (video / audio segments) ---
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    speaker: Optional[str] = None

    # --- video-specific ---
    contextual_text: Optional[str] = None
    asr_text: Optional[str] = None
    lvlm_desc: Optional[str] = None
    ocr_text: Optional[str] = None

    # --- audio-specific ---
    audio_labels: Optional[List[str]] = None

    # --- chunk_index: this chunk's sequence position within its source asset.
    # One property for every media type -- video/audio events upstream call
    # it moment_index, document events call it chunk_index; kafka_consumer.py
    # normalizes both into this single field (previously two separate
    # properties, moment_index here and this chunk_index, which for a
    # document entry ended up holding the exact same value twice). ---
    chunk_index: Optional[int] = None
    page_numbers: Optional[List[int]] = None
    section_heading: Optional[str] = None
    element_types: Optional[List[str]] = None
    char_count: Optional[int] = None


class CloneVersionRequest(BaseModel):
    """Module 04's content-dedup optimization (client.py: identical MD5 found
    at a different, archived asset_path) -- clone an already-indexed
    version's Source/Chunks/TemporalFacts onto a new version_id/asset_path
    instead of re-running the whole analysis pipeline. Always within the
    same branch/database (module 04 only triggers this for a "Global Archive
    Check within the same branch")."""
    source_version_id: str
    source_asset_path: str
    target_version_id: str
    target_asset_path: str


class SourceSummaryIndexRequest(BaseModel):
    """A whole-asset summary (video/audio/image/document alike -- every media
    type sends one embedding_type="summary" entry, previously indexed as a
    fake Chunk with no start_sec/end_sec). Written onto the Source vertex
    itself now, matching Module 20's Source.summary (neo4j_writer.py:61)
    instead of faking a moment/chunk that doesn't semantically exist."""
    version_id: str
    summary: str
    embedding: Optional[List[float]] = Field(default=None,
                                             min_length=settings.vector_dim, max_length=settings.vector_dim)
    filename: Optional[str] = None
    asset_path: Optional[str] = None
    media_type: Optional[str] = None   # documents | videos | images | audios
    status: str = "active"             # matches ChunkIndexRequest's own default
