"""Chunk index request model (PA-4, unified `Chunk`).

One `ChunkIndexRequest` covers all four media types (documents|videos|images|
audios). Media-specific fields are optional and only sent when applicable.
`sparse_*` is OPTIONAL — the 0.3 pipeline produces dense only.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings


class ChunkIndexRequest(BaseModel):
    # --- identity / common (required) ---
    chunk_id: str
    type: str                          # documents | videos | images | audios
    embedding_type: str                # text | summary
    # Optional: if omitted, the service embeds `text`/`summary` with its own BGE-M3
    # (production pipeline / Module 07 may still pass a pre-computed vector).
    embedding: list[float] | None = Field(default=None,
                                             min_length=settings.vector_dim, max_length=settings.vector_dim)

    # --- common (optional) ---
    text: str | None = None
    summary: str | None = None
    filename: str | None = None
    source: str | None = None
    asset_path: str | None = None
    version_id: str | None = None
    upload_time: str | None = None
    status: str = "active"

    # --- sparse (OPTIONAL; kept for future hybrid) ---
    sparse_indices: list[int] | None = None
    sparse_weights: list[float] | None = None

    # --- temporal (video / audio segments) ---
    start_sec: float | None = None
    end_sec: float | None = None
    speaker: str | None = None

    # --- video-specific ---
    contextual_text: str | None = None
    asr_text: str | None = None
    lvlm_desc: str | None = None
    ocr_text: str | None = None

    # --- audio-specific ---
    audio_labels: list[str] | None = None

    # --- chunk_index: this chunk's sequence position within its source asset.
    # One property for every media type -- video/audio events upstream call
    # it moment_index, document events call it chunk_index; kafka_consumer.py
    # normalizes both into this single field (previously two separate
    # properties, moment_index here and this chunk_index, which for a
    # document entry ended up holding the exact same value twice). ---
    chunk_index: int | None = None
    page_numbers: list[int] | None = None
    section_heading: str | None = None
    element_types: list[str] | None = None
    char_count: int | None = None

    # --- GD-6 / VIE01-251: PII found at ingestion (None = not flagged) ---
    pii_flag: bool | None = None

    # --- CM-3 / VIE01-406: a video moment's keyframe moderation verdict (Module 11
    # -> Module 23 image guard), as the worker sent it; None when not moderated ---
    moderation_flags: dict[str, Any] | None = None

    # --- P13: when the source was written or recorded, from the file's own metadata (as the worker read
    # it; app/temporal/anchor.py validates it before it anchors anything) ---
    capture_time: str | None = None
    capture_time_source: str | None = None   # pdf_creation_date | docx_core_created | exif_datetime_original | container_creation_time


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
    # The new upload's own primary filename -- content-dedup means the same
    # bytes, not the same name (e.g. re-uploaded under a different filename).
    # None only for a caller that predates this field; clone_version() then
    # falls back to copying the source's filename unchanged (the old,
    # buggy-for-this-case behavior), rather than refusing the clone.
    target_filename: str | None = None


class SourceSummaryIndexRequest(BaseModel):
    """A whole-asset summary (video/audio/image/document alike -- every media
    type sends one embedding_type="summary" entry, previously indexed as a
    fake Chunk with no start_sec/end_sec). Written onto the Source vertex
    itself now, matching Module 20's Source.summary (neo4j_writer.py:61)
    instead of faking a moment/chunk that doesn't semantically exist."""
    version_id: str
    summary: str
    embedding: list[float] | None = Field(default=None,
                                             min_length=settings.vector_dim, max_length=settings.vector_dim)
    filename: str | None = None
    capture_time: str | None = None          # P13: as on the chunks (a photo with no OCR text has none)
    capture_time_source: str | None = None
    asset_path: str | None = None
    media_type: str | None = None   # documents | videos | images | audios
    status: str = "active"             # matches ChunkIndexRequest's own default
    # CM-2: Module 10's image moderation verdict (Module 23 image guard), as the
    # worker sent it; None when the asset wasn't moderated.
    moderation_flags: dict[str, Any] | None = None
