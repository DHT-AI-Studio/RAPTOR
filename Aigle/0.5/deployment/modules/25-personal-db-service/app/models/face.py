"""Face trace / group write request (dev plan v1.4 §4.5, "Trace-and-cluster write").

One request carries one video's complete face analysis from Module 11's
face-detection service: every trace, the intra-video clustering result, and each
trace's representative frame. Module 11 does detection, tracking and clustering;
Module 25 only stores the result and links groups to cross-video Identities.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings

_DIM = settings.face_vector_dim


class FaceFrameIn(BaseModel):
    frame_no: int
    timestamp_sec: float
    bbox: list[float] = Field(min_length=4, max_length=4)    # x1, y1, x2, y2
    confidence: float


class RepresentativeIn(BaseModel):
    """The trace's best-quality frame. image_path/image_version_id point at the
    thumbnail Module 11 uploaded to Module 04 (absent if that upload failed)."""
    frame_no: int
    timestamp_sec: float
    bbox: list[float] = Field(min_length=4, max_length=4)
    quality: float
    min_side: float                                           # shorter side of the face box, px
    image_path: str | None = None
    image_version_id: str | None = None


class FaceTraceIn(BaseModel):
    trace_id: str
    start_sec: float
    end_sec: float
    embedding: list[float] = Field(min_length=_DIM, max_length=_DIM)
    frames: list[FaceFrameIn] = Field(min_length=1)
    representative: RepresentativeIn


class FaceGroupIn(BaseModel):
    group_id: str
    embedding: list[float] = Field(min_length=_DIM, max_length=_DIM)
    trace_ids: list[str] = Field(min_length=1)


class FaceTracesRequest(BaseModel):
    version_id: str
    asset_path: str | None = None
    filename: str | None = None
    traces: list[FaceTraceIn] = []
    groups: list[FaceGroupIn] = []

    @model_validator(mode="after")
    def _groups_reference_known_traces(self):
        trace_ids = {t.trace_id for t in self.traces}
        if len(trace_ids) != len(self.traces):
            raise ValueError("duplicate trace_id")
        seen: set[str] = set()
        for g in self.groups:
            unknown = set(g.trace_ids) - trace_ids
            if unknown:
                raise ValueError(f"group {g.group_id} references unknown traces: {sorted(unknown)}")
            if seen & set(g.trace_ids):
                raise ValueError(f"group {g.group_id} shares traces with another group")
            seen |= set(g.trace_ids)
        if len({g.group_id for g in self.groups}) != len(self.groups):
            raise ValueError("duplicate group_id")
        return self


class GroupMatch(BaseModel):
    group_id: str
    identity_id: str | None      # None: no match, the group stays unassigned (pending list)
    created_identity: bool          # True: matched another video's unassigned group, and a new
                                    # pending Identity now holds both
    score: float | None = None   # best weighted similarity found (None: matching skipped)
    skipped_small_face: bool = False


class FaceTracesResponse(BaseModel):
    version_id: str
    traces_written: int
    groups_written: int
    replaced: dict                  # counts removed from a previous write of this version
    matches: list[GroupMatch]
