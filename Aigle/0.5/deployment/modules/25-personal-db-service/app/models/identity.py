"""Identity API models (dev plan v1.4 §4.5, "Identity management", "Creating an
identity", "Binding and merging")."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# BCP-47 keys the frontend displays aliases under (1.4 §4.5 PATCH)
ALIAS_LOCALES = ("en", "zh-TW", "zh-CN", "ja", "ko")


def _check_locales(aliases: dict[str, str | None] | None) -> dict[str, str | None] | None:
    if aliases:
        unknown = sorted(set(aliases) - set(ALIAS_LOCALES))
        if unknown:
            raise ValueError(f"unsupported alias locales {unknown}; allowed: {list(ALIAS_LOCALES)}")
    return aliases


class FaceImage(BaseModel):
    face_id: str
    kind: str                                   # representative | seed
    image_path: str | None = None
    image_version_id: str | None = None


class IdentityOut(BaseModel):
    identity_id: str
    name: str
    aliases: dict[str, str]
    identity_type: str | None = None
    source: str | None = None                # auto | manual
    identity_status: str                        # pending | active | merged | deleted
    origin_version_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    avatar: FaceImage | None = None          # None: the frontend shows a default avatar
    video_count: int = 0
    group_count: int = 0


class IdentityList(BaseModel):
    total: int
    items: list[IdentityOut]


class IdentityPatch(BaseModel):
    """Only the fields present are changed. aliases: a locale set to null is removed,
    locales not in the request are left alone."""
    name: str | None = Field(None, min_length=1)
    aliases: dict[str, str | None] | None = None
    identity_status: Literal["active"] | None = None    # pending -> active only

    _locales = field_validator("aliases")(_check_locales)


class IdentityUpload(BaseModel):
    identity_id: str | None = None           # given: create-or-update that Identity (idempotent retry)
    name: str = Field(min_length=1)
    aliases: dict[str, str | None] | None = None
    identity_type: str = "person"
    source: str = "manual"

    _locales = field_validator("aliases")(_check_locales)


class PendingPersonIn(BaseModel):
    name: str = ""
    trace_id: str | None = None              # bind this trace (of the URL's video) right away
    trace_ids: list[str] | None = Field(None, min_length=1, max_length=5000)   # or these traces
    group_id: str | None = None              # or this whole group (bind/group)

    @model_validator(mode="after")
    def _at_most_one(self):
        if sum(x is not None for x in (self.trace_id, self.trace_ids, self.group_id)) > 1:
            raise ValueError("give one of trace_id, trace_ids, group_id")
        return self


class BindTraceIn(BaseModel):
    version_id: str
    trace_id: str


class UnbindIn(BaseModel):
    version_id: str
    trace_id: str


class BindGroupIn(BaseModel):
    version_id: str
    group_id: str


class BindGroupResult(BaseModel):
    identity_id: str
    version_id: str
    group_id: str
    previous_identity_id: str | None            # who held the group before (None: it was unassigned)
    removed_identities: list[str] = []          # system-created pending Identities left empty and deleted


class BindResult(BaseModel):
    identity_id: str
    version_id: str
    trace_id: str
    group_id: str | None                     # the trace's group now (None after unbind)
    previous_group_id: str | None
    removed_groups: list[str] = []              # groups left empty and deleted
    removed_identities: list[str] = []          # system-created pending Identities left empty and deleted


class PendingPersonOut(BaseModel):
    identity: IdentityOut
    bind: BindResult | None = None           # trace_id
    binds: list[BindResult] = []             # trace_ids, one per trace
    bind_group: BindGroupResult | None = None   # group_id


class TraceOut(BaseModel):
    trace_id: str
    version_id: str
    group_id: str | None
    start_sec: float
    end_sec: float
    avg_confidence: float | None = None
    representative: FaceImage | None = None
    source_status: str | None = None         # the video's Source.status; None before content indexing


class GroupOut(BaseModel):
    group_id: str
    version_id: str
    frame_count: int | None = None
    traces: list[TraceOut]
    representative: FaceImage | None = None  # best-quality frame among its traces


class PendingIdentityOut(IdentityOut):
    groups: list[GroupOut] = []                 # this identity's groups in the requested video


class PendingPersons(BaseModel):
    version_id: str
    unassigned_groups: list[GroupOut]
    pending_identities: list[PendingIdentityOut]
    active_identities: list[PendingIdentityOut] = []  # confirmed people already holding a group here


class IdentityTraces(BaseModel):
    identity_id: str
    traces: list[TraceOut]


class DeleteIdentityResult(BaseModel):
    identity_id: str
    history_id: str                             # pass to undo within the deadline
    undo_deadline: str
    groups_unassigned: list[str]


class MergeIn(BaseModel):
    target_id: str
    keep_source: bool = True                    # False: delete the source instead of marking it 'merged'


class MergeResult(BaseModel):
    merge_id: str                               # the history_id: pass to undo/redo
    undo_deadline: str
    source_id: str
    source_kept: bool
    groups_moved: list[str]
    target: IdentityOut


class HistoryReplay(BaseModel):
    history_id: str
    operation: str                              # merge | delete
    history_state: str                          # applied | reverted, after this call (unchanged if nothing applied)
    undo_deadline: str
    applied: list[str]                          # items changed by this call
    skipped: list[str]                          # items changed again since, left as they are


class PeopleSearchRequest(BaseModel):
    query: str = Field(min_length=1)            # a name or alias, whole or part of it
    limit: int = Field(50, ge=1, le=500)        # segments returned


class PeopleMatch(BaseModel):
    identity_id: str
    name: str
    matched: str                                # the name or alias that matched
    exact: bool                                 # the whole name/alias, not just part of it


class PeopleSegment(BaseModel):
    """One continuous appearance: first_sec..last_sec is directly the segment to play."""
    identity_id: str
    identity_name: str
    version_id: str
    trace_id: str
    first_sec: float
    last_sec: float
    confidence: float | None = None          # mean face-detection confidence over the trace
    keyframe: FaceImage | None = None        # the trace's representative frame
    keyframe_path: str | None = None         # keyframe.image_path (MTR-24's flat field)
    asset_path: str | None = None
    filename: str | None = None


class PeopleSearchResponse(BaseModel):
    query: str
    identities: list[PeopleMatch]
    total: int                                  # segments before the limit
    results: list[PeopleSegment]


class PeopleInVideosRequest(BaseModel):
    """Scoped counterpart to PeopleSearchRequest: these specific identities,
    restricted to these specific videos -- no limit, since the result is
    already bounded by (identities) x (videos) rather than an identity's
    total appearance count system-wide."""
    identity_ids: list[str] = Field(min_length=1)
    version_ids: list[str] = Field(min_length=1)


class PeopleInVideosResponse(BaseModel):
    results: list[PeopleSegment]


class ProfileImageIn(BaseModel):
    """A seed photo the Gateway already stored in Module 04 and Module 11 embedded."""
    image_path: str
    image_version_id: str | None = None
    embedding: list[float] = Field(min_length=512, max_length=512)
