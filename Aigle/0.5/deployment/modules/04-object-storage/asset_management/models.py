from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MediaType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"

AssetStatus = Literal["uploaded", "processing", "active", "failed", "unregistered", "archived", "destroyed"]

class ExistenceInfo(BaseModel):
    exists: bool = False
    message: str = ""
    existing_asset_path: str | None = None
    existing_version_id: str | None = None

# MTR-8 / VIE01-307: "not_found" is not one of AssetStatus's real values --
# "destroyed" is excluded here on purpose. destroy_v2()/destroy() DELETE the
# commit_history row outright (never persist status="destroyed"), so a scan
# can never actually observe it; the 6-state vocabulary scan reports against
# is uploaded/processing/active/failed/unregistered/archived.
ScanStatus = Literal["uploaded", "processing", "active", "failed", "unregistered", "archived", "not_found"]

class ScanFileRequest(BaseModel):
    path: str
    checksum: str

class ScanFileResult(BaseModel):
    path: str
    checksum: str
    status: ScanStatus
    existing_asset_path: str | None = None
    existing_version_id: str | None = None

class ScanRequest(BaseModel):
    files: list[ScanFileRequest]

class ScanResponse(BaseModel):
    results: list[ScanFileResult]

class PresignBatchItem(BaseModel):
    key: str
    version_id: str

class PresignBatchRequest(BaseModel):
    items: list[PresignBatchItem]

class PresignBatchResult(BaseModel):
    key: str
    version_id: str
    url: str
    content_type: str | None = None

class PresignBatchResponse(BaseModel):
    # Best-effort: an item missing here (failed group, or list_objects didn't
    # return it) isn't an error -- the caller falls back to /presign-url per key.
    resolved: list[PresignBatchResult]

class AssetMetadata(BaseModel):
    asset_path: str
    version_id: str
    primary_filename: str
    associated_filenames: list[tuple[str, str]] 
    upload_date: datetime
    archive_date: datetime | str | None = None
    destroy_date: datetime | str | None = None
    user: str
    branch: str
    status: AssetStatus
    checksum: str
    # Internal bookkeeping only (register_asset()'s fast-path check) -- never
    # meant for API responses, excluded from serialization so it can't leak
    # out through /fileupload or /register, neither of which use a
    # restricted response_model.
    cloned_from: str | None = Field(default=None, exclude=True)
    # CM-2: Module 10's image moderation verdict; None when not moderated.
    moderation_flags: dict[str, Any] | None = None
    existence_info: ExistenceInfo = ExistenceInfo()

class AssetMetadataResponse(BaseModel):
    asset_path: str
    version_id: str
    primary_filename: str
    associated_filenames: list[tuple[str, str]]
    upload_date: datetime
    archive_date: datetime | str | None = None
    destroy_date: datetime | str | None = None
    status: AssetStatus
    moderation_flags: dict[str, Any] | None = None

class ModerationFlags(BaseModel):
    """CM-2: body of POST /update-moderation -- Module 10's verdict for one image
    (image_moderation.py there). Model-specific detail (policies / categories /
    error) is kept as sent."""
    model_config = ConfigDict(extra="allow")

    verdict: Literal["safe", "unsafe", "error"]
    model: str
    latency_ms: float
    checked_at: str   # ISO 8601, kept as sent -- the same string Module 25 gets with the summary entry

    @field_validator("checked_at")
    @classmethod
    def _iso_8601(cls, value: str) -> str:
        datetime.fromisoformat(value)   # ValueError -> 422
        return value

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    branch: str

class User(BaseModel):
    user_id: str
    branch_id: str

def model_to_response(
    model_obj: BaseModel, 
    response_model: type[BaseModel],
    field_map: dict[str, Any] | None = None
) -> BaseModel:
    """
    Convert a Pydantic model object to another Pydantic response model dynamically.
    
    Args:
        model_obj: The Pydantic model instance to convert.
        response_model: The Pydantic model class for response.
        field_map: Optional mapping for custom transformations: 
                   {response_field_name: callable or source_field_name}
    
    Returns:
        An instance of response_model with fields populated.
    """
    data = model_obj.dict()
    field_map = field_map or {}
    response_data = {}

    for field in response_model.__fields__:
        if field in field_map:
            mapping = field_map[field]
            if callable(mapping):
                response_data[field] = mapping(model_obj)
            elif isinstance(mapping, str):
                response_data[field] = data.get(mapping)
        else:
            response_data[field] = data.get(field)

    return response_model(**response_data)