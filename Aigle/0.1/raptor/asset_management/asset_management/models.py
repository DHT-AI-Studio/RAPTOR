from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
from enum import Enum
from pydantic import BaseModel

class MediaType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"

@dataclass
class AssetMetadata:
    asset_path: str
    version_id: str
    primary_filename: str
    associated_filenames: List[Tuple[str, str]]  # List of (filename, version_id) tuples
    upload_date: datetime
    archive_date: datetime
    destroy_date: datetime
    branch: str
    status: str
    
    def to_dict(self) -> dict:
        return {
            "asset_path": self.asset_path,
            "version_id": self.version_id,
            "primary_filename": self.primary_filename,
            "associated_filenames": self.associated_filenames,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "archive_date": self.archive_date.isoformat() if self.archive_date else None,
            "destroy_date": self.destroy_date.isoformat() if self.destroy_date else None,
            "branch": self.branch,
            "status": self.status
        }

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    branch: str

class User(BaseModel):
    username: str
    password: str = ""
    password_hash: str = ""
    branch: str = ""
    permissions: List[str] = ["upload", "download", "list"]

