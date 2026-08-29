"""Request/response models for the guardrail check endpoints (GB-4)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CheckAction = Literal["pass", "block", "redact", "flag"]


class CheckRequest(BaseModel):
    content: str
    module: Optional[str] = Field(None, description="Calling module id, matched against policy scope")
    task: Optional[str] = None
    request_id: Optional[str] = None


class CheckResult(BaseModel):
    safe: bool
    action: CheckAction = "pass"
    category: Optional[str] = None
    redacted_content: Optional[str] = None
