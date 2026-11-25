"""Request/response models for the guardrail check endpoints (GB-4)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CheckAction = Literal["pass", "block", "redact", "flag"]


class CheckRequest(BaseModel):
    content: str
    module: str | None = Field(None, description="Calling module id, matched against policy scope")
    task: str | None = None
    request_id: str | None = None


class CheckResult(BaseModel):
    safe: bool
    action: CheckAction = "pass"
    category: str | None = None
    redacted_content: str | None = None
