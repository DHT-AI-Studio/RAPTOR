"""Response models for the guardrail enable/disable switches."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SystemEnabledResponse(BaseModel):
    enabled: bool = Field(..., description="Guardrail checking master switch state")


class PolicyEnabledResponse(BaseModel):
    policy_enabled: bool = Field(
        ..., description="Policy-based LLM check switch state (/policy/check/llm/*, /debug/policy/check/llm/*)"
    )


class SystemStatus(BaseModel):
    enabled: bool = Field(..., description="Guardrail checking master switch state")
    policy_enabled: bool = Field(
        ...,
        description=(
            "Policy-based LLM check switch state; when False, /policy/check/llm/* and "
            "/debug/policy/check/llm/* short-circuit to safe=true (independent of `enabled`, "
            "has no effect on /guard/check/*)"
        ),
    )
    active_policy_name: str | None = Field(None, description="Name of the active policy, or null if none")
    active_policy_version: str | None = Field(None, description="Version of the active policy, or null if none")
    active_policy_format: Literal["standard", "checker_rules"] | None = Field(
        None, description="Format of the active policy's content, or null if none is active or it parses as neither.",
    )
    conflict_strategy: Literal["strict", "majority", "permissive"] = Field(
        ..., description="How disagreeing multi-guard-model verdicts are resolved (GUARD_CONFLICT_STRATEGY).",
    )
    proxy_mounted: bool = Field(
        False, description="Whether the OpenAI-compatible proxy (/v1/chat/completions, /v1/models) is registered",
    )
