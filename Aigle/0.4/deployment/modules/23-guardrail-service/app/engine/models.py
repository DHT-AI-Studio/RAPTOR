"""Shared data models for the Policy Engine — signal, policy, decisions, trace."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ── LLM Signal ────────────────────────────────────────────────────────────────

class LLMSignal(BaseModel):
    """Structured output from the Signal Layer (LLM)."""
    category:   str        = Field(..., description="Content category")
    confidence: str        = Field(..., description="low | medium | high")
    intent:     str        = Field(..., description="informational | instructional | malicious | conversational | ambiguous | regulated_advice | personal_context")
    risk_flags: list[str]  = Field(default_factory=list, description="Risk tags identified by LLM")
    rationale:  str        = Field("", description="One-sentence explanation")
    raw:        str        = Field("", description="Raw LLM output for debug")


# ── Policy Rule ───────────────────────────────────────────────────────────────

class RuleConditions(BaseModel):
    category:       list[str] | None = None
    confidence:     list[str] | None = None
    intent:         list[str] | None = None
    risk_flags_any: list[str] | None = None   # match if ANY flag present in signal
    risk_flags_all: list[str] | None = None   # match only if ALL flags present


class PolicyRule(BaseModel):
    id:          str
    priority:    int
    description: str = ""
    conditions:  RuleConditions
    action:      Literal["allow", "block", "transform", "require_safe_completion", "escalate"]
    reason:      str
    explanation: str = ""


# ── Rule Engine Output ────────────────────────────────────────────────────────

class RuleResult(BaseModel):
    action:          str
    matched_rule:    PolicyRule | None
    default_applied: bool = False


# ── Guardrail Policy (versioned, one active at a time) ────────────────────────
#
# Only `name`/`version` are validated at upload time. Everything else in the
# uploaded YAML/JSON is stored verbatim as `raw_content` and used unprocessed
# — no schema is enforced on it. The deterministic /policy/check/* endpoints
# make a *best-effort* lazy attempt to interpret `raw_content` as a `rules`
# array (see policy_engine._parse_structured_rules); the LLM-judged
# /policy/check/llm/* endpoints just hand `raw_content` to the model as-is.

class PolicyCreateResponse(BaseModel):
    id:         UUID
    name:       str
    version:    str
    created_at: datetime


class PolicySummary(BaseModel):
    """Lightweight overview of a stored policy (no content body)."""
    id:         UUID
    name:       str
    version:    str
    is_active:  bool
    created_at: datetime


class PolicyDetail(BaseModel):
    """Full stored policy — identity fields plus content per inference target.

    `raw_content` is the shared/"original" content, used by every check
    family unchanged, and as the /policy/check/llm/* fallback for any guard
    model without its own override below."""
    id:          UUID
    name:        str
    version:     str
    raw_content: str
    content_llama_guard:       str | None = None
    content_granite_guardian:  str | None = None
    content_gpt_oss_safeguard: str | None = None
    is_active:   bool
    created_at:  datetime


class PolicyTemplate(BaseModel):
    """Suggested YAML/JSON format and example content for a policy."""
    description:  str
    example_json: str
    example_yaml: str


# ── Trace ─────────────────────────────────────────────────────────────────────

class TraceEntry(BaseModel):
    step: str
    data: dict[str, Any]


# ── Final Policy Decision (API Response) ──────────────────────────────────────

class PolicyDecision(BaseModel):
    action:                  str  = Field(..., description="allow | block | transform | require_safe_completion | escalate")
    policy_id:               UUID
    policy_name:             str
    policy_version:          str
    matched_rule_id:         str | None = Field(None)
    intent:                  str
    category:                str
    confidence:              str
    risk_flags:              list[str]
    final_action_reasoning:  str
    trace:                   list[TraceEntry]
    safe:                    bool = Field(..., description="Backward compat: True when action == 'allow'")
    rationale:               str


# ── API Request Schemas ───────────────────────────────────────────────────────

class Message(BaseModel):
    role:    str = Field(..., description="user | assistant")
    content: str


class CheckRequest(BaseModel):
    content: str  = Field(..., description="Content to classify")
    context: list[Message] | None = Field(None, description="Conversation history for context")


class ConversationCheckRequest(BaseModel):
    messages: list[Message] = Field(..., description="Conversation history (at least one message)")
