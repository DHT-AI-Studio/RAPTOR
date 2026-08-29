"""Standard Policy format — a portable, model-agnostic taxonomy category
that each guard adapter converts into its own official prompt structure
(see app/adapters/models/*.py). Upload a JSON array of these to
POST /guardrail/policies (target=original) and every guard model gets its
own correctly-formatted prompt at inference time, generated from the same
source of truth — no per-model authoring needed unless you want to override
one model's wording specifically (the content_llama_guard /
content_granite_guardian / content_gpt_oss_safeguard columns still take
precedence verbatim when set — see app/routers/check_policy_llm.py).
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class StandardPolicyExamples(BaseModel):
    violation: list[str] = Field(default_factory=list)
    allowed: list[str] = Field(default_factory=list)


class StandardPolicy(BaseModel):
    id: str
    name: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    decision: Literal["allow", "warn", "review", "block"]
    criteria: list[str]
    exceptions: list[str] = Field(default_factory=list)
    examples: StandardPolicyExamples = Field(default_factory=StandardPolicyExamples)


def parse_standard_policies(raw_content: str) -> list[StandardPolicy] | None:
    """Best-effort parse of `raw_content` as a JSON array of StandardPolicy.

    Returns None (never raises) on anything that isn't valid standard-format
    JSON — an empty/missing array, malformed JSON, or a document missing
    required fields — so callers can fall back to using raw_content
    verbatim. This keeps every pre-existing free-text/YAML policy working
    unchanged; only content that actually validates as the standard format
    triggers per-model conversion.
    """
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list) or not data:
        return None
    try:
        return [StandardPolicy(**item) for item in data]
    except (ValidationError, TypeError):
        return None
