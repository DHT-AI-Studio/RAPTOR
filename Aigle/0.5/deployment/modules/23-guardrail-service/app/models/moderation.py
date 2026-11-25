"""Moderation taxonomy (CM-1 / VIE01-404): the v0.5 content-moderation
categories (NSFW, violence, hate speech, self-harm), how each maps onto the
guard models' own labels, and a decision (plus, for images, a score
threshold) per modality.

It lives in a Standard Policy as an optional `moderation` block next to the
usual category array -- `{"categories": [...], "moderation": {...}}` (see
app/models/policy.py's parse_standard_policies) -- so it is uploaded,
activated and versioned like any other policy.

Per modality:
  - `image` / `video_frame` -- ShieldGemma 2 gives a probability per policy,
    so these carry a `threshold` (0-1): the category is flagged when the
    highest score of its mapped ShieldGemma policies reaches it. Llama Guard
    3 Vision gives no score; for it the S-codes alone decide.
  - `text` / `transcript` -- Llama Guard 3 gives S-codes, no score, so these
    carry a decision only (no threshold).
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

Modality = Literal["text", "transcript", "image", "video_frame"]
MODALITIES: tuple[str, ...] = ("text", "transcript", "image", "video_frame")
IMAGE_MODALITIES: tuple[str, ...] = ("image", "video_frame")
ImageModality = Literal["image", "video_frame"]

ModerationDecision = Literal["allow", "warn", "review", "block"]
DECISION_RANK: dict[str, int] = {"allow": 0, "warn": 1, "review": 2, "block": 3}

# ShieldGemma 2's three built-in policies (transformers'
# DEFAULT_SHIELDGEMMA2_POLICIES, the keys /infer/google-shieldgemma-2 returns).
SHIELDGEMMA2_POLICIES: tuple[str, ...] = ("dangerous", "sexual", "violence")
# Llama Guard 3's S-codes (app/adapters/base.py CATEGORY_NAMES). Kept as a
# literal set here because app/adapters/base.py imports app/models/policy.py,
# which would make importing it from this module circular; a unit test keeps
# the two equal.
LLAMA_GUARD_CODES: tuple[str, ...] = tuple(f"S{i}" for i in range(1, 15))


class ModalityRule(BaseModel):
    decision: ModerationDecision
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class ModerationCategory(BaseModel):
    description: str
    llama_guard: list[str] = Field(default_factory=list)
    shieldgemma2: list[str] = Field(default_factory=list)
    modalities: dict[Modality, ModalityRule]

    @model_validator(mode="after")
    def _check(self) -> ModerationCategory:
        unknown = [c for c in self.llama_guard if c not in LLAMA_GUARD_CODES]
        if unknown:
            raise ValueError(f"unknown Llama Guard code(s) {unknown}; expected S1-S14")
        unknown = [p for p in self.shieldgemma2 if p not in SHIELDGEMMA2_POLICIES]
        if unknown:
            raise ValueError(f"unknown ShieldGemma 2 policy(ies) {unknown}; expected one of {list(SHIELDGEMMA2_POLICIES)}")
        missing = [m for m in MODALITIES if m not in self.modalities]
        if missing:
            raise ValueError(f"missing modality rule(s) {missing}")
        for modality, rule in self.modalities.items():
            if modality in IMAGE_MODALITIES and rule.threshold is None:
                raise ValueError(f"modality '{modality}' needs a threshold (0-1)")
            if modality not in IMAGE_MODALITIES and rule.threshold is not None:
                raise ValueError(f"modality '{modality}' takes no threshold (Llama Guard gives no score)")
        return self


class ModerationPolicy(BaseModel):
    version: str = "1.0"
    categories: dict[str, ModerationCategory]

    @model_validator(mode="after")
    def _not_empty(self) -> ModerationPolicy:
        if not self.categories:
            raise ValueError("moderation.categories must not be empty")
        return self


def parse_moderation(raw_content: str | None) -> ModerationPolicy | None:
    """The `moderation` block of a Standard Policy in object form, or None
    (never raises) when the content has none or it doesn't validate --
    callers then fall back to the built-in default
    (app/services/default_moderation.py). Upload already rejects an invalid
    block with 422 (app/routers/guardrail_policies.py), so None here means
    "no block" in practice."""
    try:
        data = json.loads(raw_content or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "moderation" not in data:
        return None
    try:
        return ModerationPolicy.model_validate(data["moderation"])
    except (ValidationError, TypeError):
        return None
