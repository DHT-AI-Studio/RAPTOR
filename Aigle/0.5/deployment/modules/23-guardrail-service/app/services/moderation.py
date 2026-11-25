"""Moderation verdicts (CM-1 / VIE01-404): a guard model's raw output mapped
onto the moderation taxonomy (app/models/moderation.py) for one modality.

  - `moderate_scores()` -- ShieldGemma 2 probabilities per policy, against the
    modality's thresholds (image / video_frame).
  - `moderate_codes()` -- Llama Guard S-codes, any modality; a reported code
    flags every category that maps it (no score, so no threshold).
  - `resolve_moderation()` -- the taxonomy in force: the active policy's
    `moderation` block, else the built-in default (`moderation_for()` does the
    same for a policy already loaded, as /guardrail/policy/decide has).

A category the model can't judge at all (ShieldGemma 2 has no hate policy)
is listed in `not_covered` instead of being reported as clean.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.models.moderation import DECISION_RANK, ModerationPolicy, parse_moderation
from app.services.default_moderation import DEFAULT_MODERATION

_logger = logging.getLogger(__name__)

DEFAULT_MODERATION_POLICY = ModerationPolicy.model_validate(DEFAULT_MODERATION)


@dataclass(frozen=True)
class ResolvedModeration:
    moderation: ModerationPolicy
    policy_name: str | None = None        # None: the built-in default is in force
    policy_version: str | None = None

    @property
    def source(self) -> str:
        return "active_policy" if self.policy_name is not None else "default"


def _verdict(modality: str, categories: dict[str, dict], resolved: ResolvedModeration) -> dict:
    flagged = [name for name, c in categories.items() if c["flagged"]]
    decision = max((categories[n]["decision"] for n in flagged), key=DECISION_RANK.__getitem__, default="allow")
    return {
        "modality": modality,
        "decision": decision,
        "flagged": flagged,
        "not_covered": [name for name, c in categories.items() if not c["covered"]],
        "categories": categories,
        "source": resolved.source,
        "policy_name": resolved.policy_name,
        "policy_version": resolved.policy_version,
        "moderation_version": resolved.moderation.version,
    }


def moderate_scores(scores: dict[str, float], modality: str, resolved: ResolvedModeration) -> dict:
    """ShieldGemma 2 `{policy: unsafe_probability}` -> moderation verdict.
    A category's score is the highest of its mapped policies; it is flagged
    when that score reaches the modality's threshold."""
    categories: dict[str, dict] = {}
    for name, cat in resolved.moderation.categories.items():
        rule = cat.modalities[modality]
        mapped = {p: scores[p] for p in cat.shieldgemma2 if p in scores}
        if not mapped:
            categories[name] = {"covered": False, "flagged": False, "decision": "allow",
                                "score": None, "threshold": rule.threshold, "matched": []}
            continue
        score = max(mapped.values())
        flagged = rule.threshold is not None and score >= rule.threshold
        categories[name] = {
            "covered": True, "flagged": flagged,
            "decision": rule.decision if flagged else "allow",
            "score": score, "threshold": rule.threshold,
            "matched": sorted(p for p, s in mapped.items() if rule.threshold is not None and s >= rule.threshold),
        }
    return _verdict(modality, categories, resolved)


def moderate_codes(codes: list[str], modality: str, resolved: ResolvedModeration) -> dict:
    """Llama Guard S-codes -> moderation verdict for `modality`."""
    reported = set(codes)
    categories: dict[str, dict] = {}
    for name, cat in resolved.moderation.categories.items():
        rule = cat.modalities[modality]
        if not cat.llama_guard:
            categories[name] = {"covered": False, "flagged": False, "decision": "allow", "matched": []}
            continue
        matched = [c for c in cat.llama_guard if c in reported]
        categories[name] = {"covered": True, "flagged": bool(matched),
                            "decision": rule.decision if matched else "allow", "matched": matched}
    return _verdict(modality, categories, resolved)


async def resolve_moderation() -> ResolvedModeration:
    """The active policy's `moderation` block when there is an active policy
    with one (and the policy switch is on -- same lookup as
    /guardrail/policy/decide), else the built-in default. A failed lookup also
    falls back to the default, logged; the verdict's `source` says which."""
    from app.services import (
        policy_check_service,  # lazy: avoids importing the check stack at module load
    )

    try:
        policy = await policy_check_service.get_policy_for_branch()
    except Exception:
        _logger.warning("moderation: active policy lookup failed, using the built-in default", exc_info=True)
        return ResolvedModeration(DEFAULT_MODERATION_POLICY)
    return moderation_for(policy)


def moderation_for(policy) -> ResolvedModeration:
    """The taxonomy carried by `policy` (an already-loaded PolicyDetail, or
    None), else the built-in default. /guardrail/policy/decide uses this with
    the policy its decision was made under, so the two can't differ."""
    moderation = parse_moderation(policy.raw_content) if policy is not None else None
    if moderation is None:
        return ResolvedModeration(DEFAULT_MODERATION_POLICY)
    return ResolvedModeration(moderation, policy.name, policy.version)
