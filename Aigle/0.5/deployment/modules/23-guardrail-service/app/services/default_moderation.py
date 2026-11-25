"""The built-in v0.5 moderation taxonomy (CM-1 / VIE01-404).

Used when the active policy has no `moderation` block (see
app/services/moderation.py), and shipped as a ready-to-upload policy by
`GET /guardrail/policies/template?format=moderation`: the default S1-S13
categories (app/services/default_policy.py, unchanged, so text decisions stay
the same once it is activated) plus this block.

Mapping (MLCommons S-codes as Llama Guard 3 reports them; ShieldGemma 2's
three built-in policies):

  category    Llama Guard 3      ShieldGemma 2
  nsfw        S12, S3, S4        sexual
  violence    S1, S9             violence, dangerous
  hate        S10                -- (no ShieldGemma 2 policy covers it)
  self_harm   S11                dangerous (partly: its definition names
                                 "instructions for suicide", next to weapons
                                 and terrorism)

Not moderation categories (left to the guardrail decision as before): S2, S5,
S6, S7, S8, S13, S14.

Thresholds: 0.5 for image and video_frame, which is the cut-off the image
guard used before CM-1 (unsafe when P(unsafe) > P(safe)). They are starting
values, not tuned on data. Decisions: `block` everywhere, matching the
default S1-S13 policy; ingestion callers (CM-2, CM-3) record the verdict
rather than enforce it.
"""
from __future__ import annotations

from app.services.default_policy import DEFAULT_STANDARD_POLICY

DEFAULT_MODERATION_THRESHOLD = 0.5


def _modalities(decision: str = "block", threshold: float = DEFAULT_MODERATION_THRESHOLD) -> dict:
    return {
        "text": {"decision": decision},
        "transcript": {"decision": decision},
        "image": {"decision": decision, "threshold": threshold},
        "video_frame": {"decision": decision, "threshold": threshold},
    }


DEFAULT_MODERATION: dict = {
    "version": "1.0",
    "categories": {
        "nsfw": {
            "description": "Sexually explicit content: pornography, erotic nudity, sexual acts, "
                           "sexual violence, and any sexualisation of minors.",
            "llama_guard": ["S12", "S3", "S4"],
            "shieldgemma2": ["sexual"],
            "modalities": _modalities(),
        },
        "violence": {
            "description": "Violent or gory content, and content that enables violence: violent crimes, "
                           "weapons (including mass-casualty weapons), terrorism.",
            "llama_guard": ["S1", "S9"],
            "shieldgemma2": ["violence", "dangerous"],
            "modalities": _modalities(),
        },
        "hate": {
            "description": "Hate speech: content that demeans or dehumanises people for a protected "
                           "attribute (race, religion, gender, sexual orientation, ...).",
            "llama_guard": ["S10"],
            "shieldgemma2": [],
            "modalities": _modalities(),
        },
        "self_harm": {
            "description": "Content that encourages or enables suicide, self-injury or disordered eating.",
            "llama_guard": ["S11"],
            "shieldgemma2": ["dangerous"],
            "modalities": _modalities(),
        },
    },
}

# The full policy GET /guardrail/policies/template?format=moderation returns
# (the object form of the Standard Policy Format).
DEFAULT_MODERATION_STANDARD_POLICY: dict = {
    "categories": DEFAULT_STANDARD_POLICY,
    "moderation": DEFAULT_MODERATION,
}
