"""Structured guardrail policy models — two independent formats.

`GuardrailPolicy` (GB-2 / GB-4, detector-based): one document, many **rules**,
grouped into `input_guardrail` and `output_guardrail`. Each rule maps a
`category` to an `action`; by default the category is matched against the guard
model's (Llama Guard) classification, but a rule may set `detector: regex` +
`patterns` to be matched deterministically without an LLM call.

Policies are authored as YAML/JSON and stored as text (`raw_content`); the
checker parses that text into a `GuardrailPolicy` via `GuardrailPolicy.from_raw`.

`StandardPolicyCategory` (the "Standard Policy Format"): a portable,
model-agnostic taxonomy category that each guard adapter converts into its
own official prompt structure (see app/adapters/models/*.py). Upload a JSON
array of these to POST /guardrail/policies (target=original) and every guard
model gets its own correctly-formatted prompt, generated from the same
source of truth — no per-model authoring needed unless you want to override
one model's wording specifically (the content_llama_guard /
content_granite_guardian / content_gpt_oss_safeguard columns still take
precedence verbatim when set — see app/routers/check_policy_llm.py). See
policy_example/standard_policy_format.md for the full field reference.
"""
from __future__ import annotations

import json
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

Action = Literal["block", "redact", "flag"]
Detector = Literal["llama_guard", "regex"]


class PolicyRule(BaseModel):
    category: str                                   # Llama Guard code (S1..S14) or a custom label (jailbreak, pii)
    action: Action
    detector: Detector = "llama_guard"              # default: match against the guard model's classification
    patterns: list[str] = Field(default_factory=list)   # required when detector == "regex"

    @model_validator(mode="after")
    def _regex_needs_patterns(self) -> PolicyRule:
        if self.detector == "regex" and not self.patterns:
            raise ValueError(f"regex rule '{self.category}' must define at least one pattern")
        return self


class GuardrailBlock(BaseModel):                     # shared shape for input_guardrail / output_guardrail
    enabled: bool = True
    model: str = "llama-guard3:8b"                   # default detector when a rule doesn't set one
    timeout_ms: int = 3000
    rules: list[PolicyRule] = Field(default_factory=list)


class GuardrailScope(BaseModel):
    modules: list[str] = Field(default_factory=lambda: ["*"])
    tasks: list[str] = Field(default_factory=lambda: ["*"])

    def applies_to(self, module: str | None) -> bool:
        """True if this scope covers the given module ("*" matches everything)."""
        return "*" in self.modules or (module is not None and module in self.modules)


class AuditLogConfig(BaseModel):
    enabled: bool = True
    include_content: bool = False


class GuardrailPolicy(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    scope: GuardrailScope = Field(default_factory=GuardrailScope)
    input_guardrail: GuardrailBlock = Field(default_factory=GuardrailBlock)
    output_guardrail: GuardrailBlock = Field(default_factory=GuardrailBlock)
    audit_log: AuditLogConfig = Field(default_factory=AuditLogConfig)

    @classmethod
    def from_raw(cls, raw_content: str) -> GuardrailPolicy:
        """Parse a policy document (YAML, JSON is a subset) into a GuardrailPolicy.

        Deliberately does NOT also accept the Standard Policy Format here:
        app/services/policy_check_service.py's final_verdict() relies on this
        raising for anything that isn't checker_rules-shaped to decide
        whether its own Standard-Policy-aware decision logic
        (_guard_decision() in app/routers/guardrail.py, which correctly
        blocks an unmapped/uncategorized unsafe verdict) should run instead
        of the rule engine. app/services/checker.py converts a Standard
        Policy into one of these itself, for its own GB-4 rule-engine
        callers (/guardrail/check/*, the OpenAI-proxy check) only -- this is
        the one and only way to build a GuardrailPolicy anywhere else."""
        data = yaml.safe_load(raw_content) or {}
        if not isinstance(data, dict):
            raise ValueError("policy content must be a YAML/JSON object")
        return cls.model_validate(data)

    def block_for(self, direction: Literal["input", "output"]) -> GuardrailBlock:
        return self.input_guardrail if direction == "input" else self.output_guardrail


# ── Standard Policy Format ────────────────────────────────────────────────────

class StandardPolicyExamples(BaseModel):
    violation: list[str] = Field(default_factory=list)
    allowed: list[str] = Field(default_factory=list)


class StandardPolicyCategory(BaseModel):
    id: str
    name: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    decision: Literal["allow", "warn", "review", "block"]
    criteria: list[str]
    exceptions: list[str] = Field(default_factory=list)
    examples: StandardPolicyExamples = Field(default_factory=StandardPolicyExamples)


# A Standard Policy (the thing stored/uploaded/activated) is a JSON array of
# StandardPolicyCategory. CM-1 (VIE01-404) also accepts an object form,
# {"categories": [...], "moderation": {...}}, to carry the moderation
# taxonomy (app/models/moderation.py) next to the same array; the parsed
# value is the category list either way, so this stays a plain type alias.
StandardPolicy = list[StandardPolicyCategory]

# The only keys the object form may have.
STANDARD_OBJECT_KEYS = frozenset({"categories", "moderation"})


def standard_categories_json(data):
    """The category array of a decoded Standard Policy: the array itself, or
    `categories` of the object form. None for anything else -- including an
    object with keys other than STANDARD_OBJECT_KEYS, so a checker_rules
    YAML/JSON document is never mistaken for one."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "categories" in data and set(data) <= STANDARD_OBJECT_KEYS:
        return data["categories"]
    return None


def parse_standard_policies(raw_content: str) -> StandardPolicy | None:
    """Best-effort parse of `raw_content` as a JSON array of StandardPolicyCategory
    (or the object form's `categories` -- see STANDARD_OBJECT_KEYS above).

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
    items = standard_categories_json(data)
    if not isinstance(items, list) or not items:
        return None
    try:
        return [StandardPolicyCategory(**item) for item in items]
    except (ValidationError, TypeError):
        return None
