"""Structured guardrail policy model (GB-2 / GB-4, detector-based).

A **policy** is one document (one active at a time). It contains many **rules**,
grouped into `input_guardrail` and `output_guardrail`. Each rule maps a
`category` to an `action`; by default the category is matched against the guard
model's (Llama Guard) classification, but a rule may set `detector: regex` +
`patterns` to be matched deterministically without an LLM call.

Policies are authored as YAML/JSON and stored as text (`raw_content`); the
checker parses that text into a `GuardrailPolicy` via `GuardrailPolicy.from_raw`.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Action = Literal["block", "redact", "flag"]
Detector = Literal["llama_guard", "regex"]


class PolicyRule(BaseModel):
    category: str                                   # Llama Guard code (S1..S14) or a custom label (jailbreak, pii)
    action: Action
    detector: Detector = "llama_guard"              # default: match against the guard model's classification
    patterns: list[str] = Field(default_factory=list)   # required when detector == "regex"

    @model_validator(mode="after")
    def _regex_needs_patterns(self) -> "PolicyRule":
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
    def from_raw(cls, raw_content: str) -> "GuardrailPolicy":
        """Parse a policy document (YAML, JSON is a subset) into a GuardrailPolicy."""
        data = yaml.safe_load(raw_content) or {}
        if not isinstance(data, dict):
            raise ValueError("policy content must be a YAML/JSON object")
        return cls.model_validate(data)

    def block_for(self, direction: Literal["input", "output"]) -> GuardrailBlock:
        return self.input_guardrail if direction == "input" else self.output_guardrail
