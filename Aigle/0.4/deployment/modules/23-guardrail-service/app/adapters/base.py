"""Guard-model adapter contract — one adapter per guard-model family, owning
all family-specific prompt construction and response parsing. Nothing
outside app/adapters/ should know a model's prompt format or output shape;
the orchestrator (app/services/guard_classifier.py) only ever calls these
methods, never branches on model name or family itself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

from app.adapters.standard_policy import StandardPolicy

# MLCommons hazard taxonomy (S1-S14) — shared across every adapter that
# reports S-codes; not specific to any one model, so it lives here rather
# than inside any single adapter.
CATEGORY_NAMES: dict[str, str] = {
    "S1":  "Violent Crimes",
    "S2":  "Non-Violent Crimes",
    "S3":  "Sex Crimes",
    "S4":  "Child Exploitation",
    "S5":  "Defamation",
    "S6":  "Specialized Advice",
    "S7":  "Privacy",
    "S8":  "Intellectual Property",
    "S9":  "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


@dataclass(frozen=True)
class ChatMessage:
    role: str        # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class PolicyContent:
    """A policy handed to an adapter in place of its fixed guard prompt.

    `standard_policies`, when set, is the parsed Standard Policy Format (see
    app/adapters/standard_policy.py) — an adapter renders this into its OWN
    officially-documented prompt structure. When it's None, `raw` is used
    verbatim instead (a per-model override column, or free-text/YAML content
    that doesn't parse as the standard format) — adapters never re-interpret
    `raw` text themselves, only `standard_policies`."""
    raw: str
    standard_policies: list[StandardPolicy] | None = None


@dataclass(frozen=True)
class GuardRequest:
    """Everything app.adapters.ollama_client needs to make the call — the
    adapter decides both the payload and which Ollama endpoint shape this
    model needs (e.g. a raw-completion call instead of a chat call for
    conversation checks)."""
    endpoint: Literal["chat", "generate"]
    payload: dict


@dataclass(frozen=True)
class GuardVerdict:
    safe: bool
    categories: list[str]
    raw: str


class BaseGuardAdapter(ABC):
    """One adapter per guard-model family."""

    family: ClassVar[str]
    priority: ClassVar[int] = 100   # lower = preferred as "primary" when models disagree

    @abstractmethod
    def build_request(self, content: str, role: str, *, policy: PolicyContent | None = None) -> GuardRequest:
        """Single-message check. `policy`, when given, replaces this
        adapter's own fixed guard prompt with the caller-supplied Policy."""

    def build_conversation_request(self, messages: list[ChatMessage], *, policy: PolicyContent | None = None) -> GuardRequest:
        """Default: reduce to a single-message check on the last message.
        Override only if a model needs real multi-turn handling."""
        last = messages[-1]
        return self.build_request(last.content, last.role, policy=policy)

    @abstractmethod
    def parse(self, raw_text: str, *, used_policy_prompt: bool = False) -> GuardVerdict:
        """Parse this model's raw output into a normalized verdict.
        `used_policy_prompt=True` when this output was produced under a
        caller-supplied Policy instead of the fixed guard prompt — lets an
        adapter choose a different parsing strategy if that changes its
        output format (only gpt-oss needs this today)."""
