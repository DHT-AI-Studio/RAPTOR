"""Guard-model adapter contract — one adapter per guard-model family, owning
all family-specific prompt construction and response parsing. Nothing
outside app/adapters/ should know a model's prompt format or output shape;
the orchestrator (app/services/guard_classifier.py) only ever calls these
methods, never branches on model name or family itself.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

from app.models.policy import StandardPolicyCategory

# Where a composed prompt template's rendered text is inserted for a check:
# "system" (a system-role message), "final_user_message" (appended as the
# LAST user-role message, after the actual conversation — Granite Guardian's
# official format), or "raw_completion" (the entire /api/generate prompt,
# not a chat message list at all — Llama Guard 3's format).
PromptPlacement = Literal["system", "final_user_message", "raw_completion"]

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
class ComposedPromptTemplate:
    """A per-model prompt pre-composed at policy-activation time (see
    app/services/policy_translator.py) and persisted to the
    guardrail_policies content_* columns, instead of being rebuilt from
    scratch on every check call.

    `template` is Jinja2 source. For `placement in ("system",
    "final_user_message")` it's already fully resolved (no variables left —
    GPT-OSS/Granite's policy text has no dependency on the actual content
    being checked). For `placement == "raw_completion"` (Llama Guard 3, whose
    official format merges the category rules and the actual conversation
    into one prompt) the category rules are already baked in, but
    `check_role`/`conversation` remain real Jinja2 variables, rendered a
    second time at check time — see render_composed_template()."""
    placement: PromptPlacement
    template: str


@dataclass(frozen=True)
class PolicyContent:
    """A policy handed to an adapter in place of its fixed guard prompt.

    `standard_policies`, when set, is the parsed Standard Policy Format (see
    app/models/policy.py) — an adapter renders this into its OWN
    officially-documented prompt structure. `composed_template`, when set,
    is a ready-to-render prompt already composed at activation time (see
    ComposedPromptTemplate above) — an adapter uses it directly instead of
    calling its own `_render_*` functions again. When neither is set, `raw`
    is used verbatim instead (a per-model override column, or free-text/YAML
    content that doesn't parse as the standard format) — adapters never
    re-interpret `raw` text themselves."""
    raw: str
    standard_policies: list[StandardPolicyCategory] | None = None
    composed_template: ComposedPromptTemplate | None = None


def parse_policy_content(column_value: str | None) -> PolicyContent | None:
    """Build the PolicyContent an adapter should use for one guardrail_policies
    content column's value.

    Returns None when the column itself is unset (caller should fall back to
    the shared "original" content). Otherwise: tries to JSON-decode the value
    as a ComposedPromptTemplate envelope (`{"placement": ..., "template": ...}`)
    written by app/services/policy_translator.py at activation time; on
    anything else — a plain manual override string, free text, YAML, or
    malformed JSON — falls back to using the value verbatim as `raw`, exactly
    as every content_* column has always behaved. This keeps every
    pre-existing manually-configured policy working unchanged."""
    if not column_value:
        return None
    try:
        data = json.loads(column_value)
    except (json.JSONDecodeError, TypeError):
        return PolicyContent(raw=column_value)
    if not isinstance(data, dict) or set(data) != {"placement", "template"}:
        return PolicyContent(raw=column_value)
    try:
        composed = ComposedPromptTemplate(placement=data["placement"], template=data["template"])
    except (KeyError, TypeError):
        return PolicyContent(raw=column_value)
    return PolicyContent(raw=column_value, composed_template=composed)


def render_composed_template(composed: ComposedPromptTemplate, **variables: str) -> str:
    """Second-pass Jinja2 render at check time — a no-op substitution for
    "system"/"final_user_message" templates (they have no variables left),
    and the real check_role/conversation substitution for "raw_completion"
    (Llama Guard 3)."""
    import jinja2

    # autoescape is an HTML/XSS mitigation -- irrelevant here, this renders a
    # plain-text guard-model prompt, never HTML. Turning it on would corrupt
    # prompt content containing "<"/">"/"&" by HTML-entity-escaping it.
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)  # nosec B701
    return env.from_string(composed.template).render(**variables)


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
