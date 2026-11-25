"""Granite Guardian adapter — `<score> yes/no </score>` verdict, via a fixed
system-prompt criteria list, a caller-supplied Policy in its place, or (new)
an auto-converted Standard Policy.

Caller-supplied Policy, standard format (PolicyContent.standard_policies
set): rendered into Granite Guardian's own official "guardian block"
(app/official_prompts/granite_guardian.txt: `<guardian>{think_or_nothink}
### Criteria: {criteria_text} ### Scoring Schema: ...`), confirmed via the
upstream README/DeepWiki to be appended as the FINAL USER-ROLE message —
not a system message, unlike this adapter's other two paths below. Multiple
StandardPolicyCategory criteria are combined into one criteria_text block (Granite's
official format has no per-category ID field, so — same as today — a
violation still can't be attributed to a specific category; `think_or_nothink`
is normally an HF-tokenizer-level boolean, not reproducible via Ollama's
plain /api/chat, so it's simply omitted here, keeping output as plain
'yes'/'no' with no <think> block to strip).

Caller-supplied Policy, verbatim (PolicyContent.raw only) or no policy at
all: unchanged — system-message based, as before.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from app.adapters.base import (
    BaseGuardAdapter,
    ChatMessage,
    GuardRequest,
    GuardVerdict,
    PolicyContent,
    render_composed_template,
)
from app.adapters.registry import register
from app.models.policy import StandardPolicyCategory

_logger = logging.getLogger(__name__)
_SYSTEM_PROMPT = (Path(__file__).parent.parent.parent / "guard_prompts" / "granite_guardian.txt").read_text(encoding="utf-8")


def _render_criteria_text(policies: list[StandardPolicyCategory]) -> str:
    """Combine every StandardPolicyCategory's criteria/exceptions into one prose
    block for Granite's single criteria_text slot."""
    parts = []
    for p in policies:
        block = f"[{p.id}] {p.name}: {p.description}\nViolates if:\n" + "\n".join(f"  - {c}" for c in p.criteria)
        if p.exceptions:
            block += "\nExceptions (do NOT flag):\n" + "\n".join(f"  - {e}" for e in p.exceptions)
        parts.append(block)
    return "\n\n".join(parts)


def _render_guardian_block(policies: list[StandardPolicyCategory]) -> str:
    criteria_text = _render_criteria_text(policies)
    return (
        f"<guardian>\n\n"
        f"### Criteria: {criteria_text}\n\n"
        f"### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
    )


def _parse_granite_verdict(text: str) -> tuple[bool, list[str]]:
    """Parse granite4.1-guardian output: '<score> yes/no </score>'.

    'yes' = harmful (unsafe). Falls back to first-word check. Fails closed on empty.
    """
    if not text.strip():
        _logger.warning("granite returned empty response — treating as unsafe")
        return False, []
    m = re.search(r"<score>\s*(\w+)\s*</score>", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).lower()
        if verdict in ("yes", "harmful", "unsafe"):
            return False, []
        return True, []
    first_word = text.strip().split()[0].rstrip(".,;:").lower()
    if first_word in ("yes", "harmful", "unsafe"):
        return False, []
    return True, []


class GraniteGuardianAdapter(BaseGuardAdapter):
    family = "granite"
    priority = 2

    @staticmethod
    def _resolve_guardian_block(policy: PolicyContent | None) -> str | None:
        """None means "no guardian block" — fall back to the system-prompt path."""
        if policy is not None and policy.composed_template is not None:
            # Pre-composed at policy-activation time (app/services/policy_translator.py)
            # — already the exact text _render_guardian_block would produce
            # (plus the official <no-think> mode instruction); no variables
            # left to substitute for this model.
            return render_composed_template(policy.composed_template)
        if policy is not None and policy.standard_policies:
            return _render_guardian_block(policy.standard_policies)
        return None

    def build_request(self, content: str, role: str, *, policy: PolicyContent | None = None) -> GuardRequest:
        guardian_block = self._resolve_guardian_block(policy)
        if guardian_block is not None:
            return GuardRequest(endpoint="chat", payload={
                "messages": [{"role": role, "content": content}, {"role": "user", "content": guardian_block}],
                "stream": False, "options": {"temperature": 0.0},
            })
        system = policy.raw if policy is not None else _SYSTEM_PROMPT
        return GuardRequest(endpoint="chat", payload={
            "messages": [{"role": "system", "content": system}, {"role": role, "content": content}],
            "stream": False, "options": {"temperature": 0.0},
        })

    def build_conversation_request(self, messages: list[ChatMessage], *, policy: PolicyContent | None = None) -> GuardRequest:
        guardian_block = self._resolve_guardian_block(policy)
        if guardian_block is not None:
            chat_messages = [{"role": m.role, "content": m.content} for m in messages]
            return GuardRequest(endpoint="chat", payload={
                "messages": [*chat_messages, {"role": "user", "content": guardian_block}],
                "stream": False, "options": {"temperature": 0.0},
            })
        system = policy.raw if policy is not None else _SYSTEM_PROMPT
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        return GuardRequest(endpoint="chat", payload={
            "messages": [{"role": "system", "content": system}, *chat_messages],
            "stream": False, "options": {"temperature": 0.0},
        })

    def parse(self, raw_text: str, *, used_policy_prompt: bool = False) -> GuardVerdict:
        safe, categories = _parse_granite_verdict(raw_text)
        return GuardVerdict(safe=safe, categories=categories, raw=raw_text)


register(GraniteGuardianAdapter(), match_substrings=["granite", "guardian"])
