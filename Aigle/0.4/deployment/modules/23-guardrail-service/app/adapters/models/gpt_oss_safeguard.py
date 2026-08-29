"""GPT-OSS Safeguard adapter — 0/1 verdict.

Fixed guard prompt: strict '0'/'1[\\nS-codes]' output contract.

Caller-supplied Policy, standard format (PolicyContent.standard_policies
set): rendered into gpt-oss-safeguard's own official policy-document
structure (app/official_prompts/gpt_oss_safeguard.txt: `# Name / ##
INSTRUCTIONS / ## DEFINITIONS / ## VIOLATES (1) / ## SAFE (0) / ##
EXAMPLES`). OpenAI's cookbook guide confirms multiple policies can be
combined in one system message, "each policy as a separate section" — one
section per StandardPolicy is rendered here, with the output-format
instruction repeated near the top and bottom per the guide's own advice.
The guide offers three output-format options (plain 0/1, a JSON
{"violation":..,"policy_category":..} object, or JSON+rationale); this
keeps the plain '0'/'1\\n<ids>' convention already used everywhere else in
this system (which `_parse_gpt_oss_policy_verdict` below already parses
robustly) rather than switching to JSON, since that's explicitly one of the
guide's own supported options and avoids a parser rewrite for no benefit
here — written directly into the INSTRUCTIONS section, per the template's
own instruction to define the output contract there.

Caller-supplied Policy, verbatim (PolicyContent.raw only): the model
sometimes refuses in plain language instead of following the strict format,
so parsing needs a second, policy-aware variant (see `parse`).

Continuation-trap fix (all three system-prompt variants above): a chat
message list that ends on an assistant-authored turn gets treated by Ollama's
chat templating as a request to continue that turn, not classify it — for
output checks (role="assistant") this made the model keep writing as itself
instead of returning a verdict. build_request/build_conversation_request
therefore always end the message list on a single "user"-role message (the
checked content labeled "[User Input]"/"[AI Response]" for single-message
checks, or the whole conversation flattened into one "Classify the final
[...]" instruction for conversation mode) regardless of which system-prompt
branch is selected — only the system-message content varies by branch.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.adapters.base import BaseGuardAdapter, ChatMessage, GuardRequest, GuardVerdict, PolicyContent
from app.adapters.registry import register
from app.adapters.standard_policy import StandardPolicy

_logger = logging.getLogger(__name__)
_SYSTEM_PROMPT = (Path(__file__).parent.parent.parent / "guard_prompts" / "gpt_oss_safeguard.txt").read_text(encoding="utf-8")

_OUTPUT_INSTRUCTION = (
    "Respond with exactly one character on the first line: 0 or 1.\n"
    "0 = the content does NOT violate any policy below.\n"
    "1 = the content violates at least one policy below.\n"
    "If 1, add a second line with the comma-separated id(s) of every violated policy.\n"
    "No other text."
)


def _render_policy_section(p: StandardPolicy) -> str:
    criteria = "\n".join(f"- {c}" for c in p.criteria)
    exceptions = "\n".join(f"- {e}" for e in p.exceptions) if p.exceptions else "(none specified)"
    violation_examples = "\n".join(f"- {e}" for e in p.examples.violation) if p.examples.violation else "(none provided)"
    allowed_examples = "\n".join(f"- {e}" for e in p.examples.allowed) if p.examples.allowed else "(none provided)"
    return (
        f"# Policy {p.id}: {p.name}\n\n"
        f"## INSTRUCTIONS\n{_OUTPUT_INSTRUCTION}\n\n"
        f"## DEFINITIONS\n{p.description}\n\n"
        f"## VIOLATES [{p.id}]\n{criteria}\n\n"
        f"## SAFE [{p.id}]\n{exceptions}\n\n"
        f"## EXAMPLES\nViolation examples:\n{violation_examples}\n\nAllowed examples:\n{allowed_examples}"
    )


def _render_standard_policy_prompt(policies: list[StandardPolicy]) -> str:
    """Combine every StandardPolicy into one system prompt, one official-
    structure section each, with the output instruction repeated top+bottom
    per OpenAI's cookbook guidance for multi-policy prompts."""
    sections = "\n\n---\n\n".join(_render_policy_section(p) for p in policies)
    return f"{_OUTPUT_INSTRUCTION}\n\n{sections}\n\n{_OUTPUT_INSTRUCTION}"


def _parse_gpt_oss_verdict(text: str) -> tuple[bool, list[str]]:
    """Parse gpt-oss-safeguard output under the fixed guard prompt: '0'
    (safe) or '1\\nS1,S9' (unsafe + S-codes)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        _logger.warning("gpt-oss returned empty — treating as unsafe")
        return False, []
    first = lines[0].rstrip(".,;:")
    if first == "0":
        return True, []
    codes: list[str] = []
    for ln in lines[1:]:
        for part in ln.split(","):
            code = part.strip().upper()
            if code:
                codes.append(code)
    return False, codes


def _parse_gpt_oss_policy_verdict(text: str) -> tuple[bool, list[str]]:
    """Parse gpt-oss-safeguard output when the system prompt is a
    caller-supplied Policy rather than the fixed guard prompt that demands
    strict '0'/'1' output. Under an arbitrary policy prompt, two things the
    fixed-prompt format doesn't need to worry about can happen:

    1. The model isn't reliably constrained to its native '0'/'1\\n<codes>'
       vocabulary — depending on how the policy is worded it may answer in
       llama-guard-style 'safe'/'unsafe\\n<codes>' instead. Both are accepted
       here, case-insensitively.
    2. The verdict line isn't guaranteed to be first — the model sometimes
       prepends free-form reasoning/continuation text before it. So lines
       are scanned from the END backward for the last line that reduces
       (after stripping trailing punctuation and lowercasing) to exactly
       one of the four recognized tokens; everything after that line is
       the comma-separated codes. A verdict is never expected to be
       followed by more prose per the prompt's own instructions, so
       "last matching line" is the right anchor in both the
       verdict-first and prose-then-verdict cases.

    If no line anywhere matches — e.g. a plain-language refusal like "I'm
    sorry, but I can't help with that." with no explicit verdict token at
    all — it's an implicit unsafe verdict with no codes to extract, so
    refusal prose never gets mis-parsed as comma-separated codes.
    """
    if not text.strip():
        _logger.warning("gpt-oss(policy) returned empty — treating as unsafe")
        return False, []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        token = lines[i].rstrip(".,;:").lower()
        if token in ("safe", "0"):
            return True, []
        if token in ("unsafe", "1"):
            codes: list[str] = []
            for ln in lines[i + 1:]:
                for part in ln.split(","):
                    code = part.strip().upper()
                    if code:
                        codes.append(code)
            return False, codes
    _logger.info(
        "gpt-oss(policy) non-standard output (likely a safety refusal) — treating as unsafe: %s",
        lines[0][:200],
    )
    return False, []


class GptOssSafeguardAdapter(BaseGuardAdapter):
    family = "gpt_oss"
    priority = 1

    def build_request(self, content: str, role: str, *, policy: PolicyContent | None = None) -> GuardRequest:
        if policy is not None and policy.standard_policies:
            system = _render_standard_policy_prompt(policy.standard_policies)
        elif policy is not None:
            system = policy.raw
        else:
            system = _SYSTEM_PROMPT
        # Always wrap as user role to prevent the model from continuing an assistant turn.
        role_label = "AI Response" if role == "assistant" else "User Input"
        user_msg = {"role": "user", "content": f"[{role_label}]: {content}"}
        return GuardRequest(endpoint="chat", payload={
            "messages": [{"role": "system", "content": system}, user_msg],
            "stream": False, "options": {"temperature": 0.0},
        })

    def build_conversation_request(self, messages: list[ChatMessage], *, policy: PolicyContent | None = None) -> GuardRequest:
        if policy is not None and policy.standard_policies:
            system = _render_standard_policy_prompt(policy.standard_policies)
        elif policy is not None:
            system = policy.raw
        else:
            system = _SYSTEM_PROMPT
        # Flatten conversation to avoid the assistant-role continuation trap.
        conv_text = "\n".join(f"[{'AI' if m.role == 'assistant' else 'User'}]: {m.content}" for m in messages)
        check_label = "AI Response" if messages[-1].role == "assistant" else "User Input"
        user_msg = {"role": "user", "content": (
            f"Conversation history:\n{conv_text}\n\nClassify the final [{check_label}] in this conversation."
        )}
        return GuardRequest(endpoint="chat", payload={
            "messages": [{"role": "system", "content": system}, user_msg],
            "stream": False, "options": {"temperature": 0.0},
        })

    def parse(self, raw_text: str, *, used_policy_prompt: bool = False) -> GuardVerdict:
        if used_policy_prompt:
            safe, categories = _parse_gpt_oss_policy_verdict(raw_text)
        else:
            safe, categories = _parse_gpt_oss_verdict(raw_text)
        return GuardVerdict(safe=safe, categories=categories, raw=raw_text)


register(GptOssSafeguardAdapter(), match_substrings=["gpt-oss", "safeguard"])
