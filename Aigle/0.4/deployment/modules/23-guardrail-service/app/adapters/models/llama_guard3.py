"""Llama Guard 3 adapter — safe/unsafe verdict.

Fixed guard prompt: single-message checks go through /api/chat with no
system prompt (Ollama applies the model's own chat template); conversation
checks go through /api/generate instead, since /api/chat treats a completed
conversation (last role = assistant) as a continuation request, which makes
the model always return 'safe'.

/api/generate calls here must NOT set raw=True. That flag means "no
formatting applied to the prompt", which — confirmed by a real bug report —
also disables Ollama's special-token *parsing* of the prompt text: our
hand-authored literal strings like <|header_start|>/<|eot|> get tokenized as
ordinary sub-word text instead of the real special-token IDs the model
needs to recognize turn boundaries, so it can't reliably read the
<BEGIN UNSAFE CONTENT CATEGORIES> block and falls back to its own trained-in
taxonomy (e.g. answering S2 while ignoring custom F1-F6 categories
entirely). A/B-confirmed via POST /guardrail/prompt_test (same prompt text,
no `raw` key at all) returning the correct category vs. this adapter's old
raw=True request returning a trained-in default — so `raw` is simply
omitted from the payload now, matching /guardrail/prompt_test's request
shape exactly.

The <BEGIN CONVERSATION> transcript also needs at least one turn before an
Agent (assistant) turn being evaluated. A single, standalone "Agent: ..."
line with nothing preceding it (what an output-only check with no prior
context naturally produces) isn't a shape any real Llama Guard example ever
shows — every official example pairs a User turn before the Agent turn it
demonstrates checking — and isn't reliably parsed: confirmed by a real bug
report where such a transcript made llama-guard3:8b ignore a custom
category block entirely and fall back to a trained-in default (S1), while
the other two guard models checking the identical content both correctly
cited a real custom category, and while the otherwise-identical single
"User: ..." shape (used for input-only checks, which never has this
problem) works fine. build_conversation_request therefore prepends a
minimal placeholder "User: (no prior message)" turn whenever the transcript
would otherwise start on an assistant turn — this only changes the
rendered transcript text; check_role (which role the instructions tell the
model to evaluate) still comes from the real last message, unaffected.

Caller-supplied Policy, standard format (PolicyContent.standard_policies
set): rendered into a custom-category block matching the real prompt
structure in app/official_prompts/llama_guard.txt (updated with the actual
official template) and confirmed against a real request/response pair
(llama-guard3:8b correctly returning 'unsafe\nM1' for custom M1-M5 medical-
misinformation categories, where the earlier "Should not"/"Can" format —
adopted from a Medium worked example in a previous session — had the model
ignore the custom categories and fall back to a trained-in default like S1):

    M1 - Medical Misinformation

    Description:
    Providing false, misleading, or unsupported medical information...

    Violation criteria:
    - Claims scientifically unverified treatments can cure diseases.
    - ...

    Allowed:
    - Medical research, clinical trials, or academic discussions.
    - ...

    Examples of violations:
    - Drinking bleach can cure COVID-19.
    - ...

`criteria` -> "Violation criteria:", `exceptions` -> "Allowed:",
`examples.violation` -> "Examples of violations:" are all confirmed
field-for-field against the real request. `examples.allowed` isn't exercised
by that request (none of its 5 categories set it) but still needs somewhere
to go, so it's rendered as a symmetric "Examples of allowed:" section, same
omit-if-empty rule as the other three. Categories are separated by two
blank lines; sections within one category by one blank line — verified
line-by-line against both the official .txt file and the real request's
exact text.

The outer scaffolding (Task:/<BEGIN CONVERSATION>/... instructions) also
gained a "Classification Rules:" + "Output Format:" block between
<END CONVERSATION> and the final "Provide your safety assessment..." line,
per the same official .txt file — see _CUSTOM_RAW_TEMPLATE.

Not adopted: the .txt file's vision/image placeholder tokens (multimodal
Llama Guard 4 variant) — not relevant to this service's text-only checks,
so the conversation section here stays plain "User: .../Agent: ..." text,
same as before. `<|header_start|>`/`<|header_end|>`/`<|eot|>` (confirmed in
a previous session to be what the deployed llama-guard3:8b model actually
needs, despite the v3-suggesting tag name) are unchanged.

Also per the real format: the content being checked is ALWAYS wrapped in
<BEGIN CONVERSATION>...<END CONVERSATION> inside a single raw-completion
prompt — there's no separate "no conversation wrapper" mode. So
build_request's standard-policies branch below delegates to
build_conversation_request with a one-message list, rather than building a
second, chat-style prompt.

Caller-supplied Policy, verbatim (PolicyContent.raw only): unchanged —
injected as-is as the system message.
"""
from __future__ import annotations

from pathlib import Path

from app.adapters.base import BaseGuardAdapter, ChatMessage, GuardRequest, GuardVerdict, PolicyContent
from app.adapters.registry import register
from app.adapters.standard_policy import StandardPolicy

_ROLE_MAP: dict[str, str] = {"user": "User", "assistant": "Agent"}
_RAW_TEMPLATE = (Path(__file__).parent.parent.parent / "guard_prompts" / "llama_guard3.txt").read_text(encoding="utf-8")

_CUSTOM_RAW_TEMPLATE = (
    "<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
    "Task: Check if there is unsafe content in '{check_role}' messages in conversations according to our "
    "safety policy with the below categories.\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n{unsafe_categories}\n<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN CONVERSATION>\n\n{conv}\n\n<END CONVERSATION>\n\n"
    "Classification Rules:\n\n"
    "- Evaluate only the specified role.\n"
    "- Determine whether the specified role violates our safety policy.\n"
    "- A response may violate multiple categories.\n"
    "- Use only category IDs defined in our safety policy.\n"
    "- Do not invent new categories.\n"
    "- Do not provide explanations.\n"
    "- Do not provide reasoning.\n"
    "- Do not output any additional text.\n\n\n"
    "Output Format:\n\n"
    "If the content is safe:\n\n"
    "safe\n\n\n"
    "If the content is unsafe:\n\n"
    "unsafe\n"
    "<comma-separated category IDs>\n\n"
    "Provide your safety assessment for ONLY THE LAST {check_role} in the above conversation:\n"
    "- First line must read 'safe' or 'unsafe'.\n"
    "- If unsafe, a second line must include a comma-separated list of violated categories."
    "<|eot|><|header_start|>assistant<|header_end|>\n\n"
)


def _render_category_block(policies: list[StandardPolicy]) -> str:
    """Custom-category block matching the real official prompt structure
    (see module docstring): "{id}: {name}" then "Description:", "Violation
    criteria:" (from `criteria`), "Allowed:" (from `exceptions`), "Examples
    of violations:" (from `examples.violation`), "Examples of allowed:"
    (from `examples.allowed`) — each of the last four sections omitted
    entirely when it would have no bullets. Categories are separated by two
    blank lines; sections within one category by one blank line.
    """
    blocks: list[str] = []
    for p in policies:
        sections = [f"{p.id}: {p.name}", f"Description:\n{p.description}"]
        if p.criteria:
            sections.append("Violation criteria:\n" + "\n".join(f"- {c}" for c in p.criteria))
        if p.exceptions:
            sections.append("Allowed:\n" + "\n".join(f"- {e}" for e in p.exceptions))
        if p.examples.violation:
            sections.append("Examples of violations:\n" + "\n".join(f"- {e}" for e in p.examples.violation))
        if p.examples.allowed:
            sections.append("Examples of allowed:\n" + "\n".join(f"- {e}" for e in p.examples.allowed))
        blocks.append("\n\n".join(sections))
    return "\n\n\n".join(blocks)


def _parse_verdict(text: str) -> tuple[bool, list[str]]:
    """Parse llama-guard3 output: 'safe' or 'unsafe\\nS1,S9'."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return True, []
    if lines[0].lower() == "safe":
        return True, []
    codes: list[str] = []
    for ln in lines[1:]:
        for part in ln.split(","):
            code = part.strip().upper()
            if code:
                codes.append(code)
    return False, codes


class LlamaGuard3Adapter(BaseGuardAdapter):
    family = "llama_guard3"
    priority = 0

    def build_request(self, content: str, role: str, *, policy: PolicyContent | None = None) -> GuardRequest:
        if policy is not None and policy.standard_policies:
            # The real format has no separate "single message, no conversation
            # wrapper" mode — content is always inside <BEGIN CONVERSATION>.
            return self.build_conversation_request([ChatMessage(role=role, content=content)], policy=policy)
        if policy is not None:
            return GuardRequest(endpoint="chat", payload={
                "messages": [{"role": "system", "content": policy.raw}, {"role": role, "content": content}],
                "stream": False, "options": {"temperature": 0.0},
            })
        return GuardRequest(endpoint="chat", payload={
            "messages": [{"role": role, "content": content}],
            "stream": False, "options": {"temperature": 0.0},
        })

    def build_conversation_request(self, messages: list[ChatMessage], *, policy: PolicyContent | None = None) -> GuardRequest:
        if policy is not None and policy.standard_policies:
            check_role = _ROLE_MAP.get(messages[-1].role, "User")
            conv = "\n\n".join(f"{_ROLE_MAP.get(m.role, m.role.capitalize())}: {m.content}" for m in messages)
            if messages[0].role == "assistant":
                # A transcript starting (and, for a single-message output
                # check, consisting solely of) an Agent turn with no prior
                # User turn isn't a shape the model reliably parses — see
                # module docstring.
                conv = f"User: (no prior message)\n\n{conv}"
            prompt = _CUSTOM_RAW_TEMPLATE.format(
                check_role=check_role, conv=conv,
                unsafe_categories=_render_category_block(policy.standard_policies),
            )
            return GuardRequest(endpoint="generate", payload={
                "prompt": prompt, "stream": False, "options": {"temperature": 0.0},
            })
        if policy is not None:
            chat_messages = [{"role": m.role, "content": m.content} for m in messages]
            return GuardRequest(endpoint="chat", payload={
                "messages": [{"role": "system", "content": policy.raw}, *chat_messages],
                "stream": False, "options": {"temperature": 0.0},
            })
        check_role = _ROLE_MAP.get(messages[-1].role, "User")
        conv = "\n\n".join(f"{_ROLE_MAP.get(m.role, m.role.capitalize())}: {m.content}" for m in messages)
        if messages[0].role == "assistant":
            conv = f"User: (no prior message)\n\n{conv}"
        prompt = _RAW_TEMPLATE.format(check_role=check_role, conv=conv)
        return GuardRequest(endpoint="generate", payload={
            "prompt": prompt, "stream": False, "options": {"temperature": 0.0},
        })

    def parse(self, raw_text: str, *, used_policy_prompt: bool = False) -> GuardVerdict:
        safe, categories = _parse_verdict(raw_text)
        return GuardVerdict(safe=safe, categories=categories, raw=raw_text)


register(LlamaGuard3Adapter(), match_substrings=["llama-guard", "llama_guard"])
