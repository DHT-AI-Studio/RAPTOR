"""Unit tests for guard-model adapters (app/adapters/) — pure prompt-building
and response-parsing logic, no HTTP/Ollama involved."""
import pytest

from app.adapters.base import ChatMessage, PolicyContent
from app.adapters.models.gpt_oss_safeguard import GptOssSafeguardAdapter
from app.adapters.models.granite_guardian import GraniteGuardianAdapter
from app.adapters.models.llama_guard3 import LlamaGuard3Adapter
from app.adapters.registry import UnknownGuardModelError, get_adapter
from app.adapters.standard_policy import StandardPolicy, StandardPolicyExamples

_TWO_POLICIES = [
    StandardPolicy(
        id="M1", name="Medical Misinformation", description="Dangerous unproven medical claims.",
        severity="high", decision="block",
        criteria=["Claims a substance cures a disease without evidence."],
        exceptions=["Academic discussion of debunked claims."],
        examples=StandardPolicyExamples(
            violation=["Drinking bleach cures viral infections."],
            allowed=["Bleach is dangerous if ingested."],
        ),
    ),
    StandardPolicy(
        id="F2", name="Financial Fraud", description="Content promoting scams.",
        severity="critical", decision="block",
        criteria=["Promotes a guaranteed-return investment scheme."],
    ),
]
_STANDARD_POLICY = PolicyContent(raw="[...]", standard_policies=_TWO_POLICIES)


# ── LlamaGuard3Adapter ───────────────────────────────────────────────────────

def test_llama_guard3_build_request_no_policy_has_no_system_message():
    req = LlamaGuard3Adapter().build_request("hello", "user")
    assert req.endpoint == "chat"
    assert req.payload["messages"] == [{"role": "user", "content": "hello"}]


def test_llama_guard3_build_request_with_policy_injects_system_message():
    req = LlamaGuard3Adapter().build_request("hello", "user", policy=PolicyContent(raw="my policy"))
    assert req.payload["messages"][0] == {"role": "system", "content": "my policy"}
    assert req.payload["messages"][1] == {"role": "user", "content": "hello"}


def test_llama_guard3_build_conversation_no_policy_uses_raw_generate_endpoint():
    messages = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]
    req = LlamaGuard3Adapter().build_conversation_request(messages)
    assert req.endpoint == "generate"
    # raw=True must NOT be set — it disables Ollama's special-token parsing of
    # our hand-authored <|header_start|>/<|eot|> text (confirmed via a real
    # bug report: the model ignored custom categories under raw=True but
    # correctly read them when raw was omitted, matching /guardrail/prompt_test).
    assert "raw" not in req.payload
    assert "hi" in req.payload["prompt"] and "hello" in req.payload["prompt"]
    # Fixed prompt (app/guard_prompts/llama_guard3.txt) also uses v4-style tokens.
    assert "<|header_start|>user<|header_end|>" in req.payload["prompt"]
    assert "<|eot|><|header_start|>assistant<|header_end|>" in req.payload["prompt"]
    assert "<|start_header_id|>" not in req.payload["prompt"]
    assert "<|eot_id|>" not in req.payload["prompt"]


def test_llama_guard3_build_conversation_with_policy_uses_chat_endpoint():
    messages = [ChatMessage(role="user", content="hi")]
    req = LlamaGuard3Adapter().build_conversation_request(messages, policy=PolicyContent(raw="pol"))
    assert req.endpoint == "chat"
    assert req.payload["messages"][0] == {"role": "system", "content": "pol"}


def test_llama_guard3_build_request_with_standard_policies_uses_raw_generate_with_conversation_wrapper():
    # No separate "single message" mode for standard policies — content is
    # always inside <BEGIN CONVERSATION>, via the same raw-completion path
    # conversation-mode uses (delegates to build_conversation_request).
    req = LlamaGuard3Adapter().build_request("hello", "user", policy=_STANDARD_POLICY)
    assert req.endpoint == "generate"
    assert "raw" not in req.payload   # see the no-policy test above for why
    prompt = req.payload["prompt"]
    assert "<BEGIN CONVERSATION>" in prompt and "<END CONVERSATION>" in prompt
    assert "User: hello" in prompt
    # A standalone User turn (input check) needs no placeholder prepended.
    conv_start = prompt.index("<BEGIN CONVERSATION>")
    assert "(no prior message)" not in prompt[conv_start:]


def test_llama_guard3_build_request_output_check_gets_placeholder_user_turn():
    # Bug: a transcript consisting solely of a standalone "Agent: ..." turn
    # (what an output-only check with no prior context naturally produces)
    # made llama-guard3:8b ignore custom categories and fall back to a
    # trained-in default — confirmed by a real request/response pair. Fix:
    # prepend a placeholder User turn whenever the transcript would
    # otherwise start on an assistant turn.
    req = LlamaGuard3Adapter().build_request("hello", "assistant", policy=_STANDARD_POLICY)
    prompt = req.payload["prompt"]
    conv_start = prompt.index("<BEGIN CONVERSATION>")
    conv_end = prompt.index("<END CONVERSATION>")
    conv_block = prompt[conv_start:conv_end]
    assert "User: (no prior message)" in conv_block
    assert "Agent: hello" in conv_block
    assert conv_block.index("User: (no prior message)") < conv_block.index("Agent: hello")
    # check_role (what the instructions tell the model to evaluate) is still
    # the real last message's role, unaffected by the placeholder.
    assert "Task: Check if there is unsafe content in 'Agent' messages" in prompt
    assert "Provide your safety assessment for ONLY THE LAST Agent in the above conversation:" in prompt


def test_llama_guard3_build_conversation_with_standard_policies_real_conversation_unaffected():
    # Regression: a genuine multi-turn conversation that already starts with
    # a user turn must not get a placeholder prepended.
    messages = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]
    req = LlamaGuard3Adapter().build_conversation_request(messages, policy=_STANDARD_POLICY)
    prompt = req.payload["prompt"]
    conv_start = prompt.index("<BEGIN CONVERSATION>")
    assert "(no prior message)" not in prompt[conv_start:]
    assert "User: hi" in prompt and "Agent: hello" in prompt


def test_llama_guard3_build_conversation_no_policy_output_check_gets_placeholder_user_turn():
    # Same fix applied to the fixed/no-policy raw-completion template, for
    # the same solitary-Agent-turn edge case.
    messages = [ChatMessage(role="assistant", content="hello")]
    req = LlamaGuard3Adapter().build_conversation_request(messages)
    prompt = req.payload["prompt"]
    conv_start = prompt.index("<BEGIN CONVERSATION>")
    conv_end = prompt.index("<END CONVERSATION>")
    conv_block = prompt[conv_start:conv_end]
    assert "User: (no prior message)" in conv_block
    assert "Agent: hello" in conv_block


def test_llama_guard3_build_request_with_standard_policies_uses_v4_style_tokens():
    # The deployed llama-guard3:8b model empirically requires Llama Guard
    # 4-style special tokens despite the v3-suggesting tag name.
    req = LlamaGuard3Adapter().build_request("hello", "user", policy=_STANDARD_POLICY)
    prompt = req.payload["prompt"]
    assert "<|header_start|>user<|header_end|>" in prompt
    assert "<|eot|><|header_start|>assistant<|header_end|>" in prompt
    assert "<|start_header_id|>" not in prompt
    assert "<|end_header_id|>" not in prompt
    assert "<|eot_id|>" not in prompt


def test_llama_guard3_category_block_matches_real_official_template():
    # Custom-category format confirmed against a real request/response pair
    # (llama-guard3:8b correctly returning 'unsafe\nF2' for custom F1-F6
    # categories under this exact shape) and app/official_prompts/llama_guard.txt:
    # "{id}: {name}" / "Description:" / "Violation criteria:" (criteria) /
    # "Allowed:" (exceptions) / "Examples of violations:" (examples.violation) /
    # "Examples of allowed:" (examples.allowed) — each of the last four
    # sections omitted when it would be empty.
    req = LlamaGuard3Adapter().build_request("hello", "user", policy=_STANDARD_POLICY)
    prompt = req.payload["prompt"]
    assert "<BEGIN UNSAFE CONTENT CATEGORIES>" in prompt
    assert "<END UNSAFE CONTENT CATEGORIES>" in prompt

    # M1 has criteria + exceptions + examples.violation + examples.allowed
    # -> all four sections present, in that order.
    assert "M1: Medical Misinformation" in prompt
    m1_start = prompt.index("M1: Medical Misinformation")
    f2_start = prompt.index("F2: Financial Fraud")
    m1_block = prompt[m1_start:f2_start]
    assert "Description:\nDangerous unproven medical claims." in m1_block
    assert "Violation criteria:\n- Claims a substance cures a disease without evidence." in m1_block
    assert "Allowed:\n- Academic discussion of debunked claims." in m1_block
    assert "Examples of violations:\n- Drinking bleach cures viral infections." in m1_block
    assert "Examples of allowed:\n- Bleach is dangerous if ingested." in m1_block
    assert (m1_block.index("Violation criteria:") < m1_block.index("Allowed:")
            < m1_block.index("Examples of violations:") < m1_block.index("Examples of allowed:"))

    # F2 has criteria only (no exceptions, no examples at all)
    # -> "Violation criteria:" present, the other three sections omitted entirely.
    assert "F2: Financial Fraud" in prompt
    f2_block = prompt[f2_start:f2_start + 200]
    assert "Violation criteria:\n- Promotes a guaranteed-return investment scheme." in f2_block
    assert "Allowed:" not in f2_block
    assert "Examples of violations:" not in f2_block
    assert "Examples of allowed:" not in f2_block

    # Two blank lines between categories.
    assert "ingested.\n\n\nF2: Financial Fraud" in prompt


def test_llama_guard3_build_request_with_standard_policies_has_classification_rules_and_output_format():
    # Outer scaffolding added per app/official_prompts/llama_guard.txt: a
    # "Classification Rules:" + "Output Format:" block between
    # <END CONVERSATION> and the final assessment instruction.
    req = LlamaGuard3Adapter().build_request("hello", "user", policy=_STANDARD_POLICY)
    prompt = req.payload["prompt"]
    assert "according to our safety policy" in prompt   # wording fix (was missing "to")
    assert "Classification Rules:" in prompt
    assert "- Do not output any additional text." in prompt
    assert "Output Format:" in prompt
    assert "If the content is safe:\n\nsafe" in prompt
    assert "If the content is unsafe:\n\nunsafe\n<comma-separated category IDs>" in prompt
    assert "Provide your safety assessment for ONLY THE LAST User in the above conversation:" in prompt
    assert (prompt.index("<END CONVERSATION>") < prompt.index("Classification Rules:")
            < prompt.index("Output Format:") < prompt.index("Provide your safety assessment"))


def test_llama_guard3_build_conversation_with_standard_policies_uses_raw_generate_with_category_block():
    messages = [ChatMessage(role="user", content="hi")]
    req = LlamaGuard3Adapter().build_conversation_request(messages, policy=_STANDARD_POLICY)
    assert req.endpoint == "generate"
    assert "raw" not in req.payload
    assert "M1: Medical Misinformation" in req.payload["prompt"]
    assert "F2: Financial Fraud" in req.payload["prompt"]


def test_llama_guard3_verbatim_override_unaffected_by_standard_policies_support():
    # Regression: a raw-only PolicyContent (verbatim override / legacy free
    # text) must still behave exactly as before.
    req = LlamaGuard3Adapter().build_request("hello", "user", policy=PolicyContent(raw="my policy"))
    assert req.payload["messages"][0] == {"role": "system", "content": "my policy"}


def test_llama_guard3_parse_safe():
    v = LlamaGuard3Adapter().parse("safe")
    assert v.safe is True and v.categories == []


def test_llama_guard3_parse_unsafe_with_codes():
    v = LlamaGuard3Adapter().parse("unsafe\nS1,S9")
    assert v.safe is False and v.categories == ["S1", "S9"]


# ── GraniteGuardianAdapter ───────────────────────────────────────────────────

def test_granite_build_request_no_policy_uses_fixed_system_prompt():
    req = GraniteGuardianAdapter().build_request("hello", "user")
    assert "content safety evaluator" in req.payload["messages"][0]["content"]


def test_granite_build_request_with_policy_overrides_system_prompt():
    req = GraniteGuardianAdapter().build_request("hello", "user", policy=PolicyContent(raw="pol"))
    assert req.payload["messages"][0] == {"role": "system", "content": "pol"}


def test_granite_build_request_with_standard_policies_appends_guardian_block_as_final_user_message():
    # Per upstream README/DeepWiki: the <guardian> block is a final USER-role
    # message, not a system message — different from the verbatim-override path.
    req = GraniteGuardianAdapter().build_request("hello", "user", policy=_STANDARD_POLICY)
    messages = req.payload["messages"]
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "user"
    guardian_block = messages[1]["content"]
    assert guardian_block.startswith("<guardian>")
    assert "### Criteria:" in guardian_block
    assert "### Scoring Schema:" in guardian_block
    assert "[M1] Medical Misinformation" in guardian_block
    assert "[F2] Financial Fraud" in guardian_block
    assert "Promotes a guaranteed-return investment scheme." in guardian_block


def test_granite_build_conversation_with_standard_policies_appends_guardian_block_last():
    messages = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]
    req = GraniteGuardianAdapter().build_conversation_request(messages, policy=_STANDARD_POLICY)
    payload_messages = req.payload["messages"]
    assert payload_messages[0] == {"role": "user", "content": "hi"}
    assert payload_messages[1] == {"role": "assistant", "content": "hello"}
    assert payload_messages[2]["role"] == "user"
    assert payload_messages[2]["content"].startswith("<guardian>")


def test_granite_verbatim_override_unaffected_by_standard_policies_support():
    # Regression: verbatim override stays system-message based, unchanged.
    req = GraniteGuardianAdapter().build_request("hello", "user", policy=PolicyContent(raw="pol"))
    assert req.payload["messages"] == [
        {"role": "system", "content": "pol"}, {"role": "user", "content": "hello"},
    ]


@pytest.mark.parametrize("raw,expected_safe", [
    ("<score> yes </score>", False),
    ("<score> no </score>", True),
    ("yes", False),
    ("no", True),
    ("", False),          # fails closed on empty
])
def test_granite_parse(raw, expected_safe):
    v = GraniteGuardianAdapter().parse(raw)
    assert v.safe is expected_safe


# ── GptOssSafeguardAdapter ───────────────────────────────────────────────────

def test_gpt_oss_build_request_no_policy_wraps_role_label():
    req = GptOssSafeguardAdapter().build_request("hello", "assistant")
    user_msg = req.payload["messages"][1]
    assert user_msg["content"] == "[AI Response]: hello"


def test_gpt_oss_build_request_with_policy_still_wraps_role_to_avoid_continuation_trap():
    # Bug 3 fix: policy-driven branches must not end the message list on an
    # assistant-authored turn (Ollama treats that as "continue writing as the
    # assistant", not "classify this") — same user-role wrapping as the fixed
    # (no-policy) branch, only the system content differs.
    req = GptOssSafeguardAdapter().build_request("hello", "assistant", policy=PolicyContent(raw="pol"))
    assert req.payload["messages"] == [
        {"role": "system", "content": "pol"}, {"role": "user", "content": "[AI Response]: hello"},
    ]


def test_gpt_oss_build_request_with_standard_policies_renders_per_policy_sections():
    req = GptOssSafeguardAdapter().build_request("hello", "user", policy=_STANDARD_POLICY)
    system = req.payload["messages"][0]["content"]
    # official structure: one "# Policy {id}: {name}" section per StandardPolicy
    assert "# Policy M1: Medical Misinformation" in system
    assert "# Policy F2: Financial Fraud" in system
    assert "## INSTRUCTIONS" in system and "## DEFINITIONS" in system
    assert "## VIOLATES [M1]" in system and "## SAFE [M1]" in system
    assert "## VIOLATES [F2]" in system and "## SAFE [F2]" in system
    assert "Claims a substance cures a disease without evidence." in system   # M1 criteria -> VIOLATES
    assert "Academic discussion of debunked claims." in system               # M1 exceptions -> SAFE
    assert "Drinking bleach cures viral infections." in system               # M1 violation example
    assert "Bleach is dangerous if ingested." in system                      # M1 allowed example
    # output instruction repeated near top and bottom, per the official guide
    assert system.count("Respond with exactly one character on the first line: 0 or 1.") >= 2
    assert req.payload["messages"][1] == {"role": "user", "content": "[User Input]: hello"}


def test_gpt_oss_build_request_with_standard_policies_wraps_assistant_role_to_avoid_continuation_trap():
    req = GptOssSafeguardAdapter().build_request("hello", "assistant", policy=_STANDARD_POLICY)
    assert req.payload["messages"][1] == {"role": "user", "content": "[AI Response]: hello"}


def test_gpt_oss_build_conversation_with_standard_policies_renders_system_prompt():
    messages = [ChatMessage(role="user", content="hi")]
    req = GptOssSafeguardAdapter().build_conversation_request(messages, policy=_STANDARD_POLICY)
    system = req.payload["messages"][0]["content"]
    assert "# Policy M1: Medical Misinformation" in system
    assert "# Policy F2: Financial Fraud" in system


@pytest.mark.parametrize("policy", [None, PolicyContent(raw="pol"), _STANDARD_POLICY])
def test_gpt_oss_build_conversation_ending_in_assistant_never_ends_on_assistant_role(policy):
    # Bug 3: for every system-prompt branch (fixed / verbatim-override /
    # standard-policy), a conversation whose last turn is the assistant's
    # (i.e. an /output check) must still end the outgoing messages list on a
    # user-role turn, or Ollama treats it as a request to continue writing as
    # the assistant instead of classifying it.
    messages = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="dangerous reply")]
    req = GptOssSafeguardAdapter().build_conversation_request(messages, policy=policy)
    assert req.payload["messages"][-1]["role"] == "user"
    assert "dangerous reply" in req.payload["messages"][-1]["content"]


def test_gpt_oss_parse_fixed_prompt_strict_format():
    v = GptOssSafeguardAdapter().parse("0", used_policy_prompt=False)
    assert v.safe is True
    v = GptOssSafeguardAdapter().parse("1\nS9", used_policy_prompt=False)
    assert v.safe is False and v.categories == ["S9"]


def test_gpt_oss_parse_policy_prompt_handles_free_text_refusal():
    v = GptOssSafeguardAdapter().parse("I'm sorry, but I can't help with that.", used_policy_prompt=True)
    assert v.safe is False and v.categories == []


def test_gpt_oss_parse_policy_prompt_still_reads_strict_format():
    v = GptOssSafeguardAdapter().parse("1\nS4,S12", used_policy_prompt=True)
    assert v.safe is False and v.categories == ["S4", "S12"]


def test_gpt_oss_parse_policy_prompt_extracts_codes_after_leading_prose():
    # Reported bug: the model prepends free-form reasoning before the verdict
    # line instead of leading with it — codes must still be extracted from
    # after the LAST matching verdict line, not lost because line 0 isn't "1".
    raw = (
        "It can also be used to treat bacterial infections, and it has no side "
        "effects if taken in small amounts. If you're feeling sick, just add a "
        "few drops of bleach to your water and drink it.\n\n1\nM1,M4"
    )
    v = GptOssSafeguardAdapter().parse(raw, used_policy_prompt=True)
    assert v.safe is False and v.categories == ["M1", "M4"]


@pytest.mark.parametrize("raw,expected_safe,expected_categories", [
    ("safe", True, []),
    ("SAFE", True, []),
    ("unsafe\nS1,S9", False, ["S1", "S9"]),
    ("Unsafe\nS1", False, ["S1"]),
])
def test_gpt_oss_parse_policy_prompt_accepts_llama_guard_style_vocabulary(raw, expected_safe, expected_categories):
    # Bug 2: under a policy prompt the model isn't reliably constrained to
    # its native 0/1 vocabulary and may answer "safe"/"unsafe" instead.
    v = GptOssSafeguardAdapter().parse(raw, used_policy_prompt=True)
    assert v.safe is expected_safe and v.categories == expected_categories


def test_gpt_oss_parse_policy_prompt_still_reads_verdict_first_with_no_prose():
    # Regression: the already-working case (verdict on line 0, no leading
    # prose) must keep working exactly as before.
    v = GptOssSafeguardAdapter().parse("1\nM1,M6", used_policy_prompt=True)
    assert v.safe is False and v.categories == ["M1", "M6"]

    v = GptOssSafeguardAdapter().parse("0", used_policy_prompt=True)
    assert v.safe is True and v.categories == []


def test_gpt_oss_parse_fixed_prompt_unaffected_by_policy_parser_changes():
    # /guard/check/* (used_policy_prompt=False) keeps its own strict-first-line
    # parser untouched — prose-before-verdict is NOT expected/handled there.
    v = GptOssSafeguardAdapter().parse("0", used_policy_prompt=False)
    assert v.safe is True and v.categories == []


# ── registry ──────────────────────────────────────────────────────────────────

def test_get_adapter_resolves_known_models():
    assert get_adapter("llama-guard3:8b").family == "llama_guard3"
    assert get_adapter("granite4.1-guardian:8b").family == "granite"
    assert get_adapter("gpt-oss-safeguard:20b").family == "gpt_oss"


def test_get_adapter_raises_on_unknown_model():
    with pytest.raises(UnknownGuardModelError):
        get_adapter("some-other-model:1b")
