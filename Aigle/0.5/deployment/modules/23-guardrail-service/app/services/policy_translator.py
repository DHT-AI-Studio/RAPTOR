"""Translate a Standard Policy Format policy into each guard model's own
prompt template, ONCE, at policy-activation time — not on every check call.

The Jinja2 template skeletons live in app/prompt_templates/*.jinja2 and
mirror the exact structure already validated against each vendor's official
documentation (and, before this module existed, already implemented as
Python f-strings in app/adapters/models/*.py — see that package for the
per-vendor citations):
  - GPT-OSS-Safeguard: https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide
  - Llama Guard 4/3:   https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama-guard-4
  - Granite Guardian:  https://www.ibm.com/granite/docs/models/guardian

Only GPT-OSS-Safeguard's and Granite Guardian's policy prompts are fully
resolvable at activation time — they don't depend on the actual content
being checked. Llama Guard 3's official format merges the category rules
and the real conversation into a single raw-completion prompt, so its
composed template still carries two genuine Jinja2 variables
(`check_role`/`conversation`), rendered a second time at check time via
app/adapters/base.py's render_composed_template(). The category-block text
itself is still baked in as literal text at activation time and wrapped in
`{% raw %}...{% endraw %}` so that if a category's own text happens to
contain `{{`/`{%` characters, the second-pass render treats them as inert
text rather than re-evaluating them as Jinja2 syntax.
"""
from __future__ import annotations

from pathlib import Path

import jinja2

from app.adapters.base import ComposedPromptTemplate
from app.engine.policy_targets import COLUMN_BY_ADAPTER_FAMILY
from app.models.policy import StandardPolicyCategory, parse_standard_policies

_TEMPLATE_DIR = Path(__file__).parent.parent / "prompt_templates"
_ENV = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)  # nosec B701 -- plain-text guard-model prompts, not HTML; see base.py's render_composed_template


def _load(name: str) -> jinja2.Template:
    return _ENV.from_string((_TEMPLATE_DIR / name).read_text(encoding="utf-8"))


_LLAMA_GUARD3_CATEGORIES_TEMPLATE = _load("llama_guard3_categories.jinja2")
_LLAMA_GUARD3_SKELETON_TEMPLATE = _load("llama_guard3.jinja2")
_GRANITE_CRITERIA_TEMPLATE = _load("granite_guardian_criteria.jinja2")
_GRANITE_GUARDIAN_TEMPLATE = _load("granite_guardian.jinja2")
_GPT_OSS_SAFEGUARD_TEMPLATE = _load("gpt_oss_safeguard.jinja2")

_GPT_OSS_OUTPUT_INSTRUCTION = (
    "Respond with exactly one character on the first line: 0 or 1.\n"
    "0 = the content does NOT violate any policy below.\n"
    "1 = the content violates at least one policy below.\n"
    "If 1, add a second line with the comma-separated id(s) of every violated policy.\n"
    "No other text."
)


def compose_gpt_oss(categories: list[StandardPolicyCategory]) -> ComposedPromptTemplate:
    template = _GPT_OSS_SAFEGUARD_TEMPLATE.render(
        categories=categories, output_instruction=_GPT_OSS_OUTPUT_INSTRUCTION,
    ).strip("\n")
    return ComposedPromptTemplate(placement="system", template=template)


def compose_granite(categories: list[StandardPolicyCategory]) -> ComposedPromptTemplate:
    criteria_text = _GRANITE_CRITERIA_TEMPLATE.render(categories=categories).rstrip("\n")
    template = _GRANITE_GUARDIAN_TEMPLATE.render(criteria_text=criteria_text)
    return ComposedPromptTemplate(placement="final_user_message", template=template)


def compose_llama_guard(categories: list[StandardPolicyCategory]) -> ComposedPromptTemplate:
    unsafe_categories = _LLAMA_GUARD3_CATEGORIES_TEMPLATE.render(categories=categories).rstrip("\n")
    # Wrapped in {% raw %} so a category's own text containing literal
    # "{{"/"{%" can't be mis-evaluated by the second, check-time render.
    unsafe_categories_safe = "{% raw %}" + unsafe_categories + "{% endraw +%}"
    template = _LLAMA_GUARD3_SKELETON_TEMPLATE.render(
        unsafe_categories=unsafe_categories_safe,
        check_role="{{ check_role }}", conversation="{{ conversation }}",
    )
    return ComposedPromptTemplate(placement="raw_completion", template=template)


_COMPOSERS = {
    "llama_guard3": compose_llama_guard,
    "granite": compose_granite,
    "gpt_oss": compose_gpt_oss,
}


def translate_policy_to_prompts(raw_content: str) -> dict[str, ComposedPromptTemplate] | None:
    """Compose every guard model's prompt from `raw_content`, if it parses as
    the Standard Policy Format. Returns None (no composition performed) for
    anything else — legacy checker_rules YAML, free text, or malformed JSON —
    so callers leave the content_* columns exactly as they already are.

    Keys are guardrail_policies column names (app.engine.policy_targets),
    ready to hand to app.db.repo.update_policy_content."""
    categories = parse_standard_policies(raw_content)
    if categories is None:
        return None
    return {
        COLUMN_BY_ADAPTER_FAMILY[family]: composer(categories)
        for family, composer in _COMPOSERS.items()
    }
