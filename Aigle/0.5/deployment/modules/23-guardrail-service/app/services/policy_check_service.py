"""Policy-aware guard check (GRD-X1) — the one implementation shared by
`/api/v0.2/check/*` (app/routers/check_v0_2.py) and
`POST /guardrail/policy/decide` (app/routers/guardrail.py).

Both endpoints run the same six steps and differ only in how they present the
result: v0.2 keeps the `/guard/check/*` response shape (`CheckResponse`) it
replaces, decide maps the same outcome onto `allow|redact|warn|block`.

  1. **Deterministic detector** — the regex jailbreak/PII pass
     (`app/services/regex_detector.py` + `app/services/redactor.py`). Always
     the shipped default patterns: this endpoint family has no policy-scoped
     regex rule set of its own (that concept — `GuardrailPolicy.input_guardrail`/
     `output_guardrail` — belongs to the separate schema the GB-4 checker at
     `/guardrail/check/*` uses, not to the Standard Policy Format /
     `PolicyDetail` threaded through here). A match redacts, it doesn't
     block; only the (possibly redacted) result is ever sent to Ollama.
  2. **Active policy lookup** — `get_policy_for_branch()`: the active policy
     when there is one *and* the narrower `guardrail:policy_enabled` switch is
     on, otherwise None (the fixed-guard-prompt branch).
  3. **Guard-model evaluation** — `classify_per_model`/
     `classify_conversation_per_model`, which already accept `policy=None`;
     that is precisely what makes each adapter fall back to its fixed default
     guard prompt (see app/adapters/models/*.py), so there is deliberately no
     if/else duplicating the two call shapes.
  4. **Conflict resolution** — GB-16's `guard_classifier.combine()` across
     guard models, then `final_verdict()`: a rule-engine-shaped active policy
     re-evaluates the combined verdict via `checker.evaluate()`.
  5. **Action determination** — left to each endpoint (see above).
  6. **Audit** — `audit_if_unsafe()`, fire-and-forget via
     app/services/audit_sink.py.

Dependencies are looked up as attributes of this module (`is_enabled`,
`policy_engine`, `guard_classifier`, `audit_sink`, ...) so tests can
monkeypatch them here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from app.adapters.base import ChatMessage, PolicyContent, parse_policy_content
from app.engine import policy_engine
from app.engine.models import PolicyDetail
from app.engine.policy_targets import COLUMN_BY_ADAPTER_FAMILY
from app.models.policy import GuardrailPolicy, parse_standard_policies
from app.services import audit_sink, checker, guard_classifier, redactor, regex_detector
from app.services.guard_classifier import CombinedVerdict, ModelVerdict
from app.services.state import is_enabled, is_policy_enabled

_logger = logging.getLogger(__name__)


@dataclass
class RuleVerdict:
    """The authoritative verdict from `final_verdict()`.

    `safe`/`category` are what v0.2 has always reported. `action`/
    `redacted_content` are the rule engine's own result — only set when it
    actually ran (`rule_engine_ran`), for decide's allow/redact/warn/block."""
    safe: bool
    category: str | None
    action: str | None = None
    redacted_content: str | None = None
    rule_engine_ran: bool = False


@dataclass
class PolicyCheckOutcome:
    """Everything one check produced, for each endpoint to present its own way."""
    disabled: bool                                  # global guardrail switch off — nothing else ran
    combined: CombinedVerdict | None = None          # GB-16 combined guard verdict
    results: list[tuple[ModelVerdict, str | None]] = field(default_factory=list)  # (verdict, prompt) per model
    safe: bool = True                                # FINAL verdict (after final_verdict)
    category: str | None = None                      # FINAL category
    policy: PolicyDetail | None = None               # policy applied; None → fixed guard prompt branch
    regex_matched: bool = False                      # step 1 hit (content was redacted before inference)
    matched_pattern: str | None = None
    screened_content: str = ""                       # what was sent to the guard model(s)
    rule_action: str | None = None                   # checker.evaluate action, when the rule engine ran
    rule_redacted_content: str | None = None


# ── Steps ──────────────────────────────────────────────────────────────────────

def _screen(content: str) -> tuple[str, regex_detector.RegexMatch]:
    match = regex_detector.detect(content)
    if match.matched:
        _logger.info("policy check: regex matched pattern=%r — redacting before inference", match.matched_pattern)
    return redactor.redact(content), match


def preprocess(content: str) -> str:
    """Regex jailbreak/PII pass before the content ever reaches an LLM."""
    return _screen(content)[0]


async def get_policy_for_branch() -> PolicyDetail | None:
    """None means "take the fixed-guard-prompt branch" — either because
    there's no active policy, or because the narrower guardrail:policy_enabled
    switch (app/services/state.py) is off. Unlike /policy/check/llm/*, this
    endpoint family never reports "disabled" just because policy-checking
    specifically was turned off; it always answers via one branch or the
    other."""
    if not await is_policy_enabled():
        return None
    try:
        return await policy_engine._get_active_policy()
    except HTTPException:
        return None


def original_policy_content(policy: PolicyDetail) -> PolicyContent:
    """Same as check_policy_llm.py's helper of the same name."""
    return PolicyContent(raw=policy.raw_content, standard_policies=parse_standard_policies(policy.raw_content))


def policy_by_family(policy: PolicyDetail, original: PolicyContent) -> dict[str, PolicyContent]:
    """Same as check_policy_llm.py's helper of the same name."""
    return {
        family: parse_policy_content(getattr(policy, column)) or original
        for family, column in COLUMN_BY_ADAPTER_FAMILY.items()
    }


async def final_verdict(
    policy: PolicyDetail | None, content: str, rule_direction: str, combined: CombinedVerdict,
) -> RuleVerdict:
    """The authoritative safe/category for the response and audit trail —
    *replaces* `combined`'s own safe/categories, it doesn't just gate on top
    of them.

    With an active policy that parses as a `GuardrailPolicy`
    (app/models/policy.py — the regex + llama_guard rule schema
    app/services/checker.py's GB-4 rule engine understands), `checker.evaluate()`
    re-evaluates the already-computed guard verdict against that policy's
    `input_guardrail`/`output_guardrail` rules and its decision wins, whatever
    `combined.safe` said. The already-computed verdict is reused via a shim
    (`_reuse_combined` below) instead of letting the rule engine call the
    guard model again — `checker.evaluate()` only needs something with
    `.safe`/`.categories`, and `combined` already is that.

    Falls back to `combined.safe`/first category when there's no active
    policy, or when the active policy's `raw_content` isn't in the
    `GuardrailPolicy` schema (same `guardrail_policies.raw_content` column,
    used in an incompatible shape by the Standard Policy Format this file
    also threads through for the LLM-judged branch — see README.md's
    "Policy Schema 比較"). This mirrors `checker.py`'s own fail-open
    behaviour for an unparseable policy, but yields to the guard model's own
    verdict rather than unconditionally forcing `safe=True` (which is what
    `checker.evaluate(policy=None, ...)` does) — that would silently wave
    through genuinely unsafe content just because the active policy happens
    to be in the wrong format, and would also break this endpoint family's
    original "no active policy → behave like /guard/check/*" contract.
    """
    guard_only = RuleVerdict(safe=combined.safe, category=combined.categories[0] if combined.categories else None)
    if policy is None:
        return guard_only

    try:
        rule_policy = GuardrailPolicy.from_raw(policy.raw_content)
    except (yaml.YAMLError, ValueError, ValidationError) as exc:
        _logger.warning("policy check: active policy is not GuardrailPolicy-shaped, rule engine skipped: %s", exc)
        return guard_only

    async def _reuse_combined(_content: str, _role: str) -> CombinedVerdict:
        return combined

    result = await checker.evaluate(content, rule_direction, None, rule_policy, _reuse_combined)
    return RuleVerdict(safe=result.safe, category=result.category, action=result.action,
                       redacted_content=result.redacted_content, rule_engine_ran=True)


def audit_if_unsafe(policy: PolicyDetail | None, safe: bool, category: str | None, direction: str,
                    module: str | None = None, request_id: str | None = None, *,
                    content: str | None = None) -> None:
    """Always write a guardrail_violations row when unsafe — the audit trail
    must be complete regardless of which branch produced the verdict.

    `safe`/`category` are the FINAL verdict from `final_verdict()`, not the
    raw guard verdict — the audit trail reflects what the rule engine
    actually decided, not what the guard model alone said.

    With an active policy, `policy_id` is `policy.id`. Without one there's no
    real policy to attribute the row to; `guardrail_violations.policy_id` is a
    nullable UUID foreign key (`ON DELETE SET NULL` — see app/db/schema.py),
    so the no-policy case writes `policy_id=None` rather than a sentinel
    string the column can't hold as a UUID.

    `module`/`request_id` are the caller's own (MessageRequest), so a row can
    be traced back to whoever asked. `direction` is caller-supplied —
    /check/conversation always passes "conversation" here, so a
    conversation-triggered row is never mislabeled "input"/"output".

    `content` is the screened text that was judged; the audit record only
    ever keeps its SHA-256 (CF-4)."""
    if safe:
        return
    audit_sink.record_decision(
        policy_id=policy.id if policy is not None else None,
        policy_version=policy.version if policy is not None else None,
        module=module, direction=direction, action_taken="block",
        category=category, request_id=request_id, content=content,
    )


# ── Entry points ───────────────────────────────────────────────────────────────

async def evaluate_message(content: str, direction: str, *, module: str | None = None,
                           request_id: str | None = None, with_prompts: bool = False,
                           audit: bool = True) -> PolicyCheckOutcome:
    """Steps 1-4 (+6 when `audit`) for one message; `direction` is
    "input" (checked as the user) or "output" (checked as the assistant).
    `with_prompts` uses the *_with_prompt classifiers (the /debug variants)."""
    if not await is_enabled():
        return PolicyCheckOutcome(disabled=True)
    screened, match = _screen(content)
    policy = await get_policy_for_branch()
    original = original_policy_content(policy) if policy is not None else None
    family = policy_by_family(policy, original) if policy is not None else None
    role = "user" if direction == "input" else "assistant"
    if with_prompts:
        results = await guard_classifier.classify_per_model_with_prompt(
            screened, role=role, policy=original, policy_by_family=family)
    else:
        results = [(v, None) for v in await guard_classifier.classify_per_model(
            screened, role=role, policy=original, policy_by_family=family)]
    combined = guard_classifier.combine([v for v, _ in results])
    verdict = await final_verdict(policy, screened, direction, combined)
    if audit:
        audit_if_unsafe(policy, verdict.safe, verdict.category, direction, module, request_id, content=screened)
    return _outcome(combined, results, verdict, policy, match, screened)


async def evaluate_conversation(messages: list[ChatMessage], *, module: str | None = None,
                                request_id: str | None = None, with_prompts: bool = False,
                                audit: bool = True) -> PolicyCheckOutcome:
    """Same as evaluate_message() for a whole conversation; the rule engine
    judges the last message, as input or output by its role."""
    if not await is_enabled():
        return PolicyCheckOutcome(disabled=True)
    screened = [_screen(m.content) for m in messages]
    messages = [ChatMessage(role=m.role, content=text) for m, (text, _) in zip(messages, screened)]
    policy = await get_policy_for_branch()
    original = original_policy_content(policy) if policy is not None else None
    family = policy_by_family(policy, original) if policy is not None else None
    if with_prompts:
        results = await guard_classifier.classify_conversation_per_model_with_prompt(
            messages, policy=original, policy_by_family=family)
    else:
        results = [(v, None) for v in await guard_classifier.classify_conversation_per_model(
            messages, policy=original, policy_by_family=family)]
    combined = guard_classifier.combine([v for v, _ in results])
    rule_direction = "input" if messages[-1].role == "user" else "output"
    verdict = await final_verdict(policy, messages[-1].content, rule_direction, combined)
    if audit:
        audit_if_unsafe(policy, verdict.safe, verdict.category, "conversation", module, request_id,
                        content=messages[-1].content)
    return _outcome(combined, results, verdict, policy, screened[-1][1], messages[-1].content)


def _outcome(combined: CombinedVerdict, results: list[tuple[ModelVerdict, str | None]], verdict: RuleVerdict,
             policy: PolicyDetail | None, match: regex_detector.RegexMatch, screened: str) -> PolicyCheckOutcome:
    return PolicyCheckOutcome(
        disabled=False, combined=combined, results=results,
        safe=verdict.safe, category=verdict.category, policy=policy,
        regex_matched=match.matched, matched_pattern=match.matched_pattern, screened_content=screened,
        rule_action=verdict.action if verdict.rule_engine_ran else None,
        rule_redacted_content=verdict.redacted_content,
    )
