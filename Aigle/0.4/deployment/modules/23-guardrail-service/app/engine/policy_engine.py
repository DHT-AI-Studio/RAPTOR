"""Policy Engine — orchestrates Signal Layer → active policy lookup → rule match → PolicyDecision.

This is the single entry point for deterministic safety evaluation. There is
exactly one active policy system-wide (see app/routers/guardrail_policies.py);
this module reads it from Redis (falling back to Postgres on a cache miss or
a Redis error — see _get_active_policy).

A policy's content isn't required to have structured rules (see
app/engine/models.py's note on PolicyDetail) — this module makes a
best-effort, lazy attempt to interpret the active policy's raw_content as a
`rules` array, and returns a clear 409 pointing at /policy/check/llm/* when
it can't. The LLM-judged path (app/routers/check_policy_llm.py) has no such
requirement — it sends raw_content verbatim as the model's system prompt.
"""
from __future__ import annotations

import json
import logging
import time
from uuid import UUID

import yaml
from fastapi import HTTPException

from app.db import connection, redis_conn, repo
from app.engine.models import LLMSignal, PolicyDetail, PolicyDecision, PolicyRule, RuleConditions, RuleResult, TraceEntry
from app.engine.policy_store import ACTIVE_POLICY_CACHE_KEY, ACTIVE_POLICY_CACHE_TTL
from app.engine.signal import extract_signal
from app.services.state import is_enabled

_logger = logging.getLogger(__name__)

_DISABLED_POLICY_ID = UUID(int=0)


def _disabled_decision() -> PolicyDecision:
    return PolicyDecision(
        action="allow", policy_id=_DISABLED_POLICY_ID, policy_name="", policy_version="",
        matched_rule_id=None, intent="", category="", confidence="", risk_flags=[],
        final_action_reasoning="Guardrail checking is globally disabled.",
        trace=[], safe=True, rationale="",
    )


async def _get_active_policy() -> PolicyDetail:
    """Redis-first (bounded-TTL cache written by app/engine/policy_store.py),
    falling back to Postgres on a cache miss OR on any Redis error — Redis is
    a fail-safe accelerator here, never a hard dependency, so a hiccup on the
    read side must degrade to the source of truth rather than break every
    check request."""
    redis = await redis_conn.get_redis()
    try:
        cached = await redis.get(ACTIVE_POLICY_CACHE_KEY)
    except Exception:
        _logger.exception("policy_engine: failed to read active-policy Redis cache — falling back to Postgres")
        cached = None
    if cached:
        return PolicyDetail(**json.loads(cached))

    pool = await connection.get_pool()
    row = await repo.get_active_policy(pool)
    if row is None:
        raise HTTPException(
            status_code=503,
            detail="No active policy — activate one via PUT /guardrail/policies/{id}/activate",
        )
    policy = PolicyDetail(
        id=row["id"], name=row["name"], version=row["version"],
        raw_content=row["raw_content"],
        content_llama_guard=row["content_llama_guard"],
        content_granite_guardian=row["content_granite_guardian"],
        content_gpt_oss_safeguard=row["content_gpt_oss_safeguard"],
        is_active=row["is_active"], created_at=row["created_at"],
    )
    try:
        await redis.set(ACTIVE_POLICY_CACHE_KEY, policy.model_dump_json(), ex=ACTIVE_POLICY_CACHE_TTL)
    except Exception:
        _logger.exception("policy_engine: failed to re-warm active-policy Redis cache — continuing with the Postgres result")
    return policy


def _parse_structured_rules(raw_content: str, policy_name: str) -> tuple[list[PolicyRule], str, str]:
    """Best-effort interpretation of a policy's raw_content as a structured
    rules document. Raises 409 if it can't be interpreted this way — the
    deterministic endpoints need this shape; /policy/check/llm/* doesn't."""
    not_structured = HTTPException(
        status_code=409,
        detail=(
            f"Policy '{policy_name}' has no structured 'rules' — this endpoint requires them. "
            f"Use /policy/check/llm/* instead, which reads the policy content directly."
        ),
    )

    try:
        parsed = yaml.safe_load(raw_content)
    except yaml.YAMLError:
        raise not_structured

    if not isinstance(parsed, dict) or "rules" not in parsed:
        raise not_structured

    try:
        rules = [PolicyRule(**r) for r in parsed["rules"]]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Policy '{policy_name}' has a 'rules' field but it doesn't match the expected shape: {exc}",
        )

    default_action = str(parsed.get("default_action", "allow"))
    default_reason = str(parsed.get("default_reason", "No policy rules matched"))
    return rules, default_action, default_reason


def _matches(conditions: RuleConditions, signal: LLMSignal) -> bool:
    """Return True if ALL specified conditions match the signal."""
    if conditions.category and signal.category not in conditions.category:
        return False

    if conditions.confidence and signal.confidence not in conditions.confidence:
        return False

    if conditions.intent and signal.intent not in conditions.intent:
        return False

    if conditions.risk_flags_any:
        signal_flags = set(signal.risk_flags)
        if not signal_flags.intersection(conditions.risk_flags_any):
            return False

    if conditions.risk_flags_all:
        signal_flags = set(signal.risk_flags)
        if not set(conditions.risk_flags_all).issubset(signal_flags):
            return False

    return True


def _evaluate_rules(rules: list[PolicyRule], default_action: str, signal: LLMSignal) -> RuleResult:
    sorted_rules = sorted(rules, key=lambda r: -r.priority)
    for rule in sorted_rules:
        if _matches(rule.conditions, signal):
            _logger.info(
                "policy_engine: matched=%s action=%s flags=%s",
                rule.id, rule.action, signal.risk_flags,
            )
            return RuleResult(action=rule.action, matched_rule=rule, default_applied=False)

    _logger.info("policy_engine: no_match → default_action=%s", default_action)
    return RuleResult(action=default_action, matched_rule=None, default_applied=True)


async def evaluate(
    content: str,
    model: str,
    ollama_url: str,
    timeout: float,
) -> PolicyDecision:
    """Full pipeline: content → signal → active policy → rule match → PolicyDecision with trace."""
    if not await is_enabled():                # master off → pass immediately, no signal/policy lookup
        return _disabled_decision()

    trace: list[TraceEntry] = []
    t0 = time.monotonic()

    # ── Step 1: Signal Extraction ─────────────────────────────────────────────
    signal: LLMSignal = await extract_signal(
        content=content,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
    )
    signal_ms = int((time.monotonic() - t0) * 1000)
    trace.append(TraceEntry(step="signal_extraction", data={
        "model":      model,
        "category":   signal.category,
        "confidence": signal.confidence,
        "intent":     signal.intent,
        "risk_flags": signal.risk_flags,
        "rationale":  signal.rationale,
        "latency_ms": signal_ms,
    }))
    _logger.info(
        "signal: category=%s confidence=%s intent=%s flags=%s",
        signal.category, signal.confidence, signal.intent, signal.risk_flags,
    )

    # ── Step 2: Active Policy Lookup + Rule Evaluation ────────────────────────
    t1 = time.monotonic()
    policy = await _get_active_policy()
    rules, default_action, default_reason = _parse_structured_rules(policy.raw_content, policy.name)
    result = _evaluate_rules(rules, default_action, signal)
    rule_ms = int((time.monotonic() - t1) * 1000)

    matched_id     = result.matched_rule.id if result.matched_rule else None
    matched_reason = result.matched_rule.reason if result.matched_rule else default_reason
    matched_expl   = result.matched_rule.explanation if result.matched_rule else ""

    trace.append(TraceEntry(step="rule_evaluation", data={
        "policy_id":       str(policy.id),
        "policy_name":     policy.name,
        "policy_version":  policy.version,
        "rules_evaluated": len(rules),
        "matched_rule_id": matched_id,
        "action":          result.action,
        "reason":          matched_reason,
        "default_applied": result.default_applied,
        "latency_ms":      rule_ms,
    }))
    _logger.info(
        "decision: policy=%s action=%s rule=%s reason=%s",
        policy.name, result.action, matched_id, matched_reason,
    )

    # ── Step 3: Build final reasoning string ──────────────────────────────────
    if result.matched_rule:
        reasoning_parts = [
            f"Rule {matched_id} triggered.",
            f"Reason: {matched_reason}",
        ]
        if matched_expl:
            reasoning_parts.append(f"({matched_expl})")
        if signal.risk_flags:
            reasoning_parts.append(f"Risk flags: {', '.join(signal.risk_flags)}")
    else:
        reasoning_parts = [
            f"No rules matched for policy '{policy.name}' v{policy.version}.",
            f"Default action '{result.action}' applied.",
            f"Signal: category={signal.category}, intent={signal.intent}",
        ]

    final_reasoning = " ".join(reasoning_parts)

    total_ms = int((time.monotonic() - t0) * 1000)
    trace.append(TraceEntry(step="decision_complete", data={
        "total_latency_ms": total_ms,
        "final_action":     result.action,
    }))

    return PolicyDecision(
        action                 = result.action,
        policy_id              = policy.id,
        policy_name            = policy.name,
        policy_version         = policy.version,
        matched_rule_id        = matched_id,
        intent                 = signal.intent,
        category               = signal.category,
        confidence             = signal.confidence,
        risk_flags             = signal.risk_flags,
        final_action_reasoning = final_reasoning,
        trace                  = trace,
        safe                   = result.action == "allow",
        rationale              = signal.rationale,
    )
