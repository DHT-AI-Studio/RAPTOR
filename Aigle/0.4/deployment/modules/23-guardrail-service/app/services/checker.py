"""Policy checker (GB-4) — the detector-based 'rule engine'.

Given the active policy, the checker filters rules by scope, dispatches each rule
to its detector (regex → deterministic; llama_guard → the guard model), then
aggregates the triggered rules' actions worst-case (block > redact > flag > pass).

`evaluate()` is pure (policy + guard_classify injected) so it unit-tests without
Redis/Ollama. `check()` is the I/O wrapper: Redis enabled flag + active policy +
the in-process guard classifier.
"""
from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable, Optional
from uuid import UUID

from app.models.check import CheckRequest, CheckResult
from app.models.policy import GuardrailPolicy
from app.services import redactor, regex_detector

logger = logging.getLogger(__name__)

_SEVERITY = {"pass": 0, "flag": 1, "redact": 2, "block": 3}

# guard_classify(content, role) -> object with `.safe` and `.categories`
GuardClassify = Callable[[str, str], Awaitable]


def _passed() -> CheckResult:
    return CheckResult(safe=True, action="pass")


async def evaluate(content: str, direction: str, module: Optional[str],
                   policy: Optional[GuardrailPolicy], guard_classify: GuardClassify) -> CheckResult:
    """Core detector-based evaluation (no I/O)."""
    if policy is None:
        logger.warning("[checker] no active policy — passing through")
        return _passed()
    if not policy.scope.applies_to(module):
        return _passed()

    block = policy.block_for("input" if direction == "input" else "output")
    if not block.enabled or not block.rules:
        return _passed()

    # only pay for the guard model if at least one rule needs it
    guard_categories: set[str] = set()
    guard_flagged = False                                 # guard said unsafe, whatever the categories
    if any(r.detector == "llama_guard" for r in block.rules):
        role = "user" if direction == "input" else "assistant"
        resp = await guard_classify(content, role)
        guard_categories = set(getattr(resp, "categories", None) or [])
        guard_flagged = getattr(resp, "safe", True) is False

    triggered: list[tuple[str, str, list[str]]] = []   # (action, category, patterns)
    for rule in block.rules:
        if rule.detector == "regex":
            if regex_detector.detect(content, rule.patterns).matched:
                triggered.append((rule.action, rule.category, rule.patterns))
        elif rule.category in guard_categories:          # llama_guard
            triggered.append((rule.action, rule.category, []))

    if not triggered:
        if guard_flagged:
            # The guard model called this unsafe but no rule matched — every
            # llama_guard rule keys off `category`, so a model that reports a
            # verdict without categories (granite4.1-guardian always does) can
            # never trigger one. Rules are still the source of truth; log it so
            # the pass-through is at least visible.
            logger.warning(
                "[checker] guard flagged unsafe but no rule matched — passing through "
                "(direction=%s module=%s guard_categories=%s rule_categories=%s)",
                direction, module, sorted(guard_categories) or "none",
                sorted({r.category for r in block.rules if r.detector == "llama_guard"}),
            )
        return _passed()

    final = max((t[0] for t in triggered), key=lambda a: _SEVERITY[a])
    category = next((t[1] for t in triggered if t[0] == final), None)

    redacted_content = None
    if final == "redact":
        patterns = [p for action, _, pats in triggered if action == "redact" for p in pats]
        # span-precise redaction for regex rules; llama_guard redact has no span (see note)
        redacted_content = redactor.redact(content, patterns) if patterns else content

    return CheckResult(safe=(final == "pass"), action=final,
                       category=category, redacted_content=redacted_content)


def _parse_policy(raw_json: str) -> tuple[Optional[UUID], Optional[GuardrailPolicy]]:
    data = json.loads(raw_json)
    raw = data.get("raw_content")
    policy = GuardrailPolicy.from_raw(raw) if raw else None
    policy_id = UUID(data["id"]) if policy is not None and "id" in data else None
    return policy_id, policy


async def _load_active_policy(redis) -> tuple[Optional[UUID], Optional[GuardrailPolicy]]:
    """Active policy from the Redis cache, falling back to Postgres.

    The fallback is what makes the policy stay in force: the cache carries a
    60-second TTL and is only written when a policy is activated, so a
    cache-only read means enforcement silently stops one minute after
    activation while Postgres still reports the policy as active. Postgres is
    the source of truth; Redis is only there to keep the hot path off the DB.
    """
    from app.engine import policy_store                          # lazy: keeps evaluate() DB-free
    from app.engine.policy_store import ACTIVE_POLICY_CACHE_KEY

    cached = await redis.get(ACTIVE_POLICY_CACHE_KEY)        # PolicyDetail JSON (has id + raw_content)
    if cached:
        try:
            return _parse_policy(cached)
        except Exception as exc:                              # malformed cache — fall through to the DB
            logger.error("[checker] cached policy unparseable, re-reading Postgres: %s", exc)

    try:
        detail = await policy_store.get_active_policy()       # raises 404 when there genuinely is none
    except Exception as exc:
        logger.warning("[checker] no active policy available from Postgres either: %s", exc)
        return None, None

    try:
        result = _parse_policy(detail.model_dump_json())
    except Exception as exc:                                  # malformed policy — fail open, log
        logger.error("[checker] failed to parse active policy: %s", exc)
        return None, None

    # Re-warm so the next request stays on the fast path.
    try:
        await redis.set(ACTIVE_POLICY_CACHE_KEY, detail.model_dump_json(),
                        ex=policy_store.ACTIVE_POLICY_CACHE_TTL)
    except Exception:
        logger.warning("[checker] could not re-warm the active-policy cache", exc_info=True)

    return result


async def check(req: CheckRequest, direction: str) -> CheckResult:
    """I/O wrapper: honour the Redis master switch, load the active policy, evaluate, audit-log."""
    from app.db.redis_conn import get_redis                # lazy: keeps evaluate() DB-free
    from app.services import audit_log
    from app.services.guard_classifier import classify
    from app.services.state import is_enabled

    if not await is_enabled():                              # master off → pass immediately, no guard call
        return _passed()
    redis = await get_redis()
    policy_id, policy = await _load_active_policy(redis)
    result = await evaluate(req.content, direction, req.module, policy, classify)

    if result.action != "pass" and policy_id is not None and policy.audit_log.enabled:
        audit_log.record_violation(
            policy_id=policy_id, module=req.module, direction=direction,
            category=result.category, action_taken=result.action, request_id=req.request_id,
            content=req.content if policy.audit_log.include_content else None,
        )
    return result
