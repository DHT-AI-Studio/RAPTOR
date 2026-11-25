"""Canonical policy-aware decision endpoint (GRD-X1) + policy-aware guardrail
check endpoints (GB-4) + violation audit log (GB-6) + a raw prompt-test utility.

Mounted under /guardrail.

POST /guardrail/policy/decide is the canonical decision API: one
allow|redact|warn|block answer per piece of content. It runs the exact same
check as /api/v0.2/check/* (app/services/policy_check_service.py — regex pass,
active policy, guard model(s), GB-16 conflict resolution, rule engine), so
the two never disagree; it only presents the outcome as a decision.

POST /guardrail/check/input and /guardrail/check/output:
These run the active policy's rules via the detector-based checker (distinct from
/guard/* which is raw guard-model classification, and /policy/check/llm/* which is
the LLM-judge path). /guardrail/violations* reads the audit trail those checks write.
/guardrail/prompt_test is unrelated to policy checking — it's a debugging utility
that sends a prompt straight to a chosen Ollama model and returns its raw output,
with no guard prompt, policy, or parsing involved.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.models.check import CheckRequest, CheckResult
from app.models.decide import DecideRequest, DecideResponse, Decision
from app.models.policy import parse_standard_policies
from app.models.prompt_test import PromptTestRequest
from app.models.violation import (
    ViolationListResponse,
    ViolationRecord,
    ViolationSummaryEntry,
    ViolationSummaryResponse,
)
from app.services import (
    audit_log,
    audit_sink,
    checker,
    moderation,
    policy_check_service,
)
from app.services.policy_check_service import PolicyCheckOutcome

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Guardrail (policy checker)"])


# ── POST /guardrail/policy/decide (GRD-X1) ───────────────────────────────────────

_DECISION_RANK: dict[str, int] = {"allow": 0, "warn": 1, "redact": 2, "block": 3}
# checker.evaluate() actions → decisions
_RULE_ACTION_DECISION: dict[str, Decision] = {"pass": "allow", "flag": "warn", "redact": "redact", "block": "block"}
# Standard Policy Format per-category decisions → decisions ("review" has no
# reviewer queue to go to, so it lets the content through with a warning)
_STANDARD_DECISION: dict[str, Decision] = {"allow": "allow", "warn": "warn", "review": "warn", "block": "block"}


def _guard_reason(outcome: PolicyCheckOutcome) -> str:
    combined = outcome.combined
    if not combined.categories:
        return "unsafe"
    return ", ".join(f"{c}: {combined.category_names.get(c, c)}" for c in combined.categories)


def _guard_decision(outcome: PolicyCheckOutcome) -> tuple[Decision, str]:
    """Combined guard verdict is unsafe and no rule engine ran.

    With a Standard Policy Format policy, each reported S-code's category
    `decision` applies (worst wins). No policy, a policy that isn't in the
    standard format, or a verdict none of whose codes the policy lists
    (granite reports none at all) → block."""
    reason = _guard_reason(outcome)
    standard = parse_standard_policies(outcome.policy.raw_content) if outcome.policy is not None else None
    if not standard:
        return "block", reason
    by_id = {c.id: c for c in standard}
    matched = [by_id[code] for code in outcome.combined.categories if code in by_id]
    if not matched:
        return "block", reason
    worst = max(matched, key=lambda c: _DECISION_RANK[_STANDARD_DECISION[c.decision]])
    decision = _STANDARD_DECISION[worst.decision]
    if worst.decision != decision:                         # e.g. review → warn: keep what the policy said
        reason = f"{reason} (policy: {worst.decision})"
    return decision, reason


def _decide(outcome: PolicyCheckOutcome) -> tuple[Decision, str, str | None]:
    """(decision, reason, redacted_content) for one policy-check outcome."""
    if outcome.disabled:
        return "allow", "guardrail disabled", None

    redacted: str | None = None
    if outcome.rule_action is not None:                    # a rule-engine-shaped policy decided
        decision = _RULE_ACTION_DECISION[outcome.rule_action]
        reason = f"rule: {outcome.category}" if outcome.category else "no violation"
        if decision == "redact":
            redacted = outcome.rule_redacted_content
    elif not outcome.combined.safe:
        decision, reason = _guard_decision(outcome)
    else:
        decision, reason = "allow", "no violation"

    # The regex pass already redacted what it matched before inference; when
    # nothing stronger was decided, report that redaction.
    if outcome.regex_matched and _DECISION_RANK[decision] < _DECISION_RANK["redact"]:
        return "redact", f"regex: {outcome.matched_pattern}", outcome.screened_content
    return decision, reason, redacted


# response_model_exclude_unset: `moderation` is only set when the request
# names a modality, so a request without one gets exactly the old seven fields.
@router.post("/policy/decide", response_model=DecideResponse, response_model_exclude_unset=True,
             summary="Canonical policy-aware decision: allow | redact | warn | block (GRD-X1)")
async def policy_decide(body: DecideRequest) -> DecideResponse:
    """
    依目前啟用中的 policy，判定一段內容應該 **allow**、**redact**、**warn** 或 **block**。

    與 `POST /api/v0.2/check/*` 共用同一套判定邏輯（`app/services/policy_check_service.py`），依序：

    1. regex 偵測（命中的片段先遮蔽，再送進模型）
    2. 讀取 active policy（`guardrail:policy_enabled` 關閉時視同沒有 policy）
    3. guard model 判定（有 policy 時使用 policy 轉譯出的 prompt）
    4. 衝突處理（GB-16 `GUARD_CONFLICT_STRATEGY`；rule engine 格式的 policy 再以規則重新判定）
    5. 決定動作（見 README 的對應表）
    6. 寫入稽核紀錄（非阻塞；`allow` 不寫）

    選填 `modality`（`text` | `transcript`，CM-1）：回應多一個 `moderation` 欄位，把 guard model 的 S-code
    對應到 moderation 分類，並套用該類型的 decision（與 image / video_frame 的 `moderation` 同格式）。
    只是附加資訊，不改變 `decision`。

    判定本身失敗（guard model、Redis、Postgres 無法使用）時回 **503**
    `{"detail": {"type": "guardrail_unavailable", "message": "..."}}`。
    """
    started = time.perf_counter()
    request_id = body.request_id or str(uuid.uuid4())
    try:
        outcome = await policy_check_service.evaluate_message(
            body.content, body.direction, module=body.module, request_id=request_id, audit=False)
    except Exception as exc:  # any failure means no decision; reported as 503 guardrail_unavailable
        _logger.error("policy decide: check failed, guardrail unavailable: %r", exc)
        raise HTTPException(status_code=503, detail={
            "type": "guardrail_unavailable",
            "message": f"Guardrail check unavailable: {str(exc) or type(exc).__name__}",
        }) from exc

    decision, reason, redacted = _decide(outcome)
    if decision != "allow":
        audit_sink.record_decision(
            policy_id=outcome.policy.id if outcome.policy is not None else None,
            policy_version=outcome.policy.version if outcome.policy is not None else None,
            module=body.module, direction=body.direction, action_taken=decision,
            category=outcome.category or (outcome.combined.categories[0]
                                          if outcome.combined and outcome.combined.categories else None),
            request_id=request_id, content=outcome.screened_content,
            user_id=body.user_id, session_id=body.session_id,
        )
    codes = list(outcome.combined.categories) if outcome.combined and not outcome.combined.safe else []
    extra: dict = {}
    if body.modality is not None:
        # CM-1: the same S-codes, read through the moderation taxonomy of the
        # policy this decision used. Disabled guardrail = nothing was checked.
        extra["moderation"] = None if outcome.disabled else moderation.moderate_codes(
            codes, body.modality, moderation.moderation_for(outcome.policy))
    return DecideResponse(
        decision=decision, reason=reason,
        policy_version=outcome.policy.version if outcome.policy is not None else None,
        request_id=request_id,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        redacted_content=redacted,
        # What the guard model(s) reported, whatever the policy then decided —
        # module 12's PII scan keys off S7 even when the policy allows it (GRD-X1 Q23).
        categories=codes,
        **extra,
    )


@router.post("/check/input", response_model=CheckResult, summary="Policy check on an input prompt")
async def check_input(body: CheckRequest) -> CheckResult:
    return await checker.check(body, "input")


@router.post("/check/output", response_model=CheckResult, summary="Policy check on a generated response")
async def check_output(body: CheckRequest) -> CheckResult:
    return await checker.check(body, "output")


@router.post("/prompt_test", summary="Send a raw prompt straight to a chosen Ollama model, return its raw output")
async def prompt_test(body: PromptTestRequest) -> dict:
    """
    把 `prompt` 原封不動送給 `model`（Ollama `/api/generate`），並把 Ollama
    回傳的結果回傳——不套用任何 guard prompt、policy，也不做任何解析。

    唯一的例外是 Ollama 回傳的 `context` 欄位：那是一串 token id（數字陣列），
    是給 Ollama 自己接續對話用的內部狀態，不是人類看得懂的文字，因此這裡直接
    移除，只保留 `response` 等原本就是文字/可讀內容的欄位。
    """
    settings = get_settings()
    payload = {"model": body.model, "prompt": body.prompt, "stream": False}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate",
                json=payload,
                timeout=settings.request_timeout,
            )
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail="Ollama request timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.status_code}") from exc
    data = resp.json()
    data.pop("context", None)
    return data


@router.get("/violations", response_model=ViolationListResponse, summary="List logged policy violations")
async def list_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    module: str | None = None,
    direction: Literal["input", "output"] | None = None,
    category: str | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
) -> ViolationListResponse:
    rows, total = await audit_log.list_violations(
        page=page, page_size=page_size, module=module, direction=direction,
        category=category, from_ts=from_, to_ts=to,
    )
    return ViolationListResponse(
        items=[ViolationRecord(**dict(r)) for r in rows],
        total=total, page=page, page_size=page_size,
    )


@router.get("/violations/summary", response_model=ViolationSummaryResponse,
            summary="Violation counts by category + action, last 24h")
async def violations_summary() -> ViolationSummaryResponse:
    since, until, rows = await audit_log.summary_last_24h()
    return ViolationSummaryResponse(
        since=since, until=until,
        counts=[ViolationSummaryEntry(**dict(r)) for r in rows],
    )
