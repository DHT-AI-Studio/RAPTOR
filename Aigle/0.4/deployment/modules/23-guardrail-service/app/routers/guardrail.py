"""Policy-aware guardrail check endpoints (GB-4) + violation audit log (GB-6)
+ a raw prompt-test utility.

Mounted under /guardrail → POST /guardrail/check/input and /guardrail/check/output.
These run the active policy's rules via the detector-based checker (distinct from
/guard/* which is raw guard-model classification, and /policy/check/llm/* which is
the LLM-judge path). /guardrail/violations* reads the audit trail those checks write.
/guardrail/prompt_test is unrelated to policy checking — it's a debugging utility
that sends a prompt straight to a chosen Ollama model and returns its raw output,
with no guard prompt, policy, or parsing involved.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.models.check import CheckRequest, CheckResult
from app.models.prompt_test import PromptTestRequest
from app.models.violation import ViolationListResponse, ViolationRecord, ViolationSummaryEntry, ViolationSummaryResponse
from app.services import audit_log, checker

router = APIRouter(tags=["Guardrail (policy checker)"])


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
    module: Optional[str] = None,
    direction: Optional[Literal["input", "output"]] = None,
    category: Optional[str] = None,
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = None,
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
