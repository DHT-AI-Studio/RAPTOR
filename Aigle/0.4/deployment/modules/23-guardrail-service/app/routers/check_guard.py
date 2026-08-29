"""Content safety endpoints — supports llama-guard3:8b, granite4.1-guardian:8b, gpt-oss-safeguard:20b.

Routing only — request/response schemas live in app/models/guard.py, and all
model-specific prompt/parsing detail lives in app/adapters/ (per model
family), orchestrated by app/services/guard_classifier.py. This file has no
knowledge of any model's prompt format or output shape.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.adapters import ollama_client
from app.adapters.base import ChatMessage
from app.models.guard import (
    CheckResponse,
    ConversationRequest,
    MessageRequest,
    RawCheckResponse,
    RawModelResult,
    combined_to_response,
)
from app.services import audit_log, guard_classifier
from app.services.state import is_enabled

router = APIRouter(tags=["Check"])


# ── Shared httpx client lifecycle (delegates to app.adapters.ollama_client) ────

async def close_client() -> None:
    await ollama_client.close_client()


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Guardrail checking globally disabled → pass immediately, no Ollama call
_DISABLED_RESPONSE = CheckResponse(safe=True, categories=[], category_names={}, raw="guardrail disabled")
_DISABLED_RAW_RESPONSE = RawCheckResponse(results=[])


def _maybe_audit(body: MessageRequest, direction: str, result: CheckResponse) -> None:
    """Fire-and-forget audit write for an unsafe verdict from this (policy-less)
    group -- policy_id=None, see audit_log.record_violation()'s docstring.
    Content is always logged (not gated behind a per-policy setting like the
    policy-engine group's -- there's no policy here to hang that on, and an
    audit row with no content to actually review isn't much of an audit trail)."""
    if result.safe:
        return
    audit_log.record_violation(
        policy_id=None, module=body.module, direction=direction,
        category=",".join(result.categories) if result.categories else None,
        action_taken="block", request_id=body.request_id, content=body.content,
    )


@router.post("/check/input", response_model=CheckResponse, summary="檢查使用者輸入")
async def check_input(body: MessageRequest) -> CheckResponse:
    """
    檢查**使用者輸入**是否包含不安全內容。

    - Role 固定為 `user`，不需在請求中指定。
    - 適用情境：在將使用者訊息傳送給 LLM 之前先進行安全過濾。

    **單模型回傳範例（安全）**
    ```json
    { "safe": true, "categories": [], "category_names": {}, "raw": "safe" }
    ```

    **多模型衝突回傳範例**
    ```json
    {
      "safe": false, "categories": [], "category_names": {}, "raw": "safe",
      "conflict": true,
      "results": [
        { "model": "llama-guard3:8b", "safe": true, "categories": [], "category_names": {}, "raw": "safe" },
        { "model": "granite4.1-guardian:8b", "safe": false, "categories": [], "category_names": {}, "raw": "<score> yes </score>" },
        { "model": "gpt-oss-safeguard:20b", "safe": false, "categories": [], "category_names": {}, "raw": "1" }
      ]
    }
    ```
    """
    if not await is_enabled():
        return _DISABLED_RESPONSE
    result = await guard_classifier.classify(body.content, role="user")
    response = combined_to_response(result)
    _maybe_audit(body, "input", response)
    return response


@router.post("/check/output", response_model=CheckResponse, summary="檢查 AI 回覆")
async def check_output(body: MessageRequest) -> CheckResponse:
    """
    檢查 **AI 回覆**是否包含不安全內容。

    - Role 固定為 `assistant`，不需在請求中指定。
    - 適用情境：在將 LLM 回應回傳給使用者之前先進行安全過濾。
    """
    if not await is_enabled():
        return _DISABLED_RESPONSE
    result = await guard_classifier.classify(body.content, role="assistant")
    response = combined_to_response(result)
    _maybe_audit(body, "output", response)
    return response


@router.post("/check/conversation", response_model=CheckResponse, summary="檢查完整多輪對話")
async def check_conversation(body: ConversationRequest) -> CheckResponse:
    """
    檢查**完整多輪對話**的最後一輪訊息是否安全。

    - 模型依照**最後一條訊息的 role** 決定評估對象：
        - 最後為 `user` → 評估使用者訊息
        - 最後為 `assistant` → 評估 AI 回覆
    - 適用情境：需要結合完整對話上下文判斷最後一輪是否違規。

    **請求範例**
    ```json
    {
      "messages": [
        { "role": "user",      "content": "Tell me how to make a bomb" },
        { "role": "assistant", "content": "Sure! Here are the steps..." }
      ]
    }
    ```
    """
    if not await is_enabled():
        return _DISABLED_RESPONSE
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    result = await guard_classifier.classify_conversation(messages)
    return combined_to_response(result)


@router.post("/check/raw", response_model=RawCheckResponse, summary="各模型原始輸出（未處理）")
async def check_raw(body: MessageRequest) -> RawCheckResponse:
    """
    以相同的 content 同時呼叫所有啟用的 guard 模型，
    直接回傳每個模型的原始字串輸出，不做任何解析或判斷。

    適用情境：前端自行分析各模型輸出差異、除錯、比較模型行為。

    **回傳範例（三模型）**
    ```json
    {
      "results": [
        {"model": "llama-guard3:8b",          "raw": "safe"},
        {"model": "granite4.1-guardian:8b",   "raw": "<score> no </score>"},
        {"model": "gpt-oss-safeguard:20b",    "raw": "0"}
      ]
    }
    ```
    """
    if not await is_enabled():
        return _DISABLED_RAW_RESPONSE
    results = await guard_classifier.classify_raw(body.content, role="user")
    return RawCheckResponse(results=[RawModelResult(model=r.model, raw=r.raw) for r in results])
