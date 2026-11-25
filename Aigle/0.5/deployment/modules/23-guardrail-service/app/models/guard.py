"""Request/response models for the guard-model check endpoints
(app/routers/check_guard.py, app/routers/check_policy_llm.py), plus the
guard_classifier.CombinedVerdict -> CheckResponse conversion both routers
share."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.services import guard_classifier


class Message(BaseModel):
    role:    str = Field(..., description="訊息角色：`user` 或 `assistant`")
    content: str = Field(..., description="訊息內容")


class MessageRequest(BaseModel):
    """Role 由 endpoint 決定，請求中只需提供 content。

    module / request_id 是可選的呼叫端資訊（例如 module 07、13 打這幾條端點時
    帶的 caller module 名稱與請求 ID），只用來寫進稽核紀錄（audit_log），
    對安全檢查本身完全沒有影響。"""
    content: str = Field(..., description="要檢查的文字內容")
    module: str | None = Field(None, description="呼叫端模組名稱（僅供稽核紀錄使用）")
    request_id: str | None = Field(None, description="呼叫端請求 ID（僅供稽核紀錄使用）")


class ConversationRequest(BaseModel):
    """多輪對話訊息列表。模型只評估最後一條訊息是否安全。"""
    messages: list[Message] = Field(
        ...,
        description="對話歷史（依序排列，至少一則）。role 使用 `user` 或 `assistant`。",
    )


class SingleModelResult(BaseModel):
    """單一模型的安全檢查結果。"""
    model:          str            = Field(..., description="產生此結果的模型名稱")
    safe:           bool           = Field(..., description="True = 通過安全檢查")
    categories:     list[str]      = Field(..., description="違規類別代碼（llama-guard3 S1-S14；其他模型為空列表）")
    category_names: dict[str, str] = Field(..., description="類別代碼對應名稱")
    raw:            str            = Field(..., description="模型原始輸出")


class CheckResponse(BaseModel):
    safe: bool = Field(
        ..., description="True = 通過安全檢查；False = 偵測到違規（多模型時任一不安全即 False）"
    )
    categories: list[str] = Field(
        ..., description="違規類別代碼列表（來自 llama-guard3）；安全時為空列表"
    )
    category_names: dict[str, str] = Field(
        ..., description="類別代碼對應名稱"
    )
    raw: str = Field(
        ..., description="主模型（llama-guard3 優先）原始輸出"
    )
    conflict: bool | None = Field(
        None, description="多模型模式：True = 模型間結果衝突；單模型模式為 null"
    )
    results: list[SingleModelResult] | None = Field(
        None, description="多模型模式：各模型獨立結果；單模型模式為 null"
    )


class RawModelResult(BaseModel):
    model: str = Field(..., description="模型名稱")
    raw:   str = Field(..., description="模型原始輸出（未解析）")


class RawCheckResponse(BaseModel):
    results: list[RawModelResult] = Field(..., description="各模型原始輸出列表")


def combined_to_response(result: guard_classifier.CombinedVerdict) -> CheckResponse:
    return CheckResponse(
        safe=result.safe, categories=result.categories, category_names=result.category_names,
        raw=result.raw, conflict=result.conflict,
        results=None if result.results is None else [
            SingleModelResult(model=r.model, safe=r.safe, categories=r.categories,
                               category_names=r.category_names, raw=r.raw)
            for r in result.results
        ],
    )
