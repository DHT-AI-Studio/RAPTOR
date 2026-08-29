"""Response models for the debug variant of /policy/check/llm/*
(app/routers/debug_policy_check_llm.py). Kept out of app/models/guard.py so
removing the debug feature never touches that file.

Same shape as app.models.guard.CheckResponse/SingleModelResult (used by
/guard/check/* and /policy/check/llm/*), plus `prompt` on each per-model
result — the exact, human-readable prompt sent to that model. Unlike
CheckResponse, `results` here is always populated (never null), even with a
single active guard model, so the per-model `prompt` is never dropped."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DebugSingleModelResult(BaseModel):
    """單一模型的安全檢查結果，額外附上送給該模型的 prompt。"""
    model:          str            = Field(..., description="產生此結果的模型名稱")
    safe:           bool           = Field(..., description="True = 通過安全檢查")
    categories:     list[str]      = Field(..., description="違規類別代碼（llama-guard3 S1-S14；其他模型為空列表）")
    category_names: dict[str, str] = Field(..., description="類別代碼對應名稱")
    raw:            str            = Field(..., description="模型原始輸出")
    prompt:         str            = Field(..., description="實際送往此模型的 prompt（人類可讀）")


class DebugCheckResponse(BaseModel):
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
    results: list[DebugSingleModelResult] | None = Field(
        None, description="各模型獨立結果（含 prompt）；停用時為 null"
    )
