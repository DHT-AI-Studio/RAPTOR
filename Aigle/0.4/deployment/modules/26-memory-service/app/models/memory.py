"""
Pydantic schemas for Module 26 memory compaction (MV-12).

Pulled out of services/compact_memory.py so the request/response contract
lives separately from the service logic that implements it.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field


class CompactEvaluateRequest(BaseModel):
    messages: list[dict] = Field(
        default_factory=list,
        description="要估算 token 用量的訊息列表（不限格式，僅用於估算大小）；"
        "有帶 session_id 時忽略此欄位，改用該 session 的合併算法",
    )
    session_id: Optional[str] = Field(
        None,
        description="省略時維持舊行為，只估算 messages 大小；有帶時改用合併算法，"
        "彙總該 session 的歸檔對話 + 使用者的 long-term facts + multimedia snippets "
        "（等同呼叫 POST /memory/sessions/{session_id}/compact/evaluate）",
    )
    context_window: int = Field(128000, ge=4096, description="目標 LLM 的 context window 大小（tokens）")
    extra_tokens: int = Field(
        0, ge=0, description="Additional tokens from current turn or injected context"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="本次請求實際的輸出 max_tokens；省略時 reserved output 使用設定檔預設值",
    )


class SessionCompactEvaluateRequest(BaseModel):
    context_window: int = Field(128000, ge=4096, description="目標 LLM 的 context window 大小（tokens）")
    extra_tokens: int = Field(
        0, ge=0, description="尚未寫入 session 的當前這輪 turn 的估算 token 數（可省略）"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="本次請求實際的輸出 max_tokens；省略時 reserved output 使用設定檔預設值",
    )


class CompactEvaluation(BaseModel):
    token_count: int = Field(..., description="messages + extra_tokens 估算出的總 token 數")
    context_window: int = Field(..., description="請求中指定的 context window 大小")
    auto_compact_threshold: int = Field(
        ..., description="觸發壓縮的門檻 = context_window - reserved_output_tokens - buffer_tokens"
    )
    should_compact: bool = Field(..., description="token_count 是否已達門檻，建議觸發壓縮")
    tokens_over_threshold: int = Field(..., description="超出門檻的 token 數；未超過則為 0")


class CompactRequest(BaseModel):
    trigger: str = Field("auto", description="觸發來源，例如 auto | manual | reactive，僅作記錄用途")
    context_window: int = Field(128000, ge=4096, description="目標 LLM 的 context window 大小（tokens）")
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="本次請求實際的輸出 max_tokens；省略時 reserved output 使用設定檔預設值",
    )
    last_summarized_frame_id: Optional[str] = Field(
        None,
        description="上次壓縮邊界的 frame_id；省略時會自動尋找該 session 最新的 summary frame 作為邊界",
    )
    custom_instructions: Optional[str] = Field(
        None, description="附加給 LLM 摘要器的額外指示（可省略）"
    )
    dry_run: bool = Field(
        False, description="true 時只計算結果、不寫入 summary frame，用於預覽壓縮效果"
    )


class CompactResponse(BaseModel):
    compacted: bool = Field(..., description="本次是否實際執行了壓縮")
    source: str = Field(
        ...,
        description="under_budget | no_session | no_turns | nothing_to_compact | "
        "session_memory（既有摘要增量更新）| llm_compact（全新 LLM 摘要）| llm_failed（LLM 失敗，fail open）",
    )
    summary_id: Optional[str] = Field(None, description="新寫入 summary frame 的 ID；未壓縮則為 null")
    boundary_id: Optional[str] = Field(
        None, description="新寫入 compact_boundary frame 的 ID；未壓縮或寫入失敗則為 null"
    )
    pre_compact_tokens: int = Field(..., description="壓縮前 session 全部 turns 的估算 token 數")
    post_compact_tokens: int = Field(..., description="壓縮後（摘要 + tail）的估算 token 數")
    turns_compacted: int = Field(..., description="被摘要掉的對話輪數")
    turns_kept: int = Field(..., description="保留原文（tail）的對話輪數")
    summary_frame_written: bool = Field(..., description="summary frame 是否成功寫入 .mv2")
    fallback_used: bool = Field(..., description="是否使用了 LLM 全新摘要（而非既有摘要增量更新）")
    threshold_exceeded: bool = Field(
        False,
        description="壓縮後（含摘要裁切等縮減嘗試）token 數是否仍 ≥ threshold；"
        "true 代表 fail open——已盡力縮減但因 tail 本身已超標而無法完全達標，呼叫方可能需自行再裁切",
    )
    summary_tokens: int = Field(
        0, description="新摘要文字本身的估算 token 數（不含保留的 tail）；未壓縮則為 0"
    )

    # Aliases matching the simplified {turns_before, turns_after, summary_tokens,
    # status} shape some callers expect — derived from the fields above rather
    # than tracked separately, so they can't drift out of sync with them.
    @computed_field  # type: ignore[misc]
    @property
    def turns_before(self) -> int:
        return self.turns_compacted + self.turns_kept

    @computed_field  # type: ignore[misc]
    @property
    def turns_after(self) -> int:
        return self.turns_kept

    @computed_field  # type: ignore[misc]
    @property
    def status(self) -> Literal["ok", "fallback"]:
        return "ok" if self.compacted else "fallback"


class SummaryFrame(BaseModel):
    frame_id: str = Field(..., description="summary frame 在 .mv2 中的 frame ID")
    summary_id: str = Field(..., description="壓縮產生的 summary ID")
    source: str = Field(..., description="摘要來源：session_memory | llm_compact | manual_compact")
    source_turn_range: dict = Field(..., description="被摘要涵蓋的 turn_index 範圍，格式 {\"from\": int, \"to\": int}")
    preserved_tail_from_turn: Optional[int] = Field(
        None, description="本次壓縮保留原文的 tail 起始 turn_index"
    )
    token_count_before: int = Field(..., description="壓縮前的估算 token 數")
    token_count_after: int = Field(..., description="壓縮後的估算 token 數")
    created_at: str = Field(..., description="摘要建立時間（ISO 8601）")
    text: str = Field(..., description="摘要內容（結構化 markdown：Current State / User Intent / ...）")
