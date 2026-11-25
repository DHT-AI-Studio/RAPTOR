"""Request/response models for POST /guardrail/policy/decide (GRD-X1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["allow", "redact", "warn", "block"]


class DecideRequest(BaseModel):
    content: str = Field(..., description="要判定的文字內容")
    direction: Literal["input", "output"] = Field(
        "input", description="`input` = 使用者輸入；`output` = AI 回覆")
    module: str | None = Field(None, description="呼叫端模組名稱（寫入稽核紀錄）")
    request_id: str | None = Field(None, description="呼叫端請求 ID；未提供時自動產生")
    user_id: str | None = Field(None, description="使用者 ID；只以 SHA-256 雜湊形式寫入稽核紀錄（audit_events.actor_id_hash，CF-4）")
    session_id: str | None = Field(None, description="Session ID；寫入稽核紀錄的 metadata.session_id（resource_id 用 request_id）")
    modality: Literal["text", "transcript"] | None = Field(
        None,
        description="CM-1 內容類型（`text` | `transcript`）。提供時，回應多一個 `moderation` 欄位："
                    "依 active policy 的 moderation 分類（nsfw / violence / hate / self_harm）與該類型的 decision "
                    "判定的結果，格式與 image / video_frame 相同。不影響 `decision`；未提供時回應不變")


class DecideResponse(BaseModel):
    decision: Decision = Field(..., description="allow | redact | warn | block")
    reason: str = Field(..., description="判定原因，例如 `S1: Violent Crimes`、`regex: <pattern>`、`no violation`")
    policy_version: str | None = Field(None, description="套用的 active policy 版本；未套用 policy 時為 null")
    request_id: str = Field(..., description="請求 ID（呼叫端提供的，或自動產生的）")
    latency_ms: float = Field(..., description="判定耗時（毫秒）")
    redacted_content: str | None = Field(None, description="decision 為 `redact` 時，遮蔽後的內容")
    categories: list[str] = Field(
        default_factory=list,
        description="guard model 回報的類別代碼（例如 `S7`），與 `/guard/check/*` 的 `categories` 相同；"
                    "不受 policy 影響（policy 判定 allow 時仍會列出）。guard 判定 safe 或 guardrail 關閉時為空")
    moderation: dict | None = Field(
        None,
        description="只在請求有 `modality` 時出現：CM-1 moderation 判定（`decision`、`flagged`、`categories`、"
                    "`not_covered`、`source`、`policy_name`、`policy_version`、`moderation_version`），"
                    "由 guard model 回報的 S-code 對應到 moderation 分類。guardrail 關閉時為 null（沒有檢查）")
