"""
Classification Models
"""

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """候選 Intent"""
    intent_id: str
    description: str | None = None
    confidence: float = 0.0
    score: float | None = None  # Raw score (for rule-based, this is the rule weight)
    target_pipeline: str | None = None  # 目標 pipeline，如果為 None 則使用 "unknown"
    rank: int = 1


class ClassificationResult(BaseModel):
    """分類結果"""
    confidence: float = 0.0
    intent_type: list[str] = Field(default_factory=list)  # 例如: ["GraphRAG"] 或 ["GraphRAG", "RDBMS"] 或 ["VideoRAG"]
    intents: list[Candidate] = Field(default_factory=list)  # 候選 Intent 列表
    routing_strategy: str = "rule_based"  # 路由策略: "rule_based" | "llm"
    matched_rules: list[str] = Field(default_factory=list)  # 匹配的規則 ID 列表
    fallback_reason: str | None = None  # Fallback 原因
