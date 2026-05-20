from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.services.intent_classifier import IntentClassifierService
from app.services.signal_extractor import extract_signals

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Classify"])


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1)


class QueryPlanResponse(BaseModel):
    modes: List[str]
    sort_by: str
    group_by_doc: bool
    top_k_override: Optional[int]
    time_from: Optional[str]
    time_to: Optional[str]
    rel_position: Optional[str]
    tail_seconds: Optional[int]
    confidence: float
    routing_strategy: str = "rule_based"   # "rule_based" | "llm"
    matched_rules: List[str] = []         # 匹配的規則 ID 列表


@router.post("/classify", response_model=QueryPlanResponse)
def classify_endpoint(body: ClassifyRequest, request: Request):
    classifier: IntentClassifierService = request.app.state.intent_classifier
    signals = extract_signals(body.query)

     # Classifier (rule-based) returns GraphRAG / TKG / RDBMS or empty list.
    qic_result = classifier.classify(body.query, k=3)
    modes: set[str] = set(qic_result.intent_type) - {"unknown"}

     # VideoRAG fallback: if no structured pipeline matched, treat as a visual
     # content search.  Having an extracted date range does NOT imply TKG —
     # a year can equally indicate a metadata filter (RDBMS).
    if not modes:
        modes.add("VideoRAG")

    return QueryPlanResponse(
        modes=sorted(modes),
        sort_by=signals.sort_by,
        group_by_doc=signals.group_by_doc,
        top_k_override=signals.top_k_override,
        time_from=signals.time_from.isoformat() if signals.time_from else None,
        time_to=signals.time_to.isoformat() if signals.time_to else None,
        rel_position=signals.rel_position,
        tail_seconds=signals.tail_seconds,
        confidence=qic_result.confidence,
        routing_strategy=qic_result.routing_strategy,
        matched_rules=qic_result.matched_rules,
    )
