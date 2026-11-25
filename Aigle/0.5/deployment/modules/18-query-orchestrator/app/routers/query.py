from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.intent_classifier import IntentClassifierService
from app.services.signal_extractor import extract_signals
from app.services.temporal_parse import ParseFailure, parse_client
from app.services.time_window import window_end, window_start

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Classify"])


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1)


class QueryPlanResponse(BaseModel):
    modes: list[str]
    sort_by: str
    group_by_doc: bool
    top_k_override: int | None
    time_from: str | None
    time_to: str | None
    keywords: list[str] = []               # TMP-4: terms a TKG search filters the evidence by
    match: str | None = None               # P07: TKG Allen match mode -- "overlaps" for a closed window
    # P16: the event the answer is relative to ("after Google acquired Android") and the Allen relations a
    # fact must have to it; TKG contract v2 `reference: {"text": ...}` / `reference_relations`
    reference: str | None = None
    reference_relations: list[str] | None = None
    # MTR-25: a question about who appears in the videos -- "people" (PeopleAgent), "people+semantic"
    # (PeopleAgent + vector search) or "people+temporal" (PeopleAgent + TKG); None otherwise
    people_intent: str | None = None
    rel_position: str | None
    tail_seconds: int | None
    # TMP-7: who read the time condition -- "parse" (Module 25, the single normaliser) or "regex" (the interim
    # TMP-2 reader, until parse is proven available) -- and what came of it. UNRESOLVED: the question must not
    # be answered without its time condition (`answer` null, no window, no reference, no keywords).
    temporal_source: str = "regex"
    temporal_status: str = "NO_TEMPORAL"   # "RESOLVED" | "NO_TEMPORAL" | "UNRESOLVED"
    temporal_reason: str | None = None     # UNRESOLVED: "parse_failure", or Module 25's reason
    temporal_constraint_applied: bool = False
    answer: str | None = None
    confidence: float
    routing_strategy: str = "rule_based"   # "rule_based" | "llm"
    matched_rules: list[str] = []         # 匹配的規則 ID 列表


def _temporal(query: str, signals) -> dict:
    """The query's time condition: from Module 25's `parse` once it is proven available, else from the
    interim regex (TMP-2). A failed or unreadable parse is UNRESOLVED, never replaced by the regex."""
    tz = get_settings().timezone
    if not parse_client.available():
        time_from, time_to = window_start(signals.time_from, tz), window_end(signals.time_to, tz)
        applied = bool(time_from or time_to or signals.reference)
        return {"temporal_source": "regex", "temporal_status": "RESOLVED" if applied else "NO_TEMPORAL",
                "temporal_reason": None, "temporal_constraint_applied": applied,
                # whole days as instants with an explicit offset (TMP-3), e.g. "2011" ->
                # 2011-01-01T00:00:00.000+08:00 .. 2011-12-31T23:59:59.999+08:00
                "time_from": time_from, "time_to": time_to, "match": signals.match, "keywords": signals.keywords,
                "reference": signals.reference, "reference_relations": signals.reference_relations}
    unresolved = {"temporal_source": "parse", "temporal_status": "UNRESOLVED", "temporal_constraint_applied": False,
                  "time_from": None, "time_to": None, "match": None, "keywords": [],
                  "reference": None, "reference_relations": None}
    try:
        parsed = parse_client.parse(query)
    except ParseFailure as exc:
        _logger.warning("temporal parse failed for the query (%s): UNRESOLVED", exc)
        return {**unresolved, "temporal_reason": "parse_failure"}
    if parsed["status"] == "UNRESOLVED":
        return {**unresolved, "temporal_reason": parsed.get("reason") or "unresolved"}
    window = parsed.get("window") or {}
    return {"temporal_source": "parse", "temporal_status": parsed["status"], "temporal_reason": None,
            "temporal_constraint_applied": parsed["status"] == "RESOLVED",
            "time_from": window.get("start"), "time_to": window.get("end"), "match": parsed.get("match"),
            "keywords": [str(k) for k in parsed.get("keywords") or []],
            "reference": parsed.get("reference"), "reference_relations": parsed.get("reference_relations")}


@router.post("/classify", response_model=QueryPlanResponse)
def classify_endpoint(body: ClassifyRequest, request: Request):
    classifier: IntentClassifierService = request.app.state.intent_classifier
    signals = extract_signals(body.query)
    temporal = _temporal(body.query, signals)

     # Classifier (rule-based) returns GraphRAG / TKG / RDBMS or empty list.
    qic_result = classifier.classify(body.query, k=3)
    modes: set[str] = set(qic_result.intent_type) - {"unknown"}

    # P16: a question relative to an event is a temporal question, whatever the classifier made of it
    if temporal["reference"]:
        modes.add("TKG")

     # VideoRAG fallback: if no structured pipeline matched, treat as a visual
     # content search.  Having an extracted date range does NOT imply TKG —
     # a year can equally indicate a metadata filter (RDBMS).
    if not modes:
        modes.add("VideoRAG")

    people_intent = classifier.people_intent(body.query, temporal=temporal["temporal_constraint_applied"])

    return QueryPlanResponse(
        modes=sorted(modes),
        sort_by=signals.sort_by,
        group_by_doc=signals.group_by_doc,
        top_k_override=signals.top_k_override,
        **temporal,
        people_intent=people_intent,
        rel_position=signals.rel_position,
        tail_seconds=signals.tail_seconds,
        confidence=qic_result.confidence,
        routing_strategy=qic_result.routing_strategy,
        matched_rules=qic_result.matched_rules,
    )
