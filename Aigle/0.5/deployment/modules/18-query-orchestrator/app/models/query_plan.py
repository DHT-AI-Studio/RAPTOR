from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueryPlan:
    modes: list[str]                   # e.g. ["VideoRAG"] or ["VideoRAG", "RDBMS"] or ["GraphRAG", "TKG"]
    sort_by: str = "relevance"         # "relevance" | "time_asc" | "time_desc"
    group_by_doc: bool = False
    top_k_override: int | None = None
    time_from: datetime | None = None
    time_to: datetime | None = None
    rel_position: str | None = None  # "end" | "start"
    tail_seconds: int | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "modes": self.modes,
            "sort_by": self.sort_by,
            "group_by_doc": self.group_by_doc,
            "top_k_override": self.top_k_override,
            "time_from": self.time_from.isoformat() if self.time_from else None,
            "time_to": self.time_to.isoformat() if self.time_to else None,
            "rel_position": self.rel_position,
            "tail_seconds": self.tail_seconds,
            "confidence": self.confidence,
        }
