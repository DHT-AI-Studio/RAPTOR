from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentItem(BaseModel):
    id: str | None = None
    vector: list[float] | None = None
    payload: dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = 10
    payload_schema: str = "contextual"

    # filters
    branch_id: str | None = None
    embedding_type: str | None = None
    type: str | list[str] | None = None
    filename: list[str] | None = None
    speaker: list[str] | None = None
    source: str | None = None

class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]

class SearchResponse(BaseModel):
    results: list[SearchResult]
    timing: dict | None = None

class RerankDocument(BaseModel):
    id: str
    text: str
    payload: dict[str, Any] | None = None

class RerankRequest(BaseModel):
    query: str
    documents: list[RerankDocument]
    top_k: int | None = None  # None = return all

class RerankResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] | None = None

class RerankResponse(BaseModel):
    results: list[RerankResult]

class PayloadSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    content_extractor: Callable[[dict[str, Any]], str]
    bm25_fields: list[str] = []
    nested_paths: list[str] = []
    skip_status_filter: bool = False  # True 時不加 status=active 條件
    skip_req_filters: bool = False    # True 時忽略 req 裡的 embedding_type/type/filename 等條件


def contextual_extractor(payload: dict[str, Any]) -> str:
    return payload.get("text") or payload.get("summary") or ""

def temporal_reasoning_extractor(payload: dict[str, Any]) -> str:
    events = payload.get("events", [])
    descriptions = [e.get("description", "") for e in events]
    return " ".join(descriptions)


DEFAULT_EXTRACTORS = {
    "contextual": PayloadSchema(
        name="contextual",
        description="上下文相關文件",
        content_extractor=contextual_extractor,
        bm25_fields=["payload.text", "payload.summary"],
        nested_paths=[]
    ),
    "temporal": PayloadSchema(
        name="temporal",
        description="時間推理相關文件",
        content_extractor=temporal_reasoning_extractor,
        bm25_fields=["payload.events.description"],
        nested_paths=["payload.events"],
        skip_status_filter=True,
        skip_req_filters=True
    ),
}