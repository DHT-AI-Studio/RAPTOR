from pydantic import BaseModel
from pydantic import ConfigDict
from typing import Optional, Dict, Any, List, Union, Callable


class DocumentItem(BaseModel):
    id: Optional[str] = None
    vector: Optional[List[float]] = None
    payload: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    payload_schema: str = "contextual"

    # filters
    branch_id: Optional[str] = None
    embedding_type: Optional[str] = None
    type: Optional[Union[str, List[str]]] = None
    filename: Optional[List[str]] = None
    speaker: Optional[List[str]] = None
    source: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    timing: Optional[Dict] = None

class RerankDocument(BaseModel):
    id: str
    text: str
    payload: Optional[Dict[str, Any]] = None

class RerankRequest(BaseModel):
    query: str
    documents: List[RerankDocument]
    top_k: Optional[int] = None  # None = return all

class RerankResult(BaseModel):
    id: str
    score: float
    payload: Optional[Dict[str, Any]] = None

class RerankResponse(BaseModel):
    results: List[RerankResult]

class PayloadSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    content_extractor: Callable[[Dict[str, Any]], str]
    bm25_fields: List[str] = []
    nested_paths: List[str] = []
    skip_status_filter: bool = False  # True 時不加 status=active 條件
    skip_req_filters: bool = False    # True 時忽略 req 裡的 embedding_type/type/filename 等條件


def contextual_extractor(payload: Dict[str, Any]) -> str:
    return payload.get("text") or payload.get("summary") or ""

def temporal_reasoning_extractor(payload: Dict[str, Any]) -> str:
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