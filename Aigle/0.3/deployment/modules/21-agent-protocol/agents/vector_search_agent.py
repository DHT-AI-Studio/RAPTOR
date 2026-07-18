"""
agents/vector_search_agent.py  —  port 52001
A2A wrapper for vector search via Module 17 (raptor-hybridsearch-api).
Backend: POST /api/v1/search/vector
"""
from __future__ import annotations
import os
import httpx
import uvicorn
from loguru import logger

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import get_message_text, new_agent_text_message

from models import VectorSearchRequest, SearchResult, SearchHit
from a2a_helpers import unpack_request, pack_result

PORT = 52001

_EXCLUDE_PAYLOAD_KEYS = frozenset({
    "doc_id", "text", "summary", "enriched_content",
    "start_time", "end_time", "storage_uri", "asset_path", "version_id",
})


def _do_vector_search(req: VectorSearchRequest) -> SearchResult:
    hybrid_url = os.environ.get("HYBRID_SEARCH_URL", "http://raptor-hybridsearch-api:8000")
    body: dict = {"query": req.query, "top_k": req.top_k}
    if req.type is not None:
        body["type"] = req.type
    if req.branch_id:
        body["branch_id"] = req.branch_id
    url  = f"{hybrid_url}/api/v1/search/vector"
    logger.info("[VectorSearch] → {} body={}", url, body)

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=body)
    resp.raise_for_status()
    data = resp.json()

    hits: list[SearchHit] = []
    for r in data.get("results", []):
        score = float(r.get("score", 0.0))
        if score < req.score_threshold:
            continue
        p = r.get("payload", {})
        hits.append(SearchHit(
            id=str(r["id"]),
            doc_id=p.get("doc_id") or str(r["id"]),
            score=score,
            content=p.get("text") or p.get("summary") or p.get("enriched_content", ""),
            metadata={k: v for k, v in p.items() if k not in _EXCLUDE_PAYLOAD_KEYS},
            start_time=p.get("start_time"),
            end_time=p.get("end_time"),
            asset_path=p.get("asset_path"),
            version_id=p.get("version_id"),
            storage_uri=p.get("storage_uri"),
        ))
    logger.info("[VectorSearch] returned {}/{} hits", len(hits), len(data.get("results", [])))
    return SearchResult(hits=hits, total=len(hits), source="vector")


class VectorSearchExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, VectorSearchRequest)
            result = _do_vector_search(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:
            logger.error("[VectorSearch] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(SearchResult(hits=[], total=0, source="vector"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="vector_search_agent",
    description="Semantic vector search via Module 17 (BAAI/bge-m3, Qdrant).",
    version="0.3.0",
    url=f"http://raptor-vector-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="vector_search", name="Vector Search",
        description="Semantic vector search. type filter: 'documents'|'audios'|'images'|'videos'.",
        tags=["search", "vector", "semantic"],
        examples=['{"query": "cooling system", "top_k": 5, "type": "documents"}'],
        inputModes=["text"], outputModes=["text"],
    )],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(agent_executor=VectorSearchExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, handler)
    logger.info("VectorSearchAgent starting on port {}", PORT)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)
