"""
agents/keyword_search_agent.py  —  port 52002
A2A wrapper for BM25 keyword search via Module 25 (raptor-personal-db-service).
Backend: POST /personal/search/bm25 (X-Branch-ID header, per-caller isolated DB)
Was Module 17 (raptor-hybridsearch-api) -- see the commented-out block below
_do_keyword_search for the old implementation, kept for rollback.
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

from models import BM25SearchRequest, SearchResult, SearchHit
from a2a_helpers import unpack_request, pack_result

PORT = 52002

_EXCLUDE_PAYLOAD_KEYS = frozenset({
    "doc_id", "text", "summary", "enriched_content",
    "start_time", "end_time", "storage_uri", "asset_path", "version_id",
})


_PROVISIONED: set[str] = set()


def _ensure_database(personal_db_url: str, branch_id: str) -> None:
    """Provision the caller's module-25 database on first use (idempotent).
    Never raises -- a provisioning failure surfaces as whatever the search
    call itself then returns."""
    if not branch_id or branch_id in _PROVISIONED:
        return
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{personal_db_url}/internal/db/init",
                headers={"X-Branch-ID": branch_id},
            )
        resp.raise_for_status()
        _PROVISIONED.add(branch_id)
    except Exception as exc:
        logger.warning("[KeywordSearch] module 25 db init failed for {}: {}", branch_id, exc)


def _do_keyword_search(req: BM25SearchRequest) -> SearchResult:
    personal_db_url = os.environ.get("PERSONAL_DB_SERVICE_URL", "http://raptor-personal-db-service:8000")
    body: dict = {"query": req.query, "top_k": req.top_k}
    if req.type is not None:
        body["type"] = req.type
    branch_id = req.branch_id or ""
    _ensure_database(personal_db_url, branch_id)
    url  = f"{personal_db_url}/personal/search/bm25"
    logger.info("[KeywordSearch] → {} body={}", url, body)

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=body, headers={"X-Branch-ID": branch_id})
    resp.raise_for_status()
    data = resp.json()

    hits: list[SearchHit] = []
    for r in data.get("results", []):
        p = r.get("payload", {})
        hits.append(SearchHit(
            id=str(r["id"]),
            doc_id=p.get("doc_id") or str(r["id"]),
            score=float(r.get("score", 0.0)),
            content=p.get("text") or p.get("summary") or p.get("enriched_content", ""),
            metadata={k: v for k, v in p.items() if k not in _EXCLUDE_PAYLOAD_KEYS},
            start_time=p.get("start_time"),
            end_time=p.get("end_time"),
            asset_path=p.get("asset_path"),
            version_id=p.get("version_id"),
            storage_uri=p.get("storage_uri"),
        ))
    logger.info("[KeywordSearch] returned {} hits", len(hits))
    return SearchResult(hits=hits, total=len(hits), source="keyword")


# ---------------------------------------------------------------------------
# Old module 17 implementation -- commented out, not deleted, for rollback.
# ---------------------------------------------------------------------------
# def _do_keyword_search(req: BM25SearchRequest) -> SearchResult:
#     hybrid_url = os.environ.get("HYBRID_SEARCH_URL", "http://raptor-hybridsearch-api:8000")
#     body: dict = {"query": req.query, "top_k": req.top_k}
#     if req.type is not None:
#         body["type"] = req.type
#     if req.branch_id:
#         body["branch_id"] = req.branch_id
#     url  = f"{hybrid_url}/api/v1/search/bm25"
#     logger.info("[KeywordSearch] → {} body={}", url, body)
#
#     with httpx.Client(timeout=30.0) as client:
#         resp = client.post(url, json=body)
#     resp.raise_for_status()
#     data = resp.json()
#
#     hits: list[SearchHit] = []
#     for r in data.get("results", []):
#         p = r.get("payload", {})
#         hits.append(SearchHit(
#             id=str(r["id"]),
#             doc_id=p.get("doc_id") or str(r["id"]),
#             score=float(r.get("score", 0.0)),
#             content=p.get("text") or p.get("summary") or p.get("enriched_content", ""),
#             metadata={k: v for k, v in p.items() if k not in _EXCLUDE_PAYLOAD_KEYS},
#             start_time=p.get("start_time"),
#             end_time=p.get("end_time"),
#             asset_path=p.get("asset_path"),
#             version_id=p.get("version_id"),
#             storage_uri=p.get("storage_uri"),
#         ))
#     logger.info("[KeywordSearch] returned {} hits", len(hits))
#     return SearchResult(hits=hits, total=len(hits), source="keyword")


class KeywordSearchExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, BM25SearchRequest)
            result = _do_keyword_search(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:
            logger.error("[KeywordSearch] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(SearchResult(hits=[], total=0, source="keyword"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="keyword_search_agent",
    description="BM25 keyword search via Module 25 (ArcadeDB, per-user isolated).",
    version="0.3.0",
    url=f"http://raptor-keyword-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="keyword_search", name="Keyword Search",
        description="BM25 full-text search. type filter: 'documents'|'audios'|'images'|'videos'.",
        tags=["search", "keyword", "bm25", "opensearch"],
        examples=['{"query": "error code 404", "top_k": 5, "type": "documents"}'],
        inputModes=["text"], outputModes=["text"],
    )],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(agent_executor=KeywordSearchExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, handler)
    logger.info("KeywordSearchAgent starting on port {}", PORT)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)
