"""
agents/reranker_agent.py  —  port 52003
A2A wrapper for reranking.
Primary:  Module 25 /personal/search/rerank (was Module 17 /api/v1/search/rerank
          -- see the commented-out block below _rerank_module17, kept for rollback)
Fallback: local cross-encoder (sentence-transformers)
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

from models import RerankerRequest, SearchResult, SearchHit
from a2a_helpers import unpack_request, pack_result

PORT = 52005


def _deduplicate(candidates: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in candidates:
        cid = c.get("id", "")
        if cid not in seen or c.get("score", 0) > seen[cid].get("score", 0):
            seen[cid] = c
    return list(seen.values())


_PROVISIONED: set[str] = set()


def _ensure_database(personal_db_url: str, branch_id: str) -> None:
    """Provision the caller's module-25 database on first use (idempotent).
    Never raises -- a provisioning failure surfaces as whatever the rerank
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
        logger.warning("[Reranker] module 25 db init failed for {}: {}", branch_id, exc)


def _rerank_module25(query: str, candidates: list[dict], top_k: int, branch_id: str = "") -> list[dict] | None:
    """Call Module 25 rerank endpoint. Returns None on failure."""
    personal_db_url = os.environ.get("PERSONAL_DB_SERVICE_URL", "http://raptor-personal-db-service:8000")
    _ensure_database(personal_db_url, branch_id)
    url = f"{personal_db_url}/personal/search/rerank"
    documents = [{"id": c.get("id", ""), "text": c.get("content", "")} for c in candidates]
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"query": query, "documents": documents}, headers={"X-Branch-ID": branch_id})
        resp.raise_for_status()
        # Module 25 returns {results: [{id, score}]} — map scores back to candidates
        score_map = {item["id"]: item["score"] for item in resp.json().get("results", [])}
        if not score_map:
            return None
        reranked = []
        for c in candidates:
            cid = c.get("id", "")
            if cid in score_map:
                c = c.copy()
                c["score"] = score_map[cid]
                reranked.append(c)
        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked[:top_k]
    except Exception as exc:
        logger.warning("[Reranker] Module 25 rerank failed: {} — falling back to local", exc)
        return None


# ---------------------------------------------------------------------------
# Old module 17 implementation -- commented out, not deleted, for rollback.
# ---------------------------------------------------------------------------
# def _rerank_module17(query: str, candidates: list[dict], top_k: int) -> list[dict] | None:
#     """Call Module 17 rerank endpoint. Returns None on failure."""
#     hybrid_url = os.environ.get("HYBRID_SEARCH_URL", "http://raptor-hybridsearch-api:8000")
#     url = f"{hybrid_url}/api/v1/search/rerank"
#     documents = [{"id": c.get("id", ""), "text": c.get("content", "")} for c in candidates]
#     try:
#         with httpx.Client(timeout=30.0) as client:
#             resp = client.post(url, json={"query": query, "documents": documents})
#         resp.raise_for_status()
#         # Module 17 returns {results: [{id, score}]} — map scores back to candidates
#         score_map = {item["id"]: item["score"] for item in resp.json().get("results", [])}
#         if not score_map:
#             return None
#         reranked = []
#         for c in candidates:
#             cid = c.get("id", "")
#             if cid in score_map:
#                 c = c.copy()
#                 c["score"] = score_map[cid]
#                 reranked.append(c)
#         reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
#         return reranked[:top_k]
#     except Exception as exc:
#         logger.warning("[Reranker] Module 17 rerank failed: {} — falling back to local", exc)
#         return None


def _rerank_local(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Local cross-encoder fallback."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs  = [(query, c.get("content", "")) for c in candidates]
        scores = model.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = []
        for c, score in scored[:top_k]:
            c = c.copy()
            c["score"] = float(score)
            result.append(c)
        return result
    except Exception as exc:
        logger.warning("[Reranker] local cross-encoder failed: {} — returning RRF order", exc)
        return candidates[:top_k]


def _do_rerank(req: RerankerRequest) -> SearchResult:
    deduped = _deduplicate(req.candidates)
    logger.info("[Reranker] {} candidates → {} unique, top_k={}", len(req.candidates), len(deduped), req.top_k)

    reranked = _rerank_module25(req.query, deduped, req.top_k)
    source   = "reranker:module25"
    if reranked is None:
        reranked = _rerank_local(req.query, deduped, req.top_k)
        source   = "reranker:local"

    hits = [
        SearchHit(
            id=c.get("id", ""), doc_id=c.get("doc_id", ""),
            score=c.get("score", 0.0), content=c.get("content", ""),
            metadata=c.get("metadata", {}),
            start_time=c.get("start_time"), end_time=c.get("end_time"),
            asset_path=c.get("asset_path"), version_id=c.get("version_id"),
            storage_uri=c.get("storage_uri"),
        )
        for c in reranked
    ]
    return SearchResult(hits=hits, total=len(hits), source=source)


class RerankerExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, RerankerRequest)
            result = _do_rerank(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:
            logger.error("[Reranker] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(SearchResult(hits=[], total=0, source="reranker"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="reranker_agent",
    description="Re-scores merged candidates. Primary: Module 25 rerank. Fallback: local cross-encoder.",
    version="0.3.0",
    url=f"http://raptor-reranker-agent:52005",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="rerank", name="Reranker",
        description="Cross-encoder reranking of merged search candidates.",
        tags=["rerank", "cross-encoder", "fusion"],
        examples=['{"query": "cooling system", "candidates": [...], "top_k": 5}'],
        inputModes=["text"], outputModes=["text"],
    )],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(agent_executor=RerankerExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, handler)
    logger.info("RerankerAgent starting on port {}", PORT)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)
