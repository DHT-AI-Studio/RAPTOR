"""
agents/graphrag_agent.py  —  port 52003
A2A wrapper for Graph RAG via Module 25 (raptor-personal-db-service).
Backend: POST /personal/search/graphrag (X-Branch-ID header, per-caller isolated DB)
Was Module 20 (raptor-graph-service) /query/graph_rag -- see the commented-out
block below _do_graphrag for the old implementation, kept for rollback.

Module 25's response shape differs from Module 20's: relationships live in a
separate `relationships` list (`{type, from_id, to_id, properties}`, semantic
relation text in `properties.relation` -- the generic `type` is usually just
"RELATION"/"MENTIONS"), not flattened onto each node as `source`/`rel_types`/
`target` the way Module 20 does. Building triples now requires a node-id ->
name lookup from `nodes` first.

Returns entity relationships as GraphResult (triples + Ollama summary).
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

from models import GraphRAGRequest, GraphResult
from a2a_helpers import unpack_request, pack_result

PORT = 52003

_PROVISIONED: set[str] = set()


def _ensure_database(personal_db_url: str, branch_id: str) -> None:
    """Provision the caller's module-25 database on first use (idempotent).
    Never raises -- a provisioning failure surfaces as whatever the GraphRAG
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
        logger.warning("[GraphRAG] module 25 db init failed for {}: {}", branch_id, exc)


def _do_graphrag(req: GraphRAGRequest) -> GraphResult:
    personal_db_url = os.environ.get("PERSONAL_DB_SERVICE_URL", "http://raptor-personal-db-service:8000")
    body: dict = {
        "query":           req.query,
        "max_depth":       req.max_depth,
        "limit":           50,
        "score_threshold": req.score_threshold,
    }
    # req.entity has no equivalent on Module 25's GraphRAGRequest -- entity
    # seeding there comes only from the query's own fulltext match, not a
    # caller-supplied seed -- so it's accepted on this agent's own request
    # model (A2A protocol compatibility) but not forwarded.
    branch_id = req.branch_id or ""
    _ensure_database(personal_db_url, branch_id)

    triples: list[dict] = []
    with httpx.Client(timeout=30.0) as client:
        try:
            resp = client.post(
                f"{personal_db_url}/personal/search/graphrag",
                json=body,
                headers={"X-Branch-ID": branch_id},
            )
            resp.raise_for_status()
            data = resp.json()

            # node id -> display name, so relationships (from_id/to_id) can be
            # rendered as readable triples. Entity nodes carry `name` at the
            # top level; other node types fall back to their id.
            id_to_name = {n["id"]: (n.get("name") or n["id"]) for n in data.get("nodes", [])}

            for edge in data.get("relationships", []):
                props = edge.get("properties", {}) or {}
                predicate = props.get("relation") or props.get("predicate") or edge.get("type", "")
                from_id = edge.get("from_id", "")
                to_id   = edge.get("to_id", "")
                triples.append({
                    "subject":   id_to_name.get(from_id, from_id),
                    "predicate": predicate,
                    "object":    id_to_name.get(to_id, to_id),
                })
        except Exception as exc:
            logger.warning("[GraphRAG] Module 25 call failed: {}", exc)

    summary = _summarise(req.query, triples)
    entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
    logger.info("[GraphRAG] {} triples, {} entities", len(triples), len(entities))
    return GraphResult(triples=triples, summary=summary, entity_count=len(entities), source="graphrag")


# ---------------------------------------------------------------------------
# Old module 20 implementation -- commented out, not deleted, for rollback.
# ---------------------------------------------------------------------------
# def _do_graphrag(req: GraphRAGRequest) -> GraphResult:
#     graph_url = os.environ.get("GRAPH_SERVICE_URL", "http://raptor-graph-service:8843")
#     body: dict = {
#         "query":           req.query,
#         "max_depth":       req.max_depth,
#         "limit":           50,
#         "score_threshold": req.score_threshold,
#     }
#     if req.entity:
#         body["entity"] = req.entity
#     if req.branch_id:
#         body["branch_id"] = req.branch_id
#
#     triples: list[dict] = []
#     with httpx.Client(timeout=30.0) as client:
#         try:
#             resp = client.post(f"{graph_url}/query/graph_rag", json=body)
#             resp.raise_for_status()
#             data = resp.json()
#             for node in data.get("nodes", []):
#                 triples.append({
#                     "subject":   node.get("source", ""),
#                     "predicate": str(node.get("rel_types", [])),
#                     "object":    node.get("target", ""),
#                 })
#         except Exception as exc:
#             logger.warning("[GraphRAG] Module 20 call failed: {}", exc)
#
#     summary = _summarise(req.query, triples)
#     entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
#     logger.info("[GraphRAG] {} triples, {} entities", len(triples), len(entities))
#     return GraphResult(triples=triples, summary=summary, entity_count=len(entities), source="graphrag")


def _summarise(query: str, triples: list[dict]) -> str:
    if not triples:
        return "No relevant graph relationships found."
    # Module 07 (ai-lifecycle-api), not Ollama's native /api/chat directly --
    # think=false by default there, unlike this call site which set no think
    # field at all and used a 30s timeout (shorter than 20/25's 90s, so even
    # more exposed to qwen3.x's default "thinking" mode adding 10-20x latency
    # for the same final answer, measured live at 4.8s vs 59s on an
    # equivalent extraction prompt). Same INFERENCE_URL pipeline.py's
    # _llm_answer() already uses.
    inference_url = os.environ.get("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5")
    think = os.environ.get("INFERENCE_THINK", "false").lower() == "true"
    text = "\n".join(
        f"  {t['subject']} --[{t['predicate']}]--> {t['object']}"
        for t in triples[:20]
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{inference_url}/v1/chat/completions",
                json={
                    "model": model,
                    "engine": "ollama",
                    "messages": [{"role": "user", "content":
                        f"Summarise these graph relationships relevant to '{query}':\n{text}\n"
                        "3-5 sentences, highlight the most important connections."}],
                    "temperature": 0.0,
                    "think": think,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("[GraphRAG] Ollama summary failed: {}", exc)
        return f"Found {len(triples)} graph relationships."

    # ollama_url = os.environ.get("OLLAMA_HOST", "http://<host_ip>:11434")
    # try:
    #     with httpx.Client(timeout=30.0) as client:
    #         resp = client.post(
    #             f"{ollama_url}/api/chat",
    #             json={"model": model, "stream": False,
    #                   "messages": [{"role": "user", "content":
    #                       f"Summarise these graph relationships relevant to '{query}':\n{text}\n"
    #                       "3-5 sentences, highlight the most important connections."}]},
    #         )
    #         resp.raise_for_status()
    #         return resp.json()["message"]["content"].strip()
    # except Exception as exc:
    #     logger.warning("[GraphRAG] Ollama summary failed: {}", exc)
    #     return f"Found {len(triples)} graph relationships."


class GraphRAGExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, GraphRAGRequest)
            result = _do_graphrag(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:
            logger.error("[GraphRAG] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(GraphResult(triples=[], summary=str(e), entity_count=0, source="graphrag"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="graphrag_agent",
    description="Graph RAG via Module 25 (ArcadeDB, per-user isolated). Entity relationships and structural context.",
    version="0.3.0",
    url=f"http://raptor-graphrag-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="graphrag", name="Graph RAG",
        description="Entity relationship retrieval from Module 25 (ArcadeDB) per-user graph.",
        tags=["graph", "neo4j", "relationships", "rag"],
        examples=['{"query": "how does A relate to B", "max_depth": 2}'],
        inputModes=["text"], outputModes=["text"],
    )],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(agent_executor=GraphRAGExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, handler)
    logger.info("GraphRAGAgent starting on port {}", PORT)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)
