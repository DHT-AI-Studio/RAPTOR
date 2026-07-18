"""
agents/graphrag_agent.py  —  port 52003
A2A wrapper for Graph RAG via Module 20 /query/graph_rag.
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


def _do_graphrag(req: GraphRAGRequest) -> GraphResult:
    graph_url = os.environ.get("GRAPH_SERVICE_URL", "http://raptor-graph-service:8843")
    body: dict = {
        "query":           req.query,
        "max_depth":       req.max_depth,
        "limit":           50,
        "score_threshold": req.score_threshold,
    }
    if req.entity:
        body["entity"] = req.entity
    if req.branch_id:
        body["branch_id"] = req.branch_id

    triples: list[dict] = []
    with httpx.Client(timeout=30.0) as client:
        try:
            resp = client.post(f"{graph_url}/query/graph_rag", json=body)
            resp.raise_for_status()
            data = resp.json()
            for node in data.get("nodes", []):
                triples.append({
                    "subject":   node.get("source", ""),
                    "predicate": str(node.get("rel_types", [])),
                    "object":    node.get("target", ""),
                })
        except Exception as exc:
            logger.warning("[GraphRAG] Module 20 call failed: {}", exc)

    summary = _summarise(req.query, triples)
    entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
    logger.info("[GraphRAG] {} triples, {} entities", len(triples), len(entities))
    return GraphResult(triples=triples, summary=summary, entity_count=len(entities), source="graphrag")


def _summarise(query: str, triples: list[dict]) -> str:
    if not triples:
        return "No relevant graph relationships found."
    ollama_url = os.environ.get("OLLAMA_HOST", "http://192.168.157.135:11434")
    model      = os.environ.get("OLLAMA_MODEL", "qwen2.5")
    text = "\n".join(
        f"  {t['subject']} --[{t['predicate']}]--> {t['object']}"
        for t in triples[:20]
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={"model": model, "stream": False,
                      "messages": [{"role": "user", "content":
                          f"Summarise these graph relationships relevant to '{query}':\n{text}\n"
                          "3-5 sentences, highlight the most important connections."}]},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.warning("[GraphRAG] Ollama summary failed: {}", exc)
        return f"Found {len(triples)} graph relationships."


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
    description="Graph RAG via Module 20 /query/graph_rag. Entity relationships and structural context.",
    version="0.3.0",
    url=f"http://raptor-graphrag-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="graphrag", name="Graph RAG",
        description="Entity relationship retrieval from Neo4j knowledge graph via Module 20.",
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
