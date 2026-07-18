"""
agents/tkg_agent.py  —  port 52004
A2A wrapper for Temporal Knowledge Graph via Module 20 /tkg/query.
Returns entity relationships and temporal facts as GraphResult (triples + Ollama summary).
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

from models import TKGRequest, GraphResult
from a2a_helpers import unpack_request, pack_result

PORT = 52004


def _do_tkg(req: TKGRequest) -> GraphResult:
    graph_url = os.environ.get("GRAPH_SERVICE_URL", "http://raptor-graph-service:8843")
    body: dict = {
        "query":           req.query,
        "max_depth":       req.max_depth,
        "limit":           50,
        "score_threshold": req.score_threshold,
    }
    if req.time_start:
        body["time_start"] = req.time_start
    if req.time_end:
        body["time_end"] = req.time_end
    if req.branch_id:
        body["branch_id"] = req.branch_id

    triples: list[dict] = []
    with httpx.Client(timeout=30.0) as client:
        try:
            resp = client.post(f"{graph_url}/tkg/query", json=body)
            resp.raise_for_status()
            data = resp.json()

            # subgraph_edges: {start_id, type, end_id}
            for edge in data.get("subgraph_edges", []):
                triples.append({
                    "subject":   edge.get("start_id", ""),
                    "predicate": edge.get("type", ""),
                    "object":    edge.get("end_id", ""),
                })

            # temporal_facts: {entity, relation, value, time_start, time_end}
            for fact in data.get("temporal_facts", []):
                triples.append({
                    "subject":    fact.get("entity", ""),
                    "predicate":  fact.get("relation", ""),
                    "object":     fact.get("value", ""),
                    "valid_from": fact.get("time_start"),
                    "valid_to":   fact.get("time_end"),
                })

        except Exception as exc:
            logger.warning("[TKG] Module 20 call failed: {}", exc)

    summary = _summarise(req.query, triples)
    entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
    entities.discard("")
    logger.info("[TKG] {} triples, {} entities", len(triples), len(entities))
    return GraphResult(triples=triples, summary=summary, entity_count=len(entities), source="tkg")


def _summarise(query: str, triples: list[dict]) -> str:
    if not triples:
        return "No relevant temporal graph relationships found."
    ollama_url = os.environ.get("OLLAMA_HOST", "http://192.168.157.135:11434")
    model      = os.environ.get("OLLAMA_MODEL", "qwen2.5")
    text = "\n".join(
        f"  {t['subject']} --[{t['predicate']}]--> {t['object']}"
        + (f" ({t.get('valid_from', '')} ~ {t.get('valid_to', '')})" if t.get("valid_from") or t.get("valid_to") else "")
        for t in triples[:20]
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{ollama_url}/api/chat",
                json={"model": model, "stream": False,
                      "messages": [{"role": "user", "content":
                          f"Summarise these temporal knowledge graph relationships relevant to '{query}':\n{text}\n"
                          "3-5 sentences, highlight key entities, relationships, and any time references."}]},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.warning("[TKG] Ollama summary failed: {}", exc)
        return f"Found {len(triples)} temporal graph relationships."


class TKGExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, TKGRequest)
            result = _do_tkg(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:
            logger.error("[TKG] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(GraphResult(triples=[], summary=str(e), entity_count=0, source="tkg"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="tkg_agent",
    description="Temporal Knowledge Graph via Module 20 /tkg/query. Entity relationships with time-range filtering.",
    version="0.3.0",
    url=f"http://raptor-tkg-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="tkg", name="Temporal KG",
        description="Temporal entity relationship retrieval from Neo4j with optional time-range filtering.",
        tags=["graph", "neo4j", "temporal", "tkg", "relationships"],
        examples=['{"query": "events involving entity X", "time_start": "2024-01-01", "max_depth": 2}'],
        inputModes=["text"], outputModes=["text"],
    )],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(agent_executor=TKGExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, handler)
    logger.info("TKGAgent starting on port {}", PORT)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)
