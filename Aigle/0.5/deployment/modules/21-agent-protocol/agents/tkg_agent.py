"""
agents/tkg_agent.py  —  port 52004
A2A wrapper for Temporal Knowledge Graph via Module 25 (raptor-personal-db-service).
Backend: POST /personal/search/tkg (X-Branch-ID header, per-caller isolated DB)
Was Module 20 (raptor-graph-service) /tkg/query -- see the commented-out block
below _do_tkg for the old implementation, kept for rollback.
Returns entity relationships and temporal facts as GraphResult (triples + a summary).
P17: the summary is grounded -- verified claims rendered by template (grounded_claims.py), or the
facts as recorded -- unless TKG_GROUNDED_ANSWERS=false brings back the free-text model summary.
"""
from __future__ import annotations

import os

import grounded_claims
import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import get_message_text, new_agent_text_message
from a2a_helpers import pack_result, unpack_request
from loguru import logger
from models import GraphResult, TKGRequest

PORT = 52004

_PROVISIONED: set[str] = set()


def _ensure_database(personal_db_url: str, branch_id: str) -> None:
    """Provision the caller's module-25 database on first use (idempotent).
    Never raises -- a provisioning failure surfaces as whatever the TKG
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
    except httpx.HTTPError as exc:
        logger.warning("[TKG] module 25 db init failed for {}: {}", branch_id, exc)


def _do_tkg(req: TKGRequest) -> GraphResult:
    personal_db_url = os.environ.get("PERSONAL_DB_SERVICE_URL", "http://raptor-personal-db-service:8000")
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
    if req.entity:
        body["entity"] = req.entity
    if req.keywords:
        body["keywords"] = req.keywords
    if req.match:
        body["match"] = req.match
    if req.reference and req.reference_relations:
        # P16: a reference event is a contract-v2 feature (Module 25 resolves it, or says why it can't)
        body.update({"version": 2, "reference": {"text": req.reference},
                     "reference_relations": req.reference_relations})
    if req.source_filter:
        body.update({"version": 2, "source_filter": req.source_filter})       # P21: these sources only
    branch_id = req.branch_id or ""
    reference: dict | None = None
    _ensure_database(personal_db_url, branch_id)

    triples: list[dict] = []
    facts: list[dict] = []
    complete = False
    with httpx.Client(timeout=30.0) as client:
        try:
            resp = client.post(
                f"{personal_db_url}/personal/search/tkg",
                json=body,
                headers={"X-Branch-ID": branch_id},
            )
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
                    # P07: Module 25's Allen relations to the window and whether the fact certainly matches
                    **({"allen": fact["allen"]} if fact.get("allen") else {}),
                    # D8: the dates at the precision they have ("2024-07"); valid_from/valid_to pad a
                    # month or a year to a day. Absent from an older Module 25.
                    **({"when": fact["when"]} if fact.get("when") else {}),
                })
            reference = data.get("reference")
            facts = list(data.get("temporal_facts", []))
            complete = bool(data.get("complete"))

        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("[TKG] Module 25 call failed: {}", exc)

    if reference and reference.get("status") != "resolved":
        # no facts were returned: say why, never let a summary guess which event was meant
        summary = _unresolved_reference(reference)
        logger.info("[TKG] reference {}: {}", reference.get("status"), reference.get("text"))
        return GraphResult(triples=[], summary=summary, entity_count=0, source="tkg")
    if reference:
        triples.insert(0, _reference_triple(reference))
    claims: list[dict] = []
    if os.environ.get("TKG_GROUNDED_ANSWERS", "true").lower() != "false":
        summary, claims = grounded_claims.grounded_summary(req.query, facts, complete, reference,
                                                           personal_db_url, branch_id)
        if reference and reference.get("fact"):
            summary = f"Relative to: {grounded_claims.evidence_line(reference['fact'])}\n{summary}"
    else:
        summary = _summarise(req.query, triples)
    entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
    entities.discard("")
    logger.info("[TKG] {} triples, {} entities", len(triples), len(entities))
    return GraphResult(triples=triples, summary=summary, entity_count=len(entities), source="tkg", claims=claims)


# ---------------------------------------------------------------------------
# Old module 20 implementation -- commented out, not deleted, for rollback.
# ---------------------------------------------------------------------------
# def _do_tkg(req: TKGRequest) -> GraphResult:
#     graph_url = os.environ.get("GRAPH_SERVICE_URL", "http://raptor-graph-service:8843")
#     body: dict = {
#         "query":           req.query,
#         "max_depth":       req.max_depth,
#         "limit":           50,
#         "score_threshold": req.score_threshold,
#     }
#     if req.time_start:
#         body["time_start"] = req.time_start
#     if req.time_end:
#         body["time_end"] = req.time_end
#     if req.branch_id:
#         body["branch_id"] = req.branch_id
#
#     triples: list[dict] = []
#     with httpx.Client(timeout=30.0) as client:
#         try:
#             resp = client.post(f"{graph_url}/tkg/query", json=body)
#             resp.raise_for_status()
#             data = resp.json()
#
#             # subgraph_edges: {start_id, type, end_id}
#             for edge in data.get("subgraph_edges", []):
#                 triples.append({
#                     "subject":   edge.get("start_id", ""),
#                     "predicate": edge.get("type", ""),
#                     "object":    edge.get("end_id", ""),
#                 })
#
#             # temporal_facts: {entity, relation, value, time_start, time_end}
#             for fact in data.get("temporal_facts", []):
#                 triples.append({
#                     "subject":    fact.get("entity", ""),
#                     "predicate":  fact.get("relation", ""),
#                     "object":     fact.get("value", ""),
#                     "valid_from": fact.get("time_start"),
#                     "valid_to":   fact.get("time_end"),
#                 })
#
#         except Exception as exc:
#             logger.warning("[TKG] Module 20 call failed: {}", exc)
#
#     summary = _summarise(req.query, triples)
#     entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
#     entities.discard("")
#     logger.info("[TKG] {} triples, {} entities", len(triples), len(entities))
#     return GraphResult(triples=triples, summary=summary, entity_count=len(entities), source="tkg")


def _maybe(t: dict) -> str:
    """P16: a fact whose dates allow the requested relation to the reference event without proving it."""
    vs_ref = (t.get("allen") or {}).get("reference") or {}
    return " [possibly]" if vs_ref.get("certainty") == "possible" else ""


def _reference_triple(reference: dict) -> dict:
    """P16: the resolved reference event, first, marked so the summary relates the rest to it."""
    f = reference.get("fact") or {}
    return {"subject": f.get("entity", ""), "predicate": f.get("relation", ""), "object": f.get("value", ""),
            "valid_from": f.get("time_start"), "valid_to": f.get("time_end"), "reference": True,
            **({"when": f["when"]} if f.get("when") else {})}


def _unresolved_reference(reference: dict) -> str:
    """P16: why there are no facts relative to the event the question named."""
    text = reference.get("text") or reference.get("fact_id") or ""
    status = reference.get("status")
    if status == "ambiguous":
        options = "; ".join(f"{c.get('entity', '')} {c.get('relation', '')} {c.get('value', '')}".strip()
                            for c in reference.get("candidates") or [])
        return (f"The event \"{text}\" matches several different events in the data, so nothing can be placed "
                f"relative to it until it is made specific. Candidates: {options}.")
    if status == "undated":
        return f"The event \"{text}\" is in the data, but without a date, so nothing can be placed relative to it."
    return f"The event \"{text}\" was not found in the data."


def _dates(t: dict) -> str:
    """A triple's dates for the summary prompt. `when` (D8) carries their precision -- "2024-07",
    "2008 to 2013"; without it (an older Module 25) the padded days are all there is."""
    if t.get("when"):
        return f" ({t['when']})"
    if t.get("valid_from") or t.get("valid_to"):
        return f" ({t.get('valid_from', '')} ~ {t.get('valid_to', '')})"
    return ""


def _summarise(query: str, triples: list[dict]) -> str:
    if not triples:
        return "No relevant temporal graph relationships found."
    # Module 07 (ai-lifecycle-api), not Ollama's native /api/chat directly --
    # see graphrag_agent.py's _summarise() for the full rationale (think=false
    # by default via 07; this call site had no think control and a 30s
    # timeout, measured live at 4.8s vs 59s with/without think on an
    # equivalent prompt). Same INFERENCE_URL pipeline.py's _llm_answer() uses.
    inference_url = os.environ.get("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
    model = os.environ.get("OLLAMA_MODEL", "gemma4:12b")
    think = os.environ.get("INFERENCE_THINK", "false").lower() == "true"
    text = "\n".join(f"  {'REFERENCE EVENT: ' if t.get('reference') else ''}"
                     f"{t['subject']} --[{t['predicate']}]--> {t['object']}{_dates(t)}{_maybe(t)}" for t in triples[:20])
    relative = any(t.get("reference") for t in triples)
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{inference_url}/v1/chat/completions",
                json={
                    "model": model,
                    "engine": "ollama",
                    "messages": [{"role": "user", "content":
                        f"Summarise these temporal knowledge graph relationships relevant to '{query}':\n{text}\n"
                        "3-5 sentences, highlight key entities, relationships, and any time references. "
                        "State each date exactly as precisely as it is given: '2024-07' is July 2024, "
                        "not 1 July; '2008' is the year 2008."
                        + (" The REFERENCE EVENT is what the question is relative to; the other lines were "
                           "selected by their time relation to it. A line marked [possibly] has dates that allow "
                           "that relation without proving it: say so, never state it as certain." if relative else "")}],
                    "temperature": 0.0,
                    "think": think,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
        logger.warning("[TKG] Ollama summary failed: {}", exc)
        return f"Found {len(triples)} temporal graph relationships."

    # ollama_url = os.environ.get("OLLAMA_HOST", "http://<host_ip>:11434")
    # try:
    #     with httpx.Client(timeout=30.0) as client:
    #         resp = client.post(
    #             f"{ollama_url}/api/chat",
    #             json={"model": model, "stream": False,
    #                   "messages": [{"role": "user", "content":
    #                       f"Summarise these temporal knowledge graph relationships relevant to '{query}':\n{text}\n"
    #                       "3-5 sentences, highlight key entities, relationships, and any time references."}]},
    #         )
    #         resp.raise_for_status()
    #         return resp.json()["message"]["content"].strip()
    # except Exception as exc:
    #     logger.warning("[TKG] Ollama summary failed: {}", exc)
    #     return f"Found {len(triples)} temporal graph relationships."


class TKGExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, TKGRequest)
            result = _do_tkg(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:  # noqa: BLE001 -- top-level A2A executor safety net (request parsing + downstream calls); already logged + returns structured fallback result
            logger.error("[TKG] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(GraphResult(triples=[], summary=str(e), entity_count=0, source="tkg"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="tkg_agent",
    description="Temporal Knowledge Graph via Module 25 (ArcadeDB, per-user isolated). Entity relationships with time-range filtering.",
    version="0.3.0",
    url=f"http://raptor-tkg-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="tkg", name="Temporal KG",
        description="Temporal entity relationship retrieval from Module 25 (ArcadeDB) with optional time-range filtering.",
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
