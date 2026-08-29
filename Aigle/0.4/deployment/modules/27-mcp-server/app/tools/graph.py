from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.core.config import get_settings
from app.tools import get_client

logger = logging.getLogger(__name__)

_GRAPH_DESC = """\
GraphRAG entity and relationship query — finds matched entities and expands a subgraph.

Use for "Who is X?" or "How are X and Y related?" style questions.
Returns semantic triples (subject–relation–object), a short summary, the entity
count, and co-occurring entities.
"""

_TKG_DESC = """\
Temporal Knowledge Graph query — time-indexed entity events and relationships.

Use for "What happened to X between date A and date B?" style questions.
Returns time-ordered temporal facts, semantic triples, and a short summary.
"""

# Personal-db (module 25) variants -- were added alongside the module
# 17/20-backed tools above without touching them (13/27 were held back from
# the modules 15/21/22 migration). Now commented out, not deleted: 13's
# `/personal-db/search/graphrag` and `/personal-db/search/temporal` routes
# these called are themselves commented out (see personal_db.py) since
# `/search/graphrag` and `/search/tkg` above now ARE module 25 under the old
# 0.3-style naming -- raptor_graph_query/raptor_tkg_query need no changes,
# they already call those paths.
# _PERSONAL_DB_GRAPH_DESC = """\
# GraphRAG entity and relationship query over the caller's own personal database (module 25).

# Use for "Who is X?" or "How are X and Y related?" style questions, scoped to
# this user's own uploaded content only (physically isolated per-user database,
# not the shared global graph raptor_graph_query queries).
# Returns semantic triples (subject–relation–object), a short summary, the entity
# count, and co-occurring entities.
# """

# _PERSONAL_DB_TKG_DESC = """\
# Temporal Knowledge Graph query over the caller's own personal database (module 25).

# Use for "What happened to X between date A and date B?" style questions,
# scoped to this user's own uploaded content only (physically isolated
# per-user database, not the shared global graph raptor_tkg_query queries).
# Returns time-ordered temporal facts, semantic triples, and a short summary.
# """

# Graph-plumbing edge types that are not semantic knowledge — kept out of
# `triples` so the payload stays small and useful for an LLM. CO_OCCURS_WITH is
# summarised separately as `co_occurs_with`; the rest are dropped.
_STRUCTURAL_RELATIONS = {
    "CO_OCCURS_WITH", "MENTIONED_IN", "APPEARS_IN", "HAS_MOMENT", "HAS_TEMPORAL_FACT",
}
_COOCCUR_LIMIT = get_settings().cooccur_limit


def _map_edge(rel: Any) -> Optional[dict]:
    """Map one backend edge to {subject, relation, object[, confidence]}.

    Raptor edges look like::

        {"type": "RELATION", "start_id": "xi", "end_id": "trump",
         "properties": {"predicate": "RELATION_GOOD_WITH", "confidence": 0.9}}

    subject/object come from start_id/end_id, the relation from
    properties.predicate (falling back to the edge ``type`` for structural edges).
    Other key spellings are accepted; an unmappable edge returns None.

    Module 25 (personal-db) edges set the semantic relation text at
    properties.relation, not properties.predicate (graph_indexer.py's
    index_relationship(): `SET relation=:r`) -- without this key in the
    chain, every personal-db edge fell through to the generic `type` field
    (always the literal string "RELATION"), silently losing the actual
    relation text. Confirmed via a full read of graph_indexer.py, not
    assumed.
    """
    if not isinstance(rel, dict):
        return None
    props = rel.get("properties") if isinstance(rel.get("properties"), dict) else {}
    subj = (rel.get("start_id") or rel.get("source") or rel.get("from")
            or rel.get("subject") or rel.get("head"))
    obj = (rel.get("end_id") or rel.get("target") or rel.get("to")
           or rel.get("object") or rel.get("tail"))
    pred = (props.get("predicate") or props.get("relation") or rel.get("relation")
            or rel.get("predicate") or rel.get("type") or rel.get("label"))
    if not (subj and obj and pred):
        return None
    triple = {"subject": subj, "relation": pred, "object": obj}
    conf = props.get("confidence", rel.get("confidence"))
    if conf is not None:
        triple["confidence"] = conf
    return triple


def _semantic_triples(edges: list[Any]) -> tuple[list[dict], list[str]]:
    """Split edges into semantic triples and a capped co-occurrence list.

    Structural edges are dropped from `triples`; CO_OCCURS_WITH targets are
    collected (deduped, capped) as lightweight "related entities" context.
    """
    triples: list[dict] = []
    cooccur: list[str] = []
    for rel in edges or []:
        t = _map_edge(rel)
        if t is None:
            continue
        if t["relation"] in _STRUCTURAL_RELATIONS:
            if t["relation"] == "CO_OCCURS_WITH" and t["object"] not in cooccur:
                cooccur.append(t["object"])
            continue
        triples.append(t)
    return triples, cooccur[:_COOCCUR_LIMIT]


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_GRAPH_DESC.strip())
    async def raptor_graph_query(
        query: Annotated[str, "Natural language query, e.g. 'Who is X?' or 'How are X and Y related?'"],
        entity: Annotated[Optional[str], "Optional entity name to focus the query on."] = None,
        max_depth: Annotated[int, "Subgraph expansion depth (1–4)."] = 2,
        limit: Annotated[int, "Maximum nodes to return (1–200)."] = 50,
        strategy: Annotated[str, "Search strategy: 'hybrid', 'literal', or 'semantic'."] = "hybrid",
        ctx: Context = None,
    ) -> str:
        effective_query = f"{query} {entity}".strip() if entity else query
        await ctx.info(f"raptor_graph_query: query={effective_query!r} depth={max_depth}")

        body = {"query": effective_query, "max_depth": max_depth, "limit": limit, "strategy": strategy}

        try:
            client = await get_client(ctx)
            data = await client.post_json(
                "/search/graphrag", body, tool_name="raptor_graph_query")
        except Exception as exc:
            await ctx.error(f"raptor_graph_query failed: {exc}")
            raise

        entities = data.get("matched_entities", [])
        edges = data.get("relationships") or data.get("subgraph_edges") or []
        triples, cooccurs = _semantic_triples(edges)
        result = {
            "triples": triples,
            "entity_count": len(entities),
            "co_occurs_with": cooccurs,
            "summary": (
                f"Matched {len(entities)} entities and {len(triples)} semantic "
                f"relationships for query {effective_query!r}."
            ),
            "entities": entities,
        }
        await ctx.info(f"raptor_graph_query: {len(entities)} entities, {len(triples)} semantic triples")
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TKG_DESC.strip())
    async def raptor_tkg_query(
        query: Annotated[str, "Natural language query about time-ordered events."],
        time_start: Annotated[Optional[str], "Start date filter (ISO 8601, e.g. '2025-01-01')."] = None,
        time_end: Annotated[Optional[str], "End date filter (ISO 8601, e.g. '2026-12-31')."] = None,
        max_depth: Annotated[int, "Subgraph expansion depth (1–4)."] = 2,
        limit: Annotated[int, "Maximum nodes to return (1–200)."] = 50,
        ctx: Context = None,
    ) -> str:
        await ctx.info(
            f"raptor_tkg_query: query={query!r} time_start={time_start} time_end={time_end}")

        body: dict = {"query": query, "max_depth": max_depth, "limit": limit}
        if time_start:
            body["time_start"] = time_start
        if time_end:
            body["time_end"] = time_end

        try:
            client = await get_client(ctx)
            data = await client.post_json(
                "/search/tkg", body, tool_name="raptor_tkg_query")
        except Exception as exc:
            await ctx.error(f"raptor_tkg_query failed: {exc}")
            raise

        entities = data.get("matched_entities", [])
        facts = data.get("temporal_facts", [])
        edges = data.get("subgraph_edges") or data.get("relationships") or []
        triples, cooccurs = _semantic_triples(edges)
        window = ""
        if time_start or time_end:
            window = f" between {time_start or '…'} and {time_end or '…'}"
        result = {
            "temporal_facts": facts,
            "triples": triples,
            "co_occurs_with": cooccurs,
            "summary": (
                f"Matched {len(entities)} entities, {len(facts)} temporal facts and "
                f"{len(triples)} semantic relationships for query {query!r}{window}."
            ),
            "entities": entities,
        }
        await ctx.info(f"raptor_tkg_query: {len(entities)} entities, {len(facts)} facts, {len(triples)} triples")
        return json.dumps(result, ensure_ascii=False, indent=2)

#     @mcp.tool(description=_PERSONAL_DB_GRAPH_DESC.strip())
#     async def raptor_personal_db_graph_query(
#         query: Annotated[str, "Natural language query, e.g. 'Who is X?' or 'How are X and Y related?'"],
#         entity: Annotated[Optional[str], "Optional entity name to focus the query on."] = None,
#         max_depth: Annotated[int, "Subgraph expansion depth (1–4)."] = 2,
#         limit: Annotated[int, "Maximum nodes to return (1–200)."] = 50,
#         strategy: Annotated[str, "Search strategy: 'hybrid', 'literal', or 'semantic'."] = "hybrid",
#         ctx: Context = None,
#     ) -> str:
#         effective_query = f"{query} {entity}".strip() if entity else query
#         await ctx.info(f"raptor_personal_db_graph_query: query={effective_query!r} depth={max_depth}")

#         body = {"query": effective_query, "max_depth": max_depth, "limit": limit, "strategy": strategy}

#         try:
#             client = await get_client(ctx)
#             data = await client.post_json(
#                 "/personal-db/search/graphrag", body, tool_name="raptor_personal_db_graph_query")
#         except Exception as exc:
#             await ctx.error(f"raptor_personal_db_graph_query failed: {exc}")
#             return json.dumps({"error": str(exc)}, ensure_ascii=False)

#         entities = data.get("matched_entities", [])
#         edges = data.get("relationships") or data.get("subgraph_edges") or []
#         triples, cooccurs = _semantic_triples(edges)
#         result = {
#             "triples": triples,
#             "entity_count": len(entities),
#             "co_occurs_with": cooccurs,
#             "summary": (
#                 f"Matched {len(entities)} entities and {len(triples)} semantic "
#                 f"relationships for query {effective_query!r} (personal database)."
#             ),
#             "entities": entities,
#         }
#         await ctx.info(
#             f"raptor_personal_db_graph_query: {len(entities)} entities, {len(triples)} semantic triples")
#         return json.dumps(result, ensure_ascii=False, indent=2)

#     @mcp.tool(description=_PERSONAL_DB_TKG_DESC.strip())
#     async def raptor_personal_db_tkg_query(
#         query: Annotated[str, "Natural language query about time-ordered events."],
#         time_start: Annotated[Optional[str], "Start date filter (ISO 8601, e.g. '2025-01-01')."] = None,
#         time_end: Annotated[Optional[str], "End date filter (ISO 8601, e.g. '2026-12-31')."] = None,
#         max_depth: Annotated[int, "Subgraph expansion depth (1–4)."] = 2,
#         limit: Annotated[int, "Maximum nodes to return (1–200)."] = 50,
#         ctx: Context = None,
#     ) -> str:
#         await ctx.info(
#             f"raptor_personal_db_tkg_query: query={query!r} time_start={time_start} time_end={time_end}")

#         body: dict = {"query": query, "max_depth": max_depth, "limit": limit}
#         if time_start:
#             body["time_start"] = time_start
#         if time_end:
#             body["time_end"] = time_end

#         try:
#             client = await get_client(ctx)
#             # module 25's TKG route is named "temporal" on module 13's gateway,
#             # not "tkg" like the module 20-backed /search/tkg above.
#             data = await client.post_json(
#                 "/personal-db/search/temporal", body, tool_name="raptor_personal_db_tkg_query")
#         except Exception as exc:
#             await ctx.error(f"raptor_personal_db_tkg_query failed: {exc}")
#             return json.dumps({"error": str(exc)}, ensure_ascii=False)

#         entities = data.get("matched_entities", [])
#         facts = data.get("temporal_facts", [])
#         edges = data.get("subgraph_edges") or data.get("relationships") or []
#         triples, cooccurs = _semantic_triples(edges)
#         window = ""
#         if time_start or time_end:
#             window = f" between {time_start or '…'} and {time_end or '…'}"
#         result = {
#             "temporal_facts": facts,
#             "triples": triples,
#             "co_occurs_with": cooccurs,
#             "summary": (
#                 f"Matched {len(entities)} entities, {len(facts)} temporal facts and "
#                 f"{len(triples)} semantic relationships for query {query!r}{window} (personal database)."
#             ),
#             "entities": entities,
#         }
#         await ctx.info(
#             f"raptor_personal_db_tkg_query: {len(entities)} entities, {len(facts)} facts, {len(triples)} triples")
#         return json.dumps(result, ensure_ascii=False, indent=2)
