from __future__ import annotations

import hashlib
import json
import logging
from typing import Annotated, Any
from urllib.parse import quote

from app.core.config import get_settings
from app.tools import get_client
from mcp.server.fastmcp import Context, FastMCP

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
With a time window each fact carries `allen`: its Allen relations to the window and
whether it certainly matches. `match: overlaps` also returns facts that only share
time with the window ("What was X doing in 2010?" finds a 2008-2013 role).
"""

_TEMPORAL_CAPABILITIES_DESC = """\
What RAPTOR's temporal features are in this deployment: TKG contract versions, match modes, Allen
relations, sorts and paging (`raptor_tkg_query` with `sort` / `cursor`), the temporal operations available,
and the extraction pipeline's media types, model and versions. Check here before relying on a feature;
`tkg.unsupported` lists what is defined but not served.
"""

_TEMPORAL_REASON_DESC = """\
What RAPTOR inferred about one uploaded source's undated events from its dates and the before / after /
during links its text states ("joined later, after Android Inc. was founded" -> after 2003-10).
Returns `status` (consistent, or inconsistent with a `witness` naming the conflicting statements), the
`derived_facts` with their sound bounds and the facts they rest on, and links used or left unresolved.
`recompute: true` queues a rebuild instead (no model call).
"""

_TEMPORAL_STATUS_DESC = """\
Temporal processing status of one uploaded source (document, video, audio or image).

Use after an upload, before relying on raptor_tkg_query for that source: its temporal facts exist
only once the status is `complete` (or `partial`). Pass the upload's version_id.
Returns `status` (pending | running | complete | partial | failed | cancelled), `reason`, segment
counts, the number of facts written, the extractor / normaliser versions and model, and the
segments that failed or were in an unsupported language.
"""

_TEMPORAL_PARSE_DESC = """\
The time a question asks about, read with the same grammar RAPTOR extracts dates with. Use it to see how a
question's time condition is understood before a raptor_tkg_query, or to build that query's window.
Returns `status`: RESOLVED (a `window` -- `start` / `end` instants, a null side unbounded -- with its
`match`, and/or a `reference` event with `reference_relations`), NO_TEMPORAL, or UNRESOLVED (a date the
grammar cannot read, with its `reason`: do not answer the question without its time condition).
"after 2011" -> from 2012-01-01; "before 2013" -> up to 2012-12-31; since / until include the period;
「2011年以後」, 「2013年之前」 alike. Pass `anchor` (ISO 8601 with an offset, the question's own time) for
"last year" / 去年.
"""

_TEMPORAL_INGEST_DESC = """\
Give RAPTOR new source material for temporal processing: either inline text (`content`, plain text or
Markdown -- notes, prose, a pasted article; up to 1 MB) or an asset already uploaded (`asset_path` +
`version_id`). Inline text is stored as a source of yours for 7 days (`expires_at`), then dropped from
every search and answer. Returns `source_version_id`: poll raptor_temporal_status with it, then query the
facts with raptor_tkg_query. HTML, email, chat exports and CSV / JSON / logs are refused, not guessed at.
With `question`, RAPTOR waits for processing (up to `wait_seconds`) and answers from this source only; if
it is still processing, `answer_status` is `pending` -- ask again later with the same input.
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


def _map_edge(rel: Any) -> dict | None:
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
        entity: Annotated[str | None, "Optional entity name to focus the query on."] = None,
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
        time_start: Annotated[str | None, "Start date filter (ISO 8601, e.g. '2025-01-01')."] = None,
        time_end: Annotated[str | None, "End date filter (ISO 8601, e.g. '2026-12-31')."] = None,
        max_depth: Annotated[int, "Subgraph expansion depth (1–4)."] = 2,
        limit: Annotated[int, "Maximum nodes to return (1–200)."] = 50,
        match: Annotated[str | None, "How a fact must relate to the window: 'inside' (default), "
                                     "'overlaps' (shares any time with it), 'contains' or 'any'."] = None,
        relations: Annotated[list[str] | None, "Explicit Allen relations to the window, e.g. ['after']; "
                                               "overrides match."] = None,
        certainty: Annotated[str | None, "'possible' (default) or 'certain': only facts that "
                                         "certainly match."] = None,
        sort: Annotated[str | None, "'time' (default) or 'relevance' -- ranks the whole result, then pages "
                                    "it (contract v2)."] = None,
        cursor: Annotated[str | None, "The `cursor` of the previous call, for the next page (contract v2)."] = None,
        reference: Annotated[str | None, "An event the answer is relative to, in words, e.g. 'Google acquired "
                                         "Android' (contract v2). Needs reference_relations."] = None,
        reference_relations: Annotated[list[str] | None, "Allen relations a fact must have to the reference "
                                                         "event, e.g. ['after'] or ['before']."] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(
            f"raptor_tkg_query: query={query!r} time_start={time_start} time_end={time_end} match={match}")

        body: dict = {"query": query, "max_depth": max_depth, "limit": limit}
        if time_start:
            body["time_start"] = time_start
        if time_end:
            body["time_end"] = time_end
        # P07: passed on unchanged; the gateway and Module 25 validate them
        body.update({k: v for k, v in (("match", match), ("relations", relations), ("certainty", certainty)) if v})
        # P08: asking for a sort or a next page is asking for contract v2 (ranked and paged)
        if sort or cursor:
            body.update({"version": 2, **{k: v for k, v in (("sort", sort), ("cursor", cursor)) if v}})
        # P16: relative to an event -- Module 25 resolves it, or reports it ambiguous / not found
        if reference or reference_relations:
            body["version"] = 2
            if reference:
                body["reference"] = {"text": reference}
            if reference_relations:
                body["reference_relations"] = reference_relations

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
        if data.get("version") == 2:                       # P08: where this page sits in the whole result
            result.update({"complete": data.get("complete"), "total": data.get("total"),
                           "next_cursor": data.get("cursor")})
        if data.get("reference"):                          # P16: how the reference event was resolved
            result["reference"] = data["reference"]
        await ctx.info(f"raptor_tkg_query: {len(entities)} entities, {len(facts)} facts, {len(triples)} triples")
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TEMPORAL_CAPABILITIES_DESC.strip())
    async def raptor_temporal_capabilities(ctx: Context = None) -> str:
        try:
            client = await get_client(ctx)
            data = await client.get_json("/temporal/capabilities", tool_name="raptor_temporal_capabilities")
        except Exception as exc:
            await ctx.error(f"raptor_temporal_capabilities failed: {exc}")
            raise
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TEMPORAL_REASON_DESC.strip())
    async def raptor_temporal_reason(
        source_version_id: Annotated[str, "The version_id of the uploaded source."],
        recompute: Annotated[bool, "Queue a recompute instead of reading the stored result."] = False,
        ctx: Context = None,
    ) -> str:
        path = f"/temporal/reason/{quote(source_version_id, safe='')}"
        try:
            client = await get_client(ctx)
            if recompute:
                data = await client.post_json(f"{path}/recompute", {}, tool_name="raptor_temporal_reason")
            else:
                data = await client.get_json(path, tool_name="raptor_temporal_reason")
        except Exception as exc:
            await ctx.error(f"raptor_temporal_reason failed: {exc}")
            raise
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TEMPORAL_INGEST_DESC.strip())
    async def raptor_temporal_ingest(
        content: Annotated[str | None, "Inline source text. Give this, or asset_path + version_id."] = None,
        content_type: Annotated[str, "'text/plain' (default) or 'text/markdown'."] = "text/plain",
        filename: Annotated[str | None, "A name for the inline source."] = None,
        asset_path: Annotated[str | None, "An uploaded asset's asset_path."] = None,
        version_id: Annotated[str | None, "That asset's version_id."] = None,
        idempotency_key: Annotated[str | None, "Repeat a call safely: the same key returns the first result. "
                                               "Defaults to a hash of the input."] = None,
        question: Annotated[str | None, "Ask this about the ingested source only, once it is processed."] = None,
        wait_seconds: Annotated[int, "With question: how long to wait for processing (0-300)."] = 60,
        ctx: Context = None,
    ) -> str:
        body: dict[str, Any] = {}
        if question:
            body.update({"operation": "ingest_and_query", "question": question, "wait_seconds": wait_seconds})
        if content is not None:
            body.update({"content": content, "content_type": content_type,
                         **({"filename": filename} if filename else {})})
        if asset_path or version_id:
            body["asset"] = {"asset_path": asset_path or "", "version_id": version_id or ""}
        # the gateway refuses both / neither; a retry of the same input reuses its key
        keyed = {k: v for k, v in body.items() if k != "wait_seconds"}      # a longer wait is the same request
        body["idempotency_key"] = idempotency_key or "mcp-" + hashlib.sha256(
            json.dumps(keyed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]
        try:
            client = await get_client(ctx)
            data = await client.post_json("/temporal/ingest", body, tool_name="raptor_temporal_ingest")
        except Exception as exc:
            await ctx.error(f"raptor_temporal_ingest failed: {exc}")
            raise
        await ctx.info(f"raptor_temporal_ingest: {data.get('source_version_id')} {data.get('status')}")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TEMPORAL_PARSE_DESC.strip())
    async def raptor_temporal_parse(
        text: Annotated[str, "The question, e.g. 'What happened after 2011?' or '2011年以後發生什麼'."],
        timezone: Annotated[str | None, "IANA zone calendar dates are read in; default: the deployment's."] = None,
        anchor: Annotated[str | None, "ISO 8601 with an offset: the question's own time, for 'last year'."] = None,
        ctx: Context = None,
    ) -> str:
        body = {k: v for k, v in (("text", text), ("timezone", timezone), ("anchor", anchor)) if v}
        try:
            client = await get_client(ctx)
            data = await client.post_json("/temporal/parse", body, tool_name="raptor_temporal_parse")
        except Exception as exc:
            await ctx.error(f"raptor_temporal_parse failed: {exc}")
            raise
        await ctx.info(f"raptor_temporal_parse: {data.get('status')} {data.get('window')}")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_TEMPORAL_STATUS_DESC.strip())
    async def raptor_temporal_status(
        source_version_id: Annotated[str, "The version_id of the uploaded source."],
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_temporal_status: {source_version_id}")
        try:
            client = await get_client(ctx)
            data = await client.get_json(f"/temporal/status/{quote(source_version_id, safe='')}",
                                         tool_name="raptor_temporal_status")
        except Exception as exc:
            await ctx.error(f"raptor_temporal_status failed: {exc}")
            raise
        await ctx.info(f"raptor_temporal_status: {data.get('status')} ({data.get('facts')} facts)")
        return json.dumps(data, ensure_ascii=False, indent=2)

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
