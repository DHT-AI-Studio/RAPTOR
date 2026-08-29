from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.tools import get_client

logger = logging.getLogger(__name__)


def _shape_search_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    payload = hit.get("payload") or {}
    promoted = {"text", "asset_path", "start_time", "end_time"}
    return {
        "id":         hit.get("id", ""),
        "content":    payload.get("text", ""),
        "score":      hit.get("score", 0.0),
        "metadata":   {k: v for k, v in payload.items() if k not in promoted},
        "asset_path": payload.get("asset_path", ""),
        "start_time": payload.get("start_time"),
        "end_time":   payload.get("end_time"),
    }


def _shape_video_moments(
    video_results: List[Dict[str, Any]],
    asset_path_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filter_stem = (
        asset_path_filter.rstrip("/").rsplit("/", 1)[-1].lower()
        if asset_path_filter else None
    )
    moments: List[Dict[str, Any]] = []
    for video in video_results:
        filename = video.get("filename") or ""
        if filter_stem and filter_stem not in os.path.splitext(filename)[0].lower():
            continue
        for seg in video.get("segments") or []:
            moments.append({
                "clip_url":   video.get("asset_url", ""),
                "start_time": seg.get("start_time"),
                "end_time":   seg.get("end_time"),
                "filename":   filename,
                "video_id":   video.get("video_id", ""),
                "score":      seg.get("score", 0.0),
                "text":       seg.get("text", ""),
                "sources":    seg.get("sources") or [],
            })
    moments.sort(key=lambda m: m["score"], reverse=True)
    return moments


_SEARCH_DESC = """\
Hybrid semantic + keyword search across Raptor's multimodal knowledge base.

Runs BM25 (OpenSearch) and vector (Qdrant) in parallel, merges via RRF,
re-ranks with a cross-encoder. Supports video, audio, document, and image assets.

Each result: score, content (text chunk), asset_path, start_time, end_time, metadata.
"""

_BM25_SEARCH_DESC = """\
Keyword-only BM25 search using OpenSearch. Best for exact keyword matching.

Score is a BM25 relevance score (higher = more relevant).
Use raptor_search (hybrid) for general queries; use this when exact keyword precision matters.
"""

_VECTOR_SEARCH_DESC = """\
Semantic vector search using Qdrant. Best for conceptual or paraphrase queries.

Score is cosine similarity (0–1). Does not require exact keyword matches.
Use raptor_search (hybrid) for general queries; use this when semantic meaning matters most.
"""

_VIDEO_SEARCH_DESC = """\
Video-specific multi-retriever search (BM25 + vector + GraphRAG + TKG → rerank).

Returns results grouped by video with timestamped segments and presigned download URLs.
Prefer over raptor_search when the user asks about video clips or needs timestamps.
"""

# Personal-db (module 25) variants -- were added alongside the module
# 17-backed tools above without touching them (13/27 were held back from the
# modules 15/21/22 migration). Now commented out, not deleted: 13's
# `/personal-db/search/*` routes these called are themselves commented out
# (see personal_db.py) since `/search/*` above now IS module 25 under the
# old 0.3-style naming -- raptor_search/raptor_search_bm25/
# raptor_search_vector/raptor_video_search need no changes, they already
# call `/search/*`.
# _PERSONAL_DB_SEARCH_DESC = """\
# Hybrid semantic + keyword search over the caller's own personal database (module 25).

# Runs dense vector + BM25 in parallel inside module 25 itself, fused and reranked
# there before this tool ever sees the results. Scoped to this user's own uploaded
# content only (physically isolated per-user database), unlike raptor_search which
# queries the shared global index. Supports video, audio, document, and image assets.

# Each result: score, content (text chunk), asset_path, start_time, end_time, metadata.
# """

# _PERSONAL_DB_BM25_SEARCH_DESC = """\
# Keyword-only BM25 search over the caller's own personal database (module 25).

# Score is a BM25 relevance score (higher = more relevant), scoped to this user's
# own uploaded content only (physically isolated per-user database).
# Use raptor_personal_db_search (hybrid) for general queries; use this when exact
# keyword precision matters.
# """

# _PERSONAL_DB_VECTOR_SEARCH_DESC = """\
# Semantic vector search over the caller's own personal database (module 25).

# Score is cosine similarity (0–1), scoped to this user's own uploaded content
# only (physically isolated per-user database). Does not require exact keyword matches.
# Use raptor_personal_db_search (hybrid) for general queries; use this when semantic
# meaning matters most.
# """

# _PERSONAL_DB_VIDEO_SEARCH_DESC = """\
# Video-specific multi-retriever search over the caller's own personal database (module 25).

# Same 4-way retrieval + rerank as raptor_video_search, scoped to this user's own
# uploaded content only (physically isolated per-user database, not the shared
# global index). Returns results grouped by video with timestamped segments and
# presigned download URLs.
# """


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_SEARCH_DESC.strip())
    async def raptor_search(
        query: Annotated[str, "Search query text"],
        top_k: Annotated[int, "Maximum results to return (1–50)"] = 10,
        type: Annotated[Optional[str], "Media type filter: 'videos', 'audios', 'documents', 'images'"] = None,
        speaker: Annotated[Optional[str], "Filter by speaker name"] = None,
        source: Annotated[Optional[str], "Filter by file format, e.g. 'mp4', 'pdf'"] = None,
        embedding_type: Annotated[Optional[str], "Embedding type: 'text' or 'summary'"] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_search: query={query!r} top_k={top_k} type={type}")

        body: dict = {"query": query, "top_k": top_k}
        if type:
            body["type"] = type
        if speaker:
            body["speaker"] = [speaker]
        if source:
            body["source"] = source
        if embedding_type:
            body["embedding_type"] = embedding_type

        try:
            client = await get_client(ctx)
            data = await client.post_json("/search/hybrid", body, tool_name="raptor_search")
        except Exception as exc:
            await ctx.error(f"raptor_search failed: {exc}")
            raise

        shaped = [_shape_search_hit(h) for h in data.get("results", [])]
        await ctx.info(f"raptor_search: {len(shaped)} results")
        return json.dumps(shaped, ensure_ascii=False, indent=2)

    @mcp.tool(description=_BM25_SEARCH_DESC.strip())
    async def raptor_search_bm25(
        query: Annotated[str, "Keyword search query"],
        top_k: Annotated[int, "Maximum results to return (1–50)"] = 10,
        type: Annotated[Optional[str], "Media type filter: 'videos', 'audios', 'documents', 'images'"] = None,
        speaker: Annotated[Optional[str], "Filter by speaker name"] = None,
        source: Annotated[Optional[str], "Filter by file format, e.g. 'mp4', 'pdf'"] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_search_bm25: query={query!r} top_k={top_k} type={type}")

        body: dict = {"query": query, "top_k": top_k}
        if type:
            body["type"] = type
        if speaker:
            body["speaker"] = [speaker]
        if source:
            body["source"] = source

        try:
            client = await get_client(ctx)
            data = await client.post_json("/search/bm25", body, tool_name="raptor_search_bm25")
        except Exception as exc:
            await ctx.error(f"raptor_search_bm25 failed: {exc}")
            raise

        shaped = [_shape_search_hit(h) for h in data.get("results", [])]
        await ctx.info(f"raptor_search_bm25: {len(shaped)} results")
        return json.dumps(shaped, ensure_ascii=False, indent=2)

    @mcp.tool(description=_VECTOR_SEARCH_DESC.strip())
    async def raptor_search_vector(
        query: Annotated[str, "Natural language semantic query"],
        top_k: Annotated[int, "Maximum results to return (1–50)"] = 10,
        type: Annotated[Optional[str], "Media type filter: 'videos', 'audios', 'documents', 'images'"] = None,
        speaker: Annotated[Optional[str], "Filter by speaker name"] = None,
        source: Annotated[Optional[str], "Filter by file format, e.g. 'mp4', 'pdf'"] = None,
        embedding_type: Annotated[Optional[str], "Embedding type: 'text' or 'summary'"] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_search_vector: query={query!r} top_k={top_k} type={type}")

        body: dict = {"query": query, "top_k": top_k}
        if type:
            body["type"] = type
        if speaker:
            body["speaker"] = [speaker]
        if source:
            body["source"] = source
        if embedding_type:
            body["embedding_type"] = embedding_type

        try:
            client = await get_client(ctx)
            data = await client.post_json("/search/vector", body, tool_name="raptor_search_vector")
        except Exception as exc:
            await ctx.error(f"raptor_search_vector failed: {exc}")
            raise

        shaped = [_shape_search_hit(h) for h in data.get("results", [])]
        await ctx.info(f"raptor_search_vector: {len(shaped)} results")
        return json.dumps(shaped, ensure_ascii=False, indent=2)

    @mcp.tool(description=_VIDEO_SEARCH_DESC.strip())
    async def raptor_video_search(
        query: Annotated[str, "Natural language description of video content to find"],
        top_k: Annotated[int, "Maximum video files to return (1–50)"] = 5,
        asset_path: Annotated[Optional[str], "LakeFS path to restrict to a single video"] = None,
        candidate_multiplier: Annotated[int, "Fan-out factor per retriever (1–20)"] = 5,
        score_threshold: Annotated[float, "Minimum segment score (0.0–1.0)"] = 0.52,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_video_search: query={query!r} top_k={top_k}")

        body = {
            "query": query,
            "top_k": top_k,
            "candidate_multiplier": candidate_multiplier,
            "score_threshold": score_threshold,
        }

        try:
            client = await get_client(ctx)
            data = await client.post_json("/search/video_search", body, tool_name="raptor_video_search")
        except Exception as exc:
            await ctx.error(f"raptor_video_search failed: {exc}")
            raise

        moments = _shape_video_moments(data.get("results", []), asset_path_filter=asset_path)
        await ctx.info(f"raptor_video_search: {data.get('total', 0)} video(s) → {len(moments)} moment(s)")
        return json.dumps(moments, ensure_ascii=False, indent=2)

#     @mcp.tool(description=_PERSONAL_DB_SEARCH_DESC.strip())
#     async def raptor_personal_db_search(
#         query: Annotated[str, "Search query text"],
#         top_k: Annotated[int, "Maximum results to return (1–50)"] = 10,
#         type: Annotated[Optional[str], "Media type filter: 'videos', 'audios', 'documents', 'images'"] = None,
#         speaker: Annotated[Optional[str], "Filter by speaker name"] = None,
#         source: Annotated[Optional[str], "Filter by file format, e.g. 'mp4', 'pdf'"] = None,
#         embedding_type: Annotated[Optional[str], "Embedding type: 'text' or 'summary'"] = None,
#         ctx: Context = None,
#     ) -> str:
#         await ctx.info(f"raptor_personal_db_search: query={query!r} top_k={top_k} type={type}")

#         body: dict = {"query": query, "top_k": top_k}
#         if type:
#             body["type"] = type
#         if speaker:
#             body["speaker"] = [speaker]
#         if source:
#             body["source"] = source
#         if embedding_type:
#             body["embedding_type"] = embedding_type

#         try:
#             client = await get_client(ctx)
#             data = await client.post_json(
#                 "/personal-db/search/hybrid", body, tool_name="raptor_personal_db_search")
#         except Exception as exc:
#             await ctx.error(f"raptor_personal_db_search failed: {exc}")
#             return json.dumps({"error": str(exc)}, ensure_ascii=False)

#         shaped = [_shape_search_hit(h) for h in data.get("results", [])]
#         await ctx.info(f"raptor_personal_db_search: {len(shaped)} results")
#         return json.dumps(shaped, ensure_ascii=False, indent=2)

#     @mcp.tool(description=_PERSONAL_DB_BM25_SEARCH_DESC.strip())
#     async def raptor_personal_db_search_bm25(
#         query: Annotated[str, "Keyword search query"],
#         top_k: Annotated[int, "Maximum results to return (1–50)"] = 10,
#         type: Annotated[Optional[str], "Media type filter: 'videos', 'audios', 'documents', 'images'"] = None,
#         speaker: Annotated[Optional[str], "Filter by speaker name"] = None,
#         source: Annotated[Optional[str], "Filter by file format, e.g. 'mp4', 'pdf'"] = None,
#         ctx: Context = None,
#     ) -> str:
#         await ctx.info(f"raptor_personal_db_search_bm25: query={query!r} top_k={top_k} type={type}")

#         body: dict = {"query": query, "top_k": top_k}
#         if type:
#             body["type"] = type
#         if speaker:
#             body["speaker"] = [speaker]
#         if source:
#             body["source"] = source

#         try:
#             client = await get_client(ctx)
#             data = await client.post_json(
#                 "/personal-db/search/bm25", body, tool_name="raptor_personal_db_search_bm25")
#         except Exception as exc:
#             await ctx.error(f"raptor_personal_db_search_bm25 failed: {exc}")
#             return json.dumps({"error": str(exc)}, ensure_ascii=False)

#         shaped = [_shape_search_hit(h) for h in data.get("results", [])]
#         await ctx.info(f"raptor_personal_db_search_bm25: {len(shaped)} results")
#         return json.dumps(shaped, ensure_ascii=False, indent=2)

#     @mcp.tool(description=_PERSONAL_DB_VECTOR_SEARCH_DESC.strip())
#     async def raptor_personal_db_search_vector(
#         query: Annotated[str, "Natural language semantic query"],
#         top_k: Annotated[int, "Maximum results to return (1–50)"] = 10,
#         type: Annotated[Optional[str], "Media type filter: 'videos', 'audios', 'documents', 'images'"] = None,
#         speaker: Annotated[Optional[str], "Filter by speaker name"] = None,
#         source: Annotated[Optional[str], "Filter by file format, e.g. 'mp4', 'pdf'"] = None,
#         embedding_type: Annotated[Optional[str], "Embedding type: 'text' or 'summary'"] = None,
#         ctx: Context = None,
#     ) -> str:
#         await ctx.info(f"raptor_personal_db_search_vector: query={query!r} top_k={top_k} type={type}")

#         body: dict = {"query": query, "top_k": top_k}
#         if type:
#             body["type"] = type
#         if speaker:
#             body["speaker"] = [speaker]
#         if source:
#             body["source"] = source
#         if embedding_type:
#             body["embedding_type"] = embedding_type

#         try:
#             client = await get_client(ctx)
#             data = await client.post_json(
#                 "/personal-db/search/vector", body, tool_name="raptor_personal_db_search_vector")
#         except Exception as exc:
#             await ctx.error(f"raptor_personal_db_search_vector failed: {exc}")
#             return json.dumps({"error": str(exc)}, ensure_ascii=False)

#         shaped = [_shape_search_hit(h) for h in data.get("results", [])]
#         await ctx.info(f"raptor_personal_db_search_vector: {len(shaped)} results")
#         return json.dumps(shaped, ensure_ascii=False, indent=2)

#     @mcp.tool(description=_PERSONAL_DB_VIDEO_SEARCH_DESC.strip())
#     async def raptor_personal_db_video_search(
#         query: Annotated[str, "Natural language description of video content to find"],
#         top_k: Annotated[int, "Maximum video files to return (1–50)"] = 5,
#         asset_path: Annotated[Optional[str], "LakeFS path to restrict to a single video"] = None,
#         candidate_multiplier: Annotated[int, "Fan-out factor per retriever (1–20)"] = 5,
#         score_threshold: Annotated[float, "Minimum segment score (0.0–1.0)"] = 0.52,
#         ctx: Context = None,
#     ) -> str:
#         await ctx.info(f"raptor_personal_db_video_search: query={query!r} top_k={top_k}")

#         body = {
#             "query": query,
#             "top_k": top_k,
#             "candidate_multiplier": candidate_multiplier,
#             "score_threshold": score_threshold,
#         }

#         try:
#             client = await get_client(ctx)
#             data = await client.post_json(
#                 "/personal-db/search/video_search", body, tool_name="raptor_personal_db_video_search")
#         except Exception as exc:
#             await ctx.error(f"raptor_personal_db_video_search failed: {exc}")
#             return json.dumps({"error": str(exc)}, ensure_ascii=False)

#         moments = _shape_video_moments(data.get("results", []), asset_path_filter=asset_path)
#         await ctx.info(
#             f"raptor_personal_db_video_search: {data.get('total', 0)} video(s) → {len(moments)} moment(s)")
#         return json.dumps(moments, ensure_ascii=False, indent=2)
