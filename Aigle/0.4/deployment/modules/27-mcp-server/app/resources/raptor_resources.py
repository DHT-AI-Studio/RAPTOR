from __future__ import annotations

import json

import httpx
from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings

CAPABILITIES_TEXT = """\
# Raptor 0.4 — Platform Pipelines

## VideoRAG
Processes video assets end-to-end: transcription (WhisperX), frame description (InternVL), scene segmentation, and OCR. Stores timestamp-aligned chunks in hybrid search (OpenSearch + Qdrant). Enables retrieval of video segments by semantic or keyword query, with presigned playback URLs and precise start/end timestamps.

## DocumentRAG
Ingests PDFs and Office documents via Docling: layout parsing, table extraction, and image captioning. Chunks are embedded and stored in hybrid search. Supports passage-level Q&A, cross-document synthesis, and citation-grounded answers. Best for policy documents, reports, and structured knowledge bases.

## GraphRAG
Extracts entities and relationships from content using LLM-powered NLP and builds a Neo4j knowledge graph. Supports multi-hop traversal queries ("How are X and Y related?"), entity disambiguation, and subgraph extraction. Complements vector search for relational reasoning tasks.

## TKG (Temporal Knowledge Graph)
A time-indexed extension of GraphRAG. Entity states and events are timestamped so queries can filter by date range. Use for event timelines, tracking how entities change over time, and answering "What happened to X between date A and date B?" questions.

## RDBMS
Structured data ingested from relational sources (CSV, SQL exports) is queryable via a natural-language-to-SQL pipeline. Supports aggregations, filters, and joins over tabular data. Best for numeric KPIs, ledgers, and any data with a well-defined schema.
"""


async def _server_token() -> str | None:
    """Obtain a server-level Keycloak token for resource reads."""
    s = get_settings()
    if not (s.keycloak_username and s.keycloak_password):
        return None
    try:
        url = f"{s.keycloak_url}/realms/{s.realm_name}/protocol/openid-connect/token"
        async with httpx.AsyncClient(timeout=s.timeout_resource_list) as c:
            r = await c.post(url, data={
                "grant_type": "password",
                "client_id": s.client_id,
                "username": s.keycloak_username,
                "password": s.keycloak_password,
            })
            r.raise_for_status()
            return r.json()["access_token"]
    except Exception:
        return None


async def _gateway_get(path: str) -> str:
    s = get_settings()
    headers: dict = {}
    token = await _server_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=s.timeout_default) as c:
        r = await c.get(f"{s.api_gateway_url}{path}", headers=headers)
        r.raise_for_status()
    return json.dumps(r.json(), ensure_ascii=False, indent=2)


def register(mcp: FastMCP) -> None:

    @mcp.resource(
        "raptor://capabilities",
        mime_type="text/markdown",
        description="Static Markdown describing Raptor's RAG pipelines. "
                    "Inject into LLM system context to guide tool selection.",
    )
    async def get_capabilities() -> str:
        return CAPABILITIES_TEXT

    @mcp.resource(
        "raptor://assets",
        mime_type="application/json",
        description="JSON list of assets uploaded by the server's service account.",
    )
    async def list_assets() -> str:
        return await _gateway_get("/asset/users/commits")

    @mcp.resource(
        "raptor://assets/{asset_path}/{version_id}",
        mime_type="application/json",
        description="Metadata and presigned download URL for a specific asset version.",
    )
    async def get_asset(asset_path: str, version_id: str) -> str:
        return await _gateway_get(f"/asset/filedownload/{asset_path}/{version_id}")
