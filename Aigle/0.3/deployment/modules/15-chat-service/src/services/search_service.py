"""
Hybrid search service — proxy to module 17.
Copied from module 12; storage_service is always None here (no presigned URL resolution).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

_logger = logging.getLogger(__name__)


class HybridSearchService:
    """Proxy to module 17 hybrid search API (BM25 + Vector + Rerank)."""

    def __init__(
        self,
        hybrid_search_url: str,
        http_client: httpx.AsyncClient,
        storage_service=None,
    ):
        self.hybrid_search_url = hybrid_search_url.rstrip("/")
        self.http_client = http_client
        self.storage_service = storage_service

    async def _search(self, mode: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self.http_client.post(
            f"{self.hybrid_search_url}/api/v1/search/{mode}",
            json=body,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()

    async def hybrid_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if user and user.get("branch_id"):
            body = {**body, "branch_id": user["branch_id"]}
        return await self._search("hybrid", body)

    async def bm25_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if user and user.get("branch_id"):
            body = {**body, "branch_id": user["branch_id"]}
        return await self._search("bm25", body)

    async def vector_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if user and user.get("branch_id"):
            body = {**body, "branch_id": user["branch_id"]}
        return await self._search("vector", body)
