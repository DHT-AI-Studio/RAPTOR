"""
Hybrid search service — proxy to module 25 (Personal DB Service).

Was a proxy to module 17 (global hybrid search, branch_id-filtered);
switched to module 25 (per-user isolated ArcadeDB, X-Branch-ID-routed) --
see the commented-out block at the bottom of this class for the old
module 17 implementation, kept for rollback rather than deleted.
Copied from module 12; storage_service is always None here (no presigned
URL resolution).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

import httpx

_logger = logging.getLogger(__name__)

# Users this *process* has already provisioned on module 25. Module-level,
# not per-instance, same reasoning as module 13's personal_db_service.py:
# init is idempotent, so a cold process or a second replica just calls it
# again.
_PROVISIONED: Set[str] = set()


class HybridSearchService:
    """Proxy to module 25 personal-db search API (BM25 + Vector + Hybrid,
    reranked inside module 25 itself for /hybrid)."""

    def __init__(
        self,
        hybrid_search_url: str,
        http_client: httpx.AsyncClient,
        storage_service=None,
    ):
        self.hybrid_search_url = hybrid_search_url.rstrip("/")
        self.http_client = http_client
        self.storage_service = storage_service

    async def _ensure_database(self, branch_id: str) -> None:
        """Provision the caller's module-25 database on first use. Never
        raises: a provisioning failure should surface as whatever the
        actual search call then returns, not as an error about a call the
        caller never made."""
        if branch_id in _PROVISIONED:
            return
        try:
            resp = await self.http_client.post(
                f"{self.hybrid_search_url}/internal/db/init",
                headers={"X-Branch-ID": branch_id},
                timeout=10.0,
            )
            resp.raise_for_status()
            _PROVISIONED.add(branch_id)
        except Exception as exc:
            _logger.warning("personal-db init failed for %s: %s", branch_id, exc)

    async def _search(self, mode: str, body: Dict[str, Any], branch_id: str) -> Dict[str, Any]:
        await self._ensure_database(branch_id)
        resp = await self.http_client.post(
            f"{self.hybrid_search_url}/personal/search/{mode}",
            json=body,
            headers={"X-Branch-ID": branch_id},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()

    async def hybrid_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        branch_id = (user or {}).get("branch_id", "")
        body = {k: v for k, v in body.items() if k != "branch_id"}
        return await self._search("hybrid", body, branch_id)

    async def bm25_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        branch_id = (user or {}).get("branch_id", "")
        body = {k: v for k, v in body.items() if k != "branch_id"}
        return await self._search("bm25", body, branch_id)

    async def vector_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        branch_id = (user or {}).get("branch_id", "")
        body = {k: v for k, v in body.items() if k != "branch_id"}
        return await self._search("vector", body, branch_id)


# ---------------------------------------------------------------------------
# Old module 17 implementation -- commented out, not deleted, for rollback.
# ---------------------------------------------------------------------------
# class HybridSearchService:
#     """Proxy to module 17 hybrid search API (BM25 + Vector + Rerank)."""
#
#     def __init__(
#         self,
#         hybrid_search_url: str,
#         http_client: httpx.AsyncClient,
#         storage_service=None,
#     ):
#         self.hybrid_search_url = hybrid_search_url.rstrip("/")
#         self.http_client = http_client
#         self.storage_service = storage_service
#
#     async def _search(self, mode: str, body: Dict[str, Any]) -> Dict[str, Any]:
#         resp = await self.http_client.post(
#             f"{self.hybrid_search_url}/api/v1/search/{mode}",
#             json=body,
#             timeout=60.0,
#         )
#         resp.raise_for_status()
#         return resp.json()
#
#     async def hybrid_search(
#         self,
#         body: Dict[str, Any],
#         user: Optional[Dict[str, str]] = None,
#     ) -> Dict[str, Any]:
#         if user and user.get("branch_id"):
#             body = {**body, "branch_id": user["branch_id"]}
#         return await self._search("hybrid", body)
#
#     async def bm25_search(
#         self,
#         body: Dict[str, Any],
#         user: Optional[Dict[str, str]] = None,
#     ) -> Dict[str, Any]:
#         if user and user.get("branch_id"):
#             body = {**body, "branch_id": user["branch_id"]}
#         return await self._search("bm25", body)
#
#     async def vector_search(
#         self,
#         body: Dict[str, Any],
#         user: Optional[Dict[str, str]] = None,
#     ) -> Dict[str, Any]:
#         if user and user.get("branch_id"):
#             body = {**body, "branch_id": user["branch_id"]}
#         return await self._search("vector", body)
