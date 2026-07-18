"""
Hybrid search service — single point of contact for module 17.
All three search modes (hybrid / bm25 / vector) go through here,
including optional asset URL resolution delegated to StorageService.
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
        storage_service=None,  # StorageService | None — optional, needed for URL resolution
    ):
        self.hybrid_search_url = hybrid_search_url.rstrip("/")
        self.http_client = http_client
        self.storage_service = storage_service

    # ── Internal: module 17 call ──────────────────────────────────────────────

    async def _search(self, mode: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST to /api/v1/search/{mode}, return raw module 17 response."""
        resp = await self.http_client.post(
            f"{self.hybrid_search_url}/api/v1/search/{mode}",
            json=body,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Public: URL resolution ────────────────────────────────────────────────

    async def resolve_version_urls(
        self,
        version_map: Dict[str, Any],  # version_id → asset_path OR (asset_path, filename)
        user_dict: Dict[str, str],
    ) -> Dict[str, Optional[str]]:
        """Core: resolve presigned URLs for a version_map.
        version_map values may be:
          - str: asset_path only → calls full /filedownload (slow, fetches all associated files)
          - (asset_path, filename): both known → calls /presign-url directly (fast, 1 LakeFS call)
        Returns {version_id: presigned_url}. Failed lookups map to None.
        """
        from app.services.storage_service import User

        if not self.storage_service or not version_map:
            return {}

        user = User(user_id=user_dict["user_id"], branch_id=user_dict["branch_id"])

        async def _fetch(version_id: str, value: Any) -> tuple:
            try:
                if isinstance(value, tuple):
                    asset_path, filename = value
                    if filename:
                        from urllib.parse import quote as _quote
                        key = f"{asset_path}/{filename}"
                        url = await self.storage_service.get_presign_url(key, version_id, user)
                        return version_id, url
                    asset_path = asset_path
                else:
                    asset_path = value

                resp = await self.storage_service.download_asset(
                    asset_path=asset_path,
                    version_id=version_id,
                    user=user,
                )
                return version_id, resp.get("primary_file", {}).get("url")
            except Exception as exc:
                _logger.warning(f"URL resolution failed for {version_id}: {exc}")
                return version_id, None

        pairs = await asyncio.gather(*[
            _fetch(vid, val) for vid, val in version_map.items()
        ])
        return {vid: url for vid, url in pairs}

    async def enrich_with_urls(
        self,
        items: List[dict],
        user_dict: Dict[str, str],
    ) -> None:
        """In-place: add asset_url to each item that has version_id + asset_path.

        Works for any flat dict format (moments, payloads, etc.).
        Deduplicates by version_id — one API call per unique video.
        """
        version_map = {
            item["version_id"]: (item["asset_path"], item.get("filename"))
            for item in items
            if item.get("version_id") and item.get("asset_path")
        }
        url_map = await self.resolve_version_urls(version_map, user_dict)
        for item in items:
            vid = item.get("version_id")
            if vid and url_map.get(vid):
                item["asset_url"] = url_map[vid]

    async def _resolve_asset_urls(
        self,
        search_results: List[dict],
        user_dict: Dict[str, str],
    ) -> List[dict]:
        """Enrich chat search_results with presigned asset URLs.

        search_results is a list of groups: [{items: [{payload: {...}}, ...]}, ...]
        Payloads are modified in-place; the same list is returned.
        """
        all_payloads = [
            hit["payload"]
            for group in search_results
            for hit in group.get("items", [])
            if hit.get("payload")
        ]
        await self.enrich_with_urls(all_payloads, user_dict)
        return search_results

    # ── Public: search with optional URL resolution ───────────────────────────

    async def hybrid_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
        resolve_urls: bool = True,
    ) -> Dict[str, Any]:
        if user and user.get("branch_id"):
            body = {**body, "branch_id": user["branch_id"]}
        data = await self._search("hybrid", body)
        if user and resolve_urls:
            payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
            await self.enrich_with_urls(payloads, user)
        return data

    async def bm25_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
        resolve_urls: bool = True,
    ) -> Dict[str, Any]:
        if user and user.get("branch_id"):
            body = {**body, "branch_id": user["branch_id"]}
        data = await self._search("bm25", body)
        if user and resolve_urls:
            payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
            await self.enrich_with_urls(payloads, user)
        return data

    async def vector_search(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, str]] = None,
        resolve_urls: bool = True,
    ) -> Dict[str, Any]:
        if user and user.get("branch_id"):
            body = {**body, "branch_id": user["branch_id"]}
        data = await self._search("vector", body)
        if user and resolve_urls:
            payloads = [hit["payload"] for hit in data.get("results", []) if hit.get("payload")]
            await self.enrich_with_urls(payloads, user)
        return data
