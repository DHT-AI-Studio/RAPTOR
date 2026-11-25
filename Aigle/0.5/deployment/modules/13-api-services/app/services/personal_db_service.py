"""Authenticated proxy to the Personal DB Service (module 25) — VIE01-191.

Module 25 is internal-only and trusts whatever `X-User-ID` it is given, so this
layer is where the identity is actually established: the header is always taken
from the verified JWT's `sub`, never from anything the client supplied. A caller
cannot reach another user's database by putting someone else's id in a body or
path — the router rejects that with 403 before this service is reached, and even
if it did not, the header sent downstream would still be the caller's own.

Every user's database is created on demand: `ensure_database()` fires
`POST /internal/db/init` once per user per gateway process, so a first-time user
searching before they have uploaded anything gets an empty result rather than a
404 about a database they were never told about.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings

_logger = logging.getLogger(__name__)

# Users this *process* has already provisioned. Module-level, not per-instance:
# the service is constructed fresh on every request via Depends, so an instance
# attribute would be empty every time and re-init on each call. Only an
# optimisation — init is idempotent, so a cold process or a second replica
# simply calls it again.
_PROVISIONED: set[str] = set()


class PersonalDBService:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.base_url = settings.personal_db_url.rstrip("/")

    def _headers(self, user_id: str) -> dict[str, str]:
        """Identity for module 25. `X-Branch-ID` repeats `sub` because the
        per-user database *is* the branch in the 0.4 personal-DB model."""
        return {"X-User-ID": user_id, "X-Branch-ID": user_id}

    async def _request(
        self,
        method: str,
        path: str,
        user_id: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = await self.client.request(
                method, url, headers=self._headers(user_id), json=json, params=params, timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            _logger.error(
                "personal-db %s %s failed: %s %s",
                method, path, exc.response.status_code, exc.response.text,
            )
            # Pass the downstream status through rather than flattening to 500 —
            # a 503 from the deletion-audit guard means "retry later", which the
            # client can only act on if it survives the proxy.
            # A structured error (P09: {"error": "UNSUPPORTED_FEATURES", ...}) reaches the caller as such;
            # anything else keeps the prefixed text it always had.
            try:
                structured = exc.response.json().get("detail")
            except (ValueError, AttributeError):
                structured = None
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=structured if isinstance(structured, dict)
                else f"Personal DB service error: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            _logger.error("personal-db %s %s unreachable: %s", method, path, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Personal DB service unreachable",
            )

    async def ensure_database(self, user_id: str) -> None:
        """Provision the caller's database on their first personal-db request.

        Never raises: a provisioning failure should surface as whatever the
        actual request then returns, not as an error about a call the client
        never made. Module 25's consumer also auto-creates on first upload, so
        this is one of two independent paths to the same idempotent result.
        """
        if user_id in _PROVISIONED:
            return
        try:
            await self._request("POST", "/internal/db/init", user_id)
            _PROVISIONED.add(user_id)
        except Exception as exc:  # noqa: BLE001 -- contract is "never raises"; catches the HTTPException _request() raises plus malformed-JSON ValueError, logged
            _logger.warning("personal-db init failed for %s: %s", user_id, exc)

    # ── search ────────────────────────────────────────────────────────────────

    async def hybrid_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/personal/search/hybrid", user_id, json=body)

    async def bm25_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/personal/search/bm25", user_id, json=body)

    async def vector_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/personal/search/vector", user_id, json=body)

    async def graph_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/personal/search/graph", user_id, json=body)

    async def temporal_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # Module 25 calls this one `/tkg` (temporal knowledge graph); the public
        # route says `temporal` because that is what it does.
        return await self._request("POST", "/personal/search/tkg", user_id, json=body)

    async def temporal_capabilities(self, user_id: str) -> dict[str, Any]:
        return await self._request("GET", "/personal/temporal/capabilities", user_id)

    async def temporal_reason(self, user_id: str, version_id: str, recompute: bool = False) -> dict[str, Any]:
        path = f"/personal/temporal/reason/{quote(version_id, safe='')}"
        if recompute:
            return await self._request("POST", f"{path}/recompute", user_id)
        return await self._request("GET", path, user_id)

    async def temporal_status(self, user_id: str, version_id: str) -> dict[str, Any]:
        # P12: the durable temporal job of one of the caller's sources. Module 25 scopes it to the
        # caller's own database, so another user's version id is simply not found (404).
        return await self._request("GET", f"/personal/temporal/status/{quote(version_id, safe='')}", user_id)

    async def temporal_expiry(self, user_id: str, version_id: str) -> dict[str, Any]:
        # P21: when an inline source expires; 404 for an upload (no expiry registered)
        return await self._request("GET", f"/personal/temporal/expiry/{quote(version_id, safe='')}", user_id)

    async def temporal_verify(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # P17: an answer's temporal claims checked against the caller's own facts
        return await self._request("POST", "/personal/temporal/verify", user_id, json=body)

    async def temporal_parse(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # TMP-7: the time a question asks about; reads no database
        return await self._request("POST", "/personal/temporal/parse", user_id, json=body)

    async def set_temporal_expiry(self, user_id: str, version_id: str, expires_at: str) -> dict[str, Any]:
        return await self._request("PUT", f"/personal/temporal/expiry/{quote(version_id, safe='')}", user_id,
                                   json={"expires_at": expires_at})

    async def graphrag_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # Batch 6 of the graph/TKG/GraphRAG parity plan -- module 25 has had
        # this endpoint since Batch 4, this gateway route just never existed.
        return await self._request("POST", "/personal/search/graphrag", user_id, json=body)

    async def rerank(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # Stateless (no real per-user database access), but routed through
        # the same _request()/header plumbing as everything else here for
        # consistency -- module 25's /personal/search/rerank ignores the
        # X-Branch-ID it doesn't need.
        return await self._request("POST", "/personal/search/rerank", user_id, json=body)

    async def people_search(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/personal/search/people", user_id, json=body)

    async def people_in_videos(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/personal/search/people/in-videos", user_id, json=body)

    # ── identity (dev plan v1.4 §4.5) ─────────────────────────────────────────

    async def identity(
        self,
        method: str,
        path: str,
        user_id: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """`path` is relative to Module 25's /personal/identity."""
        return await self._request(method, f"/personal/identity{path}", user_id, json=json, params=params)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def status(self, user_id: str) -> dict[str, Any]:
        return await self._request("GET", "/internal/db/status", user_id)

    async def delete_database(self, user_id: str) -> dict[str, Any]:
        _PROVISIONED.discard(user_id)           # next request re-provisions
        return await self._request("DELETE", "/internal/db", user_id)
