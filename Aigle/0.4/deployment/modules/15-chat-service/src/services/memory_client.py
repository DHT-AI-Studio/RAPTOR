"""
Client for Module 26 (Memory Service) — archives conversation turns.

Fire-and-forget only: any failure is logged and swallowed so chat always
succeeds even when Module 26 is unreachable or slow.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

_logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default"


class MemoryClient:
    """Thin proxy to Module 26 (Memory Service)."""

    def __init__(
        self,
        memory_service_url: str,
        http_client: httpx.AsyncClient,
        timeout: float = 3.0,
        compact_timeout: Optional[float] = None,
    ):
        self.memory_service_url = memory_service_url.rstrip("/")
        self.http_client = http_client
        self.timeout = timeout
        self.compact_timeout = compact_timeout or timeout

    async def append_turn(
        self,
        user_id: str,
        session_id: Optional[str],
        user_message: str,
        assistant_response: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        sid = session_id or DEFAULT_SESSION_ID
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/sessions/{sid}/turns",
                json={
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "search_results": search_results or [],
                    "tool_calls": tool_calls or [],
                },
                headers={"X-User-ID": user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            _logger.warning("Module 26 append_turn failed (fail open): %s: %s", type(exc).__name__, exc)

    async def search_longterm(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search cross-session long-term facts/preferences/entities for this user.

        Fail open: any error (timeout, Module 26 down, bad response) is logged
        and swallowed — the caller gets an empty list and chat proceeds with
        no long-term context rather than blocking on a non-critical read.
        """
        if not query.strip():
            return []
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/longterm/search",
                json={"query": query, "top_k": top_k},
                headers={"X-User-ID": user_id},
                timeout=timeout if timeout is not None else self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            _logger.warning("Module 26 search_longterm failed (fail open): %s: %s", type(exc).__name__, exc)
            return []

    async def search_session(
        self,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search this session's archived MemVID history for turns relevant
        to the current query, beyond what's cached in Redis.

        Fail open: any error, or an empty query, returns an empty list.
        """
        if not query.strip():
            return []
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/sessions/{session_id}/search",
                json={"query": query, "top_k": top_k},
                headers={"X-User-ID": user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("hits", [])
        except Exception as exc:
            _logger.warning("Module 26 search_session failed (fail open): %s: %s", type(exc).__name__, exc)
            return []

    async def compact_session(
        self,
        user_id: str,
        session_id: str,
        context_window: int,
        trigger: str = "auto",
    ) -> Optional[Dict[str, Any]]:
        """Ask Module 26 to compact this session (MV-12). Module 26 evaluates its own
        archived-turn token count against context_window and no-ops if under budget.

        Fail open: any error (including timeout) returns None.
        """
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/sessions/{session_id}/compact",
                json={"trigger": trigger, "context_window": context_window},
                headers={"X-User-ID": user_id},
                timeout=self.compact_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            _logger.warning("Module 26 compact_session failed (fail open): %s: %s", type(exc).__name__, exc)
            return None

    async def get_latest_summary(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recently created summary frame for a session, if any."""
        try:
            resp = await self.http_client.get(
                f"{self.memory_service_url}/memory/sessions/{session_id}/summaries",
                headers={"X-User-ID": user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            summaries = resp.json()
            if not summaries:
                return None
            return max(summaries, key=lambda s: s.get("created_at", ""))
        except Exception as exc:
            _logger.warning("Module 26 get_latest_summary failed (fail open): %s: %s", type(exc).__name__, exc)
            return None
