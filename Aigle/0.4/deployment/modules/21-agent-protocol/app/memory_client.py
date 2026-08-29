"""
Client for Module 26 (Memory Service) — memory integration for the
Agent Protocol pipeline.

Adapted from 15-chat-service/src/services/memory_client.py. Every call fails
open: any error is logged and swallowed so the RAG pipeline always answers
even when Module 26 is unreachable or slow.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

DEFAULT_SESSION_ID = "default"

# Module 13's memory gateway requires the caller's JWT — Module 21 never has
# the end user's password, only the bearer token that arrived on the
# inbound request. Threaded via ContextVar rather than a function parameter
# because it needs to reach MemoryClient calls made deep inside smolagents
# tool callbacks (RAGSearchTool.forward) that have no Request object to pass
# it through explicitly. asyncio.to_thread() and asyncio.run() both preserve
# the calling thread's context, so a value set once per inbound request stays
# visible through every call this request makes, including the throwaway
# event loop tool-mode spins up.
_auth_token: contextvars.ContextVar[str] = contextvars.ContextVar("auth_token", default="")


def set_auth_token(token: str) -> None:
    """Call once per inbound request, as early as possible, with the raw
    `Authorization` header value (e.g. "Bearer eyJ...")."""
    _auth_token.set(token or "")


def _auth_headers() -> Dict[str, str]:
    token = _auth_token.get()
    return {"Authorization": token} if token else {}

_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A persistent event loop, run on its own daemon thread, that outlives
    any single request's event loop (e.g. the throwaway loop `asyncio.run()`
    creates for smolagents tool calls). Needed because a coroutine scheduled
    with `asyncio.create_task()` on a loop that's about to be torn down by
    `asyncio.run()` never gets to actually run."""
    global _bg_loop
    with _bg_lock:
        if _bg_loop is None:
            _bg_loop = asyncio.new_event_loop()
            threading.Thread(target=_bg_loop.run_forever, name="memory-fire-and-forget", daemon=True).start()
        return _bg_loop


def fire_and_forget(coro) -> None:
    """Schedule `coro` on the persistent background loop and return
    immediately. The scheduled call is still fail-open on its own (see
    MemoryClient docstrings); this only guards against it never running at
    all when the caller's own loop is ephemeral."""
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop())

    def _log_if_failed(f: "asyncio.Future") -> None:
        exc = f.exception()
        if exc:
            logger.warning("fire_and_forget task failed: {}: {}", type(exc).__name__, exc)

    future.add_done_callback(_log_if_failed)


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
        mode: str = "agent",
    ) -> None:
        """POST /api/0.4/memory/archive — per the AC's literal contract.
        session_id is passed as a query param; Module 13's /archive alias
        uses it to pick the upstream session, falling back to the caller's
        own `default` session when omitted."""
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/archive",
                params={"session_id": session_id} if session_id else None,
                json={
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "search_results": search_results or [],
                    "tool_calls": tool_calls or [],
                    "mode": mode,
                },
                headers=_auth_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Module 26 append_turn failed (fail open): {}: {}", type(exc).__name__, exc)

    async def search_longterm(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """GET /api/0.4/memory/retrieve — per the AC's literal contract.
        This is Module 13's global-search alias: broader than long-term-only
        search, it returns {"sessions": [...], "longterm": [...],
        "multimedia": [...]} instead of a flat hit list. Merge the two text-
        bearing scopes (sessions, longterm) into the flat [{frame_type,
        text}, ...] shape callers already expect, sorted by score;
        multimedia hits have no text-context-injectable shape, so they're
        dropped here (search_multimedia below still hits them directly).

        Fail open: any error returns an empty list and the pipeline proceeds
        with no long-term context.
        """
        if not query.strip():
            return []
        try:
            resp = await self.http_client.get(
                f"{self.memory_service_url}/memory/retrieve",
                params={"query": query, "top_k": top_k},
                headers=_auth_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            merged = [
                {"frame_type": h.get("frame_type", "fact"), "text": h.get("text", ""), "score": h.get("score", 0.0)}
                for h in body.get("longterm", [])
            ] + [
                {"frame_type": "session_turn", "text": h.get("text", ""), "score": h.get("score", 0.0)}
                for h in body.get("sessions", [])
            ]
            merged.sort(key=lambda h: h.get("score", 0.0), reverse=True)
            return merged[:top_k]
        except Exception as exc:
            logger.warning("Module 26 search_longterm failed (fail open): {}: {}", type(exc).__name__, exc)
            return []

    async def search_multimedia(
        self,
        user_id: str,
        query: str,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Search the user's multimedia memory (video/audio/image references).

        Fail open: any error, or an empty query, returns an empty list.
        """
        if not query.strip():
            return []
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/multimedia/search",
                json={"query": query, "top_k": top_k},
                headers=_auth_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Module 26 search_multimedia failed (fail open): {}: {}", type(exc).__name__, exc)
            return []

    async def evaluate_compact(
        self,
        user_id: str,
        session_id: str,
        context_window: int,
    ) -> Optional[Dict[str, Any]]:
        """Ask Module 26 whether this session needs compacting (MV-12), without
        writing anything. Module 26 aggregates the session's own archived
        turns plus the user's long-term facts and multimedia snippets.

        Fail open: any error (including timeout) returns None — callers should
        treat that the same as "compaction may be needed" and fall back to
        calling compact_session() directly.

        Uses the bare POST /api/0.4/memory/compact/evaluate (not a
        session-scoped path) with session_id in the body — Module 26 treats
        that the same as the dedicated per-session evaluate endpoint.
        """
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/compact/evaluate",
                json={"session_id": session_id, "context_window": context_window},
                headers=_auth_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Module 26 evaluate_compact failed (fail open): {}: {}", type(exc).__name__, exc)
            return None

    async def compact_session(
        self,
        user_id: str,
        session_id: str,
        context_window: int,
        trigger: str = "auto",
    ) -> Optional[Dict[str, Any]]:
        """POST /api/0.4/memory/compact — per the AC's literal contract.
        session_id is passed as a query param; Module 13's /compact alias
        uses it to pick the upstream session, falling back to the caller's
        own `default` session when omitted.

        Module 26 evaluates its own archived-turn token count against
        context_window and no-ops if under budget.

        Fail open: any error (including timeout) returns None.
        """
        try:
            resp = await self.http_client.post(
                f"{self.memory_service_url}/memory/compact",
                params={"session_id": session_id} if session_id else None,
                json={"trigger": trigger, "context_window": context_window},
                headers=_auth_headers(),
                timeout=self.compact_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Module 26 compact_session failed (fail open): {}: {}", type(exc).__name__, exc)
            return None

    async def get_latest_summary(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recently created summary frame for a session, if any."""
        try:
            resp = await self.http_client.get(
                f"{self.memory_service_url}/memory/sessions/{session_id}/summaries",
                headers=_auth_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            summaries = resp.json()
            if not summaries:
                return None
            return max(summaries, key=lambda s: s.get("created_at", ""))
        except Exception as exc:
            logger.warning("Module 26 get_latest_summary failed (fail open): {}: {}", type(exc).__name__, exc)
            return None


# Keyed by id(event loop): httpx.AsyncClient pins its transport to the loop
# it was created on, and this process mixes the main uvicorn loop with
# throwaway loops from asyncio.run() inside tool-mode workers. A single
# shared singleton gets poisoned once any one of those loops closes.
_clients: Dict[int, MemoryClient] = {}


def get_memory_client() -> MemoryClient:
    """Return (creating if needed) the MemoryClient bound to the currently
    running event loop."""
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is None:
        client = MemoryClient(
            # Routed through Module 13's gateway (JWT-protected), not Module 26
            # directly — see the _auth_token ContextVar docstring above for why.
            memory_service_url=os.environ.get(
                "MEMORY_SERVICE_URL", "http://raptor-api-gateway:8012/api/0.4"
            ),
            http_client=httpx.AsyncClient(),
            timeout=float(os.environ.get("MEMORY_REQUEST_TIMEOUT", "3.0")),
            compact_timeout=float(os.environ.get("COMPACT_REQUEST_TIMEOUT", "90.0")),
        )
        _clients[loop_id] = client
    return client
