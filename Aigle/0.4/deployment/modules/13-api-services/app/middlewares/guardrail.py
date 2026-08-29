"""GuardrailMiddleware — gateway-level content-policy intercept (V04-10, Module 13).

Wraps three exact (canonical path, method) pairs — `POST /api/{v}/chat/completions`,
`POST /api/{v}/a2a/query`, and `POST /api/{v}/chat/chat` (the RAG chat pipeline,
module 15) — with a before/after check against the Guardrail Service (module 23):

    POST {GUARDRAIL_URL}/guard/check/input   (before the request reaches the LLM)
    POST {GUARDRAIL_URL}/guard/check/output  (before the response reaches the caller)

Verified against module 23's `app/routers/check_guard.py` (mounted with prefix `/guard`
in its `app/main.py`) and its README's "原始 guard-model 分類" section. This is the raw
guard-model classification group (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard,
each with its own fixed guard prompt) — role is fixed by the endpoint itself (`/input`
→ user, `/output` → assistant), so the request body is just `{"content": "..."}`.
Unlike `/guardrail/check/*` (the policy-engine group module 07's `guardrail_hook.py`
calls), this group does not go through any policy and does not write to
`guardrail_violations` — there is no policy-scoped audit trail for these calls, only
whatever this middleware itself logs.

Design:

* `GR_ENABLED=false` (default) — every request takes a single boolean branch straight
  to `call_next`; no Guardrail Service call, no body buffering, zero added latency.
* A request whose (path, method) isn't one of the three intercepted routes also passes
  straight through untouched — the middleware only buffers request/response bodies for
  the routes it actually checks.
* Any Guardrail Service failure (network error, timeout, non-2xx, malformed JSON) is
  fail-open: log a warning, treat as `safe: true`, let the request continue. A
  Guardrail Service outage must never take chat/completions, a2a/query, or the RAG
  chat pipeline down.
* This middleware must be added to the FastAPI app *after* `LegacyApiAliasMiddleware`
  (i.e. added last, so it ends up innermost) — that middleware rewrites
  `/api/0.3/{chat,a2a}/...` to the canonical `/api/{v}/...` path before anything inside
  it runs, so this middleware only ever needs to match the canonical path once and
  legacy callers are covered for free.
* `/api/{v}/a2a/query` is handled entirely through `call_next` — `agent_protocol.py`
  is not touched. `/api/{v}/chat/completions` and `/api/{v}/chat/chat` are real,
  separately-authenticated FastAPI routes (see `app/routers/chat.py`); this middleware
  does not proxy either itself, it only wraps the existing routes with `call_next` the
  same way it wraps a2a/query. `/chat/completions` forwards straight to module 07;
  `/chat/chat` forwards to module 15's RAG pipeline (`settings.chat_service_url`),
  which internally calls module 07's `/v1/chat/completions` itself, possibly more than
  once across a multi-turn LangGraph tool-calling loop — this middleware only sees the
  single incoming user message and the single final answer at this route's boundary,
  not each individual internal hop.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger(__name__)

FieldExtractor = Callable[[Dict[str, Any]], str]


def _text_of(content: Any) -> str:
    """Flatten OpenAI-style message content — a plain string, or a list of
    `{"type": "text", "text": ...}` parts (the multimodal/vision-client format).
    Mirrors module 07's `src/api/openai_api.py::_plain_message()`; without this,
    any client that sends content as parts (common — several front ends always
    use the array form, even for plain text) silently skips the input check,
    since `body.get("content")` would be a list, not a str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _last_user_message(body: Dict[str, Any]) -> str:
    for m in reversed(body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        # Case-insensitive: a real client sent role: "User" (capital U) — Ollama's
        # chat template doesn't care about role casing so the request still worked
        # end-to-end, but an exact "user" match here silently found no message,
        # emptied the input-check content, and skipped the check entirely with no
        # error or log (`_check()` treats empty content as nothing-to-check). That's
        # exactly the production symptom: only /guardrail/check/output ever showed
        # up in the logs, never /guardrail/check/input.
        if isinstance(role, str) and role.strip().lower() == "user":
            return _text_of(m.get("content"))
    return ""


def _first_choice_message(body: Dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def _field(name: str) -> FieldExtractor:
    def _get(body: Dict[str, Any]) -> str:
        value = body.get(name)
        return value if isinstance(value, str) else ""
    return _get


class GuardrailMiddleware(BaseHTTPMiddleware):
    """Pre/post Guardrail Service check for a fixed set of (path, method) routes."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.gr_enabled:
            return await call_next(request)

        v = settings.api_version
        routes: Dict[tuple, tuple] = {
            (f"/api/{v}/chat/completions", "POST"): (_last_user_message, _first_choice_message),
            (f"/api/{v}/a2a/query", "POST"): (_field("question"), _field("answer")),
            (f"/api/{v}/chat/chat", "POST"): (_field("message"), _field("response")),
        }
        extractors = routes.get((request.url.path, request.method))
        if extractors is None:
            return await call_next(request)
        input_field, output_field = extractors

        # Just calling request.body() is enough — BaseHTTPMiddleware.__call__ hands
        # dispatch() a `starlette.middleware.base._CachedRequest`, whose own
        # `wrapped_receive()` already replays `request._body` to the downstream app
        # once it has been read here (that's the whole reason `_CachedRequest` exists:
        # it's Starlette's own fix for "body read in outer middleware must still reach
        # the route handler"). An explicit `request._receive = ...` patch on top of
        # that is not just redundant, it's actively wrong: `_CachedRequest` calls the
        # *real* `self.receive()` a second time to watch for client disconnect while
        # the downstream app runs, and expects that second call to eventually yield
        # `http.disconnect`. A replacement `_receive` that unconditionally (or even
        # just once-then-falls-through) replays `http.request` answers that second,
        # disconnect-watching call with another `http.request`, which Starlette treats
        # as a protocol violation: `RuntimeError: Unexpected message received:
        # http.request` while draining `response.body_iterator` — reproduced against
        # `starlette==0.37.2` (the version `fastapi==0.111.0` in requirements.txt
        # actually pins) even with a "replay once, then delegate" version of the
        # patch. The newer Starlette installed ambiently in dev didn't surface this
        # during local testing, which is why it slipped through the first time —
        # verified this fix under a venv pinned to requirements.txt's exact versions.
        raw_body = await request.body()

        try:
            payload = json.loads(raw_body) if raw_body else {}
            if not isinstance(payload, dict):
                payload = {}
        except json.JSONDecodeError:
            payload = {}

        async with httpx.AsyncClient(timeout=settings.guardrail_timeout) as client:
            blocked = await self._check(client, settings, input_field(payload), "input")
            if blocked is not None:
                return blocked

            response = await call_next(request)
            if response.status_code >= 400:
                return response

            body = b"".join([chunk async for chunk in response.body_iterator])

            try:
                resp_payload = json.loads(body) if body else {}
                if not isinstance(resp_payload, dict):
                    resp_payload = {}
            except json.JSONDecodeError:
                resp_payload = {}

            blocked = await self._check(client, settings, output_field(resp_payload), "output")
            if blocked is not None:
                return blocked

        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    @staticmethod
    async def _check(
        client: httpx.AsyncClient,
        settings,
        content: str,
        direction: str,
    ) -> Optional[JSONResponse]:
        """Return a 422 JSONResponse if a guard model flags `content` unsafe, else None (pass/fail-open).

        `/guard/check/{direction}` fixes role by endpoint, so the request body is just
        `{"content": ...}` — no `module`/`request_id`, since this endpoint group doesn't
        go through the policy engine and doesn't write an audit trail to key those by.
        """
        if not content or not content.strip():
            return None
        try:
            resp = await client.post(
                f"{settings.guardrail_url.rstrip('/')}/guard/check/{direction}",
                json={"content": content},
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:  # network/timeout/HTTP-status/JSON — all fail open
            logger.warning("[guardrail] %s check failed, failing open: %s", direction, exc)
            return None

        if result.get("safe", True):
            return None

        # `categories` is a list (multi-model dispatch, one entry per guard model
        # that reports one) and can legitimately be empty even when safe=False —
        # e.g. granite-guardian's native format has no per-category id, only a
        # safe/unsafe score (see module 23's README). First one, if any, is the
        # closest thing this endpoint group has to a `policy_id`.
        categories = result.get("categories") or []
        return JSONResponse(
            status_code=422,
            content={"error": "content_policy_violation", "policy_id": categories[0] if categories else None},
        )
