"""GuardrailMiddleware — gateway-level content-policy intercept (V04-10, Module 13).

Wraps three exact (canonical path, method) pairs — `POST /api/{v}/chat/completions`,
`POST /api/{v}/a2a/query`, and `POST /api/{v}/chat/chat` (the RAG chat pipeline,
module 15) — with a before/after check against the Guardrail Service (module 23):

    POST {GUARDRAIL_URL}/guardrail/policy/decide  direction=input   (before the request reaches the LLM)
    POST {GUARDRAIL_URL}/guardrail/policy/decide  direction=output  (before the response reaches the caller)

decide is module 23's canonical policy-aware decision API (GRD-X1): it applies the
active policy (or, with none, each guard model's fixed guard prompt) and writes every
non-allow decision to module 23's audit trail — so each call carries `module="13"`
and the request's `X-Request-ID` (one is generated when the caller sent none). It
replaces the deprecated `/guard/check/{input,output}` this middleware used before.

decide answers allow | warn | redact | block:

* `block` — 422 `{"error": "content_policy_violation", "policy_id": <first category>}`.
* `warn` — passes, with a warning logged.
* `redact` — the checked field is replaced with decide's `redacted_content`: in the
  request body before the route runs (input), or in the response body before it is
  returned (output).

Design:

* `GR_ENABLED=true` (default) — the Guardrail Service is actually called for the
  intercepted routes; `GR_ENABLED=false` short-circuits every request to a single
  boolean branch straight to `call_next` instead, with no Guardrail Service call,
  no body buffering, zero added latency.
* A request whose (path, method) isn't one of the three intercepted routes also passes
  straight through untouched — the middleware only buffers request/response bodies for
  the routes it actually checks.
* Any Guardrail Service failure (network error, timeout, non-2xx — including decide's
  own 503 `guardrail_unavailable` — or malformed JSON) is fail-open: log a warning,
  treat as `allow`, let the request continue. A
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

GB-19 / VIE01-379 — input-only screening of more routes, audit/strict mode:

* Besides the three routes above, the POST routes in `settings.guardrail_routes`
  (`{path_prefix: body_field}`, relative to `/api/{v}`; GATEWAY_GUARDRAIL_ROUTES) get
  an input check: search hybrid/vector/bm25 (`query`) and DocAgent `/process`
  (`options.question` from its multipart form, only when `task=qa`) by default.
* `GATEWAY_GUARDRAIL_MODE` applies to those routes only: `audit` (default) runs the
  check in an `asyncio.create_task()` that is never awaited and logs an unsafe result
  as WARNING with request_id, path and category; `strict` awaits it and a `block`
  returns the same 422 as above. The three routes above keep blocking whenever
  GR_ENABLED is on, whatever the mode — the mode never weakens existing protection.
* A request with `X-Internal: true` skips this middleware entirely.
  IngressHeaderStripMiddleware (CF-3, outermost) removes that header from every
  request entering the gateway, so a client can't send it to skip the check.
* For these routes, Guardrail Service failures are skipped silently (DEBUG log), with
  a short connect timeout so an unreachable service adds no real latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.dependencies import get_current_user
from app.core.config import get_settings

logger = logging.getLogger(__name__)

FieldExtractor = Callable[[dict[str, Any]], str]
# Writes redacted text back where the paired FieldExtractor read it; returns
# whether the body had that field to write.
FieldWriter = Callable[[dict[str, Any], str], bool]


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


def _last_user(body: dict[str, Any]) -> dict[str, Any] | None:
    for m in reversed(body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        # Case-insensitive: a real client sent role: "User" (capital U) — Ollama's
        # chat template doesn't care about role casing so the request still worked
        # end-to-end, but an exact "user" match here silently found no message,
        # emptied the input-check content, and skipped the check entirely with no
        # error or log (`_decide()` treats empty content as nothing-to-check). That's
        # exactly the production symptom: only /guardrail/check/output ever showed
        # up in the logs, never /guardrail/check/input.
        if isinstance(role, str) and role.strip().lower() == "user":
            return m
    return None


def _last_user_message(body: dict[str, Any]) -> str:
    m = _last_user(body)
    return _text_of(m.get("content")) if m is not None else ""


def _set_last_user_message(body: dict[str, Any], text: str) -> bool:
    """Replace the last user message's text. A parts array keeps its non-text
    parts (e.g. images), with all its text parts replaced by one — `_text_of`
    joined them into the single string that was checked and redacted."""
    m = _last_user(body)
    if m is None:
        return False
    content = m.get("content")
    if isinstance(content, list):
        others = [p for p in content if not (isinstance(p, dict) and p.get("type") == "text")]
        m["content"] = [{"type": "text", "text": text}, *others]
    else:
        m["content"] = text
    return True


def _first_choice_message(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def _set_first_choice_message(body: dict[str, Any], text: str) -> bool:
    try:
        body["choices"][0]["message"]["content"] = text
        return True
    except (KeyError, IndexError, TypeError):
        return False


def _field(name: str) -> FieldExtractor:
    def _get(body: dict[str, Any]) -> str:
        value = body.get(name)
        return value if isinstance(value, str) else ""
    return _get


def _set_field(name: str) -> FieldWriter:
    def _set(body: dict[str, Any], text: str) -> bool:
        if not isinstance(body.get(name), str):
            return False
        body[name] = text
        return True
    return _set


def _redacted(decision: dict[str, Any] | None) -> str | None:
    if not decision or decision.get("decision") != "redact":
        return None
    text = decision.get("redacted_content")
    return text if isinstance(text, str) else None


def _blocked(decision: dict[str, Any] | None) -> JSONResponse | None:
    if not decision or decision.get("decision") != "block":
        return None
    # `categories` (guard-model codes, e.g. "S1") can legitimately be empty on a
    # block — a rule-engine policy decided it, or granite-guardian's native format
    # has no per-category id (see module 23's README). First one, if any, is the
    # closest thing decide has to a `policy_id`.
    categories = decision.get("categories") or []
    return JSONResponse(
        status_code=422,
        content={"error": "content_policy_violation", "policy_id": categories[0] if categories else None},
    )


async def _parse_json_body(request: Request) -> dict[str, Any]:
    raw_body = await request.body()
    try:
        body = json.loads(raw_body) if raw_body else {}
        return body if isinstance(body, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _run_input_guardrail(
    request: Request, body: dict[str, Any], extract: FieldExtractor, write: FieldWriter,
) -> dict[str, Any]:
    """Shared body for every *_guardrail_dependency below: block/redact *body*'s screenable
    field via Module 23 (or pass it through unchanged), the input half of what
    GuardrailMiddleware used to do for this route by itself -- moved to a FastAPI dependency
    chained after get_current_user, so an unauthenticated or UMA-denied caller's content never
    reaches the Guardrail Service at all, which the middleware alone cannot guarantee (it runs
    before FastAPI's own dependency injection). The caller must return *body* (or the typed
    model it was validated into) as the route's own body parameter via Depends(...), not
    declare a separate Body(...)/typed parameter of its own: FastAPI resolves a route's own
    body parameter independently of what a dependency does to request._body, so a redaction
    made there would never reach the route (confirmed empirically before this was written).
    GR_ENABLED=false short-circuits exactly like the middleware does.
    """
    settings = get_settings()
    if not settings.gr_enabled:
        return body
    request_id = getattr(request.state, "guardrail_request_id", None) or str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=settings.guardrail_timeout) as client:
        decision = await GuardrailMiddleware._decide(client, settings, extract(body), "input", request_id)
    if decision and decision.get("decision") == "block":
        categories = decision.get("categories") or []
        raise HTTPException(
            status_code=422,
            detail={"error": "content_policy_violation", "policy_id": categories[0] if categories else None},
        )
    redacted = _redacted(decision)
    if redacted is not None:
        write(body, redacted)
    return body


async def guardrail_input_dependency(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """chat/completions' input guardrail -- see _run_input_guardrail. The route's own `body`
    parameter must come from this dependency, not a separate Body(...) declaration."""
    body = await _parse_json_body(request)
    return await _run_input_guardrail(request, body, _last_user_message, _set_last_user_message)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ── GB-19: input-only routes from settings.guardrail_routes ──────────────────

# Extra condition per route: checked only when this form/body field has this value.
_ONLY_WHEN: dict[str, tuple[str, str]] = {"/docagent/process": ("task", "qa")}

_CONNECT_TIMEOUT = 2.0

# Strong references to running audit-mode checks; asyncio only keeps weak ones.
_BACKGROUND: set[asyncio.Task] = set()


def _match_route(path: str, api_version: str, routes: dict[str, str]) -> tuple[str, str] | None:
    """(prefix, body_field) of the configured route `path` falls under, or None."""
    base = f"/api/{api_version}"
    if not path.startswith(base + "/"):
        return None
    rel = path[len(base):]
    for prefix, body_field in routes.items():
        p = "/" + prefix.strip("/")
        if rel == p or rel.startswith(p + "/"):
            return p, body_field
    return None


async def _read_fields(request: Request) -> dict[str, Any]:
    """The request's JSON object body, or its form fields (strings only). The body
    is cached by request.body(), so the route still receives it (see dispatch)."""
    content_type = request.headers.get("content-type", "").lower()
    raw = await request.body()
    if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()  # parses the cached body above
        try:
            return {k: v for k, v in form.multi_items() if isinstance(v, str)}
        finally:
            await form.close()
    try:
        payload = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lookup(data: dict[str, Any], body_field: str) -> Any:
    """`a.b` walks nested objects; a JSON string on the way (a form field) is decoded."""
    value: Any = data
    for part in body_field.split("."):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _route_client(settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(settings.guardrail_timeout, connect=_CONNECT_TIMEOUT))


def _category(decision: dict[str, Any]) -> str | None:
    categories = decision.get("categories") or []
    return categories[0] if categories else None


def _log_unsafe(decision: dict[str, Any] | None, request_id: str, path: str) -> None:
    if not decision or decision.get("decision", "allow") == "allow":
        return
    logger.warning(
        "[guardrail] unsafe content: request_id=%s path=%s category=%s decision=%s",
        request_id, path, _category(decision), decision.get("decision"),
    )


class GuardrailMiddleware(BaseHTTPMiddleware):
    """Pre/post Guardrail Service check for a fixed set of (path, method) routes."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.gr_enabled:
            return await call_next(request)
        if request.headers.get("x-internal", "").strip().lower() == "true":
            return await call_next(request)

        v = settings.api_version
        routes: dict[tuple, tuple] = {
            # PoC: chat/completions' input half moved to guardrail_input_dependency, chained
            # after get_current_user, so an unauthenticated caller's content never reaches
            # the Guardrail Service (this middleware runs before FastAPI's own dependency
            # injection and cannot guarantee that on its own -- live-confirmed: a request with
            # no Authorization header at all still got a real 422 policy_id back here instead
            # of the 401 the route's own auth dependency would give it). None, None marks the
            # input side as "handled elsewhere" -- output screening is unchanged for all three.
            (f"/api/{v}/chat/completions", "POST"): (None, None,
                                                     _first_choice_message, _set_first_choice_message),
            (f"/api/{v}/a2a/query", "POST"): (None, None,
                                              _field("answer"), _set_field("answer")),
            (f"/api/{v}/chat/chat", "POST"): (None, None,
                                              _field("response"), _set_field("response")),
        }
        fields = routes.get((request.url.path, request.method))
        if fields is None:
            if request.method == "POST":
                route = _match_route(request.url.path, v, settings.guardrail_routes)
                if route is not None:
                    return await self._screen_input(request, call_next, settings, *route)
            return await call_next(request)

        input_field, set_input_field, output_field, set_output_field = fields
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # guardrail_input_dependency reads this so its own decide call and this middleware's
        # output decide call share one request_id in module 23's audit trail, same as when
        # both were done here together.
        request.state.guardrail_request_id = request_id

        if input_field is None:
            response = await call_next(request)
            if response.status_code >= 400:
                return response
            return await self._screen_output(response, settings, request_id, output_field, set_output_field)

        # This middleware runs before FastAPI's own dependency injection (auth is a route
        # Depends(), which only runs once call_next() reaches the handler) -- without this,
        # a caller with no Authorization header at all still gets its content screened,
        # effectively a free, unauthenticated probe of the Guardrail Service's classifier
        # (live-confirmed: such a request got a real 422 policy_id back here, not the 401
        # the route's own auth dependency would give it). This is a cheap presence/shape
        # check only, not real verification -- it must not call get_current_user or any
        # real JWT/module 06 logic directly: those are resolved through FastAPI's DI, which
        # a long-standing test convention overrides (app.dependency_overrides) to isolate
        # each router's own tests from real Keycloak/module 06, and middleware calling that
        # logic directly would bypass every one of those overrides. A token with no scheme
        # or no bearer-looking value can only ever fail auth anyway, so skipping straight to
        # call_next() for it costs nothing -- it still gets a real 401, just from the route's
        # own dependency a moment later, same as always.
        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return await call_next(request)

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
            decision = await self._decide(client, settings, input_field(payload), "input", request_id)
            blocked = _blocked(decision)
            if blocked is not None:
                return blocked
            redacted = _redacted(decision)
            if redacted is not None and set_input_field(payload, redacted):
                # Same `_CachedRequest` replay as above, just with the redacted
                # body in place of the original; the route reads Content-Length
                # from the shared scope, so that has to match the new body.
                request._body = _json_bytes(payload)
                request.scope["headers"] = [
                    *((k, val) for k, val in request.scope["headers"] if k != b"content-length"),
                    (b"content-length", str(len(request._body)).encode()),
                ]

            response = await call_next(request)
            if response.status_code >= 400:
                return response

        return await self._screen_output(response, settings, request_id, output_field, set_output_field)

    @staticmethod
    async def _screen_output(
        response: Response, settings, request_id: str, output_field: FieldExtractor, set_output_field: FieldWriter,
    ) -> Response:
        """Read *response*'s body, check it against Module 23, block/redact, and return the
        (possibly rebuilt) Response -- the half of the check every wrapped route still gets
        from this middleware regardless of where its input check runs."""
        body = b"".join([chunk async for chunk in response.body_iterator])

        try:
            resp_payload = json.loads(body) if body else {}
            if not isinstance(resp_payload, dict):
                resp_payload = {}
        except json.JSONDecodeError:
            resp_payload = {}

        async with httpx.AsyncClient(timeout=settings.guardrail_timeout) as client:
            decision = await GuardrailMiddleware._decide(client, settings, output_field(resp_payload), "output", request_id)
        blocked = _blocked(decision)
        if blocked is not None:
            return blocked

        headers = dict(response.headers)
        redacted = _redacted(decision)
        if redacted is not None and set_output_field(resp_payload, redacted):
            body = _json_bytes(resp_payload)
            headers.pop("content-length", None)            # recomputed by Response for the new body

        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
        )

    async def _screen_input(self, request: Request, call_next, settings, prefix: str, body_field: str) -> Response:
        """GB-19 input-only check of a settings.guardrail_routes route, in
        GATEWAY_GUARDRAIL_MODE (audit: background, never blocks; strict: blocks)."""
        data = await _read_fields(request)
        condition = _ONLY_WHEN.get(prefix)
        if condition is not None and data.get(condition[0]) != condition[1]:
            return await call_next(request)
        content = _lookup(data, body_field)
        if not isinstance(content, str) or not content.strip():
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        path = request.url.path

        if settings.guardrail_mode == "strict":
            async with _route_client(settings) as client:
                decision = await self._decide(client, settings, content, "input", request_id,
                                              quiet=True, log_warn=False)
            _log_unsafe(decision, request_id, path)
            blocked = _blocked(decision)
            if blocked is not None:
                return blocked
            return await call_next(request)

        task = asyncio.create_task(self._audit(settings, content, request_id, path))
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)
        return await call_next(request)

    @classmethod
    async def _audit(cls, settings, content: str, request_id: str, path: str) -> None:
        async with _route_client(settings) as client:
            decision = await cls._decide(client, settings, content, "input", request_id,
                                         quiet=True, log_warn=False)
        _log_unsafe(decision, request_id, path)

    @staticmethod
    async def _decide(
        client: httpx.AsyncClient,
        settings,
        content: str,
        direction: str,
        request_id: str,
        *,
        quiet: bool = False,
        log_warn: bool = True,
    ) -> dict[str, Any] | None:
        """Module 23's decision for `content` (its DecideResponse dict), or None
        when there is nothing to check or the call failed (fail-open). `quiet`
        logs a failure at DEBUG instead of WARNING (GB-19 routes: skip silently);
        `log_warn=False` leaves logging a `warn` decision to the caller."""
        if not content or not content.strip():
            return None
        try:
            resp = await client.post(
                f"{settings.guardrail_url.rstrip('/')}/guardrail/policy/decide",
                json={"content": content, "direction": direction, "module": "13", "request_id": request_id},
            )
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                raise ValueError(f"expected a decision object, got {type(result).__name__}")
        except (httpx.HTTPError, ValueError) as exc:  # network/timeout/HTTP-status/JSON — all fail open
            (logger.debug if quiet else logger.warning)(
                "[guardrail] %s check failed, failing open: %s", direction, exc)
            return None

        if log_warn and result.get("decision") == "warn":
            logger.warning("[guardrail] %s flagged (warn), passing through: request_id=%s reason=%s",
                           direction, request_id, result.get("reason"))
        return result
