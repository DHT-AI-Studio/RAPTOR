"""Guardrail input/output screening (V04-10 / GB-19 / VIE01-379, Module 13).

Every route listed in `settings.guardrail_routes` (`{path: body_field}`,
GATEWAY_GUARDRAIL_ROUTES) gets its input screened against the Guardrail Service
(module 23) before it runs:

    POST {GUARDRAIL_URL}/guardrail/policy/decide  direction=input

Three of those routes -- `POST /api/{v}/chat/completions`, `POST /api/{v}/a2a/query`,
and `POST /api/{v}/chat/chat` (the RAG chat pipeline, module 15) -- also get their
*output* screened the same way (direction=output), by `GuardrailMiddleware` below.
The other routes (search/docagent) are input-only: they return retrieved content,
not LLM-generated content, so there is nothing equivalent to screen on the way out.

decide is module 23's canonical policy-aware decision API (GRD-X1): it applies the
active policy (or, with none, each guard model's fixed guard prompt) and writes every
non-allow decision to module 23's audit trail -- so each call carries `module="13"`
and the request's `X-Request-ID` (one is generated when the caller sent none).

decide answers allow | warn | redact | block:

* `block` -- 422 `{"error": "content_policy_violation", "policy_id": <first category>}`
  in strict mode; in audit mode, logged as WARNING and never returned to the caller.
* `warn` -- passes, with a warning logged.
* `redact` -- the checked field is replaced with decide's `redacted_content` (input
  checks only; GuardrailMiddleware's output check also supports this).

Design:

* `GR_ENABLED=true` (default) -- master switch for every check below, input and
  output alike. `false` short-circuits everything to a zero-latency passthrough.
* `GATEWAY_GUARDRAIL_MODE` (default `strict`) is one shared setting for every route
  in `guardrail_routes`, chat/completions/a2a/query/chat/chat included: `audit` runs
  the check in a background `asyncio.create_task()` that is never awaited and logs an
  unsafe result as WARNING; `strict` awaits it and blocks/redacts. Changing this away
  from `strict` weakens the three original routes too, not just search/docagent --
  that is a deliberate consequence of unifying the setting, not an oversight.
* `GATEWAY_GUARDRAIL_ROUTES` (`{path: body_field}`, relative to `/api/{v}`) is the
  single enable registry: a path listed here that a `@guardrail_route(...)`-decorated
  endpoint also declares gets checked; removing it disables that route's check with
  no code change. It cannot add a check to a route that was never decorated --
  FastAPI has no way to attach a dependency to a route from outside its own
  registration, so a genuinely new endpoint still needs one `@guardrail_route(...)`
  line, in code, once. `warn_on_unknown_configured_routes()` (called from
  `app/main.py` after every router is included) logs a WARNING for any configured
  path that no decorator anywhere actually claimed, so a typo or a stale entry is
  visible at startup instead of silently doing nothing.
* Every input check runs as a FastAPI dependency chained after `get_current_user`
  (either directly, for chat/completions|a2a/query|chat/chat, which need the
  dependency's return value -- see `_run_input_guardrail`; or via
  `make_route_guardrail_dependency()`, attached through `@guardrail_route(...)`'s
  `dependencies=[...]`, for every other route, which only need the check's side
  effect). Either way, this guarantees an unauthenticated or UMA-denied caller's
  content never reaches the Guardrail Service -- a `BaseHTTPMiddleware` cannot
  guarantee that on its own, since it runs before FastAPI's own dependency
  injection (live-confirmed, before this design: a request with no Authorization
  header at all still got a real 422 `policy_id` back from the old
  middleware-only implementation, instead of the 401 the route's own auth
  dependency would give it moments later).
* Any Guardrail Service failure (network error, timeout, non-2xx -- including
  decide's own 503 `guardrail_unavailable` -- or malformed JSON) is fail-open: log
  a warning (DEBUG for the background/audit path), treat as `allow`, let the
  request continue. A Guardrail Service outage must never take any of these
  routes down.
* A request with `X-Internal: true` skips every input check (`_check_input`) and
  `GuardrailMiddleware`'s output check. `IngressHeaderStripMiddleware` (CF-3,
  outermost) removes that header from every request entering the gateway, so a
  client can't send it to skip the check -- only a genuinely internal caller
  ever has it. This does not bypass `get_current_user` itself: an internal
  caller still authenticates normally, it just isn't content-screened.
* `GuardrailMiddleware` must be added to the FastAPI app *after*
  `LegacyApiAliasMiddleware` (i.e. added last, so it ends up innermost) -- that
  middleware rewrites `/api/0.3/{chat,a2a}/...` to the canonical `/api/{v}/...`
  path before anything inside it runs, so this middleware only ever needs to
  match the canonical path once and legacy callers are covered for free.
* `/chat/chat` forwards to module 15's RAG pipeline, which internally calls
  module 07's `/v1/chat/completions` itself, possibly more than once across a
  multi-turn LangGraph tool-calling loop -- the output check only ever sees the
  single final answer at this route's own boundary, not each internal hop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
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


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


async def _read_fields(request: Request) -> dict[str, Any]:
    """The request's JSON object body, or its form fields (strings only). The body
    is cached by request.body(), so the route still receives it."""
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


_CONNECT_TIMEOUT = 2.0


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


# Strong references to running audit-mode checks; asyncio only keeps weak ones.
_BACKGROUND: set[asyncio.Task] = set()


async def _check_input(request: Request, path: str, content: str) -> str | None:
    """Single place every input check funnels through, called either directly by
    `_run_input_guardrail` (chat/completions, a2a/query, chat/chat -- these need
    the redacted text back) or by `make_route_guardrail_dependency()`'s dependency
    (every other route -- these only need the block/log side effect). *path* is
    this route's own key in `settings.guardrail_routes` (e.g. "/chat/completions"),
    used both to decide whether this route is enabled at all and for logging.

    Returns the redacted text to use in place of *content* (strict mode, decide
    said redact), or None (nothing to change). Raises HTTPException(422) in strict
    mode on a block. In audit mode, the check runs in the background and this
    always returns None immediately -- audit mode never redacts, since redaction
    requires waiting for the decision before the route can use it.
    """
    settings = get_settings()
    if not settings.gr_enabled or path not in settings.guardrail_routes:
        return None
    if request.headers.get("x-internal", "").strip().lower() == "true":
        return None
    if not content or not content.strip():
        return None
    request_id = (getattr(request.state, "guardrail_request_id", None)
                  or request.headers.get("x-request-id") or str(uuid.uuid4()))
    request.state.guardrail_request_id = request_id

    if settings.guardrail_mode != "strict":
        task = asyncio.create_task(GuardrailMiddleware._audit(settings, content, request_id, path))
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)
        return None

    async with _route_client(settings) as client:
        decision = await GuardrailMiddleware._decide(client, settings, content, "input", request_id, log_warn=False)
    _log_unsafe(decision, request_id, path)
    if decision and decision.get("decision") == "block":
        categories = decision.get("categories") or []
        raise HTTPException(
            status_code=422,
            detail={"error": "content_policy_violation", "policy_id": categories[0] if categories else None},
        )
    return _redacted(decision)


async def _run_input_guardrail(
    request: Request, body: dict[str, Any], path: str, extract: FieldExtractor, write: FieldWriter,
) -> dict[str, Any]:
    """Shared body for every *_guardrail_dependency below: block/redact *body*'s screenable
    field via `_check_input`, then return *body* (or the typed model it was validated into)
    for the route to consume as its own body parameter via Depends(...) -- the caller must
    not declare a separate Body(...)/typed parameter of its own: FastAPI resolves a route's
    own body parameter independently of what a dependency does to request._body, so a
    redaction made there would never reach the route (confirmed empirically before this was
    written)."""
    redacted = await _check_input(request, path, extract(body))
    if redacted is not None:
        write(body, redacted)
    return body


# Every config_key any @guardrail_route(...) call has ever declared, regardless of whether
# settings.guardrail_routes actually enables it right now -- used by
# warn_on_unknown_configured_routes() to flag a configured path nothing claims. Declared here
# (used below, before the @guardrail_route-based registrations further down) so the three
# routes that don't go through that decorator -- they need their dependency's return value,
# which only a route's own parameter (not dependencies=[...]) can give them -- still register
# themselves the same way.
_KNOWN_PATHS: set[str] = set()
_KNOWN_PATHS.add("/chat/completions")


async def guardrail_input_dependency(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """chat/completions' input guardrail -- see _run_input_guardrail. The route's own `body`
    parameter must come from this dependency, not a separate Body(...) declaration."""
    body = await _parse_json_body(request)
    return await _run_input_guardrail(request, body, "/chat/completions", _last_user_message, _set_last_user_message)


def make_route_guardrail_dependency(config_key: str, field_path: str, *, only_when: tuple[str, str] | None = None):
    """Factory for a void FastAPI dependency that screens *field_path* of the request body
    (dotted for a nested field; see `_lookup`) -- used by every route that doesn't need the
    check's return value (no redaction), attached via `@guardrail_route` below rather than
    declared directly. Chained after get_current_user so an unauthenticated caller's content
    never reaches the Guardrail Service, same guarantee as `_run_input_guardrail` above.
    *config_key* is this route's own key in `settings.guardrail_routes` -- the route's full
    logical path (e.g. "/search/hybrid"), not necessarily what `router.api_route()` was given
    (a router's own prefix, set later by `app.include_router(..., prefix=...)`, isn't known
    at decoration time). `only_when` skips the check unless
    `data[only_when[0]] == only_when[1]` (DocAgent: only task=qa).

    A route using this on a multipart/form body must read its own fields from `request.form()`
    inside the handler, never its own `Form(...)`/`Body(...)` parameters -- FastAPI resolves a
    route's own body/form parameters and its `dependencies=[...]` concurrently, not strictly
    this dependency first, so a route's own `Form(...)` param and this dependency's `_read_fields`
    call race to consume `request.stream()` independently; whichever loses sees Starlette raise
    `RuntimeError("Stream consumed")`. This is specific to multipart/form bodies (confirmed by a
    real failure in this file's own tests) -- a JSON body is unaffected (`request.body()` is safe
    to call from both sides, confirmed live against a real `search.py` route)."""
    async def _dependency(
        request: Request,
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> None:
        data = await _read_fields(request)
        if only_when is not None and data.get(only_when[0]) != only_when[1]:
            return
        content = _lookup(data, field_path)
        if not isinstance(content, str):
            return
        await _check_input(request, config_key, content)
    return _dependency


def guardrail_route(router: APIRouter, route_path: str, config_key: str, field_path: str, *,
                     only_when: tuple[str, str] | None = None, methods: tuple[str, ...] = ("POST",), **kwargs):
    """Registers *route_path* on *router* exactly like `router.post(route_path, **kwargs)`
    (every other decorator kwarg -- tags, summary, description, response_model, status_code
    -- passes through unchanged), with the guardrail input-check dependency attached iff
    *config_key* (this route's full logical path, e.g. "/search/hybrid" -- see
    `make_route_guardrail_dependency`) is listed in `settings.guardrail_routes` at import
    time. Removing *config_key* from that setting and restarting disables the check with no
    code change; it cannot add a check to a route that was never decorated with this in the
    first place -- FastAPI has no way to attach a dependency to a route from outside its own
    registration. A genuinely new endpoint still needs this decorator once, in code;
    `GATEWAY_GUARDRAIL_ROUTES` only toggles enable/field among routes that already carry it."""
    _KNOWN_PATHS.add(config_key)

    def _decorator(func):
        dependencies = list(kwargs.pop("dependencies", []))
        if config_key in get_settings().guardrail_routes:
            dependencies.append(
                Depends(make_route_guardrail_dependency(config_key, field_path, only_when=only_when)))
        return router.api_route(route_path, methods=list(methods), dependencies=dependencies, **kwargs)(func)
    return _decorator


def warn_on_unknown_configured_routes() -> None:
    """Call once from app/main.py, after every router is included, so every
    @guardrail_route(...) call has already run and registered itself in
    _KNOWN_PATHS. Logs a WARNING for each settings.guardrail_routes path that no
    decorator anywhere actually claimed -- a typo'd or stale .env entry is
    otherwise silently inert, never checked, no error."""
    settings = get_settings()
    unknown = set(settings.guardrail_routes) - _KNOWN_PATHS
    for path in sorted(unknown):
        logger.warning(
            "[guardrail] GATEWAY_GUARDRAIL_ROUTES lists %s, but no route registers a "
            "guardrail check for it -- typo, or a route that no longer exists?", path,
        )


class GuardrailMiddleware(BaseHTTPMiddleware):
    """Output-only Guardrail Service check for chat/completions, a2a/query, and chat/chat.
    Input screening for these (and every other guarded route) happens in a FastAPI
    dependency instead -- see the module docstring."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.gr_enabled:
            return await call_next(request)
        if request.headers.get("x-internal", "").strip().lower() == "true":
            return await call_next(request)

        v = settings.api_version
        output_routes: dict[tuple, tuple] = {
            (f"/api/{v}/chat/completions", "POST"): (_first_choice_message, _set_first_choice_message),
            (f"/api/{v}/a2a/query", "POST"): (_field("answer"), _set_field("answer")),
            (f"/api/{v}/chat/chat", "POST"): (_field("response"), _set_field("response")),
        }
        fields = output_routes.get((request.url.path, request.method))
        if fields is None:
            return await call_next(request)

        output_field, set_output_field = fields
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # The input dependency reads this so its own decide call and this middleware's
        # output decide call share one request_id in module 23's audit trail.
        request.state.guardrail_request_id = request_id

        response = await call_next(request)
        if response.status_code >= 400:
            return response
        return await self._screen_output(response, settings, request_id, output_field, set_output_field)

    @staticmethod
    async def _screen_output(
        response: Response, settings, request_id: str, output_field: FieldExtractor, set_output_field: FieldWriter,
    ) -> Response:
        """Read *response*'s body, check it against Module 23, block/redact, and return the
        (possibly rebuilt) Response."""
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
        logs a failure at DEBUG instead of WARNING (audit-mode routes: skip silently);
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
