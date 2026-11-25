"""GuardrailInterceptor — screens the user text of an RPC with Module 23 (GB-20).

The gRPC counterpart of Module 13's REST GuardrailMiddleware: before an RPC that
carries user text reaches its handler, the text is sent to Module 23's canonical
decision API,

    POST {guardrail_url}/guardrail/policy/decide   direction=input, module="36"

which answers allow | warn | redact | block:

* `block` -- the call is aborted with PERMISSION_DENIED, "Policy violation: {category}".
* `redact` -- the field is replaced with decide's `redacted_content` in a copy of
  the request, and the handler gets the copy.
* `warn` -- passes, with a warning logged.
* `allow` -- passes.

Which field holds the user text is a fixed table, `_TEXT_FIELDS`, keyed by full
method name. An RPC not in it (Ping, GetJob, DownloadAsset, UploadAsset, ...) passes
untouched with no Module 23 call: binary payloads, ids and oneofs are never read.
Only unary-request RPCs can be in it; a client-streaming RPC has no single request
to screen before the handler starts.

Fail-open: a decide call that times out (`timeout_s`, GR_GRPC_TIMEOUT_MS), cannot
connect, answers non-2xx or answers malformed JSON is logged as a warning and the
call proceeds with the original request -- a Module 23 outage must not take the
gateway down, the same rule Module 13 follows.

Placement: inside StreamLimitInterceptor, outside ExceptionMappingInterceptor. A call
that Authentication/Identity/Authorization already rejected (auth_error_var set) is
not screened: ExceptionMapping aborts it, and an unauthenticated caller must not be
able to use the gateway to reach Module 23.

Only the request is screened: the responses (search hits, chat tokens) are not.
"""
from __future__ import annotations

import logging

import grpc
import httpx

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.context import auth_error_var, request_id_var, user_var

_logger = logging.getLogger("grpc_gateway.guardrail")

_SERVICE = "/raptor.v1.RaptorService/"

# Full method name -> the request field holding the user's text. A new RPC that carries user
# text must be added here: test_every_rpc_with_user_text_is_screened fails until it is.
_TEXT_FIELDS: dict[str, str] = {
    _SERVICE + "Search": "query",
    _SERVICE + "SearchBm25": "query",
    _SERVICE + "SearchVector": "query",
    _SERVICE + "StreamSearch": "query",
    _SERVICE + "StreamSearchBm25": "query",
    _SERVICE + "StreamSearchVector": "query",
    _SERVICE + "GraphRagSearch": "query",
    _SERVICE + "TKGSearch": "query",
    _SERVICE + "SearchByPerson": "person_name",
    _SERVICE + "StreamSearchByPerson": "person_name",
    _SERVICE + "Chat": "message",
}

_SCREENED_SHAPES = ("unary_unary", "unary_stream")


def _extract_user_text(request, method_name: str) -> tuple[str, str] | None:
    """(field name, text) of the user text `method_name`'s request carries, or None when
    the RPC is not screened or the field is empty. Only a string field is ever read."""
    field = _TEXT_FIELDS.get(method_name)
    if field is None:
        return None
    descriptor = request.DESCRIPTOR.fields_by_name.get(field)
    if descriptor is None or descriptor.type != descriptor.TYPE_STRING:
        return None
    text = getattr(request, field)
    if not text or not text.strip():
        return None
    return field, text


def _with_field(request, field: str, text: str):
    """A copy of `request` with `field` set to `text`; the original is left untouched."""
    copy = type(request)()
    copy.CopyFrom(request)
    setattr(copy, field, text)
    return copy


class GuardrailInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, guardrail_url: str, timeout_s: float, http_client: httpx.AsyncClient | None = None):
        self._decide_url = f"{guardrail_url.rstrip('/')}/guardrail/policy/decide"
        self._timeout_s = timeout_s
        self._http = http_client or httpx.AsyncClient()

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None
        method_name = handler_call_details.method
        if method_name not in _TEXT_FIELDS:
            return handler

        def make_wrapper(behavior, shape):
            if shape not in _SCREENED_SHAPES:
                return behavior

            if shape == "unary_stream":
                async def wrapped(request, context):
                    request = await self._screen(request, context, method_name)
                    async for item in iter_streaming(behavior, request, context):
                        yield item
            else:
                async def wrapped(request, context):
                    request = await self._screen(request, context, method_name)
                    return await call_unary(behavior, request, context)

            return wrapped

        return wrap_handler(handler, make_wrapper)

    async def _screen(self, request, context, method_name: str):
        """The request the handler should get: the original, or a redacted copy. Aborts the
        call on `block`."""
        if auth_error_var.get() is not None:
            return request
        extracted = _extract_user_text(request, method_name)
        if extracted is None:
            return request
        field, text = extracted

        result = await self._decide(text, request, method_name)
        if result is None:
            return request
        decision = result.get("decision")
        if decision == "block":
            categories = result.get("categories") or []
            category = categories[0] if categories else "content_policy"
            _logger.info("[guardrail] %s blocked: request_id=%s category=%s",
                         method_name, request_id_var.get(), category)
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, f"Policy violation: {category}")
        if decision == "redact":
            redacted = result.get("redacted_content")
            if isinstance(redacted, str):
                return _with_field(request, field, redacted)
        if decision == "warn":
            _logger.warning("[guardrail] %s flagged (warn), passing through: request_id=%s reason=%s",
                            method_name, request_id_var.get(), result.get("reason"))
        return request

    async def _decide(self, text: str, request, method_name: str) -> dict | None:
        """Module 23's decision for `text`, or None when the call failed (fail-open)."""
        body = {
            "content": text,
            "direction": "input",
            "module": "36",
            "user_id": user_var.get()["user_id"],
        }
        request_id = request_id_var.get()
        if request_id:
            body["request_id"] = request_id
        session_id = getattr(request, "session_id", "")
        if isinstance(session_id, str) and session_id:
            body["session_id"] = session_id
        try:
            resp = await self._http.post(self._decide_url, json=body, timeout=self._timeout_s)
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                raise ValueError(f"expected a decision object, got {type(result).__name__}")
        except (httpx.HTTPError, ValueError) as exc:
            # The type name too: httpx's timeout exceptions have an empty message.
            _logger.warning("[guardrail] %s check failed, failing open: %s: %s",
                            method_name, type(exc).__name__, exc)
            return None
        return result
