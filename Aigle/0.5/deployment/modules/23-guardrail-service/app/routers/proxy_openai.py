"""OpenAI-compatible guardrail proxy (GB-7 / GB-12): /v1/chat/completions, /v1/models.

Mounted in app/main.py without a prefix -- the routes below already carry /v1.
Callers (module 15's ChatOpenAI) point their base_url here instead of at the
upstream LLM, and get the upstream's own response back unless a check fails:

- input: messages[-1].content is checked before anything is forwarded;
- output: choices[0].message.content is checked before it is returned;
- both run the guard model(s) (same classifier as /guard/check/*, S1-S14
  categories) and then the active policy's rules (GB-4 engine, which is what
  can ask for a redact), reusing the one guard verdict.

The upstream defaults to module 07 (PROXY_UPSTREAM_URL), not the Ollama daemon:
module 15 sends engine/num_ctx/think, which only module 07 honours. The request
body is forwarded byte-for-byte.

Error contract (OpenAI-style {"error": {...}} bodies), which module 15 relies on
to tell its fallback cases apart:

- 400 policy_violation       -- blocked; callers must never retry this elsewhere
- 503 guardrail_unavailable  -- the check itself failed (guard model, Redis, ...)
- 502 upstream_unavailable   -- the upstream LLM could not be reached
- upstream non-2xx           -- passed through unchanged
- 400 unsupported / invalid_request -- streaming, empty or non-JSON body
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.db.redis_conn import get_redis
from app.services import audit_log, checker, guard_classifier, state
from app.services.proxy_shared import forward_client

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI-compatible proxy"])

# Set by app/main.py once this router is registered; reported by
# GET /guardrail/system/status as proxy_mounted.
MOUNTED = False

_DEFAULT_MODULE = "openai-proxy"


class _GuardrailUnavailable(Exception):
    """The check itself could not produce a verdict."""


class _Verdict:
    def __init__(self, action: str, category: str | None = None,
                 redacted_content: str | None = None, policy_id: Any = None) -> None:
        self.action = action                    # pass | block | redact
        self.category = category
        self.redacted_content = redacted_content
        self.policy_id = policy_id


def _error(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": status_code}},
        status_code=status_code,
    )


def _text_of(content: Any) -> str:
    """OpenAI message content: a string, or a list of parts of which only the
    text parts are checkable (image_url parts are skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
    return ""


def _last_message_text(body: dict) -> str:
    """messages[-1].content -- except after a tool call, where messages[-1] is the
    tool's output (e.g. module 15's search results): retrieved documents are not
    user input, so the latest user message is checked instead."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return ""
    if messages[-1].get("role") == "tool":
        user = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), None)
        return _text_of(user.get("content")) if user else ""
    return _text_of(messages[-1].get("content"))


def _response_text(body: Any) -> str:
    try:
        return _text_of(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


async def _check(text: str, direction: str, module: str) -> _Verdict:
    """Guard model(s) first, then the active policy's rules on the same verdict."""
    role = "user" if direction == "input" else "assistant"
    try:
        guard = await guard_classifier.classify(text, role=role)

        async def _same_verdict(_content: str, _role: str):
            return guard

        policy_id, policy = await checker._load_active_policy(await get_redis())
        policy_result = await checker.evaluate(text, direction, module, policy, _same_verdict)
    except Exception as exc:  # any failure means no verdict; the caller decides the fallback
        _logger.error("openai-proxy: %s check failed, guardrail unavailable: %r", direction, exc)
        raise _GuardrailUnavailable(str(exc) or type(exc).__name__) from exc

    if not guard.safe:
        return _Verdict("block", ",".join(guard.categories) or policy_result.category or "unsafe")
    if policy_result.action == "block":
        return _Verdict("block", policy_result.category or "unsafe", policy_id=policy_id)
    if policy_result.action == "redact":
        return _Verdict("redact", policy_result.category,
                        redacted_content=policy_result.redacted_content, policy_id=policy_id)
    return _Verdict("pass")


def _audit(verdict: _Verdict, request: Request, direction: str, content: str) -> None:
    audit_log.record_violation(
        policy_id=verdict.policy_id,
        module=_module(request),
        direction=direction,
        category=verdict.category,
        action_taken=verdict.action,
        request_id=request.headers.get("x-request-id"),
        content=content,
    )


def _module(request: Request) -> str:
    return request.headers.get("x-raptor-module") or _DEFAULT_MODULE


def _blocked(verdict: _Verdict) -> JSONResponse:
    return _error(400, "policy_violation", f"Policy violation: {verdict.category}")


def _passthrough(resp: httpx.Response) -> Response:
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


async def _upstream(method: str, path: str, **kwargs) -> httpx.Response:
    settings = get_settings()
    url = f"{settings.proxy_upstream_url.rstrip('/')}{path}"
    return await forward_client().request(method, url, timeout=settings.proxy_upstream_timeout, **kwargs)


# ── OpenAPI documentation ─────────────────────────────────────────────────────
# The handler reads the raw body (it is forwarded byte-for-byte), so FastAPI has
# no body parameter to derive a schema from; this describes the expected body
# for Swagger UI only -- nothing here validates or changes what is accepted.

def _error_example(status_code: int, error_type: str, message: str) -> dict:
    return {"value": {"error": {"message": message, "type": error_type, "code": status_code}}}


_ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "error": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "type": {"type": "string"},
                "code": {"type": "integer"},
            },
        },
    },
}

_CHAT_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["model", "messages"],
    "additionalProperties": True,
    "description": (
        "Standard OpenAI chat completions body. Forwarded to PROXY_UPSTREAM_URL byte-for-byte, "
        "so any extra field (e.g. module 15's engine / num_ctx / think) passes through unchanged."
    ),
    "properties": {
        "model": {"type": "string", "description": "Upstream model name", "example": "gemma4:12b"},
        "messages": {
            "type": "array",
            "description": (
                "Conversation. Input check: messages[-1].content, or the latest role=user message "
                "when messages[-1] is a tool result."
            ),
            "items": {
                "type": "object",
                "required": ["role"],
                "properties": {
                    "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                    "content": {
                        "description": "Text, or a list of parts (only type=text parts are checked)",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "object", "properties": {
                                "type": {"type": "string", "enum": ["text", "image_url"]},
                                "text": {"type": "string"},
                                "image_url": {"type": "object", "properties": {"url": {"type": "string"}}},
                            }}},
                        ],
                        "nullable": True,
                    },
                    "tool_calls": {"type": "array", "items": {"type": "object"},
                                   "description": "assistant turns that call tools"},
                    "tool_call_id": {"type": "string", "description": "tool result turns"},
                },
            },
        },
        "temperature": {"type": "number", "example": 0.7},
        "max_tokens": {"type": "integer"},
        "tools": {"type": "array", "items": {"type": "object"}, "description": "OpenAI function tools"},
        "tool_choice": {"description": "\"auto\", \"none\" or a specific tool"},
        "stream": {"type": "boolean", "enum": [False],
                   "description": "Must be false or omitted -- streaming is rejected (400 unsupported)"},
    },
}

_CHAT_COMPLETIONS_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {"application/json": {
            "schema": _CHAT_REQUEST_SCHEMA,
            "examples": {
                "simple": {
                    "summary": "Plain chat",
                    "value": {"model": "gemma4:12b", "messages": [{"role": "user", "content": "你好"}]},
                },
                "module15": {
                    "summary": "Module 15 (ChatOpenAI + tools + extra_body)",
                    "value": {
                        "model": "gemma4:12b",
                        "messages": [
                            {"role": "user", "content": "今天天氣好嗎"},
                            {"role": "system", "content": "你是一個有幫助的 AI 助手。…"},
                        ],
                        "temperature": 0.7,
                        "tools": [{"type": "function", "function": {
                            "name": "get_current_time", "description": "Get the current time",
                            "parameters": {"type": "object", "properties": {}}}}],
                        "tool_choice": "auto",
                        "engine": "ollama", "num_ctx": 262144, "think": False,
                    },
                },
                "after_tool_call": {
                    "summary": "Turn after a tool call (the user message is checked, not the tool output)",
                    "value": {
                        "model": "gemma4:12b",
                        "messages": [
                            {"role": "user", "content": "幫我找義大利麵的筆記"},
                            {"role": "assistant", "content": "", "tool_calls": [{
                                "id": "call_1", "type": "function",
                                "function": {"name": "unified_search_tool", "arguments": "{\"query\": \"義大利麵\"}"}}]},
                            {"role": "tool", "tool_call_id": "call_1", "content": "[文檔] notes.pdf - 煮水、加鹽"},
                        ],
                    },
                },
            },
        }},
    },
}

_CHAT_COMPLETIONS_DESCRIPTION = """\
Guarded OpenAI-compatible chat completions.

1. Validates the body (non-empty JSON object, `stream` not true).
2. If the guardrail master switch is on, checks the input text — `messages[-1].content`,
   or the latest `role=user` message when `messages[-1]` is a tool result — with the guard
   model(s) and then the active policy's rules.
3. Forwards the original body unchanged to `{PROXY_UPSTREAM_URL}/v1/chat/completions`.
4. Checks `choices[0].message.content` the same way; returns the upstream response unchanged,
   with only the content replaced on a policy `redact`.

Optional headers: `X-Raptor-Module` (audit-log source module, default `openai-proxy`),
`X-Request-ID` (audit-log request id).
"""

_CHAT_COMPLETIONS_RESPONSES = {
    200: {"description": "The upstream's own chat completion response, unchanged (or with the answer redacted)"},
    400: {
        "description": "Blocked by the guardrail, or a request the proxy does not accept",
        "content": {"application/json": {"schema": _ERROR_SCHEMA, "examples": {
            "policy_violation": _error_example(400, "policy_violation", "Policy violation: S1"),
            "invalid_request": _error_example(400, "invalid_request", "Invalid JSON: Expecting value"),
            "unsupported": _error_example(
                400, "unsupported", "stream=true is not supported by the guardrail proxy"),
        }}},
    },
    502: {
        "description": "The upstream LLM could not be reached (upstream error statuses are passed through as-is)",
        "content": {"application/json": {"schema": _ERROR_SCHEMA, "examples": {
            "upstream_unavailable": _error_example(
                502, "upstream_unavailable", "Upstream LLM unavailable: connection refused"),
        }}},
    },
    503: {
        "description": "The guardrail check itself failed (guard model, Redis, ...)",
        "content": {"application/json": {"schema": _ERROR_SCHEMA, "examples": {
            "guardrail_unavailable": _error_example(
                503, "guardrail_unavailable", "Guardrail check unavailable: Ollama unreachable"),
        }}},
    },
}

_MODELS_RESPONSES = {
    200: {
        "description": "The upstream's model list (OpenAI format), unchanged",
        "content": {"application/json": {"example": {
            "object": "list",
            "data": [{"id": "gemma4:12b", "object": "model", "created": 0, "owned_by": "library"}],
        }}},
    },
    502: {
        "description": "The upstream LLM could not be reached",
        "content": {"application/json": {"schema": _ERROR_SCHEMA, "examples": {
            "upstream_unavailable": _error_example(
                502, "upstream_unavailable", "Upstream LLM unavailable: connection refused"),
        }}},
    },
}


@router.post(
    "/v1/chat/completions",
    summary="OpenAI chat completions, guarded and proxied upstream",
    description=_CHAT_COMPLETIONS_DESCRIPTION,
    openapi_extra=_CHAT_COMPLETIONS_OPENAPI,
    responses=_CHAT_COMPLETIONS_RESPONSES,
)
async def chat_completions(request: Request) -> Response:
    raw = await request.body()
    if not raw:
        return _error(400, "invalid_request", "Request body is empty")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error(400, "invalid_request", f"Invalid JSON: {exc}")
    if not isinstance(body, dict):
        return _error(400, "invalid_request", "Request body must be a JSON object")
    if body.get("stream") is True:
        return _error(400, "unsupported", "stream=true is not supported by the guardrail proxy")

    try:
        enabled = await state.is_enabled()
    except Exception as exc:  # noqa: BLE001 -- Redis down: the switch itself is unknown
        _logger.error("openai-proxy: reading the guardrail switch failed: %r", exc)
        return _error(503, "guardrail_unavailable", f"Guardrail check unavailable: {exc}")

    prompt = _last_message_text(body) if enabled else ""
    if prompt:
        try:
            verdict = await _check(prompt, "input", _module(request))
        except _GuardrailUnavailable as exc:
            return _error(503, "guardrail_unavailable", f"Guardrail check unavailable: {exc}")
        if verdict.action == "block":
            _audit(verdict, request, "input", prompt)
            return _blocked(verdict)
        # Input redaction is not applied: the original request is forwarded unmodified.

    try:
        resp = await _upstream(
            "POST", "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"},
        )
    except httpx.HTTPError as exc:
        _logger.error("openai-proxy: upstream unreachable: %r", exc)
        return _error(502, "upstream_unavailable", f"Upstream LLM unavailable: {exc}")

    if not resp.is_success or not enabled:
        return _passthrough(resp)
    try:
        result = resp.json()
    except ValueError:
        return _passthrough(resp)

    answer = _response_text(result)
    if not answer:
        return _passthrough(resp)
    try:
        verdict = await _check(answer, "output", _module(request))
    except _GuardrailUnavailable as exc:
        return _error(503, "guardrail_unavailable", f"Guardrail check unavailable: {exc}")
    if verdict.action == "block":
        _audit(verdict, request, "output", answer)
        return _blocked(verdict)
    if verdict.action == "redact" and verdict.redacted_content is not None:
        _audit(verdict, request, "output", answer)
        result["choices"][0]["message"]["content"] = verdict.redacted_content
        return JSONResponse(result, status_code=resp.status_code)
    return _passthrough(resp)


@router.get(
    "/v1/models",
    summary="Upstream model list, proxied unchanged",
    description="Forwards to `{PROXY_UPSTREAM_URL}/v1/models` and returns its status and body unchanged "
                "(required by LangChain / OpenAI clients).",
    responses=_MODELS_RESPONSES,
)
async def models() -> Response:
    try:
        resp = await _upstream("GET", "/v1/models")
    except httpx.HTTPError as exc:
        _logger.error("openai-proxy: upstream unreachable: %r", exc)
        return _error(502, "upstream_unavailable", f"Upstream LLM unavailable: {exc}")
    return _passthrough(resp)
