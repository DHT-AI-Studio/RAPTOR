"""LLM calls through module 23's OpenAI-compatible guardrail proxy (GB-7 / GB-12).

LLM_BASE_URL points at module 23 (/v1/chat/completions), which checks the
input and output and otherwise returns the upstream's own response. Module 23
reports what went wrong through the OpenAI error body's `type`:

- policy_violation (400)        -> never retried anywhere: answered with a refusal
- guardrail_unavailable (503)   -> the check itself failed inside module 23
- (no response at all)          -> module 23 is unreachable

Only the last two fall back to the direct upstream (LLM_FALLBACK_BASE_URL), and
only when GUARDRAIL_FALLBACK_PASSTHROUGH is on; anything else (e.g. the
upstream's own errors relayed by module 23) is raised as before.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Literal

import openai
from langchain_core.messages import AIMessage

_logger = logging.getLogger(__name__)

ErrorKind = Literal["unreachable", "guardrail_unavailable", "policy_violation"]

_POLICY_PREFIX = re.compile(r"^\s*Policy violation:\s*", re.IGNORECASE)


def _error_body(exc: openai.APIStatusError) -> dict[str, Any]:
    try:
        body = exc.response.json()
    except ValueError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def classify_error(exc: BaseException) -> ErrorKind | None:
    if isinstance(exc, openai.APIConnectionError):     # includes APITimeoutError
        return "unreachable"
    if isinstance(exc, openai.APIStatusError):
        error_type = _error_body(exc).get("type")
        if exc.status_code == 400 and error_type == "policy_violation":
            return "policy_violation"
        if exc.status_code == 503 and error_type == "guardrail_unavailable":
            return "guardrail_unavailable"
    return None


def refusal_message(exc: openai.APIStatusError) -> AIMessage:
    categories = _POLICY_PREFIX.sub("", str(_error_body(exc).get("message") or "")).strip()
    reason = f"（{categories}）" if categories else ""
    return AIMessage(
        content=f"此請求違反內容政策{reason}，無法回覆。",
        additional_kwargs={"guardrail_blocked": True},
    )


def is_blocked(message: Any) -> bool:
    return bool(getattr(message, "additional_kwargs", {}).get("guardrail_blocked"))


async def ainvoke_guarded(primary, fallback, messages, *, passthrough: bool):
    """primary.ainvoke(messages), with module 23's failures handled as above."""
    try:
        return await primary.ainvoke(messages)
    except (openai.APIConnectionError, openai.APIStatusError) as exc:
        kind = classify_error(exc)
        if kind == "policy_violation":
            _logger.warning("guardrail blocked the LLM call: %s", _error_body(exc).get("message"))
            return refusal_message(exc)
        if kind is None:
            raise
        reason = "guardrail service unreachable" if kind == "unreachable" else "guardrail check unavailable"
        if not passthrough or fallback is None:
            _logger.error("%s, no fallback (GUARDRAIL_FALLBACK_PASSTHROUGH=%s): %s", reason, passthrough, exc)
            raise
        _logger.error("%s, falling back to the direct LLM upstream: %s", reason, exc)
    return await fallback.ainvoke(messages)
