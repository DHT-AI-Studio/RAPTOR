# 21-agent-protocol/app/handlers/task_handler.py
"""
GB-21 — A2A Host Orchestrator Guardrail Pre-check.

Screens all TextPart content in an incoming A2A tasks/send request
BEFORE routing to any sub-agent (52001–52006).

Flow:
  extract TextParts → POST /guardrail/policy/decide (module=21, direction=input)
      decision=block  → JSON-RPC error -32600 (Policy violation)
      decision=redact → replace TextPart.text with redacted_content, continue
      decision=allow/warn → continue unchanged
      unreachable     → fail-open, log warning, continue

Internal calls (X-A2A-Internal: true  OR  source is a known sub-agent ID)
are skipped — avoids double-screening when sub-agents call back through
the orchestrator.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from core.config import get_settings

logger = logging.getLogger(__name__)

# ── Sub-agent identity set (MTR-25 / pipeline.py get_agent_urls keys) ────────
# These are the agent IDs that sub-agents declare in their AgentCard.
# Requests whose source matches any of these are internal and skip screening.
_INTERNAL_AGENT_IDS: frozenset[str] = frozenset({
    "raptor-vector-agent",
    "raptor-keyword-agent",
    "raptor-graphrag-agent",
    "raptor-tkg-agent",
    "raptor-reranker-agent",
    "raptor-people-agent",
})

# JSON-RPC error code per A2A spec / Ticket AC
_POLICY_VIOLATION_CODE = -32600


# ── Internal-call detector ────────────────────────────────────────────────────

def is_internal_call(headers: dict[str, str], source_agent_id: str | None = None) -> bool:
    """Return True when the request originates from a known sub-agent.

    Two signals (either is sufficient):
      1. X-A2A-Internal: true  header set by the sub-agent SDK
      2. source_agent_id matches a known internal agent ID
    """
    if headers.get("x-a2a-internal", "").lower() == "true":
        return True
    return bool(source_agent_id and source_agent_id in _INTERNAL_AGENT_IDS)

# ── TextPart extraction ───────────────────────────────────────────────────────

def extract_text_parts(task: dict[str, Any]) -> list[str]:
    """Walk task.message.parts[] and collect every TextPart.text value."""
    try:
        parts = task.get("message", {}).get("parts", [])
    except AttributeError:
        return []
    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text", "")
            if text:
                texts.append(text)
    return texts


def _replace_text_parts(task: dict[str, Any], redacted: str) -> dict[str, Any]:
    """Return a shallow-copied task with every TextPart.text replaced by *redacted*."""
    import copy
    task = copy.deepcopy(task)
    try:
        parts = task["message"]["parts"]
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = redacted
    except (KeyError, TypeError):
        pass
    return task


# ── Guardrail HTTP call ───────────────────────────────────────────────────────

async def _call_guardrail(content: str, request_id: str | None) -> dict[str, Any] | None:
    """POST /guardrail/policy/decide → DecideResponse dict, or None on failure (fail-open)."""
    settings = get_settings()
    url = f"{settings.guardrail_url.rstrip('/')}/guardrail/policy/decide"
    payload: dict[str, Any] = {
        "content": content,
        "direction": "input",
        "module": "21",
    }
    if request_id:
        payload["request_id"] = request_id

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        logger.warning("[GB-21] Guardrail timeout (fail-open): url=%s", url)
    except httpx.RequestError as exc:
        logger.warning("[GB-21] Guardrail unreachable (fail-open): %s", exc)
    except httpx.HTTPStatusError as exc:
        logger.warning("[GB-21] Guardrail HTTP error %s (fail-open): %s",
                       exc.response.status_code, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GB-21] Guardrail unexpected error (fail-open): %s", exc)
    return None


# ── Public pre-check entry point ──────────────────────────────────────────────

async def guardrail_precheck(
    task: dict[str, Any],
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
    source_agent_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Screen *task* content before routing to a sub-agent.

    Returns:
        (modified_task, error_response)

        - Normal pass:   (task_unchanged, None)
        - Redact:        (task_with_redacted_parts, None)
        - Block:         (None, jsonrpc_error_dict)
        - Fail-open:     (task_unchanged, None)

    The caller should check `error_response is not None` first and return it
    immediately; otherwise use `modified_task` for routing.
    """
    settings = get_settings()

    # ── Feature flag ──────────────────────────────────────────────────────────
    if not settings.guardrail_active:
        return task, None

    # ── Skip internal sub-agent calls ─────────────────────────────────────────
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    if is_internal_call(hdrs, source_agent_id):
        logger.debug("[GB-21] Skipping guardrail for internal call (source=%s)", source_agent_id)
        return task, None

    # ── Extract TextPart content ──────────────────────────────────────────────
    texts = extract_text_parts(task)
    if not texts:
        return task, None

    combined_content = "\n".join(texts)

    # ── Call guardrail service ────────────────────────────────────────────────
    result = await _call_guardrail(combined_content, request_id)

    # Fail-open: guardrail unreachable
    if result is None:
        return task, None

    decision = result.get("decision", "allow")

    # ── Block ─────────────────────────────────────────────────────────────────
    if decision == "block":
        category = ",".join(result.get("categories") or []) or "S1"
        logger.warning(
            "[GB-21] Blocked request: request_id=%s category=%s",
            request_id, category,
        )
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": _POLICY_VIOLATION_CODE,
                "message": "Policy violation",
                "data": {"category": category},
            },
            "id": request_id,
        }
        return None, error_response

    # ── Redact ────────────────────────────────────────────────────────────────
    if decision == "redact":
        redacted_content = result.get("redacted_content") or "[redacted]"
        logger.info(
            "[GB-21] Redacting TextPart content: request_id=%s", request_id,
        )
        modified_task = _replace_text_parts(task, redacted_content)
        return modified_task, None

    # ── Allow / warn (warn is advisory only, still routes) ────────────────────
    return task, None
