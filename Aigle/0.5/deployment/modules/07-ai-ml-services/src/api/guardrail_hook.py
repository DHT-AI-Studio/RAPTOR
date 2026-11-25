"""Guardrail hook for Module 07 inference (GB-6).

Asks the Guardrail Service's canonical policy-aware decision API
(`POST /guardrail/policy/decide`, GRD-X1) before and after LLM inference. It
replaces the deprecated raw guard-model checker (`/guard/check/{input,output}`)
this hook called before: decide applies module 23's active policy (falling back
to the guard models' own built-in taxonomy when there is none), so a policy's
per-category decision now takes effect here, and every non-allow decision is
written to module 23's audit trail with this module's `module`/`request_id`.

decide answers allow | warn | redact | block. ``block`` stops inference;
``warn`` lets it through with a log line; ``redact`` carries on with decide's
``redacted_content`` in place of the checked text (see ``apply_input_redaction``
/ ``apply_output_redaction``).

Design constraints from the ticket:

- ``GUARDRAIL_ENABLED`` (default ``false``) is the explicit master switch — mirrors
  module 13's own ``GR_ENABLED``, which gates *its* GuardrailMiddleware. The two are
  deliberately independent: ``GUARDRAIL_URL``/``GUARDRAIL_TIMEOUT`` are shared-by-name
  config (same root `deployment/modules/.env.example` values module 13 also reads,
  see its own `config.py`), so simply having a real ``GUARDRAIL_URL`` value present in
  the unified `.env` — which module 13 needs regardless of whether *this* hook should
  run — must not by itself turn this hook on. An earlier version of this hook used
  "is GUARDRAIL_URL set" as its only on/off signal; that meant populating the shared
  `.env` block to enable module 13's middleware silently enabled this hook too, with
  no way to have one on and the other off. ``GUARDRAIL_ENABLED`` fixes that: this hook
  now needs both a URL *and* an explicit "yes, run" before it ever calls out.
  ``AIML_GUARDRAIL_ENABLED`` is also accepted, as a module-07-scoped alias for anyone
  who wants that read clearly in the shared root `.env` (module 07 has no prefix
  convention of its own the way module 13's `GATEWAY_*` fields do); bare
  ``GUARDRAIL_ENABLED`` wins when both are set, see ``_enabled()``.
- Any guardrail failure (network error, timeout, bad response) is **fail-open**:
  log a warning and let inference proceed — the guard must never take the LLM down.
- Synchronous ``requests`` call, since ``unified_inference`` is a sync endpoint.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = float(os.getenv("GUARDRAIL_TIMEOUT", "20.0"))  # seconds; fail-open past this


def guardrail_url() -> str | None:
    return os.getenv("GUARDRAIL_URL") or None


def _enabled() -> bool:
    # AIML_GUARDRAIL_ENABLED is a module-07-scoped alias for readability in the
    # shared root .env (this module's other settings have no prefix convention
    # of their own, unlike module 13's GATEWAY_-prefixed ones) -- bare
    # GUARDRAIL_ENABLED wins when both are set, same first-alias-wins precedence
    # module 13's own AliasChoices("GR_ENABLED", "GATEWAY_GR_ENABLED") uses.
    raw = os.getenv("GUARDRAIL_ENABLED")
    if raw is None:
        raw = os.getenv("AIML_GUARDRAIL_ENABLED", "false")
    return raw.strip().lower() in ("1", "true", "yes")


def check(content: Any, direction: str, task: str | None = None,
          request_id: str | None = None) -> dict[str, Any] | None:
    """POST /guardrail/policy/decide; return module 23's DecideResponse dict
    ({"decision": "allow"|"warn"|"redact"|"block", "reason": str,
    "policy_version": str|None, "request_id": str, "latency_ms": float,
    "redacted_content": str|None, "categories": [...]}), or None.

    None means "no decision" — either the hook is disabled (GUARDRAIL_ENABLED isn't
    true, or no URL), the content isn't checkable text, or the call failed
    (fail-open, including decide's own 503 guardrail_unavailable). Callers must
    treat None as pass-through. `task` isn't sent -- DecideRequest has no field
    for it; `module`/`request_id` are, so module 23's audit log can attribute a
    decision to its caller.
    """
    base = guardrail_url()
    if not _enabled() or not base or not isinstance(content, str) or not content.strip():
        return None
    try:
        resp = requests.post(
            f"{base}/guardrail/policy/decide",
            json={"content": content, "direction": direction, "module": "07", "request_id": request_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        if not isinstance(result, dict):
            raise ValueError(f"expected a decision object, got {type(result).__name__}")
    except Exception as exc:  # noqa: BLE001 -- fail-open contract covers any guard-service failure, not just network errors (see tests/test_guardrail_hook.py)
        logger.warning("[guardrail] %s check failed, allowing inference (fail-open): %s",
                       direction, exc)
        return None
    if result.get("decision") == "warn":
        logger.warning("[guardrail] %s flagged (warn), allowing inference: %s",
                       direction, result.get("reason"))
    return result


def is_blocked(result: dict[str, Any] | None) -> bool:
    return bool(result) and result.get("decision") == "block"


def blocked_category(result: dict[str, Any]) -> str | None:
    """The guard model's category codes, comma-joined (what PolicyViolationError
    reported before the move to decide), or None when there are none -- e.g. a
    rule-engine policy block, or a guard model with no per-category ids."""
    return ",".join(result.get("categories") or []) or None


def _redacted(result: dict[str, Any] | None) -> str | None:
    if not result or result.get("decision") != "redact":
        return None
    text = result.get("redacted_content")
    return text if isinstance(text, str) else None


def extract_input_text(data: dict[str, Any]) -> str | None:
    """Pull the text a pre-inference check should look at out of an
    InferenceRequest's ``data``. text-generation accepts two real shapes at
    the adapter level (see ``adapters/ollama.py``'s ``"messages" in data``
    branch) -- a bare prompt string under ``inputs``, or chat-style
    ``messages`` -- and until this existed, ``check()`` was only ever called
    with ``data.get("inputs")`` directly, so any caller using the
    ``messages`` shape had its pre-check silently no-op (``check()`` treats
    non-str content as "nothing to check", not an error).

    Non-text tasks (vlm/asr/ocr/...) have no ``inputs``/``messages`` field at
    all, so this returns ``None`` for them same as before -- unchanged
    behavior, this only fixes the text-generation chat-shape gap."""
    if isinstance(data.get("inputs"), str):
        return data["inputs"]
    messages = data.get("messages")
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
    return None


def extract_output_text(result: Any) -> str | None:
    """Pull the text a post-inference check should look at out of
    ``inference_service.infer()``'s ``result`` field (itself a dict, e.g.
    ``{"response": ...}`` for Ollama/most HF text-generation paths, or
    ``{"text": ...}`` for a few HF paths -- see ``adapters/hf_transformers.py``).

    Before this existed, ``check()`` was called with that whole dict, which
    always fails ``isinstance(content, str)`` and silently no-ops -- so the
    post-inference check has never actually fired for any ``/inference/infer``
    text-generation call, module 21's or anyone else's. Non-text tasks still
    have nothing this can extract, so this returns ``None`` for them, same
    fail-open no-op as today."""
    if not isinstance(result, dict):
        return None
    for key in ("response", "text"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    return None


def apply_input_redaction(data: dict[str, Any], result: dict[str, Any] | None) -> bool:
    """On a ``redact`` decision, put decide's ``redacted_content`` back where
    ``extract_input_text`` found the checked text (``inputs``, or the last user
    message), in place. Returns whether anything was replaced."""
    text = _redacted(result)
    if text is None:
        return False
    if isinstance(data.get("inputs"), str):
        data["inputs"] = text
        return True
    messages = data.get("messages")
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = text
                return True
    return False


def apply_output_redaction(result: Any, decision: dict[str, Any] | None) -> bool:
    """On a ``redact`` decision, put decide's ``redacted_content`` back where
    ``extract_output_text`` found the checked text (``response`` or ``text``),
    in place. Returns whether anything was replaced."""
    text = _redacted(decision)
    if text is None or not isinstance(result, dict):
        return False
    for key in ("response", "text"):
        if isinstance(result.get(key), str):
            result[key] = text
            return True
    return False
