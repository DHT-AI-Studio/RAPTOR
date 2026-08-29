"""Guardrail hook for Module 07 inference (GB-6).

Calls the Guardrail Service's raw guard-model checker (`/guard/check/{input,output}`)
before and after LLM inference -- the same group module 13's GuardrailMiddleware
uses (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard's own built-in safety
taxonomy), not the policy-engine group (`/guardrail/check/*`). Switched from the
policy-engine group per a direct ask: the policy engine only does anything when
module 23 has an active policy configured, which it never has in practice so far,
and even when a policy exists, its `redact` action is a no-op for anything the
guard-model detector (rather than a regex rule) flags -- there's no matched text
span to redact from a bare safe/unsafe verdict. The one thing genuinely traded
away by this switch is retry-worthy: `/guard/check/*` writes no audit trail of
its own the way the policy engine does (`guardrail_violations`, gated per-policy)
-- module 23's `/guard/check/*` handlers were extended to write one anyway
(policy_id=None), so this switch doesn't lose that after all.

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
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = float(os.getenv("GUARDRAIL_TIMEOUT", "20.0"))  # seconds; fail-open past this


def guardrail_url() -> Optional[str]:
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


def check(content: Any, direction: str, task: Optional[str] = None,
          request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """POST /guard/check/{direction}; return module 23's CheckResponse dict
    ({"safe": bool, "categories": [...], "category_names": {...}, "raw": str,
    "conflict": bool|None, "results": [...]|None}), or None.

    None means "no decision" — either the hook is disabled (GUARDRAIL_ENABLED isn't
    true, or no URL), the content isn't checkable text, or the call failed
    (fail-open). Callers must treat None as pass-through. `task` isn't sent --
    this group's MessageRequest has no field for it (it's a per-task concept from
    the policy-engine group's scope matching, meaningless to a bare guard-model
    verdict); `module`/`request_id` are, purely so module 23's audit log can
    attribute a violation to its caller.
    """
    base = guardrail_url()
    if not _enabled() or not base or not isinstance(content, str) or not content.strip():
        return None
    try:
        resp = requests.post(
            f"{base}/guard/check/{direction}",
            json={"content": content, "module": "07", "request_id": request_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — network/timeout/HTTP/JSON all fail-open
        logger.warning("[guardrail] %s check failed, allowing inference (fail-open): %s",
                       direction, exc)
        return None


def extract_input_text(data: Dict[str, Any]) -> Optional[str]:
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


def extract_output_text(result: Any) -> Optional[str]:
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
