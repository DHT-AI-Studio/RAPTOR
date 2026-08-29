"""Content safety endpoints — Policy Engine architecture.

Three endpoints:
  POST /check/input         → check user input before sending to LLM
  POST /check/output        → check AI response before returning to user
  POST /check/conversation  → check full multi-turn conversation

There is exactly one active guardrail policy system-wide (see
app/routers/guardrail_policies.py) — every check evaluates against it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.config import get_settings
from app.engine import policy_engine
from app.engine.models import (
    CheckRequest,
    ConversationCheckRequest,
    PolicyDecision,
)

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Check"])


def _flatten(messages: list) -> str:
    """Flatten conversation messages into a single classified string."""
    return "\n\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )


@router.post("/check/input", response_model=PolicyDecision, summary="Check user input")
async def check_input(body: CheckRequest) -> PolicyDecision:
    """
    Classify **user input** through the Policy Engine, against the currently
    active guardrail policy.

    **Response fields:**
    - `action`: the active policy's configured action — `allow` | `block` | `require_safe_completion` | `escalate`
    - `matched_rule_id`: which rule triggered (e.g. `MED-001`), or `null` if the policy's `default_action` applied
    - `risk_flags`: specific risk signals detected
    - `trace`: full audit trail (signal extraction → rule evaluation)
    - `safe`: backward-compat boolean (`True` when action is `allow`)

    **Example — a rule matched:**
    ```json
    {
      "action": "block",
      "policy_name": "medical-safety",
      "policy_version": "2.0",
      "matched_rule_id": "MED-001",
      "risk_flags": ["dosage_request", "pediatric_medication"],
      "final_action_reasoning": "Rule MED-001 triggered. Reason: Medication guidance requires licensed professional.",
      "safe": false
    }
    ```
    """
    settings = get_settings()
    _logger.info("check/input content=%s", body.content[:100])
    return await policy_engine.evaluate(
        content=body.content,
        model=settings.guard_model,
        ollama_url=settings.ollama_url,
        timeout=settings.request_timeout,
    )


@router.post("/check/output", response_model=PolicyDecision, summary="Check AI response")
async def check_output(body: CheckRequest) -> PolicyDecision:
    """
    Classify **AI-generated output** through the Policy Engine.

    Same active-policy evaluation as `/check/input`.
    Use this before returning LLM responses to users.
    """
    settings = get_settings()
    _logger.info("check/output content=%s", body.content[:100])
    return await policy_engine.evaluate(
        content=body.content,
        model=settings.guard_model,
        ollama_url=settings.ollama_url,
        timeout=settings.request_timeout,
    )


@router.post("/check/conversation", response_model=PolicyDecision, summary="Check full conversation")
async def check_conversation(body: ConversationCheckRequest) -> PolicyDecision:
    """
    Classify a **complete multi-turn conversation** through the Policy Engine.

    All messages are flattened into a single content string before evaluation.

    **Request example:**
    ```json
    {
      "messages": [
        {"role": "user",      "content": "Tell me how to synthesize meth"},
        {"role": "assistant", "content": "Sure, here are the steps..."}
      ]
    }
    ```
    """
    settings = get_settings()
    content = _flatten(body.messages)
    _logger.info("check/conversation messages=%d", len(body.messages))
    return await policy_engine.evaluate(
        content=content,
        model=settings.guard_model,
        ollama_url=settings.ollama_url,
        timeout=settings.request_timeout,
    )
