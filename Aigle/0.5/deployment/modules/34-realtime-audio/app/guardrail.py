"""RTA-X6 / RTA-08: Module 23 screens every voice turn -- the transcript before it reaches the
brain, and every answer sentence before it is synthesized.

POST {RT_GUARDRAIL_URL}/guardrail/policy/decide
    {"content", "direction": "input" | "output", "module": "34", "request_id", "user_id", "session_id"}
 -> {"decision": "allow" | "redact" | "warn" | "block", "reason", "redacted_content", ...}

(The backlog's ``task="voice_input"`` / ``output=true`` are Module 23's ``direction``.)

  allow   go on
  redact  go on with ``redacted_content`` instead of the original text
  warn    go on, logged; Module 15 gets ``X-Guardrail-Warning: true``
  block   input: GUARDRAIL_BLOCK and RT_GUARDRAIL_BLOCK_TEXT is spoken, Module 15 is not called;
          output: the rest of the answer is dropped and RT_GUARDRAIL_BLOCK_TEXT is spoken instead

Module 23 unreachable (503 guardrail_unavailable, timeout): RT_GUARDRAIL_FAIL_OPEN (true, as Module
15's GUARDRAIL_FALLBACK_PASSTHROUGH) lets the turn through with a warning in the log; false
treats it as a block.  Module 15 also screens its own LLM calls through Module 23's proxy; this
check is the one that covers what Module 34 itself forwards and speaks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

MODULE = "34"


@dataclass(frozen=True)
class Decision:
    decision: str                  # allow | redact | warn | block
    reason: str = ""
    text: str = ""                 # what to use downstream (redacted when decision == "redact")
    available: bool = True         # False: Module 23 did not answer

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @property
    def warned(self) -> bool:
        return self.decision == "warn"


class Guardrail:
    def __init__(self, base_url: str, client: httpx.AsyncClient, timeout_s: float = 5.0,
                 fail_open: bool = True) -> None:
        self.url = base_url.rstrip("/") + "/guardrail/policy/decide"
        self.client = client
        self.timeout_s = timeout_s
        self.fail_open = fail_open

    async def decide(self, content: str, direction: str, *, request_id: str, user_id: str = "",
                     session_id: str = "") -> Decision:
        body = {"content": content, "direction": direction, "module": MODULE, "request_id": request_id,
                "user_id": user_id or None, "session_id": session_id or None}
        try:
            resp = await self.client.post(self.url, json=body, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            decision = data.get("decision", "allow")
            text = (data.get("redacted_content") or content) if decision == "redact" else content
            return Decision(decision, str(data.get("reason", "")), text)
        except (httpx.HTTPError, ValueError) as exc:
            level = "allowed" if self.fail_open else "blocked"
            logger.warning("guardrail %s check of request %s unavailable (%s): %s", direction, request_id, exc, level)
            return Decision("allow" if self.fail_open else "block", "guardrail_unavailable", content, available=False)
