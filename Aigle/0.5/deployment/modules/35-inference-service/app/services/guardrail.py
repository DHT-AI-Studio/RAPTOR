"""Optional Module 23 checks; infrastructure failures fail open (INF-9).

Module 23's canonical decision API is `POST /guardrail/policy/decide` (GRD-X1),
which answers allow | warn | redact | block. A path pointed back at the
deprecated `/guard/check/*` (a `safe` verdict) is still understood.
"""
from __future__ import annotations

import logging
import uuid

import httpx

from app.config import Settings
from app.errors import ServiceError

logger = logging.getLogger(__name__)

_DECISIONS = {"allow", "warn", "redact", "block"}


class Guardrail:
    """Reject a block (or unsafe) verdict, apply a redaction, log unavailable or malformed checks."""

    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self.http = http
        self.settings = settings

    async def _check(self, text: str, direction: str) -> str:
        """Return the text to carry on with: `text` itself, or decide's
        `redacted_content` on a redact decision."""
        if not self.settings.guardrail_enabled:
            return text
        path = getattr(self.settings, f"guardrail_{direction}_path")
        try:
            response = await self.http.post(
                self.settings.module23_url + path,
                json={"content": text, "direction": direction, "module": "35", "request_id": str(uuid.uuid4())},
                timeout=self.settings.guardrail_timeout_s,
            )
            response.raise_for_status()
            verdict = response.json()
            if not isinstance(verdict, dict):
                raise TypeError("expected a verdict object")
            decision = verdict.get("decision")
            if decision is not None:
                if decision not in _DECISIONS:
                    raise ValueError(f"unknown decision {decision!r}")
                unsafe = decision == "block"
            else:  # the deprecated /guard/check/* shape
                unsafe = verdict.get("safe") is False or verdict.get("verdict") == "unsafe"
                if not unsafe and verdict.get("safe") is not True and verdict.get("verdict") != "safe":
                    raise ValueError("missing decision or safe/unsafe verdict")
        except (httpx.HTTPError, ValueError, TypeError):
            logger.warning("Module 23 %s check unavailable; proceeding with inference", direction)
            return text
        if unsafe:
            reason = verdict.get("reason")
            if not isinstance(reason, str) or not reason:
                categories = verdict.get("categories")
                reason = ", ".join(str(category) for category in categories) if isinstance(categories, list) else ""
            raise ServiceError(
                400, "guardrail_rejection", reason=reason or "unsafe content", direction=direction
            )
        if decision == "warn":
            logger.warning("Module 23 flagged %s (warn); proceeding: %s", direction, verdict.get("reason"))
        if decision == "redact":
            redacted = verdict.get("redacted_content")
            if isinstance(redacted, str):
                return redacted
            logger.warning("Module 23 %s redact decision has no redacted_content; proceeding unredacted", direction)
        return text

    async def check_input(self, text: str) -> str:
        """Check text before Module 07 generation; returns the text to generate from."""
        return await self._check(text, "input")

    async def check_output(self, text: str) -> str:
        """Check generated text only when the output switch is enabled; returns the text to return."""
        if self.settings.guardrail_check_output:
            return await self._check(text, "output")
        return text
