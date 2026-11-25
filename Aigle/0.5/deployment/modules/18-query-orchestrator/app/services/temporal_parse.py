"""Module 25's `parse` operation as the query's only date reader (TMP-7, temporal plan §5.2 / §5.11).

Module 18 reads a question's time condition through Module 25 once that is proven available: its
capabilities say PD_TEMPORAL_V2 is on (`temporal_v2`) and advertise the `parse` route. Until then the
interim regex (signal_extractor, TMP-2) stays. Once in use, a `parse` call that times out or fails makes
the query UNRESOLVED -- it is never answered without its time condition, and no regex result stands in
for it (Appendix D F2).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.core.config import get_settings

_logger = logging.getLogger(__name__)


class ParseFailure(Exception):
    """`parse` did not answer usably: a timeout, a connection error, an HTTP error or an unreadable reply."""


class TemporalParseClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checked_at = 0.0
        self._available = False

    def available(self) -> bool:
        """Whether Module 25 serves `parse` with PD_TEMPORAL_V2 on. Re-checked every parse_capability_ttl
        seconds; an unreachable Module 25 is "not proven available", so the interim regex stays."""
        settings = get_settings()
        with self._lock:
            if time.monotonic() - self._checked_at < settings.parse_capability_ttl:
                return self._available
            try:
                resp = httpx.get(f"{settings.personal_db_url}/personal/temporal/capabilities", timeout=3.0)
                resp.raise_for_status()
                caps = resp.json()
                ok = bool(caps.get("temporal_v2")) and bool((caps.get("operations") or {}).get("parse"))
            except (httpx.HTTPError, ValueError, AttributeError) as exc:
                _logger.warning("Module 25 capabilities unavailable (%s): the interim regex stays", exc)
                ok = False
            if ok != self._available:
                _logger.info("temporal parse %s", "in use (Module 25)" if ok else "not available: interim regex")
            self._available, self._checked_at = ok, time.monotonic()
            return ok

    def parse(self, text: str) -> dict[str, Any]:
        """Module 25's reading of *text*, in the deployment's zone, with deictic phrases ("last year", 去年)
        read from now. Raises ParseFailure on a timeout, a connection error, any HTTP error or a reply
        without a status."""
        settings = get_settings()
        now = datetime.now(ZoneInfo(settings.timezone)).isoformat(timespec="seconds")
        try:
            resp = httpx.post(f"{settings.personal_db_url}/personal/temporal/parse",
                              json={"text": text, "timezone": settings.timezone, "anchor": now},
                              timeout=settings.parse_timeout_ms / 1000)
            resp.raise_for_status()
            out = resp.json()
        except httpx.TimeoutException as exc:
            raise ParseFailure(f"timeout after {settings.parse_timeout_ms} ms") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ParseFailure(str(exc) or type(exc).__name__) from exc
        if not isinstance(out, dict) or out.get("status") not in ("RESOLVED", "NO_TEMPORAL", "UNRESOLVED"):
            raise ParseFailure("reply without a status")
        return out


parse_client = TemporalParseClient()
