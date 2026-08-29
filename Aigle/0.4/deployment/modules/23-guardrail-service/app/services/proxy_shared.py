"""Shared evaluate() helper and forward HTTP client used by both proxy routers."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.core.config import get_settings

_logger = logging.getLogger(__name__)
from app.core.metrics import (
    guardrails_check_duration_seconds,
    guardrails_checks_total,
    guardrails_violations_total,
)
from app.services import proxy_checker

_forward_client: httpx.AsyncClient | None = None


def forward_client() -> httpx.AsyncClient:
    global _forward_client
    if _forward_client is None or _forward_client.is_closed:
        _forward_client = httpx.AsyncClient(
            timeout=httpx.Timeout(125.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _forward_client


async def close_forward_client() -> None:
    global _forward_client
    if _forward_client and not _forward_client.is_closed:
        await _forward_client.aclose()


@dataclass
class CheckResult:
    allowed: bool
    reason: str
    category: str = "none"
    confidence: float = 0.0


async def evaluate(content: str, check_type: str) -> CheckResult:
    settings = get_settings()
    with guardrails_check_duration_seconds.labels(check_type).time():
        ollama_result = await proxy_checker.check(content)

    is_violation = ollama_result.confidence >= settings.confidence_threshold

    guardrails_checks_total.labels(
        check_type,
        "violation" if is_violation else "pass",
        settings.guardrails_mode,
    ).inc()

    if is_violation:
        guardrails_violations_total.labels(check_type, ollama_result.category).inc()

    allowed = not (settings.guardrails_mode == "enforce" and is_violation)
    _logger.info(
        "guardrails check_type=%s confidence=%.2f category=%s allowed=%s | %s",
        check_type,
        ollama_result.confidence,
        ollama_result.category,
        allowed,
        ollama_result.reason or "-",
    )
    return CheckResult(
        allowed=allowed,
        reason=ollama_result.reason,
        category=ollama_result.category,
        confidence=ollama_result.confidence,
    )
