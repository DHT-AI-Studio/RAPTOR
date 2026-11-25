# services/image_orchestrator_service/image_moderation.py
"""CM-2: image moderation at ingestion -- one Module 23 image-guard call per image.

The orchestrator calls moderate_image() once the image is on local disk, alongside
the description and OCR requests. The returned moderation_flags are written to the
asset's Module 04 metadata and travel with the summary entry to Module 25's Source
record. Moderation only records a verdict; it never stops an unsafe image from
being indexed.

Settings (environment):

* IMAGE_MODERATION_ENABLED -- false (default): no Module 23 call at all.
* GUARDRAIL_URL -- Module 23's base URL, the shared name modules 07 and 13 read.
* IMAGE_MODERATION_MODEL -- "shieldgemma-2" (default) -> POST /infer/google-shieldgemma-2;
  "llama-guard-3-vision" -> POST /infer/Llama-Guard-3-11B-Vision.
* IMAGE_MODERATION_TIMEOUT -- seconds per call (default 180: the first call after
  Module 23 starts loads the model -- ~50 s for ShieldGemma 2 in 8-bit on an RTX 4090,
  ~0.4 s per image after that -- and Module 23 itself may wait for GPU memory).
* IMAGE_MODERATION_FAIL_MODE -- what an unreachable or failing Module 23 means. GRD-X3,
  which the ticket points to for this, does not define it yet; "open" (default) follows
  the platform's other guardrail calls (module 13's GuardrailMiddleware, module 07's
  hook): the image is indexed with verdict "error". "closed" fails the image's job.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
from moderation_flag_reporter import report_moderation_flags

logger = logging.getLogger(__name__)

_ENDPOINTS = {
    "shieldgemma-2": "/infer/google-shieldgemma-2",
    "llama-guard-3-vision": "/infer/Llama-Guard-3-11B-Vision",
}

# shieldgemma-2's raw policy scores have no per-policy threshold of their own (the
# "overall_verdict" is Module 23's own call); this is the bar a policy's own score
# must cross to count as one of the categories that caused that verdict, matching
# CM-1's default moderation threshold.
_FLAG_THRESHOLD = 0.5


class ModerationUnavailable(Exception):
    """Module 23 gave no verdict: unreachable, timed out, non-200, or an answer
    that isn't one of its documented shapes. Carries the flags to record when the
    fail mode is open."""

    def __init__(self, message: str, flags: dict[str, Any]):
        super().__init__(message)
        self.flags = flags


def enabled() -> bool:
    return os.environ.get("IMAGE_MODERATION_ENABLED", "false").strip().lower() == "true"


def fail_closed() -> bool:
    return os.environ.get("IMAGE_MODERATION_FAIL_MODE", "open").strip().lower() == "closed"


def _endpoint(model: str) -> str:
    if model not in _ENDPOINTS:
        raise ValueError(f"IMAGE_MODERATION_MODEL must be one of {sorted(_ENDPOINTS)}, not {model!r}")
    return _ENDPOINTS[model]


def _flags(model: str, verdict: str, started: float, **extra: Any) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "model": model,
        **extra,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse(model: str, body: Any) -> tuple[str, dict[str, Any]]:
    """(verdict, model-specific detail) from Module 23's answer; ValueError if the
    answer doesn't have the documented shape (module 23 README, "Vision guard")."""
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    if model == "shieldgemma-2":
        verdict = body.get("overall_verdict")
        policies = body.get("policies")
        if verdict not in ("safe", "unsafe") or not isinstance(policies, dict):
            raise ValueError("expected overall_verdict and policies")
        scores = {name: p.get("unsafe_probability") for name, p in policies.items() if isinstance(p, dict)}
        return verdict, {"policies": scores}
    verdict = body.get("label")
    categories = body.get("categories")
    if verdict not in ("safe", "unsafe") or not isinstance(categories, list):
        raise ValueError("expected label and categories")
    return verdict, {"categories": categories}


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


async def moderate_image(image_path: str, filename: str | None = None) -> dict[str, Any]:
    """The image's moderation_flags. Raises ModerationUnavailable when there is no
    verdict: Module 23 failed, the image can't be read, or the settings are invalid."""
    started = time.perf_counter()
    model = os.environ.get("IMAGE_MODERATION_MODEL", "shieldgemma-2").strip()
    try:
        url = os.environ.get("GUARDRAIL_URL", "http://raptor-guardrail-service:8026").rstrip("/") + _endpoint(model)
        timeout = float(os.environ.get("IMAGE_MODERATION_TIMEOUT", "180"))
        image_bytes = await asyncio.to_thread(_read_bytes, image_path)
        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename=filename or os.path.basename(image_path))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(url, data=form) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                body = await resp.json(content_type=None)
        verdict, detail = _parse(model, body)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as exc:  # asyncio's: the worker runs Python 3.10
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        flags = _flags(model, "error", started, error=reason)
        logger.warning(f"[moderation] {model} gave no verdict for {filename or image_path} "
                       f"after {flags['latency_ms']} ms: {reason}")
        raise ModerationUnavailable(reason, flags) from exc

    flags = _flags(model, verdict, started, **detail)
    logger.info(f"[moderation] {filename or image_path}: {verdict} ({model}, {flags['latency_ms']} ms)")
    return flags


async def report_flags(flags: dict[str, Any], asset_path: str, version_id: str, branch_id: str) -> None:
    """CM-5 / VIE01-408: report each category that made this verdict "unsafe" to
    Module 23's review store, alongside (not instead of) `flags` already written to
    Module 04/25 by the caller. No-op for "safe"/"error" -- nothing to review."""
    if flags.get("verdict") != "unsafe":
        return
    policies = flags.get("policies")
    if isinstance(policies, dict):
        scores = policies
        categories = [name for name, score in scores.items()
                      if isinstance(score, (int, float)) and score >= _FLAG_THRESHOLD] or list(scores.keys())
    else:
        scores = None
        categories = flags.get("categories") or []
    if not categories:
        return
    url = os.environ.get("GUARDRAIL_URL", "http://raptor-guardrail-service:8026")
    # Module 25's Source is keyed by (asset_path, version_id), not asset_path alone
    # (older versions can stay queryable after a newer one replaces them) -- version_id
    # travels as its own field (not folded into asset_path) so an admin reviewing this
    # flag, or Module 25's consumer looking it up, can use it directly.
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        await report_moderation_flags(
            session, url, branch_id=branch_id, version_id=version_id, categories=categories,
            model=flags.get("model", ""), scores=scores, asset_path=asset_path,
        )
