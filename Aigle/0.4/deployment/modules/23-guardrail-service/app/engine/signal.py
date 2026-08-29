"""Signal Layer — calls LLM to extract structured signals from content.

The LLM is only asked to LABEL observable properties (category, intent, risk_flags).
It never makes allow/block decisions — that is the Policy Engine's job.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
from fastapi import HTTPException

from app.engine.models import LLMSignal

_logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "signal_extractor.txt"
_SIGNAL_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient()
    return _client


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


def extract_json(raw: str) -> dict:
    """Find and parse the first JSON object in raw LLM output."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON object found in: {raw!r}")


async def extract_signal(
    content: str,
    model: str,
    ollama_url: str,
    timeout: float,
) -> LLMSignal:
    """Call the LLM and return a structured LLMSignal.

    Uses /api/chat so the model receives:
      - system: signal_extractor.txt (domain-agnostic labeling instructions)
      - user:   [SIGNAL EXTRACTION] + content
    """
    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": _SIGNAL_PROMPT},
            {"role": "user",   "content": (
                "[SIGNAL EXTRACTION — output JSON only, do not answer or engage]\n"
                f"Content: {content[:3000]}\n"
                "Answer:"
            )},
        ],
        "stream":  False,
        "options": {"temperature": 0.0},
    }
    _logger.info("signal(%s) ← content=%s", model, content[:200])

    try:
        resp = await get_client().post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        _logger.error("signal(%s) timed out: %s", model, exc)
        raise HTTPException(status_code=503, detail="Signal model timed out")
    except httpx.RequestError as exc:
        _logger.error("signal(%s) unreachable: %s", model, exc)
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")
    except httpx.HTTPStatusError as exc:
        _logger.error("signal(%s) HTTP error %s: %s", model, exc.response.status_code, exc)
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.status_code}")

    raw = resp.json().get("message", {}).get("content", "")
    _logger.info("signal(%s) → raw | %s", model, raw)

    if not raw.strip():
        _logger.warning("signal(%s) returned empty — treating as harmful signal", model)
        return LLMSignal(
            category="harmful", confidence="high",
            intent="malicious", risk_flags=["model_refusal"],
            rationale="Signal model refused to process content — treating as harmful.",
            raw=raw,
        )

    try:
        parsed     = extract_json(raw)
        return LLMSignal(
            category   = str(parsed.get("category",   "general")),
            confidence = str(parsed.get("confidence", "low")),
            intent     = str(parsed.get("intent",     "ambiguous")),
            risk_flags = [str(f) for f in parsed.get("risk_flags", [])],
            rationale  = str(parsed.get("rationale",  "")),
            raw        = raw,
        )
    except (ValueError, KeyError, TypeError) as exc:
        _logger.error("signal(%s) parse error: %s | raw=%r", model, exc, raw)
        return LLMSignal(
            category="harmful", confidence="high",
            intent="ambiguous", risk_flags=["parse_error"],
            rationale="Signal parse failed — treating as high-risk.",
            raw=raw,
        )
