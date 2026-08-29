from __future__ import annotations

import json
import logging

import httpx
from pydantic import BaseModel, field_validator

from app.core.config import get_settings

_logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a content safety classifier. "
    "Analyze the following content and respond ONLY with valid JSON, no other text.\n\n"

    "Categories:\n"
    "  harmful     – violence, weapons, explosives, hacking, illegal drug synthesis or instructions\n"
    "  nsfw        – sexual or explicit adult content\n"
    "  hate_speech – racism, discrimination, targeted harassment against groups\n"
    "  self_harm   – suicide methods, self-harm instructions\n"
    "  none        – safe, benign, informational content (including law, general knowledge, and education)\n\n"

    "Field definitions:\n"
    "  category    – the most appropriate category above\n\n"

    "  confidence  – risk score from 0.00 to 1.00 indicating likelihood and severity of policy violation:\n"
    "    Must be selected in 0.05 increments only (e.g., 0.00, 0.05, 0.10 ... 1.00)\n\n"

    "    0.00: definitely safe (purely informational, legal, educational, no risk)\n"
    "    0.05: negligible risk (sensitive terms mentioned in clearly safe context)\n"
    "    0.10: very low risk (historical or neutral references to sensitive topics)\n"
    "    0.15: minimal risk (adjacent topics with no actionable content)\n"
    "    0.20: low risk (related to sensitive domains but purely descriptive)\n"
    "    0.25: low-moderate risk (could be misinterpreted but still non-actionable)\n"
    "    0.30: mild gray zone (conceptual description of sensitive domains)\n"
    "    0.35: gray zone leaning safe (near sensitive content but still abstract)\n"
    "    0.40: moderate gray zone (closer to potentially misusable topics)\n"
    "    0.45: upper gray zone (dual-use concepts, unclear but non-operational)\n"
    "    0.50: ambiguous / dual-use content (uncertain intent or interpretation)\n"
    "    0.55: slight risk leaning (possible misuse potential but still abstract)\n"
    "    0.60: moderate risk (concepts that could enable misuse in theory)\n"
    "    0.65: elevated risk (early-stage procedural or structured knowledge)\n"
    "    0.70: high risk (partial procedural or operationally relevant content)\n"
    "    0.75: high risk (close to actionable guidance)\n"
    "    0.80: very high risk (near-operational instructions)\n"
    "    0.85: critical risk (almost complete actionable guidance)\n"
    "    0.90: severe risk (near fully executable harmful instructions)\n"
    "    0.95: extreme risk (nearly complete harmful method)\n"
    "    1.00: severe violation (explicit, complete harmful instructions)\n\n"

    "  Rule constraints:\n"
    "    - If category == 'none', confidence MUST be 0.00\n"
    "    - confidence MUST strictly follow 0.05 step increments\n"
    "    - 0.50 is the default value for ambiguity (dual-use or unclear intent)\n"
    "    - Only assign confidence > 0.00 if any plausible risk signal exists\n\n"

    "  reason      – one sentence explanation in the same language as input; "
    "return empty string if category is 'none'\n\n"

    "JSON schema:\n"
    '{"category": "harmful|nsfw|hate_speech|self_harm|none", '
    '"confidence": 0.00-1.00, '
    '"reason": "string"}'
)

def _extract_json(raw: str) -> dict:
    """Extract the first valid JSON object from raw LLM output.

    Handles markdown code fences (Gemma), <think> tags (Qwen3),
    leading prose, and plain JSON — model-agnostic.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no valid JSON object in checker response: {raw!r}")


class CheckerUnavailableError(Exception):
    """Raised when the guardrails checker cannot produce a result."""


class OllamaResult(BaseModel):
    category: str = "none"
    confidence: float = 0.0
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        return max(0.0, min(1.0, float(v)))


_checker_client: httpx.AsyncClient | None = None


def get_checker_client() -> httpx.AsyncClient:
    global _checker_client
    if _checker_client is None or _checker_client.is_closed:
        _checker_client = httpx.AsyncClient(timeout=httpx.Timeout(35.0))
    return _checker_client


async def close_checker_client() -> None:
    global _checker_client
    if _checker_client and not _checker_client.is_closed:
        await _checker_client.aclose()


async def check(content: str) -> OllamaResult:
    settings = get_settings()
    payload = {
        "model": settings.guardrails_model,
        "prompt": f"<content>{content[:3000]}</content>",
        "system": _SYSTEM_PROMPT,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 150},
    }
    _logger.info("checker ← content | %s", content[:500])
    try:
        resp = await get_checker_client().post(
            f"{settings.real_ollama_url}/api/generate",
            json=payload,
            timeout=35.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        _logger.info("checker → raw | %s", raw)
        return OllamaResult(**_extract_json(raw))
    except Exception as exc:
        _logger.error("Guardrails check failed: %s", exc)
        raise CheckerUnavailableError(str(exc)) from exc
