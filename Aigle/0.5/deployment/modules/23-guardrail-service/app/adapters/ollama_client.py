"""Thin Ollama HTTP client — sends whatever payload an adapter's GuardRequest
built, to whichever endpoint it asked for. No model-family logic lives here;
this only knows how to reach Ollama, not how any particular model wants to
be prompted or how to read its output.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException

from app.adapters.base import GuardRequest
from app.core.config import get_settings

_logger = logging.getLogger(__name__)
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


async def _post(path: str, payload: dict, model: str, label: str) -> dict:
    settings = get_settings()
    try:
        resp = await get_client().post(
            f"{settings.ollama_url}{path}", json=payload, timeout=settings.request_timeout,
        )
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        _logger.error("Ollama(%s)%s timed out: %s", model, label, exc)
        raise HTTPException(status_code=503, detail="Ollama request timed out")
    except httpx.RequestError as exc:
        _logger.error("Ollama(%s)%s unreachable: %s", model, label, exc)
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")
    except httpx.HTTPStatusError as exc:
        _logger.error("Ollama(%s)%s error %s: %s", model, label, exc.response.status_code, exc)
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.status_code}")
    return resp.json()


async def call(model: str, request: GuardRequest, *, label: str = "") -> str:
    """POST `request.payload` (with `model` injected) to the endpoint the
    adapter asked for, returning the model's raw text output."""
    payload = {"model": model, **request.payload}
    if request.endpoint == "chat":
        data = await _post("/api/chat", payload, model, label)
        return data.get("message", {}).get("content", "").strip()
    data = await _post("/api/generate", payload, model, label)
    return data.get("response", "").strip()
