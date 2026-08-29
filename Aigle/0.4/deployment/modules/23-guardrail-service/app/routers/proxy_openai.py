"""Reverse proxy for Ollama OpenAI-compatible API (/v1/*).

Not currently mounted in app/main.py — this router exists and works, but
its routes aren't reachable until it's added to an `app.include_router(...)`
call there.
"""
from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.services.proxy_checker import CheckerUnavailableError
from app.services.proxy_shared import evaluate, forward_client

_logger = logging.getLogger(__name__)

router = APIRouter()

_BLOCKED_CHOICE = {
    "index": 0,
    "message": {"role": "assistant", "content": ""},
    "finish_reason": "stop",
}


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> JSONResponse:
    settings = get_settings()
    raw = await request.body()
    if not raw:
        return JSONResponse({"error": "Request body is empty"}, status_code=400)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)

    messages = body.get("messages", [])
    user_content = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    try:
        if user_content:
            result = await evaluate(user_content, "prompt")
            if not result.allowed:
                blocked = dict(_BLOCKED_CHOICE)
                blocked["message"] = {"role": "assistant", "content": result.reason}
                return JSONResponse({
                    "id": "guardrails-blocked",
                    "object": "chat.completion",
                    "choices": [blocked],
                    "model": body.get("model", ""),
                })
    except CheckerUnavailableError:
        return JSONResponse({"error": {"message": "安全檢查服務暫時無法使用，請稍後再試。", "type": "service_unavailable"}}, status_code=503)

    body["stream"] = False
    try:
        resp = await forward_client().post(
            f"{settings.real_ollama_url}/v1/chat/completions",
            json=body,
            timeout=120.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.response.status_code)

    result_body = resp.json()

    try:
        response_text = result_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        response_text = ""

    try:
        if response_text:
            chk = await evaluate(response_text, "response")
            if not chk.allowed:
                result_body["choices"][0]["message"]["content"] = (
                    chk.reason or "此回應因內容政策無法顯示。"
                )
    except CheckerUnavailableError:
        return JSONResponse({"error": {"message": "安全檢查服務暫時無法使用，請稍後再試。", "type": "service_unavailable"}}, status_code=503)

    return JSONResponse(result_body)
