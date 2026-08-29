"""Unit tests for POST /guardrail/prompt_test (app/routers/guardrail.py) —
verifies the prompt/model are relayed to Ollama unmodified and its response
is returned unmodified (except the non-human-readable `context` token-id
array, which is stripped), with no guard prompt / policy / parsing involved."""
import httpx
import pytest
from fastapi import HTTPException

from app.models.prompt_test import PromptTestRequest
from app.routers.guardrail import prompt_test

pytestmark = pytest.mark.asyncio


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


async def test_prompt_test_sends_prompt_and_model_verbatim(monkeypatch):
    captured = {}

    async def fake_post(self, url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"model": "llama-guard3:8b", "response": "hello back", "done": True, "eval_count": 7})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await prompt_test(PromptTestRequest(prompt="hi there", model="llama-guard3:8b"))

    assert captured["url"].endswith("/api/generate")
    assert captured["json"] == {"model": "llama-guard3:8b", "prompt": "hi there", "stream": False}
    # Ollama's response is relayed as-is (no `context` field here to strip).
    assert result == {"model": "llama-guard3:8b", "response": "hello back", "done": True, "eval_count": 7}


async def test_prompt_test_strips_non_human_readable_context_token_ids(monkeypatch):
    async def fake_post(self, url, json, timeout):
        return FakeResponse({
            "model": "llama-guard3:8b", "response": "safe", "done": True,
            "context": [1058, 3382, 29901, 4451, 2],   # raw token ids — not human-readable
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await prompt_test(PromptTestRequest(prompt="hi", model="llama-guard3:8b"))

    assert "context" not in result
    assert result == {"model": "llama-guard3:8b", "response": "safe", "done": True}


async def test_prompt_test_raises_503_on_timeout(monkeypatch):
    async def fake_post(self, url, json, timeout):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(HTTPException) as exc_info:
        await prompt_test(PromptTestRequest(prompt="hi", model="m"))
    assert exc_info.value.status_code == 503


async def test_prompt_test_raises_503_on_unreachable(monkeypatch):
    async def fake_post(self, url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(HTTPException) as exc_info:
        await prompt_test(PromptTestRequest(prompt="hi", model="m"))
    assert exc_info.value.status_code == 503


async def test_prompt_test_raises_502_on_ollama_error_status(monkeypatch):
    async def fake_post(self, url, json, timeout):
        return FakeResponse({}, status_code=500)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(HTTPException) as exc_info:
        await prompt_test(PromptTestRequest(prompt="hi", model="m"))
    assert exc_info.value.status_code == 502
