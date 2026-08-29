"""Unit tests for the pipeline executor (BM-4) with httpx mocked."""
from __future__ import annotations

import httpx
import pytest

from app.services import executor


class _FakeResponse:
    def __init__(self, json_data, status_code=200, raise_exc=None):
        self._json = json_data
        self.status_code = status_code
        self.content = b"x"
        self._raise_exc = raise_exc

    def json(self):
        return self._json

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc


class _FakeClient:
    """Async context manager standing in for httpx.AsyncClient."""

    def __init__(self, response=None, post_exc=None, **kwargs):
        self._response = response
        self._post_exc = post_exc
        self.last_url = None
        self.last_json = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.last_url = url
        self.last_json = json
        if self._post_exc:
            raise self._post_exc
        return self._response


def _patch_client(monkeypatch, response=None, post_exc=None):
    def _factory(*args, **kwargs):
        return _FakeClient(response=response, post_exc=post_exc)

    monkeypatch.setattr(executor.httpx, "AsyncClient", _factory)


async def test_chat_pipeline_normalizes_output(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"response": "hello there"}))
    result = await executor.call_pipeline("chat", {"message": "hi", "user_id": "u1"})
    assert result["output"] == "hello there"
    assert result["latency_ms"] >= 0.0
    assert result["raw_response"] == {"response": "hello there"}
    assert "error" not in result


async def test_search_pipeline_joins_results(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"results": [{"text": "a"}, {"text": "b"}]}))
    result = await executor.call_pipeline("search", {"query": "q", "top_k": 2, "user_id": "u1"})
    assert "a" in result["output"] and "b" in result["output"]


async def test_search_pipeline_requires_a_branch_id(monkeypatch):
    """No input.branch_id/user_id and no run_branch_id -- Module 25's search is
    per-user, there's no shared corpus to silently fall back to."""
    _patch_client(monkeypatch, _FakeResponse({"results": []}))
    result = await executor.call_pipeline("search", {"query": "q"})
    assert result["output"] == ""
    assert "branch_id" in result["error"]


async def test_search_pipeline_falls_back_to_run_branch_id(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"results": [{"text": "a"}]}))
    result = await executor.call_pipeline("search", {"query": "q"}, run_branch_id="submitter-sub")
    assert "a" in result["output"]


async def test_rag_pipeline_extracts_answer(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"answer": "42"}))
    result = await executor.call_pipeline("rag", {"question": "meaning?", "user_id": "u1"})
    assert result["output"] == "42"


async def test_rag_pipeline_requires_a_branch_id(monkeypatch):
    """No input.branch_id/user_id and no run_branch_id -- Module 21's /query
    treats a missing X-Branch-ID as "no results" rather than erroring, which
    would otherwise look like a normal completed run with an ungrounded
    answer instead of a clear failure."""
    _patch_client(monkeypatch, _FakeResponse({"answer": "should not be reached"}))
    result = await executor.call_pipeline("rag", {"question": "meaning?"})
    assert result["output"] == ""
    assert "branch_id" in result["error"]


async def test_rag_pipeline_falls_back_to_run_branch_id(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"answer": "42"}))
    result = await executor.call_pipeline("rag", {"question": "meaning?"}, run_branch_id="submitter-sub")
    assert result["output"] == "42"


async def test_classify_pipeline_extracts_intent(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"intent": "search"}))
    result = await executor.call_pipeline("classify", {"query": "find docs"})
    assert result["output"] == "search"


async def test_local_infer_injects_model_path_from_config_override(monkeypatch):
    captured = {}

    class _CapClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse({"output": "Paris", "latency_ms": 5.0})

    monkeypatch.setattr(executor.httpx, "AsyncClient", lambda *a, **k: _CapClient())
    result = await executor.call_pipeline(
        "local_infer",
        {"inputs": "capital of France?"},
        config_override={"model_path": "/models/ft123", "temperature": 0.1},
    )
    assert result["output"] == "Paris"
    assert "inference/infer" in captured["url"]
    assert captured["json"]["model_path"] == "/models/ft123"   # from config_override
    assert captured["json"]["inputs"] == "capital of France?"  # from test-case input
    assert captured["json"]["temperature"] == 0.1


async def test_lifecycle_infer_builds_module07_body_and_extracts_response(monkeypatch):
    captured = {}

    class _CapClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            # Module 07 shape: the completion is nested under result.response.
            return _FakeResponse({"result": {"response": "Paris.", "metadata": {}},
                                  "engine": "hf-transformers"})

    monkeypatch.setattr(executor.httpx, "AsyncClient", lambda *a, **k: _CapClient())
    result = await executor.call_pipeline(
        "lifecycle_infer",
        {"inputs": "capital of France?"},
        config_override={"model_name": "qwen3-0.6B", "engine": "transformers", "temperature": 0.7},
    )
    # nested result.response is extracted, not the stringified dict
    assert result["output"] == "Paris."
    assert captured["url"].endswith("/inference/infer")
    body = captured["json"]
    assert body["task"] == "text-generation"
    assert body["engine"] == "transformers"                 # from config_override
    assert body["model_name"] == "qwen3-0.6B"               # from config_override
    assert body["data"]["inputs"] == "capital of France?"   # from test-case input
    assert body["options"]["temperature"] == 0.7            # transformers needs > 0


async def test_lifecycle_infer_defaults_engine_and_temperature(monkeypatch):
    captured = {}

    class _CapClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            captured["json"] = json
            return _FakeResponse({"result": {"response": "ok"}})

    monkeypatch.setattr(executor.httpx, "AsyncClient", lambda *a, **k: _CapClient())
    # no engine/temperature in override → fall back to config defaults
    await executor.call_pipeline("lifecycle_infer", {"inputs": "hi"},
                                 config_override={"model_name": "qwen3-0.6B"})
    assert captured["json"]["engine"] == "transformers"     # default
    assert captured["json"]["options"]["temperature"] > 0.0  # default, strictly positive


async def test_error_does_not_raise(monkeypatch):
    _patch_client(monkeypatch, post_exc=httpx.ConnectError("boom"))
    result = await executor.call_pipeline("chat", {"message": "hi"})
    assert result["output"] == ""
    assert result["error"]
    assert result["latency_ms"] >= 0.0


async def test_unsupported_pipeline(monkeypatch):
    result = await executor.call_pipeline("nope", {})
    assert result["output"] == ""
    assert "unsupported" in result["error"]


async def test_unload_local_infer_posts_model_path(monkeypatch):
    captured = {}

    class _CapClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse({"unloaded": True, "model_path": "/models/ft123"})

    monkeypatch.setattr(executor.httpx, "AsyncClient", lambda *a, **k: _CapClient())
    await executor.unload_local_infer(model_path="/models/ft123")
    assert captured["url"].endswith("/api/v1/inference/unload")
    assert captured["json"] == {"model_path": "/models/ft123"}


async def test_unload_local_infer_swallows_errors(monkeypatch):
    _patch_client(monkeypatch, post_exc=httpx.ConnectError("down"))
    # Must not raise — freeing VRAM is best-effort.
    await executor.unload_local_infer(model_path="/models/ft123")
