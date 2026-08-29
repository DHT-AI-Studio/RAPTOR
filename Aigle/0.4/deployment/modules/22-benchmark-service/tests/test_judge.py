"""Unit tests for the LLM judge client (BM-6) — httpx mocked at the transport level.

Covers the score-parsing edge cases (valid / non-numeric / timeout) and the
Module 07 embedding path that test_scorer.py mocks away.
"""
from __future__ import annotations

import httpx
import pytest

from app.services import judge


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=None)


class _FakeClient:
    """Async context manager standing in for httpx.AsyncClient."""

    last_request = None

    def __init__(self, response=None, post_exc=None):
        self._response = response
        self._post_exc = post_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        _FakeClient.last_request = {"url": url, "json": json}
        if self._post_exc is not None:
            raise self._post_exc
        return self._response


def _patch_client(monkeypatch, response=None, post_exc=None):
    def _factory(*args, **kwargs):
        return _FakeClient(response=response, post_exc=post_exc)

    monkeypatch.setattr(judge.httpx, "AsyncClient", _factory)


def _gen(text):
    """Module 07 /inference/infer envelope for a text-generation result."""
    return _FakeResponse({"result": {"response": text, "model": "m"}, "success": True})


# ── evaluate: score parsing ──────────────────────────────────────────

async def test_evaluate_valid_integer_score(monkeypatch):
    _patch_client(monkeypatch, _gen("4"))
    score = await judge.evaluate("Score clarity 1-5.", "some output")
    assert score == pytest.approx((4 - 1) / (5 - 1))


async def test_evaluate_extracts_first_number_from_prose(monkeypatch):
    _patch_client(monkeypatch, _gen("Score: 3.5 — fairly clear"))
    score = await judge.evaluate("Score clarity 1-5.", "some output")
    assert score == pytest.approx((3.5 - 1) / (5 - 1))


async def test_evaluate_non_numeric_response_is_zero(monkeypatch):
    _patch_client(monkeypatch, _gen("an excellent answer!"))
    assert await judge.evaluate("Score 1-5.", "out") == 0.0


async def test_evaluate_timeout_is_zero(monkeypatch):
    # Ruled 2026-07: judge timeouts degrade to 0.0 instead of raising, so one
    # slow test case cannot abort a whole benchmark run.
    _patch_client(monkeypatch, post_exc=httpx.TimeoutException("timed out"))
    assert await judge.evaluate("Score 1-5.", "out") == 0.0


async def test_evaluate_clamps_out_of_range_score(monkeypatch):
    _patch_client(monkeypatch, _gen("17"))
    assert await judge.evaluate("Score 1-5.", "out") == 1.0


async def test_evaluate_custom_score_range(monkeypatch):
    _patch_client(monkeypatch, _gen("5"))
    score = await judge.evaluate("Score 0-10.", "out", score_range=(0, 10))
    assert score == pytest.approx(0.5)


# ── embeddings via Module 07 ─────────────────────────────────────────

async def test_embed_uses_module07_embedding_task(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"result": {"embeddings": [[0.6, 0.8]]}}))
    vec = await judge.embed("hello")
    assert vec == [0.6, 0.8]
    req = _FakeClient.last_request
    assert req["url"].endswith("/inference/infer")
    assert req["json"]["task"] == "embedding"
    assert req["json"]["data"]["inputs"] == ["hello"]


async def test_embed_and_compare_batches_both_texts(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"result": {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}}))
    sim = await judge.embed_and_compare("a", "b")
    assert sim == 0.0  # orthogonal vectors
    assert _FakeClient.last_request["json"]["data"]["inputs"] == ["a", "b"]


async def test_embed_and_compare_identical_vectors(monkeypatch):
    _patch_client(monkeypatch, _FakeResponse({"result": {"embeddings": [[1.0, 1.0], [1.0, 1.0]]}}))
    assert await judge.embed_and_compare("a", "a") == pytest.approx(1.0)


async def test_embed_and_compare_transport_error_propagates(monkeypatch):
    # Unlike evaluate(), embedding errors propagate — the cosine_similarity
    # scorer catches them and degrades that dimension to 0.0.
    _patch_client(monkeypatch, post_exc=httpx.TimeoutException("timed out"))
    with pytest.raises(httpx.TimeoutException):
        await judge.embed_and_compare("a", "b")
