"""LLM-as-judge + embedding client for Module 07 (BM-6).

``evaluate`` grades an output against a rubric via the AI Lifecycle inference
endpoint and normalizes the raw score to [0, 1]. ``embed`` / ``embed_and_compare``
go through the same endpoint with ``task: "embedding"`` (used by the
cosine_similarity scoring method). All are network calls and are mocked in
unit tests.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ContentPolicyBlockedError(Exception):
    """Raised by complete()/_infer_embeddings() when module 07 returns its
    guardrail-block shape (HTTP 400, {"error_type": "PolicyViolationError",
    ...}) -- a permanent result: retrying the exact same content will never
    succeed. Not caught here (evaluate()/pairwise() already fail open on
    ANY single exception with no retry, so this needs no special handling
    in this file) -- exported so callers with their own retry loop
    (autotune/proposer.py) can treat it distinctly from a transient httpx
    error and give up immediately instead of wasting their remaining
    attempts."""


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "output", "text", "response", "generated_text", "content"):
            if key in payload and payload[key] is not None:
                return _extract_text(payload[key])
    if isinstance(payload, list) and payload:
        return _extract_text(payload[0])
    return str(payload)


async def _infer(inputs: str, model: str) -> str:
    """Short judge completion (max_length=16) — a thin alias over ``complete`` so
    the judge/pairwise scorers share the one Module 07 text-generation path."""
    return await complete(inputs, model=model, max_length=16)


async def complete(prompt: str, model: Optional[str] = None,
                   max_length: int = 256, temperature: float = 0.0,
                   timeout: Optional[int] = None) -> str:
    """Generic text completion via the Module 07 inference endpoint.

    Same path as the judge, but with caller-controlled length/timeout — the
    autotune proposer needs a longer output (JSON) than _infer's judge default
    (max_length=16). Kept here so all Module 07 LLM calls share one client.
    """
    settings = get_settings()
    model = model or settings.judge_model
    url = settings.inference_url.rstrip("/") + "/inference/infer"
    body = {
        "task": "text-generation",
        "engine": settings.judge_engine,
        "model_name": model,
        "data": {"inputs": prompt},
        "options": {"temperature": temperature, "max_length": max_length, "think": settings.judge_think},
    }
    async with httpx.AsyncClient(timeout=timeout or settings.inference_timeout) as client:
        resp = await client.post(url, json=body)
    if resp.status_code == 400:
        try:
            err_body = resp.json()
        except ValueError:
            err_body = {}
        # FastAPI's HTTPException(detail={...}) nests the payload under
        # "detail" -- it is never at the top level of the response body.
        detail = err_body.get("detail")
        if not isinstance(detail, dict):
            detail = {}
        if detail.get("error_type") == "PolicyViolationError":
            raise ContentPolicyBlockedError(
                f"內容被guardrail政策擋下 (category={detail.get('category')}, "
                f"direction={detail.get('direction')})"
            )
    resp.raise_for_status()
    return _extract_text(resp.json())


async def evaluate(
    rubric: str,
    output: str,
    expected_answer: Optional[str] = None,
    score_range: Sequence[float] = (1, 5),
    model: Optional[str] = None,
) -> float:
    """Grade ``output`` per ``rubric``; return a normalized score in [0, 1]."""
    settings = get_settings()
    model = model or settings.judge_model
    lo, hi = float(score_range[0]), float(score_range[1])

    reference = f"\nReference answer:\n{expected_answer}\n" if expected_answer else ""
    prompt = (
        f"You are a strict evaluator. {rubric}\n"
        f"Respond with ONLY a single integer between {int(lo)} and {int(hi)}.\n"
        f"{reference}"
        f"\nOutput to grade:\n{output}\n\nScore:"
    )

    try:
        text = await _infer(prompt, model)
    except Exception as exc:  # noqa: BLE001 — judge failures must not abort a run
        logger.warning("LLM judge call failed: %s", exc)
        return 0.0

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        logger.warning("LLM judge returned no parseable score: %r", text)
        return 0.0

    raw = float(match.group())
    if hi == lo:
        return 0.0
    normalized = (raw - lo) / (hi - lo)
    return max(0.0, min(1.0, normalized))


async def pairwise(
    task_input: str,
    output_a: str,
    output_b: str,
    criteria: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Ask the judge which answer is better; return 'A', 'B', or 'TIE'.

    LLMs give far more reliable relative ("which is better") judgements than
    absolute scores, which makes this the right primitive for before/after
    comparison. Position bias is handled by the caller (swap + agree).
    """
    settings = get_settings()
    model = model or settings.judge_model
    crit = f" Evaluation criteria: {criteria}." if criteria else ""
    prompt = (
        "You are comparing two answers to the same task."
        f"{crit}\n\n"
        f"Task:\n{task_input}\n\n"
        f"Answer A:\n{output_a}\n\n"
        f"Answer B:\n{output_b}\n\n"
        "Which answer is better? Reply with exactly one token: A, B, or TIE."
    )
    try:
        text = await _infer(prompt, model)
    except Exception as exc:  # noqa: BLE001 — a judge failure must not abort compare
        logger.warning("pairwise judge call failed: %s", exc)
        return "TIE"

    match = re.search(r"\b(TIE|A|B)\b", text.strip().upper())
    return match.group(1) if match else "TIE"


async def _infer_embeddings(texts: Sequence[str], model: Optional[str]) -> List[List[float]]:
    """Embed ``texts`` in one Module 07 call (``task: "embedding"``).

    The model (``bge-m3`` by default — multilingual, 1024-dim) must be
    registered in Module 07, e.g. POST /models/register_ollama with
    task="embedding".
    """
    settings = get_settings()
    url = settings.inference_url.rstrip("/") + "/inference/infer"
    body = {
        "task": "embedding",
        "model_name": model or settings.embed_model,
        "data": {"inputs": list(texts)},
    }
    async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
        resp = await client.post(url, json=body)
    resp.raise_for_status()
    result = resp.json().get("result") or {}
    return [[float(x) for x in vec] for vec in result.get("embeddings", [])]


async def embed(text: str, model: Optional[str] = None) -> List[float]:
    """Return the embedding vector of a single text via Module 07."""
    vectors = await _infer_embeddings([text], model)
    return vectors[0] if vectors else []


async def embed_and_compare(text_a: str, text_b: str, model: Optional[str] = None) -> float:
    """Cosine similarity of two texts as a float in [0, 1] (BM-6).

    One batched Module 07 call embeds both texts, then numpy dot product on
    the vectors. Transport errors propagate — the cosine_similarity scorer
    catches them and degrades to 0.0.
    """
    import numpy as np

    vectors = await _infer_embeddings([text_a, text_b], model)
    if len(vectors) != 2:
        raise ValueError(f"expected 2 embeddings from Module 07, got {len(vectors)}")
    va, vb = np.asarray(vectors[0], dtype=float), np.asarray(vectors[1], dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(va, vb) / (na * nb))))
