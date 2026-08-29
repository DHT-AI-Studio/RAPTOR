"""
Shared dedup check for long-term memory writes: embedding similarity
shortlists candidates, a second small LLM call judges DUPLICATE / UPDATE / NEW.

Used by both the manual `POST /memory/longterm/facts` endpoint and the LLM
extraction pipeline (`services/extractor.py`) so a fact written either way is
checked against the same existing memory.
"""
from __future__ import annotations

import json
import math
import re
from typing import Awaitable, Callable, Optional

import httpx

from core.config import settings

# Top-K candidates the recall step shortlists before the LLM judges them.
# A single cosine threshold can't do this alone — calibration against the
# real bge-m3 model showed contradicting preferences ("Chinese" vs "English",
# 0.92) score *higher* than genuine paraphrase-duplicates (0.78-0.82), so a
# fixed number would either miss reworded duplicates or silently drop real
# preference changes. The recall floor below only filters out obviously
# unrelated candidates before asking the LLM to make the actual call.
DEDUP_CANDIDATE_K = 3
DEDUP_RECALL_FLOOR = 0.55

DEDUP_SYSTEM_PROMPT = """\
You check whether a NEW memory item duplicates or contradicts EXISTING memory \
items belonging to the same user.

Given the NEW item and a list of EXISTING candidate items (with IDs), output \
exactly one line in one of these formats:
  DUPLICATE:<id>   NEW restates the same claim as EXISTING <id> (same meaning, different wording)
  UPDATE:<id>      NEW contradicts, refines, or supersedes EXISTING <id>'s value
  NEW              NEW is unrelated to all candidates, or asserts something none of them cover

No prose, no explanation — one line only.\
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def shortlist_candidates(text: str, existing_full: list[dict]) -> list[dict]:
    """Return up to DEDUP_CANDIDATE_K existing facts most similar to `text`.

    Pure recall step — casts a loose net so the LLM judgment call sees only
    plausible candidates, not the entire memory store.
    """
    if not existing_full:
        return []
    from services.memvid_store import _get_embedder
    import asyncio
    embedder = _get_embedder()
    texts = [f.get("text", "") for f in existing_full]
    vecs = await asyncio.to_thread(embedder.embed_documents, [text] + texts)
    query_vec, candidate_vecs = vecs[0], vecs[1:]
    scored = [
        (sim, f) for sim, f in (
            (_cosine_similarity(query_vec, v), f) for v, f in zip(candidate_vecs, existing_full)
        )
        if sim >= DEDUP_RECALL_FLOOR
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [f for _, f in scored[:DEDUP_CANDIDATE_K]]


def build_dedup_prompt(new_text: str, candidates: list[dict]) -> str:
    candidates_block = json.dumps(
        [{"id": c.get("frame_id", ""), "text": c.get("text", "")} for c in candidates],
        ensure_ascii=False,
    )
    return (
        f"{DEDUP_SYSTEM_PROMPT}\n\n"
        f"NEW item:\n{new_text}\n\n"
        f"EXISTING candidates:\n{candidates_block}\n\n"
        "Output one line only."
    )


async def default_dedup_llm(new_text: str, candidates: list[dict]) -> str:
    """Call Module 07 to classify NEW vs the shortlisted candidates. Returns raw response string."""
    prompt = build_dedup_prompt(new_text, candidates)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.module07_url.rstrip('/')}/inference/infer",
            json={
                "task": "text-generation",
                "engine": "ollama",
                "model_name": settings.extraction_model,
                "data": {"inputs": prompt},
                "options": {"temperature": 0, "max_tokens": 32, "clean_input": False, "think": settings.inference_think},
            },
        )
        resp.raise_for_status()
    body = resp.json()
    return body.get("result", {}).get("response", "").strip()


def parse_dedup_verdict(raw: str) -> tuple[str, str]:
    """Parse 'DUPLICATE:<id>' / 'UPDATE:<id>' / 'NEW' -> (verdict, id).

    Anything unparseable defaults to ('NEW', '') — fail open, never blocks a
    legitimate ADD because the classifier call was malformed or unreachable.
    """
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    match = re.match(r"(DUPLICATE|UPDATE)\s*:\s*(\S+)", line, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2)
    return "NEW", ""


async def check_duplicate(
    text: str,
    existing_full: list[dict],
    dedup_llm: Optional[Callable[[str, list[dict]], Awaitable[str]]] = None,
) -> tuple[str, str]:
    """Return (verdict, matched_frame_id) for a candidate ADD: NEW / DUPLICATE / UPDATE.

    Fails open to NEW on any error — a broken dedup check must never suppress
    a real memory write.
    """
    import logging
    logger = logging.getLogger(__name__)
    llm = dedup_llm or default_dedup_llm
    try:
        candidates = await shortlist_candidates(text, existing_full)
        if not candidates:
            return "NEW", ""
        raw = await llm(text, candidates)
        return parse_dedup_verdict(raw)
    except Exception:
        logger.warning("Dedup check failed, treating ADD as NEW (fail open)", exc_info=True)
        return "NEW", ""
