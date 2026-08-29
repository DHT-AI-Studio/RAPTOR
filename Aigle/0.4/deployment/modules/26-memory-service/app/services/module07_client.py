"""Module 07 (AI/ML inference service) client for LLM-based summarization.

Kept separate from compact_memory.py so the compaction service's logic isn't
tied to Module 07's specific request/response envelope — swapping models,
endpoints, or the inference backend only touches this file.
"""
from __future__ import annotations

import logging

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


async def summarize(prompt: str) -> str:
    """Run one text-generation pass against Module 07 and return the generated text."""
    async with httpx.AsyncClient(timeout=settings.compact_llm_timeout_seconds) as client:
        resp = await client.post(
            f"{settings.module07_url.rstrip('/')}/inference/infer",
            json={
                "task": "text-generation",
                "engine": "ollama",
                "model_name": settings.extraction_model,
                "data": {"inputs": prompt},
                # clean_input=False: Module 07's default preprocessing collapses all
                # whitespace, which would destroy the markdown heading structure the
                # summarization prompt instructs the model to follow.
                "options": {
                    "temperature": 0,
                    "max_tokens": settings.compact_summary_max_tokens,
                    "clean_input": False,
                    "think": settings.inference_think,
                },
            },
        )
        resp.raise_for_status()
    body = resp.json()
    logger.info(
        "memory_compact_llm_call model=%s processing_time=%.3fs",
        settings.extraction_model,
        body.get("processing_time", 0),
    )
    return body.get("result", {}).get("response", "").strip()
