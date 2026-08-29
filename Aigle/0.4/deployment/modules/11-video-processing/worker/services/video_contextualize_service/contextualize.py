# services/video_contextualize_service/contextualize.py

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

import httpx

from config import (
    BATCH_SIZE,
    INFERENCE_MAX_LENGTH,
    INFERENCE_MODEL,
    INFERENCE_TEMPERATURE,
    INFERENCE_THINK,
    INFERENCE_TIMEOUT,
    INFERENCE_URL,
    MAX_CONCURRENT_BATCHES,
)

logger = logging.getLogger(__name__)


def _build_prompt(global_summary: str, batch: List[Dict[str, Any]]) -> str:
    chunks = "\n".join(
        f'<chunk index="{m["moment_index"]}">'
        f'ocr: {m.get("ocr_text", "")} / '
        f'asr: {m.get("asr_text", "")} / '
        f'lvlm: {m.get("lvlm_desc", "")}'
        f'</chunk>'
        for m in batch
    )
    return (
        f"<document>\n{global_summary}\n</document>\n\n"
        "你是一個影片內容分析專家。以下是影片中數個連續的 10 秒片段，"
        "請根據上方的影片全局總結，為每個 <chunk> 產生一句精簡的繁體中文上下文"
        "（不超過 50 字），說明此片段在整支影片中的角色或主題定位。\n\n"
        "請嚴格只輸出 JSON 陣列格式，不要有任何其他文字：\n"
        '[{"index": <int>, "context": "<str>"}, ...]\n\n'
        f"{chunks}"
    )


def _parse_json_array(raw: str, batch: List[Dict[str, Any]]) -> List[str]:
    """Return contextual_prefix strings aligned with batch order.
    Falls back to empty strings on any parse failure."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            index_map = {item["index"]: item.get("context", "") for item in parsed}
            return [index_map.get(m["moment_index"], "") for m in batch]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Regex fallback: grab first [...] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            index_map = {item["index"]: item.get("context", "") for item in parsed}
            return [index_map.get(m["moment_index"], "") for m in batch]
        except Exception:
            pass

    logger.warning(
        "JSON parse failed for batch starting at moment_index=%s; using empty prefixes",
        batch[0]["moment_index"],
    )
    return [""] * len(batch)


async def _call_inference(client: httpx.AsyncClient, prompt: str) -> str:
    payload = {
        "task": "text-generation-ollama",
        "engine": "ollama",
        "model_name": INFERENCE_MODEL,
        "data": {"inputs": prompt},
        "options": {"max_length": INFERENCE_MAX_LENGTH, "temperature": INFERENCE_TEMPERATURE, "think": INFERENCE_THINK},
    }
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    response = await client.post(
        f"{INFERENCE_URL}/inference/infer",
        json=payload,
        headers=headers,
        timeout=INFERENCE_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("result", {}).get("response", "").strip()


async def _process_batch(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    global_summary: str,
    batch: List[Dict[str, Any]],
) -> List[str]:
    async with semaphore:
        prompt = _build_prompt(global_summary, batch)
        logger.info(
            "Calling inference for batch of %d moments (index %s–%s)",
            len(batch),
            batch[0]["moment_index"],
            batch[-1]["moment_index"],
        )
        try:
            raw = await _call_inference(client, prompt)
            logger.info(
                "Inference response received for batch starting at moment_index=%s",
                batch[0]["moment_index"],
            )
            return _parse_json_array(raw, batch)
        except Exception as exc:
            logger.warning(
                "Inference call failed for batch starting at moment_index=%s: %s; using empty prefixes",
                batch[0]["moment_index"],
                exc,
            )
            return [""] * len(batch)


async def generate_contextual_prefixes(
    global_summary: str,
    moments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return moments list with 'contextual_prefix' added to each dict."""
    batches = [moments[i : i + BATCH_SIZE] for i in range(0, len(moments), BATCH_SIZE)]
    logger.info("Contextualizing %d moments across %d batches", len(moments), len(batches))

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_process_batch(semaphore, client, global_summary, batch) for batch in batches]
        )

    flat_prefixes = [prefix for batch_prefixes in results for prefix in batch_prefixes]
    enriched = []
    for moment, prefix in zip(moments, flat_prefixes):
        enriched.append({**moment, "contextual_prefix": prefix})

    logger.info("Contextualization complete: %d moments enriched", len(enriched))
    return enriched
