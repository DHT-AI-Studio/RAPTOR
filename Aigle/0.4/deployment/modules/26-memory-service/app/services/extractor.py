"""
LLM-based extraction of preferences, entities, and facts from conversation turns.
Implements Mem0-style ADD/DELETE/UPDATE so stale preferences get replaced, not
accumulated. Safe for fire-and-forget: all public functions never raise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable, Optional

import httpx
from pydantic import BaseModel

from core.config import settings
from services import meta_index
from services.dedup import check_duplicate
from services.long_term_memory import FactAddRequest, LongTermMemoryService
from services.memvid_store import sync_remove

logger = logging.getLogger(__name__)


class _ContentPolicyBlockedError(Exception):
    """Raised by _default_llm() when module 07 returns its guardrail-block
    shape (HTTP 400, {"error_type": "PolicyViolationError", ...}) -- a
    permanent result: retrying the identical turn text against the same
    policy can never succeed. Caught separately in extract_and_store()'s
    retry loop so it gives up immediately instead of sleeping through 2
    more pointless attempts."""


# Top-K existing facts to retrieve for context — limits prompt size while
# ensuring the most relevant existing facts are visible to the LLM.
_CONTEXT_FACTS_K = 10

_SYSTEM_PROMPT = """\
You manage a user's long-term memory. Given a list of existing memory items \
(with IDs) and the latest conversation turn, output a JSON array of memory \
operations.

Each operation must be exactly one of:
  {"op": "ADD",    "type": "preference|entity|fact", "text": "<text>"}
  {"op": "ADD",    "type": "entity",  "name": "<name>", "context": "<context>"}
  {"op": "DELETE", "id": "<existing_id>"}
  {"op": "UPDATE", "id": "<existing_id>", "text": "<updated text>"}

Rules:
- ADD only when the turn reveals something genuinely new and durable.
- DELETE or UPDATE when the turn contradicts or refines an existing item —
  including when the new statement uses different wording but addresses the
  same dimension (e.g. reply language, timezone, dietary restriction) with a
  different value. Do not treat a differently-worded conflict as unrelated.
- Output [] if nothing warrants a change.
- Maximum 5 operations total. No prose, no markdown — JSON array only.

Example (illustrates the pattern only — always read the real "id" values from
the "Existing memory items" list below, never reuse an id from this example):
Existing memory items:
[{"id": "example-42", "type": "preference", "text": "用戶偏好使用台北時區 (UTC+8)"}]

New conversation turn:
User: 我搬到東京了，麻煩改用日本時區
Assistant: 好的，已為您更新時區設定。

Output:
[{"op": "UPDATE", "id": "example-42", "text": "用戶偏好使用東京時區 (UTC+9)"}]\
"""


# ── Models ────────────────────────────────────────────────────────────────────

class ExtractionOp(BaseModel):
    op: str             # ADD | DELETE | UPDATE
    type: str = ""      # preference | entity | fact  (ADD only)
    text: str = ""      # preference / fact / entity body
    name: str = ""      # entity name (ADD entity)
    context: str = ""   # entity context (ADD entity)
    id: str = ""        # frame_id  (DELETE / UPDATE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_turn(turn: dict) -> str:
    return (
        f"User: {turn.get('user_message', '')}\n"
        f"Assistant: {turn.get('assistant_response', '')}"
    )


def _item_text(op: ExtractionOp) -> str:
    if op.type == "entity":
        parts = [op.name, op.context]
        return ": ".join(p for p in parts if p)
    return op.text


def _parse_ops(raw: str) -> list[ExtractionOp]:
    """Parse JSON array of operations from LLM output. Returns [] on failure."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    ops: list[ExtractionOp] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        op_type = item.get("op", "").upper()
        if op_type not in ("ADD", "DELETE", "UPDATE"):
            continue
        ops.append(ExtractionOp(
            op=op_type,
            type=item.get("type", ""),
            text=item.get("text", ""),
            name=item.get("name", ""),
            context=item.get("context", ""),
            id=str(item.get("id", "")),
        ))
    return ops


# ── LLM call ─────────────────────────────────────────────────────────────────

def _build_prompt(turn_text: str, existing_facts: list[dict]) -> str:
    """Combine system + user into a single prompt for Module 07 /inference/infer."""
    facts_block = json.dumps(existing_facts, ensure_ascii=False) if existing_facts else "[]"
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Existing memory items:\n{facts_block}\n\n"
        f"New conversation turn:\n{turn_text}\n\n"
        "Output JSON array of operations only."
    )


async def _default_llm(turn_text: str, existing_facts: list[dict]) -> str:
    """Call Module 07 /inference/infer (Ollama engine). Returns raw response string."""
    prompt = _build_prompt(turn_text, existing_facts)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.module07_url.rstrip('/')}/inference/infer",
            json={
                "task": "text-generation",
                "engine": "ollama",
                "model_name": settings.extraction_model,
                "data": {"inputs": prompt},
                # clean_input=False: Module 07's default preprocessing collapses all
                # whitespace (' '.join(text.split())), which destroys the newline
                # structure separating the few-shot example from the real data below
                # it — the model then can't tell them apart and returns [].
                "options": {"temperature": 0, "max_tokens": 512, "clean_input": False, "think": settings.inference_think},
            },
        )
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
                raise _ContentPolicyBlockedError(
                    f"內容被guardrail政策擋下 (category={detail.get('category')}, "
                    f"direction={detail.get('direction')})"
                )
        resp.raise_for_status()
    body = resp.json()
    # Module 07 response: {"result": {"response": "..."}, "success": true}
    text = body.get("result", {}).get("response", "")
    logger.info(
        "Module 07 extraction call succeeded (model=%s, processing_time=%.3fs)",
        settings.extraction_model,
        body.get("processing_time", 0),
    )
    return text


# ── Shard path helper ─────────────────────────────────────────────────────────

def _find_shard_for_frame(user_id: str, frame_id: str, lt_svc: LongTermMemoryService) -> str | None:
    """Scan shards to find which one contains frame_id (via meta_index)."""
    for path in lt_svc._shard_paths(user_id):
        idx = meta_index.load(path)
        if frame_id in idx:
            return path
    return None


# ── Public API ────────────────────────────────────────────────────────────────

async def extract_and_store(
    user_id: str,
    session_id: str,
    turn: dict,
    lt_svc: LongTermMemoryService,
    *,
    _llm_fn: Optional[Callable[[str, list[dict]], Awaitable[str]]] = None,
    _dedup_llm_fn: Optional[Callable[[str, list[dict]], Awaitable[str]]] = None,
) -> int:
    """
    Extract memory operations from a single conversation turn and apply them.
    Returns the number of write operations executed. Never raises.
    """
    llm = _llm_fn or _default_llm
    turn_text = _format_turn(turn)

    existing_hits: list[dict] = []
    try:
        existing_hits = await _build_existing_facts(user_id, lt_svc, turn_text)
    except Exception:
        logger.warning("Could not fetch existing facts for extraction context", exc_info=True)

    raw: str | None = None
    for attempt in range(3):
        try:
            raw = await llm(turn_text, existing_hits)
            break
        except _ContentPolicyBlockedError as exc:
            # Permanent -- retrying the identical turn text against the same
            # policy can never succeed, so give up now instead of sleeping
            # through 2 more pointless attempts.
            logger.warning(
                "Extraction blocked by guardrail policy for %s/%s, not retrying: %s",
                user_id, session_id, exc,
            )
            return 0
        except Exception as exc:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "Extraction LLM failed after 3 attempts for %s/%s: %s",
                    user_id, session_id, exc,
                )
                return 0

    ops = _parse_ops(raw or "")

    existing_full: list[dict] = []
    if any(op.op == "ADD" for op in ops):
        try:
            existing_full = await lt_svc.get_facts(user_id)
        except Exception:
            logger.warning("Could not fetch full existing facts for dedup check", exc_info=True)

    executed = 0
    for op in ops:
        try:
            if op.op == "ADD":
                text = _item_text(op)
                if not text.strip():
                    continue

                verdict, match_id = await check_duplicate(text, existing_full, _dedup_llm_fn)

                if verdict == "DUPLICATE":
                    logger.info(
                        "extraction_dedup_skipped user=%s text=%r duplicate_of=%s",
                        user_id, text[:60], match_id,
                    )
                    continue

                if verdict == "UPDATE" and match_id:
                    shard = await asyncio.to_thread(_find_shard_for_frame, user_id, match_id, lt_svc)
                    if shard:
                        await asyncio.to_thread(sync_remove, shard, match_id)
                        existing_full = [f for f in existing_full if str(f.get("frame_id")) != match_id]
                        logger.info(
                            "extraction_dedup_superseded user=%s old_id=%s new_text=%r",
                            user_id, match_id, text[:60],
                        )
                    else:
                        logger.debug("Dedup UPDATE: frame_id %s not found in any shard", match_id)

                frame_type = op.type if op.type in ("preference", "entity", "fact") else "fact"
                resp = await lt_svc.add_fact(
                    user_id,
                    FactAddRequest(text=text, frame_type=frame_type, session_id=session_id),
                    skip_dedup=True,
                )
                existing_full.append({"frame_id": resp.frame_id, "text": text, "frame_type": frame_type})
                executed += 1

            elif op.op == "DELETE" and op.id:
                shard = await asyncio.to_thread(_find_shard_for_frame, user_id, op.id, lt_svc)
                if shard:
                    await asyncio.to_thread(sync_remove, shard, op.id)
                    existing_full = [f for f in existing_full if str(f.get("frame_id")) != op.id]
                    executed += 1
                else:
                    logger.debug("DELETE: frame_id %s not found in any shard", op.id)

            elif op.op == "UPDATE" and op.id and op.text.strip():
                shard = await asyncio.to_thread(_find_shard_for_frame, user_id, op.id, lt_svc)
                if not shard:
                    logger.debug("UPDATE: frame_id %s not found in any shard — skipping", op.id)
                    continue
                idx = await asyncio.to_thread(meta_index.load, shard)
                original_type = idx.get(op.id, {}).get("frame_type", "fact")
                await asyncio.to_thread(sync_remove, shard, op.id)
                existing_full = [f for f in existing_full if str(f.get("frame_id")) != op.id]
                frame_type = op.type if op.type in ("preference", "entity", "fact") else original_type
                resp = await lt_svc.add_fact(
                    user_id,
                    FactAddRequest(text=op.text, frame_type=frame_type, session_id=session_id),
                )
                existing_full.append({"frame_id": resp.frame_id, "text": op.text, "frame_type": frame_type})
                executed += 1

        except Exception:
            logger.warning("Failed to execute %s op for frame %s", op.op, op.id, exc_info=True)

    return executed


async def _build_existing_facts(
    user_id: str, lt_svc: LongTermMemoryService, turn_text: str
) -> list[dict]:
    """Return top-K most semantically relevant existing facts via BM25+HNSW search."""
    from services.memvid_store import sync_search
    result: list[dict] = []
    seen_ids: set[str] = set()
    for path in lt_svc._shard_paths(user_id):
        hits = await asyncio.to_thread(sync_search, path, turn_text, _CONTEXT_FACTS_K)
        for hit in hits:
            fid = str(hit.get("frame_id", ""))
            if fid and fid not in seen_ids:
                seen_ids.add(fid)
                result.append({
                    "id": fid,
                    "type": hit.get("frame_type", "fact"),
                    "text": hit.get("text", ""),
                })
    return result[:_CONTEXT_FACTS_K]
