"""
app/pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG pipeline for Module 21 (Orchestrator Agent).

Ported from 22-agent-protocol/core/pipeline.py; adapted for Raptor 0.3.
Maintains A2A agent architecture: orchestrator calls each specialized
agent via A2A JSON-RPC messages, never direct HTTP.

Agent backends (Raptor 0.3 services):
  VectorAgent   (:52001) → Module 17 /api/v1/search/vector
  KeywordAgent  (:52002) → Module 17 /api/v1/search/bm25
  GraphRAGAgent (:52003) → Module 20 /query/graph_rag
  TKGAgent      (:52004) → Module 20 /tkg/query
  RerankerAgent (:52005) → Module 17 /api/v1/search/rerank

LLM answer → Module 07 /v1/chat/completions
Intent     → Module 18 /classify (BM25+FAISS+LightGBM+LLM) → Module 07 LLM fallback → regex fallback
Video URLs → Module 04 /filedownload/{asset_path}/{version_id} (inline, no VideoAgent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from urllib.parse import quote as _url_quote

import httpx
import opencc
from loguru import logger

_s2tw_converter = opencc.OpenCC("s2twp")

# Media asset references in a query (file extensions or explicit asset paths)
# gate the multimedia-memory lookup so text-only questions skip the extra call.
_MEDIA_REF_RE = re.compile(
    r"\.(mp4|avi|mov|mkv|mp3|wav|flac|m4a|png|jpe?g|gif|webp|bmp)\b|asset_path",
    re.IGNORECASE,
)

def _s2tw(text: str) -> str:
    return _s2tw_converter.convert(text) if text else text

from a2a.client import A2AClient
from a2a.types import AgentCard, SendMessageRequest, MessageSendParams
from a2a.client.helpers import create_text_message_object

_agent_card_cache: dict[str, "AgentCard"] = {}

from models import (
    QueryPlan, SortOrder,
    VectorSearchRequest, BM25SearchRequest, RerankerRequest,
    GraphRAGRequest, TKGRequest,
    SearchHit, SearchResult, GraphResult, OrchestratorResult,
)
from a2a_helpers import pack_request, extract_text_from_response


# ── Agent URL registry ────────────────────────────────────────────────────────

def get_agent_urls() -> dict[str, str]:
    return {
        "vector":   os.environ.get("VECTOR_AGENT_URL",   "http://raptor-vector-agent:52001"),
        "keyword":  os.environ.get("KEYWORD_AGENT_URL",  "http://raptor-keyword-agent:52002"),
        "graphrag": os.environ.get("GRAPHRAG_AGENT_URL", "http://raptor-graphrag-agent:52003"),
        "tkg":      os.environ.get("TKG_AGENT_URL",      "http://raptor-tkg-agent:52004"),
        "reranker": os.environ.get("RERANKER_AGENT_URL", "http://raptor-reranker-agent:52005"),
    }


# ── Intent classification ─────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are a query analyzer for a multimodal RAG system.
Analyze the question and respond with ONLY valid JSON — no markdown, no explanation.

JSON schema (all fields required):
{
  "modes":        ["vector","keyword"],
  "sort_by":      "relevance",
  "top_k":        null,
  "rel_position": null,
  "tail_seconds": null,
  "group_by_doc": false,
  "entities":     [],
  "time_range":   {"start": null, "end": null}
}

modes: always include "vector" and "keyword";
       add "video"    for video/clip/footage questions;
       add "graphrag" for entity/relationship questions (who, how are X and Y related, etc.);
       add "tkg"      for temporal/sequence questions (before, after, timeline, history, events in order).
sort_by: "relevance" | "time_asc" | "time_desc"
rel_position: "start" | "end" | null
tail_seconds: float if a trailing window is specified, else null
group_by_doc: true only when asking across multiple documents
entities: named entities useful as graph seed nodes
time_range: ISO8601 dates if a calendar window is specified"""


async def _classify_intent_llm(question: str, inference_url: str) -> dict | None:
    """Module 18 → Module 07 → regex is the real fallback chain (see
    build_query_plan()); this only fires when Module 18 itself is
    unreachable. Was posting straight to Ollama's native /api/chat via
    OLLAMA_HOST -- no think control there, and qwen3.x-family models think
    by default (same latency bug already fixed for _llm_answer() and
    smolagents_host.py's get_ollama_model() elsewhere in this module).
    """
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5")
    think = os.environ.get("LLM_THINK", "false").lower() == "true"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{inference_url}/v1/chat/completions",
                json={
                    "model": model,
                    "engine": "ollama",
                    "messages": [
                        {"role": "system", "content": _INTENT_SYSTEM},
                        {"role": "user",   "content": f"Question: {question}"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "think": think,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw)
            return json.loads(raw)
    except Exception as exc:
        logger.warning("[Pipeline] LLM intent classification failed: {} — using regex fallback", exc)
        return None

# Old direct-Ollama implementation -- commented out, not deleted, for rollback.
# async def _classify_intent_llm(question: str, ollama_url: str) -> dict | None:
#     model = os.environ.get("OLLAMA_MODEL", "qwen2.5")
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             resp = await client.post(
#                 f"{ollama_url}/api/chat",
#                 json={
#                     "model": model,
#                     "messages": [
#                         {"role": "system", "content": _INTENT_SYSTEM},
#                         {"role": "user",   "content": f"Question: {question}"},
#                     ],
#                     "options": {"temperature": 0.0, "num_predict": 256},
#                     "stream": False,
#                 },
#             )
#             resp.raise_for_status()
#             raw = resp.json()["message"]["content"].strip()
#             raw = re.sub(r"^```(?:json)?\s*", "", raw)
#             raw = re.sub(r"\s*```\s*$", "", raw)
#             return json.loads(raw)
#     except Exception as exc:
#         logger.warning("[Pipeline] LLM intent classification failed: {} — using regex fallback", exc)
#         return None


def _build_plan_from_llm(llm: dict) -> QueryPlan:
    raw_modes = set(llm.get("modes") or [])
    modes = {"vector", "keyword"} | (raw_modes & {"video", "graphrag", "tkg"})

    sort_by = {
        "time_asc":  SortOrder.TIME_ASC,
        "time_desc": SortOrder.TIME_DESC,
    }.get(llm.get("sort_by", "relevance"), SortOrder.RELEVANCE)

    raw_top_k = llm.get("top_k")
    top_k_override = int(raw_top_k) if raw_top_k else None

    rel_position = llm.get("rel_position")
    if rel_position not in ("start", "middle", "end"):
        rel_position = None

    raw_tail = llm.get("tail_seconds")
    tail_seconds = float(raw_tail) if raw_tail else None

    if rel_position == "end" and sort_by == SortOrder.TIME_DESC:
        sort_by = SortOrder.RELEVANCE

    group_by_doc = bool(llm.get("group_by_doc", False))
    entities = [str(e) for e in (llm.get("entities") or []) if e]

    raw_tr = llm.get("time_range") or {}
    time_range: dict | None = (
        {"start": raw_tr.get("start"), "end": raw_tr.get("end")}
        if (raw_tr.get("start") or raw_tr.get("end")) else None
    )

    return QueryPlan(
        modes=modes, sort_by=sort_by, group_by_doc=group_by_doc,
        top_k_override=top_k_override, rel_position=rel_position,
        tail_seconds=tail_seconds, entities=entities, time_range=time_range,
    )


# ── Regex fallback ────────────────────────────────────────────────────────────

_GRAPHRAG_KEYWORDS = {
    "who", "which", "relate", "relation", "relationship", "connect", "connection",
    "between", "involved", "involve", "link", "linked", "associate", "associated",
}
_TKG_KEYWORDS = {
    "before", "after", "during", "sequence", "order", "timeline", "history",
    "cause", "because", "lead", "led", "result", "follow", "followed", "precede",
    "happen", "happened", "event", "when", "then", "next", "previous",
}
_CHINESE_NUMS = {
    "一": 1, "兩": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


_VIDEO_KEYWORDS = {
    "video", "clip", "footage", "film", "movie", "影片", "影像", "片段", "視頻",
}

def _build_plan_from_regex(question: str) -> QueryPlan:
    q_lower = question.lower()
    words = set(re.sub(r"[^\w\s]", " ", q_lower).split())
    modes: set[str] = {"vector", "keyword"}
    if words & _VIDEO_KEYWORDS or re.search(r"影片|片段|視頻|video|clip|footage", q_lower):
        modes.add("video")
    if words & _GRAPHRAG_KEYWORDS:
        modes.add("graphrag")
    if words & _TKG_KEYWORDS:
        modes.add("tkg")

    sort_by = SortOrder.RELEVANCE
    if any(s in question for s in ["最後", "最新", "最近", "latest", "most recent", "newest"]):
        sort_by = SortOrder.TIME_DESC
    elif any(s in question for s in ["最早", "最初", "earliest", "chronological", "oldest"]):
        sort_by = SortOrder.TIME_ASC
    elif re.search(r"前[一兩二三四五六七八九十\d]+次", question):
        sort_by = SortOrder.TIME_ASC

    top_k_override: int | None = None
    m = re.search(r"前([一兩二三四五六七八九十])次?", question)
    if m:
        top_k_override = _CHINESE_NUMS.get(m.group(1), 2)
    if top_k_override is None:
        m = re.search(r"前(\d+)[次個]?", question)
        if m:
            top_k_override = int(m.group(1))
    if top_k_override is None:
        m = re.search(r"\b(?:top|first|last|show(?:\s+me)?|give\s+me)\s+(\d+)\b", q_lower)
        if m:
            top_k_override = int(m.group(1))

    rel_position: str | None = None
    tail_seconds: float | None = None

    _end_signals = (
        r"最後\s*\d*\s*[秒分]?|last\s+\d+\s*(?:sec(?:onds?)?|min(?:utes?)?)|"
        r"last\s+minute|ending|final\s+(?:moment|scene|clip)|closing|tail\b"
    )
    if re.search(_end_signals, question, re.IGNORECASE):
        rel_position = "end"
        m = re.search(r"(?:最後|last)\s*(\d+)\s*(?:秒|sec(?:onds?)?)", question, re.IGNORECASE)
        tail_seconds = float(m.group(1)) if m else 30.0
        if sort_by == SortOrder.TIME_DESC and not any(
            s in question for s in ["最新", "最近", "latest", "most recent", "newest"]
        ):
            sort_by = SortOrder.RELEVANCE
    elif re.search(r"開頭|最開始|intro(?:duction)?|opening|beginning|first\s+few\s+sec", question, re.IGNORECASE):
        rel_position = "start"

    group_by_doc = bool(re.search(
        r"所有影片|哪些影片|哪些作品|all\s+video|every\s+video|each\s+video|"
        r"across\s+video|different\s+video|which\s+video", q_lower,
    ))

    return QueryPlan(
        modes=modes, sort_by=sort_by, group_by_doc=group_by_doc,
        top_k_override=top_k_override, rel_position=rel_position,
        tail_seconds=tail_seconds, entities=[], time_range=None,
    )


async def _build_plan_from_module18(question: str) -> QueryPlan | None:
    qo_url = os.environ.get("QUERY_ORCHESTRATOR_URL", "http://raptor-query-orchestrator:8000")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{qo_url}/classify", json={"query": question})
            resp.raise_for_status()
            data = resp.json()
        m18_modes = set(data.get("modes", []))
        modes: set[str] = {"vector", "keyword"}
        if "VideoRAG" in m18_modes:
            modes.add("video")
        if "GraphRAG" in m18_modes:
            modes.add("graphrag")
        if "TKG" in m18_modes:
            modes.add("tkg")
        sort_by = {"time_asc": SortOrder.TIME_ASC, "time_desc": SortOrder.TIME_DESC}.get(
            data.get("sort_by", "relevance"), SortOrder.RELEVANCE,
        )
        rel_position = data.get("rel_position")
        if rel_position not in ("start", "end"):
            rel_position = None
        raw_tail = data.get("tail_seconds")
        time_from, time_to = data.get("time_from"), data.get("time_to")
        logger.debug(
            "[Pipeline] Module 18 plan: m18_modes={} → modes={} sort={} strategy={}",
            sorted(m18_modes), sorted(modes), data.get("sort_by"), data.get("routing_strategy"),
        )
        return QueryPlan(
            modes=modes, sort_by=sort_by,
            group_by_doc=data.get("group_by_doc", False),
            top_k_override=data.get("top_k_override"),
            rel_position=rel_position,
            tail_seconds=float(raw_tail) if raw_tail else None,
            entities=[],
            time_range={"start": time_from, "end": time_to} if (time_from or time_to) else None,
        )
    except Exception as exc:
        logger.warning("[Pipeline] Module 18 classify failed: {} — falling back to LLM/regex", exc)
        return None


async def build_query_plan(question: str, inference_url: str) -> QueryPlan:
    plan = await _build_plan_from_module18(question)
    if plan is not None:
        return plan
    llm = await _classify_intent_llm(question, inference_url)
    if llm is not None:
        plan = _build_plan_from_llm(llm)
        logger.debug(
            "[Pipeline] LLM plan: modes={} sort={} entities={} top_k={} pos={}",
            sorted(plan.modes), plan.sort_by.value, plan.entities,
            plan.top_k_override, plan.rel_position,
        )
        return plan
    return _build_plan_from_regex(question)


# ── RRF + post-processing ─────────────────────────────────────────────────────

def _rrf_fuse(ranked_lists: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    hits_by_id: dict[str, SearchHit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            if hit.id not in hits_by_id or hit.score > hits_by_id[hit.id].score:
                hits_by_id[hit.id] = hit
    return sorted(hits_by_id.values(), key=lambda h: scores[h.id], reverse=True)


def _postprocess(hits: list[SearchHit], plan: QueryPlan, top_k: int) -> list[SearchHit]:
    if plan.rel_position:
        filtered = [
            h for h in hits
            if h.start_time is None or h.metadata.get("rel_position") == plan.rel_position
        ]
        if filtered:
            hits = filtered

    if plan.group_by_doc:
        seen: dict[str, SearchHit] = {}
        for h in hits:
            if h.doc_id not in seen:
                seen[h.doc_id] = h
        hits = list(seen.values())

    if plan.sort_by != SortOrder.RELEVANCE:
        vid  = [h for h in hits if h.start_time is not None]
        text = [h for h in hits if h.start_time is None]
        vid.sort(key=lambda h: h.start_time or 0.0, reverse=(plan.sort_by == SortOrder.TIME_DESC))
        hits = text + vid

    return hits[:plan.top_k_override or top_k]


# ── Prompt assembly ───────────────────────────────────────────────────────────

def assemble_grounded_prompt(question: str, top_chunks: list[dict],
                              graph_context: str, doc_sources: list[dict],
                              memory_context: str = "") -> str:
    chunk_text = "\n\n".join(
        f"[Chunk {i+1} | doc={c.get('doc_id','?')} score={c.get('score',0):.3f}]\n{c.get('content','')}"
        for i, c in enumerate(top_chunks)
    )
    sources_text = "\n".join(
        f"  - doc_id={s.get('doc_id')} | uri={s.get('storage_uri','?')}"
        for s in doc_sources
    )
    memory_block = (
        f"USER MEMORY (long-term facts and preferences about this user):\n{memory_context}\n\n"
        if memory_context else ""
    )
    memory_instruction = (
        " USER MEMORY holds facts remembered from prior sessions and is the "
        "only valid source for questions about the user themselves (their "
        "name, preferences, belongings, etc.) — prefer it over context chunks "
        "for those."
        if memory_context else ""
    )
    return (
        "You are a precise question-answering assistant.\n"
        "The retrieved context chunks below are excerpts from a general "
        "knowledge base of arbitrary uploaded documents, videos, and images "
        "— third-party content, not facts about the user, unless a chunk "
        "explicitly names the user.\n"
        "Answer using ONLY the provided context and graph relationships — "
        f"never invent facts absent from all of them.{memory_instruction}\n"
        "If a question asks about the user's own life, belongings, or "
        "preferences and none of the provided material actually answers it, "
        "say so honestly — do not substitute unrelated context chunks just "
        "because they mention similar things.\n"
        "If the answer is not in any of them, say so explicitly.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{memory_block}"
        f"CONTEXT CHUNKS:\n{chunk_text}\n\n"
        f"GRAPH RELATIONSHIPS:\n{graph_context or 'No graph context available.'}\n\n"
        f"DOCUMENT SOURCES:\n{sources_text or 'No sources available.'}\n\n"
        "ANSWER:"
    )


# ── A2A agent caller ──────────────────────────────────────────────────────────

async def _get_card(url: str, client: httpx.AsyncClient) -> "AgentCard | None":
    """Fetch agent card via /.well-known/agent-card.json; cache per base URL."""
    if url not in _agent_card_cache:
        try:
            resp = await client.get(f"{url.rstrip('/')}/.well-known/agent-card.json", timeout=5.0)
            resp.raise_for_status()
            card = AgentCard(**resp.json())
            # Trust the URL we successfully reached over the agent's
            # self-declared url — they diverge when discovery goes through a
            # different network path than the card's own (e.g. host-mapped
            # port vs. docker-internal hostname).
            card.url = url
            _agent_card_cache[url] = card
        except Exception as exc:
            logger.debug("[Pipeline] agent card fetch failed for {}: {}", url, exc)
            return None
    return _agent_card_cache.get(url)


async def _call_agent(url: str, payload, *, request_id: str, timeout: int = 30) -> str:
    """Send an A2A message to a remote agent, return the text response or '' on failure."""
    async with httpx.AsyncClient(timeout=timeout, headers={"X-Request-ID": request_id}) as client:
        try:
            card = await _get_card(url, client)
            a2a = A2AClient(httpx_client=client, agent_card=card) if card else A2AClient(httpx_client=client, url=url)
            req = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(
                    message=create_text_message_object(content=pack_request(payload))
                ),
            )
            resp = await a2a.send_message(req)
            return extract_text_from_response(resp)
        except Exception as exc:
            logger.warning("[Pipeline] rid={} agent={} call failed: {}", request_id, url, exc)
            return ""


# ── LLM answer (Module 07) ────────────────────────────────────────────────────

async def _llm_answer(prompt: str) -> str:
    module07 = os.environ.get("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    # num_ctx: Module 07's Ollama adapter passes this straight through to
    # Ollama's native /api/chat (ollama.py:277's _OLLAMA_PASSTHROUGH) -- unlike
    # ChatOpenAI/Ollama's OpenAI-compat layer (module 15's old client), which
    # silently ignores it, confirmed live. Shared with module 15 via the same
    # root LLM_NUM_CTX var (should match LLM_MODEL's own trained context
    # length per Ollama's /api/show model_info.*.context_length, not some
    # arbitrary number -- update it when LLM_MODEL changes).
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "32768"))
    # Deliberately its own var, not module 15's shared TEMPERATURE (0.7) --
    # that's chat conversational tone; this is grounded RAG answer synthesis
    # over retrieved facts, where a lower temperature has been the actual
    # value here all along (0.3). Reusing the shared var would silently
    # raise it to 0.7 with no discussion of whether that's wanted.
    temperature = float(os.environ.get("LLM_ANSWER_TEMPERATURE", "0.3"))
    # Shared with module 15 via the same root LLM_THINK var, same reasoning
    # as num_ctx above -- both are the "talk to Ollama via 07" knobs 15's
    # ChatOpenAI(extra_body=...) also sets.
    think = os.environ.get("LLM_THINK", "false").lower() == "true"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{module07}/v1/chat/completions",
                json={
                    "model": model_name,
                    "engine": "ollama",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "think": think,
                },
            )
            resp.raise_for_status()
            choices = resp.json().get("choices") or [{}]
            return (choices[0].get("message") or {}).get("content", "").strip()
    except Exception as exc:
        logger.error("[Pipeline] LLM answer failed: {}", exc)
        return "Answer generation failed."


# ── Video clip URL resolution (inline via Module 04) ─────────────────────────

async def _fetch_presigned_url(
    asset_path: str,
    version_id: str | None,
    filename: str | None = None,
    auth_headers: dict | None = None,
) -> str | None:
    """Fetch presigned URL from Module 04.

    Fast path (/presign-url): used when filename is known — 1 direct LakeFS call.
    Slow path (/filedownload): fallback when only asset_path + version_id are available.
    auth_headers must contain X-User-ID and X-Branch-ID (required by Module 04).
    """
    asset_mgmt = os.environ.get("ASSET_MANAGEMENT_URL", "http://raptor-asset-management:8000")
    headers = auth_headers or {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if filename and version_id:
                key = _url_quote(f"{asset_path}/{filename}", safe="")
                vid = _url_quote(version_id, safe="")
                resp = await client.get(f"{asset_mgmt}/presign-url/{key}/{vid}", headers=headers)
                resp.raise_for_status()
                return resp.json().get("url")
            else:
                path = f"{asset_path}/{version_id}" if version_id else asset_path
                resp = await client.get(f"{asset_mgmt}/filedownload/{path}", headers=headers)
                resp.raise_for_status()
                return resp.json().get("primary_file", {}).get("url")
    except Exception as exc:
        logger.debug("[Pipeline] Module 04 presign failed for {}/{}: {}", asset_path, version_id, exc)
        return None


async def _resolve_source_uri(hit: SearchHit, auth_headers: dict | None = None) -> str | None:
    """Resolve presigned download URL for a source hit (any content type)."""
    if not hit.asset_path or not hit.version_id:
        return None
    filename = hit.metadata.get("filename")
    return await _fetch_presigned_url(hit.asset_path, hit.version_id, filename, auth_headers)


# ── Memory integration (shared by direct pipeline + smolagents modes) ─────────

async def gather_memory_context(user_id: str, session_id: str, question: str,
                                  rid: str = "") -> str:
    """Retrieve long-term/multimedia memory and MV-12 compact summary, formatted
    as prompt-ready text. Fail open: no user_id or an unreachable Module 26 →
    empty string."""
    if not user_id:
        return ""

    from memory_client import get_memory_client
    mem = get_memory_client()
    memory_lines: list[str] = []

    memory_hits = await mem.search_longterm(user_id, question, top_k=3)
    memory_lines += [
        f"[{h.get('frame_type', 'fact')}] {h.get('text', '')}"
        for h in memory_hits if h.get("text")
    ]

    # Query references media assets → surface remembered clip/image references
    if _MEDIA_REF_RE.search(question):
        media_hits = await mem.search_multimedia(user_id, question, top_k=2)
        memory_lines += [
            f"[media:{h.get('media_type', '?')}] {h.get('asset_path', '')} — "
            f"{h.get('text_snippet', '')}"
            for h in media_hits
        ]

    # Compact archived session history via Module 26 (MV-12). Evaluate the
    # combined-context budget first (compactdesign.md §10.2); only skip the
    # (heavier) compact call when evaluate explicitly says it's unnecessary.
    # If evaluate itself fails (None, fail open), fall back to calling
    # compact_session directly — it has its own internal threshold no-op.
    used_session = session_id or user_id
    try:
        context_window_tokens = int(os.environ.get("MEM_COMPACT_THRESHOLD", "128000"))
        evaluation = await mem.evaluate_compact(
            user_id=user_id, session_id=used_session, context_window=context_window_tokens,
        )
        if evaluation is not None and not evaluation.get("should_compact"):
            compact_result = None
        else:
            compact_result = await mem.compact_session(
                user_id=user_id, session_id=used_session, context_window=context_window_tokens,
            )
        if compact_result and compact_result.get("compacted"):
            summary = await mem.get_latest_summary(user_id, used_session)
            if summary and summary.get("text"):
                logger.info(
                    "[Pipeline] rid={} compact: session={} {}→{} tokens",
                    rid, used_session,
                    compact_result.get("pre_compact_tokens", 0),
                    compact_result.get("post_compact_tokens", 0),
                )
                memory_lines.insert(0, f"[歷史摘要]\n{summary['text']}")
    except Exception as exc:
        logger.warning("[Pipeline] rid={} compact via Module 26 failed (fail open): {}", rid, exc)

    memory_context = "\n".join(memory_lines)
    if memory_context:
        logger.info("[Pipeline] rid={} memory: {} context lines", rid, len(memory_lines))
    return memory_context


async def archive_turn(user_id: str, session_id: str, question: str, answer: str,
                        search_results: list[dict] | None = None, mode: str = "agent") -> None:
    """Archive turn to Module 26. session_id defaults to user_id so
    invocations without an explicit session still accumulate per-user
    memory. No-op when user_id is empty.

    Scheduled via `memory_client.fire_and_forget` rather than awaited
    directly — see that function's docstring for why a plain
    `asyncio.create_task()` doesn't survive callers that run inside a
    throwaway loop (e.g. RAGSearchTool.forward's asyncio.run).
    """
    if not user_id:
        return
    from memory_client import get_memory_client
    await get_memory_client().append_turn(
        user_id=user_id,
        session_id=session_id or user_id,
        user_message=question,
        assistant_response=answer,
        search_results=search_results or [],
        mode=mode,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_rag_pipeline(
    question: str,
    top_k: int = 5,
    request_id: str | None = None,
    user_id: str = "",
    branch_id: str = "",
    session_id: str = "",
    mode: str = "direct",
) -> OrchestratorResult:
    """
    Full RAG pipeline via A2A agents:
      1. Build QueryPlan (Module 18 → Ollama LLM → regex fallback)
      2. Fan-out via A2A (always):
           vec_text  — VectorAgent  type=[documents,audios,images]
           kw_text   — KeywordAgent type=[documents,audios,images]
         conditional:
           vec_video — VectorAgent  type=videos   (if "video"   in modes)
           kw_video  — KeywordAgent type=videos   (if "video"   in modes)
           graphrag  — GraphRAGAgent              (if "graphrag" in modes)
           tkg       — TKGAgent                   (if "tkg"     in modes)
      3. RRF fusion
      4. Rerank via RerankerAgent (A2A)
      5. Post-process (rel_position / group_by_doc / sort / cap)
      6. Resolve video clips inline via Module 04
      7. Grounded prompt → LLM answer via Module 07
    """
    inference_url = os.environ.get("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
    urls = get_agent_urls()
    rid  = request_id or str(uuid.uuid4())[:8]

    plan = await build_query_plan(question, inference_url)
    logger.info(
        "[Pipeline] rid={} modes={} sort={} group={} top_k={} pos={}",
        rid, sorted(plan.modes), plan.sort_by.value,
        plan.group_by_doc, plan.top_k_override, plan.rel_position,
    )

    vec_url     = urls["vector"]
    kw_url      = urls["keyword"]
    rnk_url     = urls["reranker"]
    use_graphrag = "graphrag" in plan.modes
    use_tkg      = "tkg"      in plan.modes

    _TEXT_TYPES  = ["documents", "audios", "images"]
    _VIDEO_TYPES = "videos"

    # ── Step 1: Fan-out ───────────────────────────────────────────────────────
    tasks: list = [
        _call_agent(vec_url, VectorSearchRequest(query=question, top_k=top_k * 2, type=_TEXT_TYPES, branch_id=branch_id or None), request_id=rid),
        _call_agent(kw_url,  BM25SearchRequest(query=question,   top_k=top_k * 2, type=_TEXT_TYPES, branch_id=branch_id or None), request_id=rid),
        _call_agent(vec_url, VectorSearchRequest(query=question, top_k=top_k, type=_VIDEO_TYPES, branch_id=branch_id or None), request_id=rid),
        _call_agent(kw_url,  BM25SearchRequest(query=question,   top_k=top_k, type=_VIDEO_TYPES, branch_id=branch_id or None), request_id=rid),
    ]
    labels = ["vec_text", "kw_text", "vec_video", "kw_video"]

    if use_graphrag:
        seed_entity = plan.entities[0] if plan.entities else None
        tasks.append(_call_agent(
            urls["graphrag"],
            GraphRAGRequest(query=question, entity=seed_entity, max_depth=2, branch_id=branch_id or None),
            request_id=rid,
        ))
        labels.append("graphrag")

    if use_tkg:
        time_start = time_end = None
        if plan.time_range:
            time_start = plan.time_range.get("start")
            time_end   = plan.time_range.get("end")
        tasks.append(_call_agent(
            urls["tkg"],
            TKGRequest(query=question, max_depth=2, time_start=time_start, time_end=time_end, branch_id=branch_id or None),
            request_id=rid,
        ))
        labels.append("tkg")

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Step 2: Parse results ─────────────────────────────────────────────────
    parsed: dict = {}
    for label, raw in zip(labels, raw_results):
        if isinstance(raw, Exception):
            logger.warning("[Pipeline] rid={} {} raised: {}", rid, label, raw)
            continue
        if not raw:
            logger.warning("[Pipeline] rid={} {} returned empty", rid, label)
            continue
        try:
            if label in ("vec_text", "vec_video", "kw_text", "kw_video"):
                parsed[label] = SearchResult.model_validate_json(raw)
            elif label in ("graphrag", "tkg"):
                parsed[label] = GraphResult.model_validate_json(raw)
        except Exception as exc:
            logger.warning("[Pipeline] rid={} failed to parse {} response: {}", rid, label, exc)

    vec_text_result: SearchResult | None = parsed.get("vec_text")
    vec_vid_result:  SearchResult | None = parsed.get("vec_video")
    kw_text_result:  SearchResult | None = parsed.get("kw_text")
    kw_vid_result:   SearchResult | None = parsed.get("kw_video")
    graphrag_result:  GraphResult  | None = parsed.get("graphrag")
    tkg_result:       GraphResult  | None = parsed.get("tkg")

    # Combine graph context from both sources
    graph_triples = []
    graph_summary_parts = []
    for gr in filter(None, [graphrag_result, tkg_result]):
        graph_triples.extend(gr.triples)
        if gr.summary:
            graph_summary_parts.append(gr.summary)
    graph_context = "\n\n".join(graph_summary_parts)

    # ── Step 3: RRF Fusion ────────────────────────────────────────────────────
    ranked_lists = [
        r.hits for r in filter(None, [vec_text_result, kw_text_result, vec_vid_result, kw_vid_result])
    ]
    all_candidates: list[SearchHit] = _rrf_fuse(ranked_lists) if ranked_lists else []

    # ── Step 4: Rerank ────────────────────────────────────────────────────────
    if all_candidates:
        raw_reranked = await _call_agent(
            rnk_url,
            RerankerRequest(query=question, candidates=[h.model_dump() for h in all_candidates], top_k=top_k),
            request_id=rid,
        )
        try:
            reranked = SearchResult.model_validate_json(raw_reranked)
            top_hits = reranked.hits
        except Exception as exc:
            logger.warning("[Pipeline] rid={} rerank parse failed: {} — using RRF order", rid, exc)
            top_hits = all_candidates
    else:
        top_hits = all_candidates

    top_hits = _postprocess(top_hits, plan, top_k)
    top_chunks = [h.model_dump() for h in top_hits]

    # ── Step 5: Token budget check — trim chunks before prompt assembly ───────
    _PROMPT_BUDGET_TOKENS = int(os.environ.get("COMPACT_PROMPT_BUDGET_TOKENS", "80000"))
    chunk_tokens = sum(len(json.dumps(c, default=str)) // 4 for c in top_chunks)
    if chunk_tokens > _PROMPT_BUDGET_TOKENS and len(top_chunks) > 1:
        kept: list[dict] = []
        used = 0
        for chunk in top_chunks:
            t = len(json.dumps(chunk, default=str)) // 4
            if used + t > _PROMPT_BUDGET_TOKENS:
                break
            kept.append(chunk)
            used += t
        if kept:
            logger.warning(
                "[Pipeline] rid={} compact_context: trimmed chunks {}->{} ({}→{} tokens)",
                rid, len(top_chunks), len(kept), chunk_tokens, used,
            )
            top_chunks = kept

    # ── Step 6: Resolve presigned URLs via Module 04 ─────────────────────────
    auth_headers: dict = {}
    if user_id:
        auth_headers["X-User-ID"]   = user_id
        auth_headers["X-Branch-ID"] = branch_id or user_id
    uri_results = await asyncio.gather(
        *[_resolve_source_uri(h, auth_headers) for h in top_hits],
        return_exceptions=True,
    )
    for chunk, uri in zip(top_chunks, uri_results):
        if isinstance(uri, str) and uri:
            chunk["storage_uri"] = uri

    # ── Step 7: Long-term + multimedia memory retrieval ──────────────────────
    memory_context = await gather_memory_context(user_id, session_id, question, rid=rid)

    # ── Step 8: Grounded prompt → LLM ────────────────────────────────────────
    doc_sources = list({
        c.get("doc_id"): {"doc_id": c.get("doc_id"), "storage_uri": c.get("storage_uri", "")}
        for c in top_chunks
    }.values())

    prompt = assemble_grounded_prompt(
        question=question, top_chunks=top_chunks,
        graph_context=graph_context, doc_sources=doc_sources,
        memory_context=memory_context,
    )
    answer = await _llm_answer(prompt)

    # Convert Simplified → Traditional Chinese for LLM-generated text
    answer        = _s2tw(answer)
    graph_context = _s2tw(graph_context)

    # ── Step 9: Archive this turn to Module 26 (fire-and-forget) ──────────────
    from memory_client import fire_and_forget
    fire_and_forget(archive_turn(
        user_id, session_id, question, answer,
        search_results=[
            {"doc_id": c.get("doc_id"), "score": c.get("score")} for c in top_chunks
        ],
        mode=mode,
    ))

    logger.info("[Pipeline] rid={} done chunks_used={}", rid, len(top_chunks))
    return OrchestratorResult(
        answer=answer,
        sources=top_chunks,
        graph_context=graph_context,
        chunks_used=len(top_chunks),
    )
