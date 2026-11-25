"""
agents/people_search_agent.py  —  port 52006
A2A wrapper for People Search (MTR-25) via Module 13 (POST /api/{v}/search/people).

People Search finds a person by their name or alias, not by a whole question, so this agent first
works out who the question names: it reads the user's people (GET /identity/identities) and keeps
every name or alias that appears in the question -- the same people the user named in Module 25,
however they are written, without a name extractor guessing. Each is then searched for.

Module 13 acts as the end user of the bearer token the request carries (`auth`): People Search is
the user's own data and needs their face_detection consent (MTR-24). Without a token the agent
finds no one.
"""
from __future__ import annotations

import os
import re

import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import get_message_text, new_agent_text_message
from a2a_helpers import pack_result, unpack_request
from loguru import logger
from models import PeopleSearchRequest, SearchHit, SearchResult

PORT = 52006
_PAGE = 200                  # Module 13's /identity/identities limit


def _gateway() -> str:
    return os.environ.get("API_GATEWAY_URL", "http://raptor-api-gateway:8012/api/0.5").rstrip("/")


def _named(question: str, names: list[str]) -> bool:
    """Whether one of these names appears in the question: a whole word for a Latin-script name
    ("Al" is not in "Alice"), anywhere for a CJK one."""
    q = question.casefold()
    for n in names:
        n = (n or "").strip().casefold()
        if not n:
            continue
        if n.isascii():
            if re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", q):
                return True
        elif n in q:
            return True
    return False


def _people_in(question: str, client: httpx.Client, headers: dict) -> list[str]:
    """The names of the user's people the question names, longest first (Peter Parker before Peter)."""
    found: list[str] = []
    offset = 0
    while True:
        resp = client.get(f"{_gateway()}/identity/identities", headers=headers,
                          params={"limit": _PAGE, "offset": offset})
        resp.raise_for_status()
        data = resp.json()
        for p in data.get("items", []):
            name = p.get("name") or ""
            if name and _named(question, [name] + list((p.get("aliases") or {}).values())):
                found.append(name)
        offset += _PAGE
        if offset >= int(data.get("total", 0)):
            break
    return sorted(set(found), key=len, reverse=True)


def _do_people_search(req: PeopleSearchRequest) -> SearchResult:
    if not req.auth:
        logger.warning("[PeopleSearch] no bearer token -- Module 13 People Search needs the user's own")
        return SearchResult(hits=[], total=0, source="people")
    headers = {"Authorization": req.auth if req.auth.lower().startswith("bearer ") else f"Bearer {req.auth}"}
    hits: list[SearchHit] = []
    order: list[int] = []
    seen: set[str] = set()
    with httpx.Client(timeout=30.0) as client:
        names = _people_in(req.query, client, headers)
        logger.info("[PeopleSearch] question names {}", names)
        for rank, name in enumerate(names):
            resp = client.post(f"{_gateway()}/search/people", headers=headers,
                               json={"query": name, "limit": max(req.top_k, 1) * 5})
            if resp.status_code == 403:      # no face_detection consent (MTR-24)
                logger.warning("[PeopleSearch] People Search denied: {}", resp.text[:200])
                break
            resp.raise_for_status()
            for s in resp.json().get("results", []):
                if s["trace_id"] in seen:
                    continue
                seen.add(s["trace_id"])
                first, last = float(s["first_sec"]), float(s["last_sec"])
                where = s.get("filename") or s["version_id"]
                hits.append(SearchHit(
                    id=f"people:{s['version_id']}:{s['trace_id']}",
                    doc_id=s["version_id"],
                    score=float(s.get("confidence") or 0.0),
                    content=f"{s['identity_name']} appears in {where} from {first:.1f}s to {last:.1f}s.",
                    metadata={"identity_name": s["identity_name"], "identity_id": s.get("identity_id"),
                              "trace_id": s["trace_id"], "first_sec": first, "last_sec": last,
                              "keyframe_path": s.get("keyframe_path") or (s.get("keyframe") or {}).get("image_path"),
                              "filename": s.get("filename"), "type": "videos"},
                    start_time=first, end_time=last,
                    asset_path=s.get("asset_path"), version_id=s["version_id"],
                ))
                order.append(rank)
    # each person the question names in turn (the longest name first); within one, the surest faces first
    hits = [h for _, h in sorted(zip(order, hits), key=lambda x: (x[0], -x[1].score))][:req.top_k]
    logger.info("[PeopleSearch] returned {} segments", len(hits))
    return SearchResult(hits=hits, total=len(hits), source="people")


class PeopleSearchExecutor(AgentExecutor):
    async def execute(self, ctx: RequestContext, queue: EventQueue):
        raw = get_message_text(ctx.message) or ""
        try:
            req    = unpack_request(raw, PeopleSearchRequest)
            result = _do_people_search(req)
            await queue.enqueue_event(new_agent_text_message(pack_result(result)))
        except Exception as e:  # noqa: BLE001 -- top-level A2A executor safety net (request parsing + downstream HTTP + dict access); already logged + returns structured fallback result
            logger.error("[PeopleSearch] error: {}", e)
            await queue.enqueue_event(new_agent_text_message(
                pack_result(SearchResult(hits=[], total=0, source="people"))
            ))

    async def cancel(self, ctx: RequestContext, queue: EventQueue):
        raise NotImplementedError


agent_card = AgentCard(
    name="people_search_agent",
    description="People Search (MTR-25): the segments of the user's videos where the people a question "
                "names appear -- by the names and aliases the user gave them (Module 13 /search/people).",
    version="0.5.0",
    url=f"http://raptor-people-agent:{PORT}",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[AgentSkill(
        id="people_search", name="People Search",
        description="Find where the people a question names appear in the user's videos. Returns segments "
                    "{trace_id, version_id, first_sec, last_sec, identity_name, score}.",
        tags=["search", "people", "face", "identity"],
        examples=['{"query": "find Peter Parker in the video", "top_k": 10}'],
        inputModes=["text"], outputModes=["text"],
    )],
)

if __name__ == "__main__":
    handler = DefaultRequestHandler(agent_executor=PeopleSearchExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card, handler)
    logger.info("PeopleSearchAgent starting on port {}", PORT)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)
