"""
MTR-25: People Search as a retrieval modality -- Module 18's people intent routes to PeopleAgent:
- "people"          PeopleAgent alone (the usual fan-out only when it finds no one)
- "people+semantic" PeopleAgent + vector search, RRF-fused, then the reranker
- "people+temporal" PeopleAgent + TKG, the person's segments in time order
and agent.delegate hands a task to a sub-agent, PeopleAgent included.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_people_routing.py -v
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import discovery
import jsonrpc_handler
import memory_client
import pipeline
from models import (
    BM25SearchRequest,
    PeopleSearchRequest,
    QueryPlan,
    RerankerRequest,
    SearchHit,
    SearchResult,
    TKGRequest,
    VectorSearchRequest,
)


def people_hit(trace, version, first, last, score=0.9, name="Peter Parker"):
    return SearchHit(id=f"people:{version}:{trace}", doc_id=version, score=score,
                     content=f"{name} appears in {version}.mp4 from {first}s to {last}s.",
                     metadata={"identity_name": name, "trace_id": trace}, start_time=first, end_time=last,
                     version_id=version, asset_path=f"u/videos/{version}")


def video_hit(id_, version, first, last, content):
    return SearchHit(id=id_, doc_id=version, score=0.5, content=content, start_time=first, end_time=last,
                     version_id=version, asset_path=f"u/videos/{version}")


def run(monkeypatch, intent, people=(), video=(), modes=None):
    calls, prompts = [], []

    async def plan(question, inference_url):
        return QueryPlan(modes=modes or {"vector", "keyword"}, people_intent=intent)

    async def call_agent(url, payload, *, request_id, timeout=30):
        calls.append(payload)
        if isinstance(payload, PeopleSearchRequest):
            return SearchResult(hits=list(people), total=len(people), source="people").model_dump_json()
        if isinstance(payload, VectorSearchRequest) and payload.type == "videos":
            return SearchResult(hits=list(video), total=len(video), source="vector").model_dump_json()
        if isinstance(payload, RerankerRequest):          # keeps RRF's order
            hits = [SearchHit(**c) for c in payload.candidates][:payload.top_k]
            return SearchResult(hits=hits, total=len(hits), source="reranker").model_dump_json()
        if isinstance(payload, TKGRequest):
            return json.dumps({"triples": [], "summary": "Peter Parker joined the Daily Bugle in 2023.",
                               "entity_count": 1, "source": "tkg"})
        return SearchResult(hits=[], total=0, source="x").model_dump_json()

    async def answer(prompt):
        prompts.append(prompt)
        return "answer"

    async def nothing(*a, **kw):
        return ""

    async def no_uri(hit, auth_headers=None):
        return None

    monkeypatch.setattr(pipeline, "build_query_plan", plan)
    monkeypatch.setattr(pipeline, "_call_agent", call_agent)
    monkeypatch.setattr(pipeline, "gather_memory_context", nothing)
    monkeypatch.setattr(pipeline, "_llm_answer", answer)
    monkeypatch.setattr(pipeline, "archive_turn", nothing)
    monkeypatch.setattr(pipeline, "_resolve_source_uri", no_uri)
    monkeypatch.setattr(memory_client, "fire_and_forget", lambda coro: coro.close())

    async def go():
        memory_client.set_auth_token("Bearer user-jwt")
        return await pipeline.run_rag_pipeline("q", user_id="u1", branch_id="u1")
    return asyncio.run(go()), calls, prompts


# ── Module 18's people intent -> the plan ─────────────────────────────────────

def _plan(monkeypatch, body):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"modes": ["VideoRAG"], "sort_by": "relevance", **body}

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return Resp()

    monkeypatch.setattr(pipeline.httpx, "AsyncClient", Client)
    return asyncio.run(pipeline._build_plan_from_module18("q"))


def test_module_18s_people_intent_reaches_the_plan(monkeypatch):
    p = _plan(monkeypatch, {"people_intent": "people"})
    assert p.people_intent == "people" and "people" in p.modes and "tkg" not in p.modes
    p = _plan(monkeypatch, {"people_intent": "people+temporal"})
    assert p.people_intent == "people+temporal" and {"people", "tkg"} <= p.modes
    p = _plan(monkeypatch, {"people_intent": "nonsense"})
    assert p.people_intent is None and "people" not in p.modes
    assert _plan(monkeypatch, {}).people_intent is None


# ── "people": PeopleAgent alone ──────────────────────────────────────────────

def test_the_person_alone_goes_to_people_agent_only(monkeypatch):
    found = [people_hit("t1", "v1", 0.0, 3.5), people_hit("t3", "v2", 10.0, 12.0, 0.8)]
    res, calls, prompts = run(monkeypatch, "people", people=found)
    (req,) = calls                                                  # no vector/keyword/reranker call
    assert isinstance(req, PeopleSearchRequest) and req.auth == "Bearer user-jwt" and req.query == "q"
    # the response names the person and the clip times
    assert res.people == [
        {"identity_name": "Peter Parker", "version_id": "v1", "trace_id": "t1", "first_sec": 0.0, "last_sec": 3.5,
         "score": 0.9},
        {"identity_name": "Peter Parker", "version_id": "v2", "trace_id": "t3", "first_sec": 10.0, "last_sec": 12.0,
         "score": 0.8}]
    assert [s["id"] for s in res.sources] == ["people:v1:t1", "people:v2:t3"]
    assert "Peter Parker appears in v1.mp4 from 0.0s to 3.5s." in prompts[0]   # the LLM answers from them


def test_when_people_agent_finds_no_one_the_usual_search_runs(monkeypatch):
    res, calls, _ = run(monkeypatch, "people", people=[])
    assert isinstance(calls[0], PeopleSearchRequest)
    assert sum(isinstance(c, (VectorSearchRequest, BM25SearchRequest)) for c in calls) == 4
    assert res.people == []


# ── "people+semantic": PeopleAgent + vector search, RRF ──────────────────────

def test_a_person_and_a_topic_fuse_people_and_vector_results(monkeypatch):
    found = [people_hit("t9", "v9", 50.0, 55.0), people_hit("t1", "v1", 20.0, 30.0)]
    moments = [video_hit("m-other", "v5", 0.0, 10.0, "budget talk by someone else"),
               video_hit("m-budget", "v1", 25.0, 40.0, "...and the budget for next year...")]
    res, calls, _ = run(monkeypatch, "people+semantic", people=found, video=moments)
    kinds = [type(c).__name__ for c in calls]
    assert "PeopleSearchRequest" in kinds and "VectorSearchRequest" in kinds and kinds[-1] == "RerankerRequest"
    # the moment where Peter appears AND the budget is talked about ranks first
    assert res.sources[0]["id"] == "m-budget"
    assert {p["trace_id"] for p in res.people} == {"t9", "t1"}


def test_align_people_takes_the_overlapping_hit_only():
    p_on, p_off = people_hit("t1", "v1", 20.0, 30.0, 0.7), people_hit("t2", "v1", 100.0, 110.0)
    other = video_hit("m1", "v1", 25.0, 40.0, "x")
    elsewhere = video_hit("m2", "v2", 20.0, 30.0, "y")                 # same time, another video
    aligned = pipeline._align_people([p_on, p_off], [[elsewhere, other]])
    assert [h.id for h in aligned] == ["m1", "people:v1:t2"] and aligned[0].score == 0.7


# ── "people+temporal": PeopleAgent + TKG, time order ─────────────────────────

def test_a_person_and_a_time_go_to_people_agent_and_tkg_in_time_order(monkeypatch):
    found = [people_hit("t2", "v1", 40.0, 45.0), people_hit("t7", "v2", 3.0, 4.0), people_hit("t1", "v1", 5.0, 9.0)]
    res, calls, prompts = run(monkeypatch, "people+temporal", people=found,
                              modes={"vector", "keyword", "tkg", "people"})
    assert any(isinstance(c, TKGRequest) for c in calls) and any(isinstance(c, PeopleSearchRequest) for c in calls)
    assert [p["trace_id"] for p in res.people] == ["t1", "t2", "t7"]      # v1 in time order, then v2
    assert "Peter Parker joined the Daily Bugle in 2023." in prompts[0]


def test_a_question_without_people_intent_never_calls_people_agent(monkeypatch):
    res, calls, _ = run(monkeypatch, None)
    assert not any(isinstance(c, PeopleSearchRequest) for c in calls) and res.people == []


# ── agent.delegate ───────────────────────────────────────────────────────────

def _delegate(monkeypatch, task, target, payload=None):
    sent = []

    async def call_agent(url, payload, *, request_id, timeout=30):
        sent.append((url, payload))
        return SearchResult(hits=[people_hit("t1", "v1", 0.0, 3.5)], total=1, source="people").model_dump_json()

    monkeypatch.setattr(pipeline, "_call_agent", call_agent)
    handler = jsonrpc_handler.JSONRPCHandler(None, discovery.DiscoveryService())

    async def go():
        memory_client.set_auth_token("Bearer caller-jwt")
        return await handler.handle({"jsonrpc": "2.0", "id": 7, "method": "agent.delegate",
                                     "params": {"task": task, "payload": payload or {}, "target_agent_id": target}})
    return asyncio.run(go()), sent


def test_delegate_hands_the_task_to_people_agent_and_returns_its_result(monkeypatch):
    out, sent = _delegate(monkeypatch, {"query": "find Peter Parker in the video", "top_k": 5}, "people_search_agent")
    ((url, req),) = sent
    assert url == pipeline.get_agent_urls()["people"]
    assert isinstance(req, PeopleSearchRequest) and req.top_k == 5 and req.auth == "Bearer caller-jwt"
    r = out["result"]
    assert r["status"] == "completed" and r["result"]["hits"][0]["metadata"]["identity_name"] == "Peter Parker"


def test_delegate_by_logical_name_and_plain_text_task(monkeypatch):
    out, sent = _delegate(monkeypatch, "find Peter Parker", "people")
    assert sent[0][1].query == "find Peter Parker" and out["result"]["status"] == "completed"


def test_delegate_rejects_an_unknown_target_and_a_bad_task(monkeypatch):
    out, sent = _delegate(monkeypatch, {"query": "x"}, "nobody")
    assert out["result"]["status"] == "rejected" and sent == []
    out, sent = _delegate(monkeypatch, {"top_k": 3}, "people")          # no query
    assert out["error"]["code"] == jsonrpc_handler.INVALID_PARAMS and sent == []


# ── agent mode (smolagents) ──────────────────────────────────────────────────

def test_agent_modes_people_tool_carries_the_users_token(monkeypatch):
    import smolagents_host
    from a2a.types import AgentCapabilities, AgentCard
    sent = []

    class A2A:
        def __init__(self, httpx_client=None, agent_card=None):
            pass

        async def send_message(self, req):
            sent.append(json.loads(req.params.message.parts[0].root.text))

    monkeypatch.setattr(smolagents_host, "A2AClient", A2A)
    monkeypatch.setattr(smolagents_host, "extract_text_from_response", lambda resp: "{}")
    card = AgentCard(name="x", description="x", version="1", url="http://x", capabilities=AgentCapabilities(),
                     defaultInputModes=["text"], defaultOutputModes=["text"], skills=[])

    async def go(name):
        memory_client.set_auth_token("Bearer user-jwt")
        await smolagents_host.A2AAgentTool(name, "d", card, branch_id="u1")._call_agent("find Peter")
    asyncio.run(go("people"))
    asyncio.run(go("vector"))
    assert sent[0]["auth"] == "Bearer user-jwt" and "auth" not in sent[1]     # PeopleAgent only
