"""
P21: a question scoped to particular sources (source_version_ids) is answered from them only -- vector,
keyword and TKG search carry the scope; GraphRAG and the user's memory are left out.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_source_scope.py -v
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import keyword_search_agent
import memory_client
import pipeline
import tkg_agent
import vector_search_agent
from models import BM25SearchRequest, QueryPlan, TKGRequest, VectorSearchRequest


def _run(monkeypatch, scope):
    calls, memory = [], []

    async def plan(question, inference_url):
        return QueryPlan(modes={"vector", "keyword", "tkg", "graphrag"})

    async def call_agent(url, payload, *, request_id, timeout=30):
        calls.append(payload)
        return ""

    async def gather(user_id, session_id, question, rid=None):
        memory.append(question)
        return "the user likes tea"

    async def answer(prompt):
        return "answer"

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(pipeline, "build_query_plan", plan)
    monkeypatch.setattr(pipeline, "_call_agent", call_agent)
    monkeypatch.setattr(pipeline, "gather_memory_context", gather)
    monkeypatch.setattr(pipeline, "_llm_answer", answer)
    monkeypatch.setattr(pipeline, "archive_turn", noop)
    monkeypatch.setattr(memory_client, "fire_and_forget", lambda coro: coro.close())
    asyncio.run(pipeline.run_rag_pipeline("what happened?", user_id="u1", branch_id="u1",
                                          source_version_ids=scope))
    return calls, memory


def test_a_scoped_question_searches_only_those_sources(monkeypatch):
    calls, memory = _run(monkeypatch, ["v-inline"])
    searches = [c for c in calls if isinstance(c, (VectorSearchRequest, BM25SearchRequest))]
    assert len(searches) == 4 and all(c.version_id == ["v-inline"] for c in searches)
    tkg = [c for c in calls if isinstance(c, TKGRequest)]
    assert len(tkg) == 1 and tkg[0].source_filter == ["v-inline"]
    assert not any(type(c).__name__ == "GraphRAGRequest" for c in calls)       # cross-source graph left out
    assert memory == []                                                       # and the user's memory


def test_an_unscoped_question_is_unchanged(monkeypatch):
    calls, memory = _run(monkeypatch, None)
    assert all(getattr(c, "version_id", None) is None for c in calls)
    assert any(type(c).__name__ == "GraphRAGRequest" for c in calls) and memory == ["what happened?"]


def test_agents_pass_the_scope_on(monkeypatch):
    sent = []

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [], "subgraph_edges": [], "temporal_facts": []}

    class Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            sent.append((url, json))
            return _Resp()

    for mod in (vector_search_agent, keyword_search_agent, tkg_agent):
        monkeypatch.setattr(mod.httpx, "Client", Client)
    monkeypatch.setenv("TKG_GROUNDED_ANSWERS", "false")
    monkeypatch.setattr(tkg_agent, "_summarise", lambda q, t: "")
    vector_search_agent._do_vector_search(VectorSearchRequest(query="q", branch_id="b1", version_id=["v1"]))
    keyword_search_agent._do_keyword_search(BM25SearchRequest(query="q", branch_id="b1", version_id=["v1"]))
    tkg_agent._do_tkg(TKGRequest(query="q", branch_id="b1", source_filter=["v1"]))
    bodies = {url.rsplit("/", 1)[-1]: body for url, body in sent if "/personal/search/" in url}
    assert bodies["vector"]["version_id"] == ["v1"] and bodies["bm25"]["version_id"] == ["v1"]
    assert (bodies["tkg"]["version"], bodies["tkg"]["source_filter"]) == (2, ["v1"])



def test_the_tkg_agent_gets_longer_than_a_plain_search(monkeypatch):
    waits = {}

    async def plan(question, inference_url):
        return QueryPlan(modes={"vector", "keyword", "tkg"})

    async def call_agent(url, payload, *, request_id, timeout=30):
        waits[type(payload).__name__] = timeout
        return ""

    async def nothing(*a, **kw):
        return ""

    monkeypatch.setattr(pipeline, "build_query_plan", plan)
    monkeypatch.setattr(pipeline, "_call_agent", call_agent)
    monkeypatch.setattr(pipeline, "gather_memory_context", nothing)
    monkeypatch.setattr(pipeline, "_llm_answer", nothing)
    monkeypatch.setattr(pipeline, "archive_turn", nothing)
    monkeypatch.setattr(memory_client, "fire_and_forget", lambda coro: coro.close())
    asyncio.run(pipeline.run_rag_pipeline("what happened?", user_id="u1", branch_id="u1"))
    assert waits["TKGRequest"] == pipeline.TKG_AGENT_TIMEOUT == 90
    assert waits["VectorSearchRequest"] == waits["BM25SearchRequest"] == 30
