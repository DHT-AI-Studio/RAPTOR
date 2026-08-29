import json

import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp.server.fastmcp import FastMCP

from app.services.raptor_client import MCPToolError
from app.tools import graph, a2a, processing, pipeline


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def mock_ctx(mock_client):
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.request_context.lifespan_context = {"raptor_client": mock_client}
    return ctx


@pytest.fixture
def tools():
    mcp = FastMCP("test")
    for mod in (graph, a2a, processing, pipeline):
        mod.register(mcp)
    return {n: mcp._tool_manager._tools[n].fn for n in mcp._tool_manager._tools}


# ── raptor_graph_query ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_query_shapes_triples_and_count(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {
        "matched_entities": [{"name": "Xi"}, {"name": "Trump"}],
        "relationships": [{"source": "Trump", "relation": "met", "target": "Xi"}],
    }
    out = json.loads(await tools["raptor_graph_query"](query="relations", ctx=mock_ctx))
    assert out["entity_count"] == 2
    assert out["triples"] == [{"subject": "Trump", "relation": "met", "object": "Xi"}]
    assert "summary" in out
    assert mock_client.post_json.call_args[0][0] == "/search/graphrag"


@pytest.mark.asyncio
async def test_graph_query_keeps_semantic_drops_structural(mock_client, mock_ctx, tools):
    # Real Raptor edge shape: start_id / end_id / properties.predicate + confidence.
    mock_client.post_json.return_value = {
        "matched_entities": [{"name": "Xi"}],
        "relationships": [
            {"type": "RELATION", "start_id": "xi", "end_id": "trump",
             "properties": {"predicate": "RELATION_GOOD_WITH", "confidence": 0.9}},
            {"type": "CO_OCCURS_WITH", "start_id": "xi", "end_id": "china", "properties": {}},
            {"type": "MENTIONED_IN", "start_id": "xi", "end_id": "doc1", "properties": {}},
        ],
    }
    out = json.loads(await tools["raptor_graph_query"](query="rel", ctx=mock_ctx))
    # only the semantic RELATION survives as a triple
    assert out["triples"] == [
        {"subject": "xi", "relation": "RELATION_GOOD_WITH", "object": "trump", "confidence": 0.9}]
    # CO_OCCURS_WITH target is summarised separately; MENTIONED_IN is dropped
    assert out["co_occurs_with"] == ["china"]


@pytest.mark.asyncio
async def test_graph_query_entity_folds_into_query(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"matched_entities": [], "relationships": []}
    await tools["raptor_graph_query"](query="who is", entity="Samsung", ctx=mock_ctx)
    assert mock_client.post_json.call_args[0][1]["query"] == "who is Samsung"


@pytest.mark.asyncio
async def test_graph_query_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("boom")
    with pytest.raises(Exception, match="boom"):
        await tools["raptor_graph_query"](query="x", ctx=mock_ctx)


# ── raptor_tkg_query ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tkg_query_triples_from_edges_facts_passthrough(mock_client, mock_ctx, tools):
    # TKG returns edges under subgraph_edges and temporal_facts as their own list.
    mock_client.post_json.return_value = {
        "matched_entities": [{"name": "A"}],
        "subgraph_edges": [
            {"type": "RELATION", "start_id": "A", "end_id": "B",
             "properties": {"predicate": "MEETS_WITH", "confidence": 0.8}},
        ],
        "temporal_facts": [{"id": "tf1", "entity": "A", "value": "B", "time_end": "2018-04-30"}],
    }
    out = json.loads(await tools["raptor_tkg_query"](
        query="timeline", time_start="2025-01-01", time_end="2026-12-31", ctx=mock_ctx))
    assert out["triples"] == [
        {"subject": "A", "relation": "MEETS_WITH", "object": "B", "confidence": 0.8}]
    assert out["temporal_facts"][0]["id"] == "tf1"      # facts passed through untouched
    body = mock_client.post_json.call_args[0][1]
    assert body["time_start"] == "2025-01-01" and body["time_end"] == "2026-12-31"
    assert mock_client.post_json.call_args[0][0] == "/search/tkg"


# ── raptor_a2a_direct / agent ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a2a_direct_normalises_answer_sources(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {
        "answer": "hello", "sources": [{"id": "s1"}], "graph_context": "g", "chunks_used": 1}
    out = json.loads(await tools["raptor_a2a_direct"](question="q", ctx=mock_ctx))
    assert out["answer"] == "hello" and out["sources"] == [{"id": "s1"}]
    body = mock_client.post_json.call_args[0][1]
    assert body["mode"] == "direct" and body["question"] == "q"


@pytest.mark.asyncio
async def test_a2a_agent_surfaces_trace(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {
        "answer": "a", "sources": [], "tool_calls": [{"tool": "raptor_search"}]}
    out = json.loads(await tools["raptor_a2a_agent"](question="q", ctx=mock_ctx))
    assert out["agent_trace"] == [{"tool": "raptor_search"}]
    assert mock_client.post_json.call_args[0][1]["mode"] == "agent"


# ── raptor_check_status ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_status_normalises_complete(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {
        "key": "document_orchestrator:c1", "m_type": "document",
        "value": {"step": "complete", "branch_id": "b", "progress": 1.0,
                  "chunks": [{"id": "x"}, {"id": "y"}]},
    }
    out = json.loads(await tools["raptor_check_status"](
        correlation_id="c1", m_type="document", ctx=mock_ctx))
    assert out["status"] == "complete"
    assert out["progress"] == 1.0
    assert out["error"] is None
    assert out["result"]["chunk_count"] == 2      # heavy chunks array collapsed
    assert "chunks" not in out["result"]
    assert mock_client.get_json.call_args[0][0] == "/processing/cache/document/c1"


@pytest.mark.asyncio
async def test_check_status_not_found(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"data": {}}
    with pytest.raises(MCPToolError, match="not found"):
        await tools["raptor_check_status"](correlation_id="nope", ctx=mock_ctx)


# ── raptor_query_orchestrate ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_orchestrate_wraps_a2a_direct(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"answer": "A", "sources": [{"id": "s"}]}
    out = json.loads(await tools["raptor_query_orchestrate"](query="q", ctx=mock_ctx))
    assert out["answer"] == "A" and out["sources"] == [{"id": "s"}]
    assert out["pipeline_used"] is None and out["confidence"] is None
    path, body = mock_client.post_json.call_args[0][0], mock_client.post_json.call_args[0][1]
    assert path == "/a2a/query" and body["mode"] == "direct"
