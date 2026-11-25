import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.services.raptor_client import MCPToolError
from app.tools import a2a, graph, pipeline, processing
from mcp.server.fastmcp import FastMCP


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


# ── raptor_tkg_query: contract v2 / raptor_temporal_capabilities ─────────────

@pytest.mark.asyncio
async def test_tkg_query_sort_or_cursor_asks_for_v2_and_reports_the_page(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"matched_entities": [], "temporal_facts": [{"fact_id": "f1"}],
                                          "version": 2, "complete": False, "total": 7, "cursor": "next"}
    out = json.loads(await tools["raptor_tkg_query"](query="Android", sort="relevance", ctx=mock_ctx))
    body = mock_client.post_json.call_args[0][1]
    assert (body["version"], body["sort"]) == (2, "relevance") and "cursor" not in body
    assert (out["complete"], out["total"], out["next_cursor"]) == (False, 7, "next")
    await tools["raptor_tkg_query"](query="Android", cursor="next", ctx=mock_ctx)
    assert mock_client.post_json.call_args[0][1]["cursor"] == "next"


@pytest.mark.asyncio
async def test_tkg_query_without_sort_or_cursor_stays_v1(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"matched_entities": [], "temporal_facts": []}
    out = json.loads(await tools["raptor_tkg_query"](query="Android", ctx=mock_ctx))
    assert "version" not in mock_client.post_json.call_args[0][1] and "next_cursor" not in out


@pytest.mark.asyncio
async def test_temporal_capabilities(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"contract_versions": [1, 2]}
    out = json.loads(await tools["raptor_temporal_capabilities"](ctx=mock_ctx))
    assert out == {"contract_versions": [1, 2]}
    assert mock_client.get_json.call_args[0][0] == "/temporal/capabilities"


# ── raptor_temporal_parse (TMP-7) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_parse_posts_the_question_and_passes_the_answer_through(mock_client, mock_ctx, tools):
    parsed = {"status": "RESOLVED", "match": "inside", "window": {"start": "2012-01-01T00:00:00.000+08:00", "end": None}}
    mock_client.post_json.return_value = parsed
    out = json.loads(await tools["raptor_temporal_parse"](text="2011年以後發生什麼", ctx=mock_ctx))
    assert out == parsed
    assert mock_client.post_json.call_args[0][:2] == ("/temporal/parse", {"text": "2011年以後發生什麼"})
    await tools["raptor_temporal_parse"](text="last year", timezone="Asia/Taipei",
                                         anchor="2015-03-20T10:00:00+08:00", ctx=mock_ctx)
    assert mock_client.post_json.call_args[0][1] == {"text": "last year", "timezone": "Asia/Taipei",
                                                     "anchor": "2015-03-20T10:00:00+08:00"}


# ── raptor_temporal_reason ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_reason_reads_or_queues(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"status": "consistent", "derived_facts": [{"fact_id": "tf-der-1"}]}
    out = json.loads(await tools["raptor_temporal_reason"](source_version_id="v 1", ctx=mock_ctx))
    assert out["derived_facts"][0]["fact_id"] == "tf-der-1"
    assert mock_client.get_json.call_args[0][0] == "/temporal/reason/v%201"
    mock_client.post_json.return_value = {"queued": True}
    out = json.loads(await tools["raptor_temporal_reason"](source_version_id="v1", recompute=True, ctx=mock_ctx))
    assert out == {"queued": True} and mock_client.post_json.call_args[0][0] == "/temporal/reason/v1/recompute"


# ── raptor_temporal_status ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_status_passes_the_job_through(mock_client, mock_ctx, tools):
    job = {"status": "partial", "reason": "unsupported_language", "facts": 3,
           "segments_with_problems": [{"chunk_id": "c9", "status": "unsupported"}]}
    mock_client.get_json.return_value = job
    out = json.loads(await tools["raptor_temporal_status"](source_version_id="881212da-8593", ctx=mock_ctx))
    assert out == job
    assert mock_client.get_json.call_args[0][0] == "/temporal/status/881212da-8593"


@pytest.mark.asyncio
async def test_temporal_status_version_id_stays_one_path_segment(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {}
    await tools["raptor_temporal_status"](source_version_id="../../search/tkg?x=1", ctx=mock_ctx)
    assert mock_client.get_json.call_args[0][0] == "/temporal/status/..%2F..%2Fsearch%2Ftkg%3Fx%3D1"


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


@pytest.mark.asyncio
async def test_tkg_query_reference_asks_for_v2_and_reports_its_resolution(mock_client, mock_ctx, tools):
    ref = {"status": "ambiguous", "text": "the acquisition", "candidates": [{"fact_id": "a"}, {"fact_id": "b"}]}
    mock_client.post_json.return_value = {"matched_entities": [], "temporal_facts": [], "version": 2,
                                          "complete": True, "total": 0, "cursor": None, "reference": ref}
    out = json.loads(await tools["raptor_tkg_query"](query="what happened", reference="the acquisition",
                                                     reference_relations=["after"], ctx=mock_ctx))
    body = mock_client.post_json.call_args[0][1]
    assert (body["version"], body["reference"], body["reference_relations"]) == (
        2, {"text": "the acquisition"}, ["after"])
    assert out["reference"] == ref


# ── raptor_temporal_ingest (P21) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_ingest_inline_text_and_a_stable_retry_key(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"source_version_id": "v1", "status": "pending",
                                          "expires_at": "2026-10-28T00:00:00Z"}
    out = json.loads(await tools["raptor_temporal_ingest"](content="Google acquired Android in 2005.",
                                                           ctx=mock_ctx))
    path, body = mock_client.post_json.call_args[0][:2]
    assert path == "/temporal/ingest" and out["source_version_id"] == "v1"
    assert (body["content"], body["content_type"]) == ("Google acquired Android in 2005.", "text/plain")
    assert "asset" not in body and body["idempotency_key"].startswith("mcp-")
    key = body["idempotency_key"]
    await tools["raptor_temporal_ingest"](content="Google acquired Android in 2005.", ctx=mock_ctx)
    assert mock_client.post_json.call_args[0][1]["idempotency_key"] == key          # same input, same key


@pytest.mark.asyncio
async def test_temporal_ingest_an_uploaded_asset(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"source_version_id": "v9"}
    await tools["raptor_temporal_ingest"](asset_path="u/report", version_id="v9", idempotency_key="k", ctx=mock_ctx)
    body = mock_client.post_json.call_args[0][1]
    assert body == {"asset": {"asset_path": "u/report", "version_id": "v9"}, "idempotency_key": "k"}


@pytest.mark.asyncio
async def test_temporal_ingest_with_a_question_answers_from_that_source(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"answer_status": "pending", "answer": None}
    out = json.loads(await tools["raptor_temporal_ingest"](content="notes", question="When?", wait_seconds=0,
                                                           ctx=mock_ctx))
    body = mock_client.post_json.call_args[0][1]
    assert (body["operation"], body["question"], body["wait_seconds"]) == ("ingest_and_query", "When?", 0)
    assert out["answer_status"] == "pending"
    key = body["idempotency_key"]
    await tools["raptor_temporal_ingest"](content="notes", question="When?", wait_seconds=120, ctx=mock_ctx)
    assert mock_client.post_json.call_args[0][1]["idempotency_key"] == key      # asking again later: same request
