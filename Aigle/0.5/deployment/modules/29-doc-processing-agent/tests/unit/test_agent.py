"""DA-09a: verify correct tool chain for all 7 task types."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from app.adapters.doc12_adapter import ProcessedDoc
from app.agent.orchestrator import DocAgentOrchestrator
from app.core.config import DocAgentSettings

MOCK_SETTINGS = DocAgentSettings(
    module07_url="http://mock-m07:8010",
    module12_sync_url="http://mock-m12:8020",
    qdrant_url="http://mock-qdrant:6333",
)

SAMPLE_DOC = ProcessedDoc(
    filename="test.pdf",
    full_text="Test document content about Q1 2026 financial results.",
    chunks=[{"id": "c1", "payload": {"text": "Test content", "chunk_index": 0}}],
    total_chunks=1,
)


@pytest.fixture
def orchestrator():
    """Orchestrator with all tools mocked."""
    orch = DocAgentOrchestrator(settings=MOCK_SETTINGS)
    # Mock all LLM tools
    orch._classify    = MagicMock(run=AsyncMock(return_value={"category": "財務報告"}))
    orch._extract     = MagicMock(run=AsyncMock(return_value={"extracted": {"revenue": "1M"}}))
    orch._summarize   = MagicMock(run=AsyncMock(return_value={"summary": "Test summary"}))
    orch._compare     = MagicMock(run=AsyncMock(return_value={"similarities": [], "differences": []}))
    orch._qa          = MagicMock(run=AsyncMock(return_value={"answer": "Q1 2026"}))
    orch._parse_table = MagicMock(run=AsyncMock(return_value={"tables": []}))
    # Mock search tool
    orch._search_tool = MagicMock(forward=MagicMock(return_value={"results": [], "indexed_chunks": 1}))
    return orch


@pytest.mark.asyncio
async def test_classify_calls_only_classify_tool(orchestrator):
    result = await orchestrator.run("classify", SAMPLE_DOC)
    orchestrator._classify.run.assert_called_once()
    orchestrator._extract.run.assert_not_called()
    orchestrator._summarize.run.assert_not_called()
    assert result["task"] == "classify"
    assert result["category"] == "財務報告"


@pytest.mark.asyncio
async def test_extract_calls_only_extract_tool(orchestrator):
    result = await orchestrator.run("extract", SAMPLE_DOC, {"fields": ["revenue"]})
    orchestrator._extract.run.assert_called_once_with(
        full_text=SAMPLE_DOC.full_text, fields=["revenue"]
    )
    orchestrator._classify.run.assert_not_called()
    assert "extracted" in result


@pytest.mark.asyncio
async def test_summarize_calls_only_summarize_tool(orchestrator):
    result = await orchestrator.run("summarize", SAMPLE_DOC, {"max_length": 300})
    orchestrator._summarize.run.assert_called_once_with(
        full_text=SAMPLE_DOC.full_text, max_length=300
    )
    orchestrator._classify.run.assert_not_called()
    assert "summary" in result


@pytest.mark.asyncio
async def test_compare_calls_only_compare_tool(orchestrator):
    opts = {"compare_text": "Another document."}
    await orchestrator.run("compare", SAMPLE_DOC, opts)
    orchestrator._compare.run.assert_called_once()
    orchestrator._classify.run.assert_not_called()


@pytest.mark.asyncio
async def test_qa_calls_only_qa_tool(orchestrator):
    result = await orchestrator.run("qa", SAMPLE_DOC, {"question": "What period?"})
    orchestrator._qa.run.assert_called_once_with(
        full_text=SAMPLE_DOC.full_text, question="What period?"
    )
    orchestrator._classify.run.assert_not_called()
    assert "answer" in result


@pytest.mark.asyncio
async def test_parse_table_calls_only_parse_table_tool(orchestrator):
    result = await orchestrator.run("parse_table", SAMPLE_DOC)
    orchestrator._parse_table.run.assert_called_once()
    orchestrator._classify.run.assert_not_called()
    assert "tables" in result


@pytest.mark.asyncio
async def test_search_calls_embedding_search_not_llm(orchestrator):
    """search task must use EmbeddingSearchTool, NOT any LLM tool."""
    result = await orchestrator.run("search", SAMPLE_DOC, {"query": "financial results"})
    orchestrator._search_tool.forward.assert_called_once()
    # No LLM tool should be called
    orchestrator._classify.run.assert_not_called()
    orchestrator._summarize.run.assert_not_called()
    orchestrator._qa.run.assert_not_called()
    assert "results" in result


@pytest.mark.asyncio
async def test_search_requires_query(orchestrator):
    result = await orchestrator.run("search", SAMPLE_DOC, {})
    assert "error" in result


@pytest.mark.asyncio
async def test_unknown_task_returns_error(orchestrator):
    result = await orchestrator.run("unknown_task", SAMPLE_DOC)  # type: ignore
    assert "error" in result


@pytest.mark.asyncio
async def test_tool_error_is_caught_and_returned(orchestrator):
    from app.tools.base import LLMToolError
    orchestrator._classify.run = AsyncMock(side_effect=LLMToolError("Module 07 down", 503))
    result = await orchestrator.run("classify", SAMPLE_DOC)
    assert "error" in result
    assert result["status_code"] == 503
