"""DA-09a: test all tools with mocked Module 07 and Module 12 responses."""
from __future__ import annotations

import httpx
import pytest
import respx
from app.core.config import DocAgentSettings

M07_URL = "http://mock-module07:8010"
M12_URL = "http://mock-module12:8020"

MOCK_SETTINGS = DocAgentSettings(
    module07_url=M07_URL,
    module12_sync_url=M12_URL,
    qdrant_url="http://mock-qdrant:6333",
    llm_model="test-llm",
    llm_timeout=5.0,
    doc12_timeout=5.0,
)

SAMPLE_TEXT = "This is a test document about financial results for Q1 2026."
M07_INFER_URL = f"{M07_URL}/inference/infer"
M12_ANALYZE_URL = f"{M12_URL}/analyze"


# ── Module 07 response helpers ─────────────────────────────────────────────────
def m07_ok(response_text: str) -> dict:
    return {"result": {"response": response_text}, "success": True}


def m12_ok() -> dict:
    return {
        "filename": "test.pdf",
        "total_chunks": 1,
        "chunks": [{"id": "abc", "payload": {"text": SAMPLE_TEXT, "chunk_index": 0}}],
        "full_text": SAMPLE_TEXT,
        "summary": "Test summary",
    }


# ── ClassifyTool ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_classify_tool_success():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(
        200, json=m07_ok('{"category": "財務報告", "subcategory": "季報", "confidence": 0.9, "reason": "含財務數據"}')
    ))
    from app.tools.classify import ClassifyTool
    tool = ClassifyTool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT)
    assert result["category"] == "財務報告"
    assert result["confidence"] == 0.9


@pytest.mark.asyncio
@respx.mock
async def test_llm_tool_requests_pass_engine_ollama_for_the_mlflow_fallback():
    route = respx.post(M07_INFER_URL).mock(return_value=httpx.Response(200, json=m07_ok("ok")))
    from app.tools.classify import ClassifyTool
    await ClassifyTool(settings=MOCK_SETTINGS).run(full_text=SAMPLE_TEXT)
    assert route.calls.last.request.content
    import json
    assert json.loads(route.calls.last.request.content)["engine"] == "ollama"


@pytest.mark.asyncio
@respx.mock
async def test_classify_tool_module07_timeout():
    respx.post(M07_INFER_URL).mock(side_effect=httpx.TimeoutException("timeout"))
    from app.tools.base import LLMToolError
    from app.tools.classify import ClassifyTool
    tool = ClassifyTool(settings=MOCK_SETTINGS)
    with pytest.raises(LLMToolError, match="timeout"):
        await tool.run(full_text=SAMPLE_TEXT)


@pytest.mark.asyncio
@respx.mock
async def test_classify_tool_module07_unavailable():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(503))
    from app.tools.base import LLMToolError
    from app.tools.classify import ClassifyTool
    tool = ClassifyTool(settings=MOCK_SETTINGS)
    with pytest.raises(LLMToolError) as exc_info:
        await tool.run(full_text=SAMPLE_TEXT)
    assert exc_info.value.status_code == 503


# ── ExtractTool ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_extract_tool_with_fields():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(
        200, json=m07_ok('{"revenue": "$1,000,000", "period": "Q1 2026"}')
    ))
    from app.tools.extract import ExtractTool
    tool = ExtractTool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT, fields=["revenue", "period"])
    assert "extracted" in result
    assert result["extracted"]["revenue"] == "$1,000,000"


# ── SummarizeTool ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_summarize_tool_success():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(
        200, json=m07_ok("這是一份關於 Q1 2026 財務結果的測試文件。")
    ))
    from app.tools.summarize import SummarizeTool
    tool = SummarizeTool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT)
    assert "summary" in result
    assert len(result["summary"]) > 0


# ── CompareTool ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_compare_tool_success():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(
        200, json=m07_ok('{"similarities": ["both financial"], "differences": ["different periods"], "summary": "similar docs"}')
    ))
    from app.tools.compare import CompareTool
    tool = CompareTool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT, compare_text="Q2 2026 financial results.")
    assert "similarities" in result


@pytest.mark.asyncio
async def test_compare_tool_missing_compare_text():
    from app.tools.compare import CompareTool
    tool = CompareTool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT)
    assert "error" in result


# ── QATool ────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_qa_tool_success():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(
        200, json=m07_ok('{"answer": "Q1 2026", "confidence": 0.95, "source_hint": "第一段"}')
    ))
    from app.tools.qa import QATool
    tool = QATool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT, question="What period does this cover?")
    assert result["answer"] == "Q1 2026"


@pytest.mark.asyncio
async def test_qa_tool_missing_question():
    from app.tools.qa import QATool
    tool = QATool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT)
    assert "error" in result


# ── ParseTableTool ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_parse_table_tool_success():
    respx.post(M07_INFER_URL).mock(return_value=httpx.Response(
        200, json=m07_ok('{"tables": [{"title": "Q1", "headers": ["Q", "Rev"], "rows": [["Q1", "250k"]]}]}')
    ))
    from app.tools.parse_table import ParseTableTool
    tool = ParseTableTool(settings=MOCK_SETTINGS)
    result = await tool.run(full_text=SAMPLE_TEXT)
    assert "tables" in result
    assert len(result["tables"]) == 1


# ── Doc12Adapter ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_doc12_adapter_success():
    respx.post(M12_ANALYZE_URL).mock(return_value=httpx.Response(200, json=m12_ok()))
    from app.adapters.doc12_adapter import Doc12Adapter
    adapter = Doc12Adapter(settings=MOCK_SETTINGS)
    doc = await adapter.process(b"%PDF-1.4 test", "test.pdf")
    assert doc.full_text == SAMPLE_TEXT
    assert doc.total_chunks == 1


@pytest.mark.asyncio
@respx.mock
async def test_doc12_adapter_timeout():
    respx.post(M12_ANALYZE_URL).mock(side_effect=httpx.TimeoutException("timeout"))
    from app.adapters.doc12_adapter import Doc12Adapter
    adapter = Doc12Adapter(settings=MOCK_SETTINGS)
    with pytest.raises(httpx.TimeoutException):
        await adapter.process(b"data", "test.pdf")


@pytest.mark.asyncio
@respx.mock
async def test_doc12_adapter_unavailable():
    respx.post(M12_ANALYZE_URL).mock(side_effect=httpx.ConnectError("refused"))
    from app.adapters.doc12_adapter import Doc12Adapter
    adapter = Doc12Adapter(settings=MOCK_SETTINGS)
    with pytest.raises(RuntimeError, match="unreachable"):
        await adapter.process(b"data", "test.pdf")
