import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp.server.fastmcp import FastMCP
from app.tools.chat import register


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
    register(mcp)
    return {n: mcp._tool_manager._tools[n].fn for n in mcp._tool_manager._tools}


def _chat_response(**kwargs):
    return {
        "response": "ok",
        "session_id": "sess-1",
        "search_triggered": False,
        "search_results": [],
        **kwargs,
    }


# raptor_chat

@pytest.mark.asyncio
async def test_chat_basic_response(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response(response="hello back")

    result = json.loads(await tools["raptor_chat"](message="hello", ctx=mock_ctx))

    assert result["response"] == "hello back"
    assert result["session_id"] == "sess-1"
    assert result["search_triggered"] is False
    assert result["search_results"] == []


@pytest.mark.asyncio
async def test_chat_sends_session_id(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response(session_id="sess-99")

    await tools["raptor_chat"](message="hi", session_id="sess-99", ctx=mock_ctx)

    body = mock_client.post_json.call_args[0][1]
    assert body["session_id"] == "sess-99"


@pytest.mark.asyncio
async def test_chat_no_session_id_omitted_from_body(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response()

    await tools["raptor_chat"](message="hi", ctx=mock_ctx)

    body = mock_client.post_json.call_args[0][1]
    assert "session_id" not in body


@pytest.mark.asyncio
async def test_chat_valid_history_forwarded(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response()
    history = [
        {"role": "user",      "content": "what is raptor?"},
        {"role": "assistant", "content": "raptor is..."},
    ]

    await tools["raptor_chat"](message="tell me more", history=history, ctx=mock_ctx)

    body = mock_client.post_json.call_args[0][1]
    assert len(body["history"]) == 2


@pytest.mark.asyncio
async def test_chat_invalid_history_entries_skipped(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response()
    history = [
        {"role": "user", "content": "valid"},
        {"bad": "entry"},
        "not a dict",
    ]

    await tools["raptor_chat"](message="hi", history=history, ctx=mock_ctx)

    body = mock_client.post_json.call_args[0][1]
    assert len(body["history"]) == 1
    assert body["history"][0]["content"] == "valid"


@pytest.mark.asyncio
async def test_chat_empty_history_omitted_from_body(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response()

    await tools["raptor_chat"](message="hi", history=[], ctx=mock_ctx)

    body = mock_client.post_json.call_args[0][1]
    assert "history" not in body


@pytest.mark.asyncio
async def test_chat_uses_correct_url(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = _chat_response()

    await tools["raptor_chat"](message="hi", ctx=mock_ctx)

    assert mock_client.post_json.call_args[0][0] == "/chat/chat"


@pytest.mark.asyncio
async def test_chat_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("LLM unavailable")

    with pytest.raises(Exception, match="LLM unavailable"):
        await tools["raptor_chat"](message="hi", ctx=mock_ctx)
