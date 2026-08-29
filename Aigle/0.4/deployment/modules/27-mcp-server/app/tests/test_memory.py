import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp.server.fastmcp import FastMCP
from app.tools.memory import register


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


@pytest.mark.asyncio
async def test_retrieve_sends_query_and_top_k(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"results": []}

    await tools["raptor_memory_retrieve"](query="last quarter revenue", top_k=3, ctx=mock_ctx)

    params = mock_client.get_json.call_args[1]["params"]
    assert params["query"] == "last quarter revenue"
    assert params["top_k"] == 3


@pytest.mark.asyncio
async def test_retrieve_uses_correct_path(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"results": []}

    await tools["raptor_memory_retrieve"](query="hello", ctx=mock_ctx)

    assert mock_client.get_json.call_args[0][0] == "/memory/retrieve"


@pytest.mark.asyncio
async def test_retrieve_default_top_k(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"results": []}

    await tools["raptor_memory_retrieve"](query="hello", ctx=mock_ctx)

    assert mock_client.get_json.call_args[1]["params"]["top_k"] == 5


@pytest.mark.asyncio
async def test_retrieve_returns_results(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {
        "results": [{"text": "user prefers Traditional Chinese", "score": 0.91}],
    }

    result = json.loads(await tools["raptor_memory_retrieve"](query="language preference", ctx=mock_ctx))

    assert result["results"][0]["text"] == "user prefers Traditional Chinese"


@pytest.mark.asyncio
async def test_retrieve_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.get_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_retrieve"](query="hello", ctx=mock_ctx)


# ── raptor_memory_store ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_sends_text_and_frame_type(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"frame_id": "f1"}

    await tools["raptor_memory_store"](text="likes tea", frame_type="preference", ctx=mock_ctx)

    path, body = mock_client.post_json.call_args[0][0], mock_client.post_json.call_args[0][1]
    assert path == "/memory/store"
    assert body == {"text": "likes tea", "frame_type": "preference"}


@pytest.mark.asyncio
async def test_store_default_frame_type_is_fact(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"frame_id": "f1"}

    await tools["raptor_memory_store"](text="x", ctx=mock_ctx)

    assert mock_client.post_json.call_args[0][1]["frame_type"] == "fact"


@pytest.mark.asyncio
async def test_store_includes_session_id_when_given(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"frame_id": "f1"}

    await tools["raptor_memory_store"](text="x", session_id="sess1", ctx=mock_ctx)

    assert mock_client.post_json.call_args[0][1]["session_id"] == "sess1"


@pytest.mark.asyncio
async def test_store_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_store"](text="x", ctx=mock_ctx)


# ── raptor_memory_timeline ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeline_sends_page_params(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"turns": []}

    await tools["raptor_memory_timeline"](page=2, page_size=50, ctx=mock_ctx)

    assert mock_client.get_json.call_args[0][0] == "/memory/timeline"
    params = mock_client.get_json.call_args[1]["params"]
    assert params == {"page": 2, "page_size": 50}


@pytest.mark.asyncio
async def test_timeline_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.get_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_timeline"](ctx=mock_ctx)


# ── raptor_memory_multimedia_search ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_multimedia_search_sends_query(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"results": []}

    await tools["raptor_memory_multimedia_search"](query="whiteboard photo", ctx=mock_ctx)

    path, body = mock_client.post_json.call_args[0][0], mock_client.post_json.call_args[0][1]
    assert path == "/memory/multimedia/search"
    assert body["query"] == "whiteboard photo" and "media_type" not in body


@pytest.mark.asyncio
async def test_multimedia_search_includes_media_type_when_given(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"results": []}

    await tools["raptor_memory_multimedia_search"](query="x", media_type="image", ctx=mock_ctx)

    assert mock_client.post_json.call_args[0][1]["media_type"] == "image"


@pytest.mark.asyncio
async def test_multimedia_search_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_multimedia_search"](query="x", ctx=mock_ctx)


# ── raptor_memory_session_summaries ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_summaries_uses_correct_path(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"summaries": []}

    await tools["raptor_memory_session_summaries"](session_id="sess1", ctx=mock_ctx)

    assert mock_client.get_json.call_args[0][0] == "/memory/sessions/sess1/summaries"


@pytest.mark.asyncio
async def test_session_summaries_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.get_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_session_summaries"](session_id="sess1", ctx=mock_ctx)


# ── raptor_memory_compact ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compact_defaults_to_no_session_param(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"status": "ok"}

    await tools["raptor_memory_compact"](ctx=mock_ctx)

    path = mock_client.post_json.call_args[0][0]
    kwargs = mock_client.post_json.call_args[1]
    assert path == "/memory/compact"
    assert kwargs["params"] is None


@pytest.mark.asyncio
async def test_compact_passes_session_id_as_query_param(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"status": "ok"}

    await tools["raptor_memory_compact"](session_id="sess1", ctx=mock_ctx)

    assert mock_client.post_json.call_args[1]["params"] == {"session_id": "sess1"}


@pytest.mark.asyncio
async def test_compact_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_compact"](ctx=mock_ctx)


# ── raptor_memory_compact_evaluate ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compact_evaluate_sends_defaults(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"estimated_tokens": 1000}

    await tools["raptor_memory_compact_evaluate"](ctx=mock_ctx)

    path, body = mock_client.post_json.call_args[0][0], mock_client.post_json.call_args[0][1]
    assert path == "/memory/compact/evaluate"
    assert body["context_window"] == 128000 and "session_id" not in body


@pytest.mark.asyncio
async def test_compact_evaluate_includes_session_id_when_given(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"estimated_tokens": 1000}

    await tools["raptor_memory_compact_evaluate"](session_id="sess1", ctx=mock_ctx)

    assert mock_client.post_json.call_args[0][1]["session_id"] == "sess1"


@pytest.mark.asyncio
async def test_compact_evaluate_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_compact_evaluate"](ctx=mock_ctx)


# ── raptor_memory_archive ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_archive_sends_turn_body(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"status": "ok"}

    await tools["raptor_memory_archive"](
        user_message="hi", assistant_response="hello there", ctx=mock_ctx)

    path, body = mock_client.post_json.call_args[0][0], mock_client.post_json.call_args[0][1]
    assert path == "/memory/archive"
    assert body == {"user_message": "hi", "assistant_response": "hello there"}
    assert mock_client.post_json.call_args[1]["params"] is None


@pytest.mark.asyncio
async def test_archive_passes_session_id_as_query_param(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"status": "ok"}

    await tools["raptor_memory_archive"](
        user_message="hi", assistant_response="hello", session_id="sess1", ctx=mock_ctx)

    assert mock_client.post_json.call_args[1]["params"] == {"session_id": "sess1"}


@pytest.mark.asyncio
async def test_archive_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("boom")

    with pytest.raises(Exception, match="boom"):
        await tools["raptor_memory_archive"](
            user_message="hi", assistant_response="hello", ctx=mock_ctx)
