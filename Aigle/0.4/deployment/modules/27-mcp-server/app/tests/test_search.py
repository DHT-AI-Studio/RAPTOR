import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp.server.fastmcp import FastMCP
from app.tools.search import register


def _hit(id="h1", score=0.9, text="hello world", asset_path="video/mp4/clip",
         start_time="10.0", end_time="20.0", **extra):
    return {
        "id": id, "score": score,
        "payload": {
            "text": text, "asset_path": asset_path,
            "start_time": start_time, "end_time": end_time,
            "filename": "clip.mp4", "type": "videos", **extra,
        },
    }


def _video(video_id="v1", filename="clip.mp4", asset_url="http://cdn/v.mp4", segments=None):
    if segments is None:
        segments = [{"start_time": 0.0, "end_time": 10.0, "score": 0.8,
                     "text": "hello", "sources": ["bm25"]}]
    return {"video_id": video_id, "filename": filename,
            "asset_url": asset_url, "score": 0.8, "segments": segments}


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


# raptor_search

@pytest.mark.asyncio
async def test_search_shapes_hit(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"results": [_hit()]}

    result = json.loads(await tools["raptor_search"](query="hello", ctx=mock_ctx))
    hit = result[0]

    assert hit["id"] == "h1"
    assert hit["content"] == "hello world"
    assert hit["score"] == 0.9
    assert hit["asset_path"] == "video/mp4/clip"
    assert hit["start_time"] == "10.0"
    assert hit["end_time"] == "20.0"


@pytest.mark.asyncio
async def test_search_metadata_excludes_promoted_fields(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"results": [_hit()]}

    result = json.loads(await tools["raptor_search"](query="hello", ctx=mock_ctx))
    meta = result[0]["metadata"]

    assert "text" not in meta
    assert "asset_path" not in meta
    assert "start_time" not in meta
    assert "end_time" not in meta
    assert meta["filename"] == "clip.mp4"


@pytest.mark.asyncio
async def test_search_sends_optional_filters(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"results": []}

    await tools["raptor_search"](
        query="test", type="videos", speaker="alice", source="mp4", ctx=mock_ctx
    )

    body = mock_client.post_json.call_args[0][1]
    assert body["type"] == "videos"
    assert body["speaker"] == ["alice"]
    assert body["source"] == "mp4"


@pytest.mark.asyncio
async def test_search_omits_none_filters(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"results": []}

    await tools["raptor_search"](query="test", ctx=mock_ctx)

    body = mock_client.post_json.call_args[0][1]
    assert "type" not in body
    assert "speaker" not in body


@pytest.mark.asyncio
async def test_search_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("connection refused")

    with pytest.raises(Exception, match="connection refused"):
        await tools["raptor_search"](query="test", ctx=mock_ctx)


# raptor_video_search

@pytest.mark.asyncio
async def test_video_search_flattens_and_sorts_segments(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {
        "total": 1,
        "results": [_video(segments=[
            {"start_time": 0.0,  "end_time": 10.0, "score": 0.6, "text": "b", "sources": ["vector"]},
            {"start_time": 10.0, "end_time": 20.0, "score": 0.9, "text": "a", "sources": ["bm25"]},
        ])],
    }

    result = json.loads(await tools["raptor_video_search"](query="test", ctx=mock_ctx))

    assert len(result) == 2
    assert result[0]["score"] == 0.9
    assert result[0]["clip_url"] == "http://cdn/v.mp4"
    assert result[0]["start_time"] == 10.0


@pytest.mark.asyncio
async def test_video_search_asset_path_filter(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {
        "total": 2,
        "results": [_video(filename="alpha.mp4"), _video(filename="beta.mp4", video_id="v2")],
    }

    result = json.loads(await tools["raptor_video_search"](
        query="test", asset_path="video/mp4/alpha", ctx=mock_ctx
    ))

    assert len(result) == 1
    assert result[0]["filename"] == "alpha.mp4"


@pytest.mark.asyncio
async def test_video_search_asset_path_no_match(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"total": 1, "results": [_video(filename="gamma.mp4")]}

    result = json.loads(await tools["raptor_video_search"](
        query="test", asset_path="video/mp4/alpha", ctx=mock_ctx
    ))

    assert result == []


@pytest.mark.asyncio
async def test_video_search_uses_module13_url(mock_client, mock_ctx, tools):
    mock_client.post_json.return_value = {"total": 0, "results": []}

    await tools["raptor_video_search"](query="test", ctx=mock_ctx)

    assert mock_client.post_json.call_args[0][0] == "/search/video_search"


@pytest.mark.asyncio
async def test_video_search_api_error_returns_error_json(mock_client, mock_ctx, tools):
    mock_client.post_json.side_effect = Exception("timeout")

    with pytest.raises(Exception, match="timeout"):
        await tools["raptor_video_search"](query="test", ctx=mock_ctx)
