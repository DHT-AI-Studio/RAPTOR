import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp.server.fastmcp import FastMCP
from app.tools.asset import register, MCPToolError


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


_SMALL_B64 = base64.b64encode(b"hello").decode()


def _upload_response(asset_path="video/mp4/clip", version_id="v123", exists=False, correlation_id="cid-1"):
    """Nested response matching POST /asset/fileupload_analysis."""
    return {
        "upload_result": {
            "asset_path": asset_path,
            "version_id": version_id,
            "primary_filename": "clip.mp4",
            "checksum": "abc123",
            "existence_info": {"exists": exists, "message": ""},
        },
        "processing_result": {
            "correlation_id": correlation_id,
            "status": "queued" if not exists else "skipped",
        },
    }


# raptor_list_assets

@pytest.mark.asyncio
async def test_list_assets_sends_params(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"commits": [], "total_count": 0, "total_pages": 1}

    await tools["raptor_list_assets"](keyword="clip", page=2, page_size=5, ctx=mock_ctx)

    params = mock_client.get_json.call_args[1]["params"]
    assert params["keyword"] == "clip"
    assert params["page"] == 2
    assert params["page_size"] == 5


@pytest.mark.asyncio
async def test_list_assets_omits_none_filters(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"commits": [], "total_count": 0, "total_pages": 1}

    await tools["raptor_list_assets"](ctx=mock_ctx)

    params = mock_client.get_json.call_args[1]["params"]
    assert "keyword" not in params
    assert "start_date" not in params


# raptor_get_asset_url

@pytest.mark.asyncio
async def test_get_asset_url_builds_correct_path(mock_client, mock_ctx, tools):
    mock_client.get_json.return_value = {"primary_file": {"url": "http://cdn/file"}}
    version_id = "a" * 64

    await tools["raptor_get_asset_url"](
        asset_path="video/mp4/clip", version_id=version_id, ctx=mock_ctx
    )

    path = mock_client.get_json.call_args[0][0]
    assert path == f"/asset/filedownload/video/mp4/clip/{version_id}"


# raptor_upload_asset

@pytest.mark.asyncio
async def test_upload_returns_shaped_result(mock_client, mock_ctx, tools):
    mock_client.upload_file.return_value = _upload_response()

    result = json.loads(await tools["raptor_upload_asset"](
        filename="clip.mp4", content_base64=_SMALL_B64,
        content_type="video/mp4", ctx=mock_ctx,
    ))

    assert result["version_id"] == "v123"
    assert result["asset_path"] == "video/mp4/clip"
    assert result["size_bytes"] == len(b"hello")
    assert result["exists"] is False
    assert result["correlation_id"] == "cid-1"


@pytest.mark.asyncio
async def test_upload_uses_fileupload_analysis_endpoint(mock_client, mock_ctx, tools):
    mock_client.upload_file.return_value = _upload_response(version_id="v1", asset_path="p")

    await tools["raptor_upload_asset"](
        filename="clip.mp4", content_base64=_SMALL_B64,
        content_type="video/mp4", ctx=mock_ctx,
    )

    assert mock_client.upload_file.call_args[1]["path"] == "/asset/fileupload_analysis"


@pytest.mark.asyncio
async def test_upload_rejects_file_over_size_limit(mock_client, mock_ctx, tools):
    oversized = b"x" * 6  # 6 bytes

    with patch("app.tools.asset._MAX_UPLOAD_BYTES", 5):  # lower limit to 5 bytes for speed
        with pytest.raises(MCPToolError):
            await tools["raptor_upload_asset"](
                filename="big.mp4",
                content_base64=base64.b64encode(oversized).decode(),
                content_type="video/mp4", ctx=mock_ctx,
            )

    mock_client.upload_file.assert_not_called()


@pytest.mark.asyncio
async def test_upload_invalid_base64_returns_error_json(mock_client, mock_ctx, tools):
    with pytest.raises(MCPToolError):
        await tools["raptor_upload_asset"](
            filename="clip.mp4", content_base64="!!!invalid!!!",
            content_type="video/mp4", ctx=mock_ctx,
        )


@pytest.mark.asyncio
async def test_upload_detects_duplicate(mock_client, mock_ctx, tools):
    mock_client.upload_file.return_value = _upload_response(
        version_id="v-existing", exists=True
    )

    result = json.loads(await tools["raptor_upload_asset"](
        filename="clip.mp4", content_base64=_SMALL_B64,
        content_type="video/mp4", ctx=mock_ctx,
    ))

    assert result["exists"] is True
