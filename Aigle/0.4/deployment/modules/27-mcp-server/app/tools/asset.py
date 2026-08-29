from __future__ import annotations

import base64
import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.core.config import get_settings
from app.tools import get_client

logger = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = get_settings().max_upload_bytes


class MCPToolError(Exception):
    pass

_LIST_DESC = """\
List assets uploaded by the authenticated user, with optional keyword and date filters.

Each entry: asset_path, version_id, primary_filename, upload_date, status, checksum.
Pagination: total_count, total_pages — iterate by incrementing page.
"""

_GET_URL_DESC = """\
Get a presigned 24-hour download URL for a specific asset version.

Returns: metadata, primary_file (filename, version_id, content_type, url),
and associated_file_N entries for derived files (e.g. video frames).
"""

_UPLOAD_DESC = """\
Upload a file to Raptor and trigger async AI processing.

File is stored in LakeFS; a Kafka message starts the appropriate worker:
  video → chunking, OCR, ASR, LVLM, graph indexing
  audio → diarisation, ASR, classification
  document → layout analysis, summarisation, graph indexing
  image → feature extraction, indexing

Returns immediately with asset_path, version_id, and exists (dedup flag).
Poll raptor_check_status with the correlation_id from the processing pipeline.
"""


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_LIST_DESC.strip())
    async def raptor_list_assets(
        keyword: Annotated[Optional[str], "Partial filename filter (case-insensitive)"] = None,
        start_date: Annotated[Optional[str], "Return assets uploaded on or after this date (ISO 8601)"] = None,
        end_date: Annotated[Optional[str], "Return assets uploaded on or before this date (ISO 8601)"] = None,
        page: Annotated[int, "Page number, 1-based"] = 1,
        page_size: Annotated[int, "Results per page (1–100)"] = 10,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_list_assets: page={page} page_size={page_size} keyword={keyword!r}")

        params: dict = {"page": page, "page_size": page_size}
        if keyword:
            params["keyword"] = keyword
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            client = await get_client(ctx)
            data = await client.get_json("/asset/users/commits", params=params, tool_name="raptor_list_assets")
        except Exception as exc:
            await ctx.error(f"raptor_list_assets failed: {exc}")
            raise

        total = data.get("total_count", 0)
        pages = data.get("total_pages", 1)
        n = len(data.get("commits", []))
        await ctx.info(f"raptor_list_assets: {total} total, {n} on page {page}/{pages}")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_GET_URL_DESC.strip())
    async def raptor_get_asset_url(
        asset_path: Annotated[str, "LakeFS asset path, e.g. 'video/mp4/my-video'. From raptor_list_assets."],
        version_id: Annotated[str, "64-char hex version ID. From raptor_list_assets."],
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_get_asset_url: asset_path={asset_path!r} version_id={version_id[:16]!r}…")

        try:
            client = await get_client(ctx)
            data = await client.get_json(
                f"/asset/filedownload/{asset_path}/{version_id}",
                tool_name="raptor_get_asset_url",
            )
        except Exception as exc:
            await ctx.error(f"raptor_get_asset_url failed: {exc}")
            raise

        assoc_count = sum(1 for k in data if k.startswith("associated_file_"))
        await ctx.info(f"raptor_get_asset_url: presigned URL obtained (+{assoc_count} associated file(s))")
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.tool(description=_UPLOAD_DESC.strip())
    async def raptor_upload_asset(
        filename: Annotated[str, "Original filename including extension, e.g. 'interview.mp4'"],
        content_base64: Annotated[str, "File contents as standard Base64 (RFC 4648)"],
        content_type: Annotated[str, "MIME type, e.g. 'video/mp4', 'application/pdf'"] = "application/octet-stream",
        archive_ttl_days: Annotated[Optional[int], "Days until asset is archived. Omit for permanent storage."] = None,
        destroy_ttl_days: Annotated[Optional[int], "Days until asset is deleted. Must exceed archive_ttl_days."] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(f"raptor_upload_asset: filename={filename!r} content_type={content_type!r}")

        try:
            file_bytes = base64.b64decode(content_base64)
        except Exception as exc:
            await ctx.error(f"raptor_upload_asset: base64 decode failed: {exc}")
            raise MCPToolError(f"Invalid base64 content: {exc}") from exc

        if len(file_bytes) > _MAX_UPLOAD_BYTES:
            raise MCPToolError(
                f"File exceeds {_MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB limit "
                f"({len(file_bytes) / 1024 / 1024:.1f} MB)"
            )

        extra_data: dict = {}
        if archive_ttl_days is not None:
            extra_data["archive_date_or_ttl"] = str(archive_ttl_days)
        if destroy_ttl_days is not None:
            extra_data["destroy_date_or_ttl"] = str(destroy_ttl_days)

        try:
            client = await get_client(ctx)
            data = await client.upload_file(
                path="/asset/fileupload_analysis",
                field_name="primary_file",
                filename=filename,
                content=file_bytes,
                content_type=content_type,
                extra_data=extra_data or None,
                tool_name="raptor_upload_asset",
            )
        except Exception as exc:
            await ctx.error(f"raptor_upload_asset: upload failed: {exc}")
            raise

        exists = (data.get("upload_result", {}).get("existence_info") or {}).get("exists", False)
        await ctx.info(f"raptor_upload_asset: complete exists={exists}")
        return json.dumps({
            "asset_path":       data.get("upload_result", {}).get("asset_path", ""),
            "version_id":       data.get("upload_result", {}).get("version_id", ""),
            "size_bytes":       len(file_bytes),
            "exists":           exists,
            "correlation_id":   (data.get("processing_result") or {}).get("correlation_id"),
        }, ensure_ascii=False, indent=2)
