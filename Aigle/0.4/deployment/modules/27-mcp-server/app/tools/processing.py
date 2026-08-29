from __future__ import annotations

import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context, FastMCP

from app.services.raptor_client import MCPToolError
from app.tools import get_client

logger = logging.getLogger(__name__)

_STATUS_DESC = """\
Poll the async AI processing pipeline status for an uploaded asset.

Status values: queued → transcribing → extracting → indexing → complete | failed

Provide m_type for a direct lookup. Omit it to auto-detect across all media types.
"""


def _extract_payload(data: dict) -> dict:
    """The processing entry may be wrapped as {value:{...}} (cache/{m_type} path)
    or be the payload dict itself (cache/all scan). Return the inner payload."""
    inner = data.get("value")
    return inner if isinstance(inner, dict) else data


def _normalise_status(correlation_id: str, data: dict) -> dict:
    """Map the raw processing-cache entry onto {status, progress, result, error}.

    Status lives at ``value.step`` (queued → transcribing → extracting → indexing
    → complete | failed). On a terminal state the payload is returned as ``result``
    with the (potentially large) ``chunks`` array collapsed to a ``chunk_count``.
    """
    payload = _extract_payload(data)
    status = payload.get("step") or payload.get("status") or "unknown"
    result = None
    if status in ("complete", "failed"):
        result = dict(payload)
        chunks = result.get("chunks")
        if isinstance(chunks, list):
            result["chunk_count"] = len(chunks)
            result.pop("chunks", None)
    return {
        "correlation_id": correlation_id,
        "status": status,
        "progress": payload.get("progress"),
        "result": result,
        "error": payload.get("error"),
    }


def register(mcp: FastMCP) -> None:

    @mcp.tool(description=_STATUS_DESC.strip())
    async def raptor_check_status(
        correlation_id: Annotated[
            str,
            "Correlation ID returned by raptor_upload_asset. "
            "Poll until status is 'complete' or 'failed'.",
        ],
        m_type: Annotated[
            Optional[str],
            "Media type: 'document', 'video', 'image', or 'audio'. "
            "Omit to auto-detect (slower — scans all types).",
        ] = None,
        ctx: Context = None,
    ) -> str:
        await ctx.info(
            f"raptor_check_status: correlation_id={correlation_id!r} m_type={m_type!r}"
        )

        try:
            client = await get_client(ctx)

            if m_type:
                data = await client.get_json(
                    f"/processing/cache/{m_type}/{correlation_id}",
                    tool_name="raptor_check_status",
                )
            else:
                # Scan all types and find the matching correlation_id
                all_data = await client.get_json(
                    "/processing/cache/all",
                    tool_name="raptor_check_status",
                )
                cache = all_data.get("data", {})
                matched = {k: v for k, v in cache.items() if correlation_id in k}
                if not matched:
                    raise MCPToolError(f"correlation_id '{correlation_id}' not found in processing cache")
                # Return the first (and normally only) match
                key, data = next(iter(matched.items()))
                data = {"key": key, **data} if isinstance(data, dict) else {"key": key, "value": data}

        except Exception as exc:
            await ctx.error(f"raptor_check_status failed: {exc}")
            raise

        status_obj = _normalise_status(correlation_id, data)
        await ctx.info(f"raptor_check_status: status={status_obj['status']!r}")
        return json.dumps(status_obj, ensure_ascii=False, indent=2)
