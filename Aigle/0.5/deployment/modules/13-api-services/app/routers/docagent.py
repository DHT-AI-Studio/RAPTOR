"""DA-09a — Module 13 authenticated proxy for DocAgent (Module 29).

Forwards all /api/{v}/docagent/* requests to the DocAgent service.
Auth is enforced at the include_router() call in main.py.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.middlewares.guardrail import _KNOWN_PATHS, _check_input, _lookup, _read_fields

logger = logging.getLogger(__name__)
router = APIRouter(tags=["DocAgent"])

_PROCESS_CONFIG_KEY = "/docagent/process"
_KNOWN_PATHS.add(_PROCESS_CONFIG_KEY)


async def _docagent_process_guardrail_dependency(
    path: str,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> None:
    """Input guardrail for POST /docagent/process, only when task=qa -- this router is a
    single catch-all proxy (see proxy_docagent below), so unlike every other guarded route
    there's no dedicated route function to attach @guardrail_route to; this dependency does
    its own path/method/task matching instead. Chained after get_current_user like every
    other guardrail dependency (see app/middlewares/guardrail.py's module docstring)."""
    if path != "process" or request.method != "POST":
        return
    if _PROCESS_CONFIG_KEY not in get_settings().guardrail_routes:
        return
    data = await _read_fields(request)
    if data.get("task") != "qa":
        return
    content = _lookup(data, "options.question")
    if not isinstance(content, str):
        return
    await _check_input(request, _PROCESS_CONFIG_KEY, content)

_DOCAGENT_PATH_PREFIX = "/api/v1/docagent"


async def _proxy(request: Request, path: str) -> Response:
    settings = get_settings()
    doc_agent_url = getattr(settings, "doc_agent_url",
                            "http://raptor-doc-processing-agent:8290")
    target = f"{doc_agent_url.rstrip('/')}/api/v1/docagent/{path}"

    body = await request.body()

    # Forward all headers except Host
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    # A DocAgent task runs an LLM inference call (and, for classify/extract/
    # summarize/compare/qa/parse_table, a Module 12 document-parse step
    # first) -- both routinely slower than request.app.state.http_client's
    # short request_timeout (10s), which every other ordinary proxied route
    # can assume but this one can't. Confirmed live: a real classify call
    # timed out at exactly 10s and, since only httpx.ConnectError was ever
    # caught below, surfaced as an unhandled exception -> a generic 500 with
    # no indication it was a timeout. Dedicated client with the longer
    # upload_timeout instead, same fix as inference.py's own proxy.
    async with httpx.AsyncClient(timeout=settings.upload_timeout) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body,
                params=dict(request.query_params),
            )
        except httpx.ConnectError:
            logger.error("DocAgent unreachable at %s", doc_agent_url)
            return Response(
                content=b'{"error":"doc_agent_unavailable"}',
                status_code=503,
                media_type="application/json",
            )
        except httpx.TimeoutException:
            logger.error("DocAgent request timed out: %s", target)
            return Response(
                content=b'{"error":"doc_agent_timeout"}',
                status_code=504,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("DocAgent request failed for %s: %s", target, exc)
            return Response(
                content=b'{"error":"doc_agent_error"}',
                status_code=502,
                media_type="application/json",
            )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    dependencies=[Depends(_docagent_process_guardrail_dependency)],
)
async def proxy_docagent(request: Request, path: str = "") -> Response:
    return await _proxy(request, path)
