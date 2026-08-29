"""Contract test — every MCP tool's backend call path must exist as a real
route on Module 13's /api/0.4 surface.

Fetches Module 13's OpenAPI spec and checks each tool path against it.
Requires network access to a live Module 13 (API Gateway); intended to run
in CI against a deployed/staging gateway. Skips (not fails) when the
gateway is unreachable, so local runs without network access aren't
penalised.
"""
from __future__ import annotations

import re

import httpx
import pytest

from app.core.config import get_settings

# Every backend path an MCP tool/resource calls, relative to the API
# version prefix — e.g. "/search/hybrid" maps to the gateway route
# "/api/0.4/search/hybrid". Keep this list in sync with the actual call
# sites in app/tools/*.py and app/resources/raptor_resources.py.
TOOL_PATHS = [
    "/search/hybrid",           # tools/search.py::raptor_search
    "/search/bm25",              # tools/search.py::raptor_search_bm25
    "/search/vector",            # tools/search.py::raptor_search_vector
    "/search/video_search",      # tools/search.py::raptor_video_search
    "/search/graphrag",          # tools/graph.py::raptor_graph_query
    "/search/tkg",                # tools/graph.py::raptor_tkg_query
    "/chat/chat",                 # tools/chat.py::raptor_chat
    "/asset/users/commits",       # tools/asset.py::raptor_list_assets ; raptor_resources.py::list_assets
    "/asset/filedownload/{asset_path}/{version_id}",  # tools/asset.py ; raptor_resources.py::get_asset
    "/asset/fileupload_analysis",  # tools/asset.py::raptor_upload_asset
    "/a2a/query",                  # tools/a2a.py ; tools/pipeline.py
    "/processing/cache/{media_type}/{correlation_id}",  # tools/processing.py::raptor_check_status
    "/processing/cache/all",        # tools/processing.py
    "/memory/retrieve",             # tools/memory.py::raptor_memory_retrieve
    "/memory/store",                # tools/memory.py::raptor_memory_store
    "/memory/timeline",              # tools/memory.py::raptor_memory_timeline
    "/memory/multimedia/search",    # tools/memory.py::raptor_memory_multimedia_search
    "/memory/sessions/{session_id}/summaries",  # tools/memory.py::raptor_memory_session_summaries
    "/memory/compact",               # tools/memory.py::raptor_memory_compact
    "/memory/compact/evaluate",      # tools/memory.py::raptor_memory_compact_evaluate
    "/memory/archive",               # tools/memory.py::raptor_memory_archive
]


def _path_shape(path: str) -> str:
    """Collapse every {param} segment to a placeholder so path parameter
    *names* (which are Module 13's to choose, not ours) don't cause a false
    mismatch — only the literal segments and parameter positions matter."""
    return re.sub(r"\{[^}]+\}", "{}", path)


def _openapi_url(gateway_base_url: str) -> str:
    # api_gateway_url already includes the /api/{version} prefix (derived from
    # MCP_API_GATEWAY_BASE_URL + MCP_API_VERSION); the OpenAPI document itself
    # is served from the gateway's root.
    root = gateway_base_url.split("/api/")[0]
    return f"{root}/openapi.json"


@pytest.fixture(scope="module")
def gateway_paths() -> set[str]:
    settings = get_settings()
    url = _openapi_url(settings.api_gateway_url)
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url)
            r.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Module 13 unreachable at {url}: {exc}")
    return set(r.json().get("paths", {}).keys())


@pytest.mark.parametrize("tool_path", TOOL_PATHS)
def test_tool_path_exists_on_gateway(tool_path: str, gateway_paths: set[str]) -> None:
    full_path = f"/api/{get_settings().api_version}{tool_path}"
    shape = _path_shape(full_path)
    gateway_shapes = {_path_shape(p) for p in gateway_paths}
    assert shape in gateway_shapes, (
        f"{full_path} is not a registered Module 13 route — "
        f"this MCP tool call would 404/405 against the live gateway"
    )
