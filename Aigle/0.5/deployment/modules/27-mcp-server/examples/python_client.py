"""Standalone MCP client for Raptor — no LLM, no agent framework.

Takes an already-obtained JWT (e.g. from Raptor's `POST /api/0.4/sso/login`,
or `examples/curl_mcp.sh` which can log in for you) and drives the MCP
protocol directly over Streamable HTTP: initialize, list tools, call a tool,
read a resource.

Requires: pip install mcp
Usage:
    python examples/python_client.py --jwt <token>
    python examples/python_client.py --jwt <token> --server-url http://localhost:8027/mcp
"""
from __future__ import annotations

import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jwt", required=True, help="Bearer JWT (from Keycloak sso/login)")
    parser.add_argument("--server-url", default="http://localhost:8027/mcp", help="MCP server Streamable HTTP endpoint")
    return parser.parse_args()


def text_of(result) -> str:
    """Concatenate the text blocks of an MCP tool/resource result's content list."""
    return "".join(block.text for block in result.content if hasattr(block, "text"))


async def main() -> None:
    args = parse_args()
    async with streamablehttp_client(args.server_url, headers={"Authorization": f"Bearer {args.jwt}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"Connected to {init_result.serverInfo.name} v{init_result.serverInfo.version}")
            print(f"Protocol: {init_result.protocolVersion}\n")
            tools = await session.list_tools()
            print(f"Available tools ({len(tools.tools)}):")
            for tool in tools.tools:
                print(f"  - {tool.name}")
            print()
            search_result = await session.call_tool("raptor_search", {"query": "video", "top_k": 3})
            results = json.loads(text_of(search_result))
            print(f"raptor_search('video') -> {len(results)} hit(s):")
            for hit in results:
                print(f"  - [{hit.get('score', 0):.3f}] {hit.get('asset_path', '')}")
            print()
            resource_result = await session.read_resource("raptor://capabilities")
            capabilities_md = resource_result.contents[0].text
            print(f"raptor://capabilities ({len(capabilities_md)} chars):")
            print(capabilities_md[:300] + "...")


if __name__ == "__main__":
    asyncio.run(main())
