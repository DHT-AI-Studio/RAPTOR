"""End-to-end test: agent frameworks completing a multi-step Raptor task via MCP,
with zero custom integration code beyond the MCP tool adapter.

Requires KEYCLOAK_USERNAME / KEYCLOAK_PASSWORD env vars (skips if unset).
Hits the live raptor-mcp-server container (:8027) and real Module 13 — no mocking.

Run:
    pytest app/tests/test_agent_orchestration.py -v -s
"""
from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable

import httpx
import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from app.tests.agents.react_agent import build_react_agent, run_react_agent
from app.tests.agents.smolagents_agent import TASK_HINT, build_smolagents_agent, run_smolagents_agent

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8027/mcp")
GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://raptor_open_0_3_api.dhtsolution.com:8012")
REALM_NAME = os.getenv("REALM_NAME", "dhtsolution")
CLIENT_ID = os.getenv("CLIENT_ID", "raptor")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TASK = (
    "List all video assets, pick the most recent one, search for related content, "
    "retrieve any relevant memory about the user's preferences, then summarise "
    "in 3 bullet points"
)

# was referenced below but never defined (see commit msg) — tools TASK needs
SCOPED_TOOL_NAMES = {
    "raptor_list_assets",
    "raptor_get_asset_url",
    "raptor_search",
    "raptor_video_search",
    "raptor_memory_retrieve",
}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} not set")
    return value


async def _fetch_jwt(username: str, password: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{GATEWAY_BASE_URL}/api/0.4/sso/login?client_id={CLIENT_ID}",
            data={"username": username, "password": password, "realm_name": REALM_NAME},
        )
        r.raise_for_status()
        return r.json()["access_token"]


async def _retry(attempt: Callable[[], Awaitable[None]], attempts: int = 3) -> None:
    for i in range(attempts):
        try:
            await attempt()
            return
        except AssertionError:
            if i == attempts - 1:
                raise
            print(f"\n  attempt {i + 1} failed, retrying...")


def _assert_completed(output: str, tool_names: set) -> None:
    print(f"  tools used: {tool_names}")
    assert len(tool_names) >= 3, f"expected >=3 distinct successful tool calls, used {tool_names}"
    assert output.strip(), "agent produced no final answer"


async def _get_tools(jwt: str) -> list:
    client = MultiServerMCPClient({
        "raptor": {
            "transport": "streamable_http",
            "url": MCP_SERVER_URL,
            "headers": {"Authorization": f"Bearer {jwt}"},
        }
    })
    return await client.get_tools()


@pytest.mark.asyncio
async def test_react_agent_completes_multi_step_video_task():
    username = _require_env("KEYCLOAK_USERNAME")
    password = _require_env("KEYCLOAK_PASSWORD")
    jwt = await _fetch_jwt(username, password)

    tools = [t for t in await _get_tools(jwt) if t.name in SCOPED_TOOL_NAMES]
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    async def attempt() -> None:
        executor = build_react_agent(llm, tools)
        output, tool_names = await run_react_agent(executor, llm, TASK, tools)
        print(f"\n  ReAct final answer:\n{output}")
        _assert_completed(output, tool_names)

    await _retry(attempt)


@pytest.mark.asyncio
async def test_smolagents_agent_completes_multi_step_video_task():
    username = _require_env("KEYCLOAK_USERNAME")
    password = _require_env("KEYCLOAK_PASSWORD")
    jwt = await _fetch_jwt(username, password)

    tools = [t for t in await _get_tools(jwt) if t.name in SCOPED_TOOL_NAMES]

    async def attempt() -> None:
        agent = build_smolagents_agent(f"ollama_chat/{OLLAMA_MODEL}", OLLAMA_BASE_URL, tools)
        output, tool_names = await asyncio.to_thread(run_smolagents_agent, agent, tools, f"{TASK_HINT}\n\n{TASK}")
        print(f"\n  smolagents final answer:\n{output}")
        _assert_completed(output, tool_names)

    await _retry(attempt)
