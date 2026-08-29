"""smolagents ToolCallingAgent over Raptor's MCP tools (alternative to react_agent.py)."""
from __future__ import annotations

import asyncio
from typing import List

from langchain_core.tools import BaseTool
from smolagents import LiteLLMModel, Tool, ToolCallingAgent

from app.tests.agents.react_agent import _extract_text, _normalize_args, _shrink, synthesis_prompt, top_up_tools

TASK_HINT = (
    "Call exactly 3 different tools, each at most once, to gather information: "
    "first call two different search/lookup tools (e.g. raptor_video_search and "
    "raptor_search, or raptor_get_asset_url) to gather raw evidence, then call "
    "raptor_chat last to synthesize everything you found into a summary — "
    "raptor_chat is always your 3rd and final tool call. Never call the same "
    "tool twice, even with a reworded query. Immediately after raptor_chat "
    "responds, call final_answer; do not call a 4th tool. Empty or unhelpful "
    "results are a valid outcome: just say honestly what you did and did not find."
)


def _schema_type(field: dict) -> str:
    if field.get("type"):
        return field["type"]
    for option in field.get("anyOf", []):
        if option.get("type") and option["type"] != "null":
            return option["type"]
    return "string"


def _smol_tool(tool: BaseTool) -> Tool:
    inputs = {}
    for name, field in tool.args.items():
        spec = {"type": _schema_type(field), "description": field.get("description") or field.get("title") or name}
        if "default" in field:
            spec["nullable"] = True
        inputs[name] = spec

    class RaptorSmolTool(Tool):
        name = tool.name
        description = tool.description
        output_type = "string"
        skip_forward_signature_validation = True

        def __init__(self):
            self.inputs = inputs
            super().__init__()

        def forward(self, **kwargs) -> str:
            result = asyncio.run(tool.ainvoke(_normalize_args(tool.name, kwargs)))
            return _shrink(_extract_text(result), max_chars=1000)

    return RaptorSmolTool()


def build_smolagents_agent(model_id: str, api_base: str, tools: List[BaseTool], max_steps: int = 10) -> ToolCallingAgent:
    smol_tools = [_smol_tool(t) for t in tools]
    model = LiteLLMModel(model_id=model_id, api_base=api_base, temperature=0)
    return ToolCallingAgent(tools=smol_tools, model=model, max_steps=max_steps)


def run_smolagents_agent(agent: ToolCallingAgent, tools: List[BaseTool], task: str) -> tuple[str, set]:
    """Run the agent and return (final_answer, tools_used).

    Uses react_agent.top_up_tools (see its docstring) so the tool-diversity
    guarantee holds regardless of when the model quits. Sync (like
    `agent.run`) — callers already run this off-thread via asyncio.to_thread.
    """
    tools_by_name = {t.name: t for t in tools}
    output = str(agent.run(task))

    transcript, tools_used = [], set()
    for step in agent.memory.steps:
        calls = getattr(step, "tool_calls", None)
        if not calls or getattr(step, "error", None) or calls[0].name == "final_answer":
            continue
        tools_used.add(calls[0].name)
        transcript.append(f"Tool: {calls[0].name}\nResult: {step.observations}")

    if asyncio.run(top_up_tools(tools_used, transcript, tools_by_name, task)):
        message = {"role": "user", "content": [{"type": "text", "text": synthesis_prompt(task, transcript)}]}
        output = agent.model.generate([message]).content

    return output, tools_used
