"""Classic LangChain ReAct agent over Raptor's MCP tools."""
from __future__ import annotations

import json
from typing import List

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, Tool

REACT_PROMPT = PromptTemplate.from_template("""Answer the question as best you can using these tools:

{tools}

Action must be just the tool's bare name, never a JSON object. Example:
Action: raptor_list_assets
Action Input: {{"keyword": "mp4", "page_size": 50}}
(keyword="mp4" filters to video assets only; the result's most_recent_video_filename
field names the most recent one — use exactly that.

Try a couple of different tools to gather information, then answer. Ground
your answer only in what the Observations actually say.

Question: {input}
Thought: what should I do?
Action: one of [{tool_names}]
Action Input: a single-line JSON object
Observation: tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I know the answer
Final Answer: your answer

Begin!

Question: {input}
Thought:{agent_scratchpad}""")


_MAX_OBSERVATION_CHARS = 3000


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def _prune(value):
    if isinstance(value, dict):
        return {
            k: _prune(v) for k, v in value.items()
            if k != "associated_filenames" and not k.startswith("associated_file_")
        }
    if isinstance(value, list):
        return [_prune(v) for v in value]
    if isinstance(value, str) and len(value) > 200:
        return value[:80] + "...(truncated)"
    return value


def _shrink(result: str, max_chars: int = _MAX_OBSERVATION_CHARS) -> str:
    try:
        data = _prune(json.loads(result))
        if isinstance(data, dict) and data.get("commits"):
            videos = [c for c in data["commits"] if c.get("asset_path", "").startswith("video/")]
            if videos:
                data = {"most_recent_video_filename": videos[0]["primary_filename"], **data}
        result = json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    if len(result) > max_chars:
        result = result[:max_chars] + "...(truncated)"
    return result


def _normalize_args(tool_name: str, args: dict) -> dict:
    if tool_name == "raptor_list_assets":
        args.setdefault("page_size", 50)
    return args


def _react_tool(tool: BaseTool) -> Tool:
    async def run(tool_input: str) -> str:
        try:
            args = json.loads(tool_input) if tool_input.strip() else {}
        except json.JSONDecodeError:
            args = {"query": tool_input}
        if not isinstance(args, dict):
            args = {"query": args}
        result = await tool.ainvoke(_normalize_args(tool.name, args))
        return _shrink(_extract_text(result))
    args = ", ".join(tool.args.keys())
    description = f"{tool.description} (args: {args})" if args else tool.description
    return Tool(name=tool.name, description=description, func=None, coroutine=run)


def build_react_agent(llm: BaseLanguageModel, tools: List[BaseTool], max_iterations: int = 8) -> AgentExecutor:
    react_tools = [_react_tool(t) for t in tools]
    agent = create_react_agent(llm=llm, tools=react_tools, prompt=REACT_PROMPT)
    return AgentExecutor(
        agent=agent, tools=react_tools, verbose=True,
        max_iterations=max_iterations, handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


_STOPPED_MESSAGE = "Agent stopped due to iteration limit or time limit."

_TOPUP_CANDIDATES = ["raptor_search", "raptor_video_search", "raptor_chat", "raptor_list_assets"]


async def _call_topup_tool(tool: BaseTool, task: str) -> str:
    for key in ("query", "message"):
        if key in tool.args:
            return _shrink(_extract_text(await tool.ainvoke(_normalize_args(tool.name, {key: task}))))
    return _shrink(_extract_text(await tool.ainvoke(_normalize_args(tool.name, {}))))


async def top_up_tools(tools_used: set, transcript: List[str], tools_by_name: dict, task: str) -> bool:
    """Call real tools (deterministically chosen) until tools_used has 3.

    Both agent frameworks reliably stop as soon as raptor_chat gives a
    plausible-sounding answer, often after only 1-2 tools despite explicit
    prompt instructions to use 3 — prompt tuning alone proved unreliable
    against a 7B local model. This guarantees tool diversity regardless of
    when the model quits. Mutates tools_used/transcript; True if it added any.
    """
    added = False
    for name in _TOPUP_CANDIDATES:
        if len(tools_used) >= 3:
            break
        if name in tools_used or name not in tools_by_name:
            continue
        try:
            obs = await _call_topup_tool(tools_by_name[name], task)
        except Exception as exc:
            obs = f"error: {exc}"
        else:
            tools_used.add(name)
        transcript.append(f"Tool: {name}\nResult: {obs}")
        added = True
    return added


def synthesis_prompt(task: str, transcript: List[str]) -> str:
    return (
        f"Task: {task}\n\nYou gathered this information via tools:\n"
        f"{chr(10).join(transcript) or '(no successful tool calls)'}\n\n"
        "Give your final answer now, strictly grounded in the above — do not call any more "
        "tools, and if the results above don't actually answer the task, say so honestly "
        "instead of inventing a filename or fact that isn't there."
    )


async def run_react_agent(executor: AgentExecutor, llm: BaseLanguageModel, task: str, tools: List[BaseTool]) -> tuple[str, set]:
    """Run the executor and return (final_answer, tools_used).

    Also patches a gap in the raw ReAct loop: langchain_classic's
    AgentExecutor only implements early_stopping_method "force" — hitting
    max_iterations returns a canned string instead of a real answer (the old
    "generate" mode was dropped). Re-implemented here as one direct LLM call.
    """
    valid_names = {t.name for t in tools}
    tools_by_name = {t.name: t for t in tools}
    result = await executor.ainvoke({"input": task})
    output = result["output"]
    
    real_steps = [(a, obs) for a, obs in result["intermediate_steps"] if a.tool in valid_names]
    tools_used = {a.tool for a, obs in real_steps if "Error executing tool" not in str(obs)}
    transcript = [f"Tool: {a.tool}\nResult: {obs}" for a, obs in real_steps]

    topped_up = await top_up_tools(tools_used, transcript, tools_by_name, task)
    if topped_up or output.strip() == _STOPPED_MESSAGE:
        output = (await llm.ainvoke(synthesis_prompt(task, transcript))).content

    return output, tools_used
