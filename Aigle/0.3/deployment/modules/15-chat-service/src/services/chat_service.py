"""
LangGraph Chat Service with Unified Search Integration.
Moved from module 12 — identical logic, adapted imports.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import httpx
import opencc
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from redis.asyncio import Redis

from .search_service import HybridSearchService

_logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

_s2twp = opencc.OpenCC("s2twp")


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def to_traditional(text: str) -> str:
    return _s2twp.convert(text) if text else text


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def unified_search_tool(query: str) -> str:
    """執行統一搜尋（跨多個集合）"""
    return json.dumps({"status": "pending", "query": query}, ensure_ascii=False)


@tool
def get_current_time() -> str:
    """獲取當前時間"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """計算數學表達式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"計算結果: {result}"
    except Exception as e:
        return f"計算錯誤: {str(e)}"


# ── State ─────────────────────────────────────────────────────────────────────

class SearchResult(TypedDict):
    collection: str
    items: List[Dict[str, Any]]


class ConversationMemory(TypedDict):
    timestamp: float
    user_message: str
    assistant_response: str
    search_results: Optional[List[SearchResult]]


class ChatState(TypedDict):
    user_id: str
    session_id: Optional[str]
    messages: Annotated[List, add_messages]
    current_query: str
    search_results: Optional[List[SearchResult]]
    search_triggered: bool
    context_window: List[str]
    final_response: str
    tool_calls_log: List[Dict[str, Any]]


# ── Service ───────────────────────────────────────────────────────────────────

class UnifiedChatService:
    """LangGraph RAG chat service with hybrid search and Redis memory."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        redis_client: Redis,
        search_service: HybridSearchService,
        model_name: str = "Qwen3-14B-AWQ",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        llm_base_url: str = "http://192.168.157.135:11434/v1",
        memory_context_window: int = 5,
        memory_ttl: int = 3600,
        redis_timeout: float = 0.5,
    ):
        self.http_client = http_client
        self.redis_client = redis_client
        self.search_service = search_service
        self.memory_context_window = memory_context_window
        self.memory_ttl = memory_ttl
        self.redis_timeout = redis_timeout

        llm_config: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key or os.getenv("OPENAI_API_KEY", "test-api-key"),
        }
        if llm_base_url:
            llm_config["base_url"] = llm_base_url

        self.llm = ChatOpenAI(**llm_config)
        self.tools = [unified_search_tool, calculate, get_current_time]

        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="auto")
        except Exception:
            self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        workflow.add_node("load_memory", self._load_memory)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("call_model", self._call_model)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_memory", self._save_memory)
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "prepare_context")
        workflow.add_edge("prepare_context", "call_model")
        workflow.add_conditional_edges(
            "call_model",
            self._should_execute_tools,
            {"execute": "execute_tools", "skip": "save_memory"},
        )
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("generate_response", "save_memory")
        workflow.add_edge("save_memory", END)
        return workflow.compile()

    def _should_execute_tools(self, state: ChatState) -> Literal["execute", "skip"]:
        last_ai = state["messages"][-1]
        if hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
            return "execute"
        if self._should_trigger_search(state.get("current_query", ""), state.get("search_triggered", False)):
            return "execute"
        return "skip"

    async def _execute_tools(self, state: ChatState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        current_query = state.get("current_query", "")
        search_results: List = []
        tool_calls_log: List = list(state.get("tool_calls_log", []))

        tool_calls_to_execute = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls_to_execute.extend(last_message.tool_calls)
        if not tool_calls_to_execute and self._should_trigger_search(current_query, state.get("search_triggered", False)):
            tool_calls_to_execute.append({
                "name": "unified_search_tool",
                "args": {"query": current_query},
                "id": "auto_search_" + str(int(time.time())),
            })

        user_id = state.get("user_id", "")
        user_dict = {"user_id": user_id, "branch_id": user_id}
        for tc in tool_calls_to_execute:
            name = tc.get("name")
            if name == "unified_search_tool":
                await self._execute_search_tool(tc, current_query, messages, search_results, tool_calls_log, user=user_dict)
            elif name == "calculate":
                self._execute_calculate_tool(tc, messages, tool_calls_log)
            elif name == "get_current_time":
                self._execute_time_tool(tc, messages, tool_calls_log)

        return {
            "messages": messages,
            "search_results": search_results,
            "search_triggered": len(search_results) > 0,
            "tool_calls_log": tool_calls_log,
        }

    async def _execute_search_tool(self, tool_call, current_query, messages, search_results, tool_calls_log, user=None):
        tool_name = tool_call.get("name")
        query = tool_call.get("args", {}).get("query", current_query)
        try:
            start_time = time.time()
            search_response = await self.search_service.hybrid_search(
                {"query": query, "top_k": 2, "embedding_type": "text"}, user=user
            )
            execution_time = (time.time() - start_time) * 1000
            results = search_response.get("results", [])
            collection_stats = search_response.get("collection_stats", {})
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

            search_results.append({
                "collection": "unified", "items": sorted_results,
                "stats": collection_stats, "query": query,
                "original_user_query": current_query,
            })

            formatted_results = []
            for idx, result in enumerate(sorted_results[:5], 1):
                payload = result.get("payload", {})
                item_type = payload.get("type", "unknown")
                formatted_results.append({
                    "rank": idx,
                    "relevance_score": f"{result.get('score', 0):.2%}",
                    "type": item_type,
                    "key_info": self._extract_key_info(item_type, payload),
                    "full_payload": result,
                })

            messages.append(ToolMessage(
                content=json.dumps({
                    "success": True,
                    "user_original_question": current_query,
                    "search_query": query,
                    "results_count": len(sorted_results),
                    "formatted_results": formatted_results,
                    "summary": f"搜尋『{query}』，找到 {len(sorted_results)} 個結果。",
                    "collection_stats": collection_stats,
                }, ensure_ascii=False, default=str),
                tool_call_id=tool_call.get("id"),
                name=tool_name,
            ))
            tool_calls_log.append({
                "tool": tool_name, "query": query, "results_count": len(sorted_results),
                "execution_time_ms": execution_time,
                "top_score": sorted_results[0].get("score", 0) if sorted_results else 0,
                "status": "success",
            })
        except Exception as e:
            _logger.error(f"Search tool error: {e}")
            messages.append(ToolMessage(
                content=json.dumps({"success": False, "error": str(e)}, ensure_ascii=False),
                tool_call_id=tool_call.get("id"), name=tool_name,
            ))
            tool_calls_log.append({"tool": tool_name, "query": query, "error": str(e), "status": "error"})

    def _execute_calculate_tool(self, tool_call, messages, tool_calls_log):
        expr = tool_call.get("args", {}).get("expression", "")
        try:
            content = f"計算結果: {eval(expr, {'__builtins__': {}}, {})}"
            status = "success"
        except Exception as e:
            content = f"計算錯誤: {str(e)}"
            status = "error"
        messages.append(ToolMessage(content=content, tool_call_id=tool_call.get("id"), name=tool_call.get("name")))
        tool_calls_log.append({"tool": tool_call.get("name"), "expression": expr, "result": content, "status": status})

    def _execute_time_tool(self, tool_call, messages, tool_calls_log):
        content = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        messages.append(ToolMessage(content=content, tool_call_id=tool_call.get("id"), name=tool_call.get("name")))
        tool_calls_log.append({"tool": tool_call.get("name"), "result": content, "status": "success"})

    def _extract_key_info(self, item_type: str, payload: Dict[str, Any]) -> str:
        if item_type == "image":
            return f"[圖片] {payload.get('filename', 'N/A')} - {payload.get('text', 'N/A')[:100]}"
        if item_type == "video":
            return f"[視頻] {payload.get('filename', 'N/A')} ({payload.get('start_time', 'N/A')}-{payload.get('end_time', 'N/A')}s) - {payload.get('text', 'N/A')[:80]}"
        if item_type == "audio":
            return f"[音頻] {payload.get('filename', 'N/A')} ({payload.get('start_time', 'N/A')}-{payload.get('end_time', 'N/A')}s) - {payload.get('text', 'N/A')[:80]}"
        if item_type == "document":
            return f"[文檔] {payload.get('filename', 'N/A')} (頁: {payload.get('page_numbers', 'N/A')}) - {payload.get('text', 'N/A')[:80]}"
        return f"[{item_type.upper()}] 結果"

    async def _generate_response(self, state: ChatState) -> Dict[str, Any]:
        response = await self.llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def _memory_key(self, user_id: str, session_id: Optional[str] = None) -> str:
        return f"chat_memory:{user_id}:{session_id}" if session_id else f"chat_memory:{user_id}"

    async def _load_memory(self, state: ChatState) -> Dict[str, Any]:
        memory_key = self._memory_key(state["user_id"], state.get("session_id"))
        try:
            data = await asyncio.wait_for(self.redis_client.get(memory_key), timeout=self.redis_timeout)
            context_window = []
            if data:
                for mem in json.loads(data)[-self.memory_context_window:]:
                    context_window.append(f"用戶: {mem['user_message']}\n助手: {mem['assistant_response']}")
            return {"context_window": context_window}
        except Exception as e:
            _logger.error(f"Error loading memory: {e}")
            return {"context_window": []}

    def _merge_system_prompt(self, existing: Optional[str], user_query: Optional[str] = None) -> str:
        tool_policy = (
            "你是一個有幫助的 AI 助手。\n\n"
            "【核心職責】\n"
            "- 當用戶詢問與資料庫/檔案/影片/音訊/圖片內容相關的問題，必須優先調用 unified_search_tool 搜尋證據，再作答。\n"
            "- 包含以下情況：詢問某文件/圖片/影片/音訊的內容、要求找資料、引用來源、包含關鍵詞如: search/find/查/找/搜尋/pdf/image/video/audio/document。\n\n"
            "【回答格式要求】\n"
            "- 直接回答用戶的具體問題，用搜尋結果的具體信息支持回答\n"
            "- 必須標記引用來源（文件名、時間戳、頁碼等）和相似度分數\n"
            "- 如果搜尋結果與問題不相關，應明確說明"
        )
        if user_query:
            tool_policy += f"\n\n【當前用戶問題】\n\"{user_query}\"\n"
        if existing and existing.strip():
            return f"{existing.strip()}\n\n【工具使用規則】\n{tool_policy}"
        return tool_policy

    async def _prepare_context(self, state: ChatState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        existing_sys = None
        other_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage) and existing_sys is None:
                existing_sys = m.content
            else:
                other_msgs.append(m)

        system_prompt = self._merge_system_prompt(existing_sys, state.get("current_query", ""))
        context_window = state.get("context_window", [])
        if context_window:
            system_prompt += "\n\n【最近對話記錄】\n" + "\n".join(context_window)

        return {"messages": [SystemMessage(content=system_prompt)] + other_msgs}

    async def _call_model(self, state: ChatState) -> Dict[str, Any]:
        response = await self.llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def _should_trigger_search(self, user_text: str, already_searched: bool) -> bool:
        if already_searched or not user_text:
            return False
        text = user_text.lower()
        keywords = [
            "search", "find", "查", "找", "搜尋", "來源", "哪裡", "參考",
            "pdf", "document", "doc", "image", "圖片", "video", "影片", "audio", "音訊", "音頻",
        ]
        return any(k in text for k in keywords) or "?" in text or "？" in text

    async def _save_memory(self, state: ChatState) -> Dict[str, Any]:
        memory_key = self._memory_key(state["user_id"], state.get("session_id"))
        final_response = ""
        try:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    final_response = to_traditional(strip_think_tags(msg.content))
                    break

            memory_item: ConversationMemory = {
                "timestamp": time.time(),
                "user_message": state["current_query"],
                "assistant_response": final_response,
                "search_results": state.get("search_results"),
            }

            existing_data = await asyncio.wait_for(self.redis_client.get(memory_key), timeout=self.redis_timeout)
            memories = json.loads(existing_data) if existing_data else []
            memories.append(memory_item)
            memories = memories[-self.memory_context_window:]

            await asyncio.wait_for(
                self.redis_client.setex(memory_key, self.memory_ttl, json.dumps(memories, ensure_ascii=False, default=str)),
                timeout=self.redis_timeout,
            )
            return {"final_response": final_response}
        except Exception as e:
            _logger.error(f"Error saving memory: {e}")
            return {"final_response": strip_think_tags(final_response)}

    async def chat(
        self,
        user_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=message))

        initial_state: ChatState = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": messages,
            "current_query": message,
            "search_results": None,
            "search_triggered": False,
            "context_window": [],
            "final_response": "",
            "tool_calls_log": [],
        }

        result = await self.graph.ainvoke(initial_state)

        final_response = result.get("final_response", "")
        if not final_response:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_response = to_traditional(strip_think_tags(msg.content))
                    break

        return {
            "response": final_response,
            "user_id": user_id,
            "search_results": result.get("search_results"),
            "tool_calls": result.get("tool_calls_log", []),
            "search_triggered": result.get("search_triggered", False),
        }

    async def chat_stream(self, user_id: str, message: str, history: Optional[List[Dict[str, str]]] = None):
        messages = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=message))
        initial_state: ChatState = {
            "user_id": user_id, "messages": messages, "current_query": message,
            "search_results": None, "search_triggered": False,
            "context_window": [], "final_response": "", "tool_calls_log": [],
        }
        async for event in self.graph.astream(initial_state):
            yield event
