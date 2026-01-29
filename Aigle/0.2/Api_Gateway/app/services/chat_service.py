"""
LangGraph Chat Service with Unified Search Integration
提供對話功能、工具調用、unified_search集成和短期記憶的服務
"""

from typing import Annotated, List, Dict, Any, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import asyncio
import os
import json
import logging
import time
import re
from datetime import datetime
from redis.asyncio import Redis
import httpx

from app.services.search_service import SearchService

_logger = logging.getLogger(__name__)


# ==================== 工具函數定義 ====================

def strip_think_tags(text: str) -> str:
    """移除 <think>...</think> 及其內容。"""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


@tool
def unified_search_tool(query: str) -> str:
    """
    執行統一搜尋（跨多個集合）
    
    Args:
        query: 搜尋查詢字符串
    
    Returns:
        搜尋結果的 JSON 字符串
    """
    try:
        # 工具僅用於告知模型此為搜尋調用，實際執行在工作流中完成
        return json.dumps({
            "status": "pending",
            "query": query,
            "note": "This will be executed inside UnifiedChatService._execute_search_tool"
        }, ensure_ascii=False)
    except Exception as e:
        _logger.error(f"Search tool error: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool
def get_current_time() -> str:
    """獲取當前時間"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    計算數學表達式
    
    Args:
        expression: 要計算的數學表達式，例如 "2 + 2" 或 "10 * 5"
    
    Returns:
        計算結果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"計算結果: {result}"
    except Exception as e:
        return f"計算錯誤: {str(e)}"


# ==================== LangGraph 狀態定義 ====================

class SearchResult(TypedDict):
    """搜尋結果"""
    collection: str
    items: List[Dict[str, Any]]


class ConversationMemory(TypedDict):
    """對話記憶項"""
    timestamp: float
    user_message: str
    assistant_response: str
    search_results: Optional[List[SearchResult]]


class ChatState(TypedDict):
    """對話狀態"""
    user_id: str  # 用於區分不同用戶
    messages: Annotated[List, add_messages]  # LangGraph 消息管理
    current_query: str  # 當前查詢
    search_results: Optional[List[SearchResult]]  # 搜尋結果
    search_triggered: bool  # 是否觸發搜尋
    context_window: List[str]  # 上下文窗口（最近5條交互的摘要）
    final_response: str  # 最終回應
    tool_calls_log: List[Dict[str, Any]]  # 工具調用日誌


# ==================== LangGraph 服務類 ====================

class UnifiedChatService:
    """統一對話服務（集成 Unified Search 和短期記憶）"""
    
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        redis_client: Redis,
        search_service: SearchService,
        api_base_url: str = "http://localhost:8012",
        model_name: str = "Qwen3-14B-AWQ",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        llm_base_url: str = "http://192.168.157.169:8000/v1",
        memory_context_window: int = 5,
        memory_ttl: int = 3600,
        redis_timeout: float = 0.5,
    ):
        """
        初始化統一對話服務
        
        Args:
            http_client: httpx 非同步客戶端（用於調用 API 端點）
            redis_client: Redis 非同步客戶端
            api_base_url: API 基礎 URL（例如 http://localhost:8000）
            model_name: 模型名稱
            temperature: 溫度參數
            api_key: OpenAI API Key
            llm_base_url: LLM API 基礎 URL
            memory_context_window: 上下文窗口大小（記憶數量）
            memory_ttl: 記憶過期時間（秒）
        """
        self.http_client = http_client
        self.redis_client = redis_client
        self.search_service = search_service
        self.api_base_url = api_base_url
        self.memory_context_window = memory_context_window
        self.memory_ttl = memory_ttl
        self.redis_timeout = redis_timeout
        
        # 配置 LLM
        llm_config = {
            "model": model_name,
            "temperature": temperature,
        }
        
        if api_key:
            llm_config["api_key"] = api_key
        elif os.getenv("OPENAI_API_KEY"):
            llm_config["api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            llm_config["api_key"] = "test-api-key"  # 默認測試金鑰

        if llm_base_url:
            llm_config["base_url"] = llm_base_url
        
        self.llm = ChatOpenAI(**llm_config)
        
        # 定義工具列表（綁定所有工具）
        self.tools = [
            unified_search_tool,
            calculate,
            get_current_time,
        ]
        
        # 綁定工具到 LLM，優先使用 auto 模式（不支持則回退）
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="auto")
        except Exception:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 建立 LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """建立 LangGraph 狀態圖 - 簡化流程：調用函數 -> 取得結果 -> 回答問題"""
        workflow = StateGraph(ChatState)
        
        # 定義節點
        workflow.add_node("load_memory", self._load_memory)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("call_model", self._call_model)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_memory", self._save_memory)
        
        # 設定入口點
        workflow.set_entry_point("load_memory")
        
        # 添加邊 - 簡化流程
        workflow.add_edge("load_memory", "prepare_context")
        workflow.add_edge("prepare_context", "call_model")
        
        # 條件邊：判斷模型是否調用工具
        workflow.add_conditional_edges(
            "call_model",
            self._should_execute_tools,
            {
                "execute": "execute_tools",
                "skip": "generate_response"
            }
        )
        
        # 執行工具後直接生成回應
        workflow.add_edge("execute_tools", "generate_response")
        
        # 保存記憶
        workflow.add_edge("generate_response", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()
    
    def _should_execute_tools(self, state: ChatState) -> Literal["execute", "skip"]:
        """判斷是否執行工具"""
        messages = state["messages"]
        last_ai = messages[-1]
        current_query = state.get("current_query", "")
        
        # 1) 模型主動調用工具
        if hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
            _logger.debug("Model requested tools -> execute")
            return "execute"
        
        # 2) 規則兜底 - 檢查用戶問題是否需要搜尋
        if self._should_trigger_search(current_query, state.get("search_triggered", False)):
            _logger.debug("Heuristic triggers search -> execute")
            return "execute"
        
        _logger.debug("No tools needed -> skip")
        return "skip"
    
    async def _execute_tools(self, state: ChatState) -> Dict[str, Any]:
        """【第一步】執行所有工具調用"""
        messages = state["messages"]
        last_message = messages[-1]
        current_query = state.get("current_query", "")
        
        search_results = []
        tool_calls_log = state.get("tool_calls_log", [])
        
        # 收集所有工具調用
        tool_calls_to_execute = []
        
        # 1) 從模型的 tool_calls 收集
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls_to_execute.extend(last_message.tool_calls)
        
        # 2) 如果沒有工具調用但需要搜尋，自動添加搜尋工具
        if not tool_calls_to_execute and self._should_trigger_search(current_query, state.get("search_triggered", False)):
            tool_calls_to_execute.append({
                "name": "unified_search_tool",
                "args": {"query": current_query},
                "id": "auto_search_" + str(int(time.time()))
            })
        
        # 3) 執行所有工具調用
        for tool_call in tool_calls_to_execute:
            tool_name = tool_call.get("name")
            
            if tool_name == "unified_search_tool":
                await self._execute_search_tool(
                    tool_call, current_query, messages, search_results, tool_calls_log
                )
            
            elif tool_name == "calculate":
                self._execute_calculate_tool(tool_call, messages, tool_calls_log)
            
            elif tool_name == "get_current_time":
                self._execute_time_tool(tool_call, messages, tool_calls_log)
        
        return {
            "messages": messages,
            "search_results": search_results,
            "search_triggered": len(search_results) > 0,
            "tool_calls_log": tool_calls_log,
        }
    
    async def _execute_search_tool(
        self,
        tool_call: Dict[str, Any],
        current_query: str,
        messages: List,
        search_results: List,
        tool_calls_log: List
    ) -> None:
        """執行搜尋工具"""
        tool_name = tool_call.get("name")
        query = tool_call.get("args", {}).get("query", current_query)
        
        try:
            _logger.info(f"Executing unified search - Query: '{query}', User question: '{current_query}'")
            start_time = time.time()
            search_response = await self.search_service.unified_search(
                query_text=query,
                embedding_type="text",
                limit_per_collection=2,
            )
            execution_time = (time.time() - start_time) * 1000
            
            # 處理搜尋結果
            results = search_response.get("results", [])
            collection_stats = search_response.get("collection_stats", {})
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            _logger.debug(f"Found {len(sorted_results)} results, top score: {sorted_results[0].get('score', 0) if sorted_results else 0}")
            
            # 儲存搜尋結果
            search_results.append({
                "collection": "unified",
                "items": sorted_results,
                "stats": collection_stats,
                "query": query,
                "original_user_query": current_query,
            })
            
            # 格式化結果
            formatted_results = []
            for idx, result in enumerate(sorted_results[:5], 1):
                item_type = result.get("type", "unknown")
                payload = result.get("payload", {})
                score = result.get("score", 0)
                
                key_info = self._extract_key_info(item_type, payload)
                
                formatted_results.append({
                    "rank": idx,
                    "relevance_score": f"{score:.2%}",
                    "type": item_type,
                    "key_info": key_info,
                    "full_payload": result,
                })
            
            # 添加工具結果消息
            tool_result = ToolMessage(
                content=json.dumps({
                    "success": True,
                    "user_original_question": current_query,
                    "search_query": query,
                    "results_count": len(sorted_results),
                    "formatted_results": formatted_results,
                    "summary": f"根據問題『{current_query}』搜尋『{query}』，"
                              f"找到 {len(sorted_results)} 個相關結果。"
                              f"最相關的結果相似度為 {sorted_results[0].get('score', 0):.2%}" 
                              if sorted_results else "未找到相關結果",
                    "collection_stats": collection_stats,
                }, ensure_ascii=False, default=str),
                tool_call_id=tool_call.get("id"),
                name=tool_name,
            )
            
            messages.append(tool_result)
            
            # 記錄工具調用
            tool_calls_log.append({
                "tool": tool_name,
                "query": query,
                "user_original_query": current_query,
                "results_count": len(sorted_results),
                "execution_time_ms": execution_time,
                "top_score": sorted_results[0].get("score", 0) if sorted_results else 0,
                "status": "success"
            })
            
        except Exception as e:
            _logger.error(f"Search tool error: {str(e)}")
            
            tool_result = ToolMessage(
                content=json.dumps({
                    "success": False,
                    "user_query": current_query,
                    "error": str(e)
                }, ensure_ascii=False),
                tool_call_id=tool_call.get("id"),
                name=tool_name,
            )
            
            messages.append(tool_result)
            tool_calls_log.append({
                "tool": tool_name,
                "query": query,
                "user_query": current_query,
                "error": str(e),
                "status": "error"
            })
    
    def _execute_calculate_tool(
        self,
        tool_call: Dict[str, Any],
        messages: List,
        tool_calls_log: List
    ) -> None:
        """執行計算工具"""
        tool_name = tool_call.get("name")
        args = tool_call.get("args", {})
        expr = args.get("expression", "")
        
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            content = f"計算結果: {result}"
            status = "success"
        except Exception as e:
            content = f"計算錯誤: {str(e)}"
            status = "error"
        
        tool_result = ToolMessage(
            content=content,
            tool_call_id=tool_call.get("id"),
            name=tool_name,
        )
        
        messages.append(tool_result)
        
        tool_calls_log.append({
            "tool": tool_name,
            "expression": expr,
            "result": content,
            "status": status
        })
    
    def _execute_time_tool(
        self,
        tool_call: Dict[str, Any],
        messages: List,
        tool_calls_log: List
    ) -> None:
        """執行時間工具"""
        tool_name = tool_call.get("name")
        content = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tool_result = ToolMessage(
            content=content,
            tool_call_id=tool_call.get("id"),
            name=tool_name,
        )
        
        messages.append(tool_result)
        
        tool_calls_log.append({
            "tool": tool_name,
            "result": content,
            "status": "success"
        })
    
    def _extract_key_info(self, item_type: str, payload: Dict[str, Any]) -> str:
        """提取結果的關鍵信息"""
        if item_type == "image":
            return f"[圖片] {payload.get('filename', 'N/A')} - {payload.get('text', 'N/A')[:100]}"
        
        elif item_type == "video":
            return f"[視頻] {payload.get('filename', 'N/A')} ({payload.get('start_time', 'N/A')}-{payload.get('end_time', 'N/A')}s) - {payload.get('text', 'N/A')[:80]}"
        
        elif item_type == "audio":
            return f"[音頻] {payload.get('filename', 'N/A')} ({payload.get('start_time', 'N/A')}-{payload.get('end_time', 'N/A')}s) - {payload.get('text', 'N/A')[:80]}"
        
        elif item_type == "document":
            return f"[文檔] {payload.get('filename', 'N/A')} (頁: {payload.get('page_numbers', 'N/A')}) - {payload.get('text', 'N/A')[:80]}"
        
        return f"[{item_type.upper()}] 結果"
    
    async def _generate_response(self, state: ChatState) -> Dict[str, Any]:
        """【第二步】基於工具結果生成回答"""
        messages = state["messages"]
        
        # 調用模型生成最終回答
        response = await self.llm_with_tools.ainvoke(messages)
        
        _logger.debug(f"Generated response: {response.content[:100]}...")
        
        return {"messages": [response]}
    
    async def _load_memory(self, state: ChatState) -> Dict[str, Any]:
        """載入用戶短期記憶"""
        user_id = state["user_id"]
        memory_key = f"chat_memory:{user_id}"
        
        try:
            memory_data = await asyncio.wait_for(
                self.redis_client.get(memory_key), timeout=self.redis_timeout
            )
            context_window = []
            
            if memory_data:
                memories = json.loads(memory_data)
                # 只保留最近的 context_window 條記憶
                for mem in memories[-self.memory_context_window :]:
                    context_window.append(
                        f"用戶: {mem['user_message']}\n助手: {mem['assistant_response']}"
                    )
            
            _logger.debug(f"Loaded {len(context_window)} memory items for user {user_id}")
            
            return {
                "context_window": context_window,
            }
        except Exception as e:
            _logger.error(f"Error loading memory: {str(e)}")
            return {"context_window": []}
    
    def _merge_system_prompt(self, existing: Optional[str], user_query: Optional[str] = None) -> str:
        """合併/生成系統提示，確保工具使用規則生效，並明確指導模型基於用戶問題回答"""
        tool_policy = (
            "你是一個有幫助的 AI 助手。\n\n"
            "【核心職責】\n"
            "- 當用戶詢問與資料庫/檔案/影片/音訊/圖片內容相關的問題，必須優先調用 unified_search_tool 搜尋證據，再作答。\n"
            "- 包含以下情況：詢問某文件/圖片/影片/音訊的內容、要求找資料、引用來源、比對內容、包含關鍵詞如: search/find/查/找/搜尋/pdf/image/video/audio/document。\n"
            "- 請避免憑空回答需要查證的內容。\n\n"
            "【搜尋結果處理指南】\n"
            "1. 收到搜尋結果後，仔細分析每個結果的 score（相似度）和 payload 內容。\n"
            "2. 優先參考相似度最高的結果，特別是 score > 0.5 的結果。\n"
            "3. 【重要】必須將搜尋結果與原始問題進行關聯性對應：\n"
            "   - 分析用戶問題的核心關鍵詞和意圖\n"
            "   - 從搜尋結果中找出與問題最相關的部分\n"
            "   - 優先引用相似度最高且內容最相關的結果\n\n"
            "【回答格式要求】\n"
            "- 直接回答用戶的具體問題\n"
            "- 用搜尋結果中的具體信息支持回答，避免模糊表述\n"
            "- 若有多個相關結果，按相似度排序呈現，並說明為什麼這些結果與問題相關\n"
            "- 必須標記引用來源（文件名、時間戳、頁碼等）和相似度分數\n"
            "- 若搜尋結果中有時間戳、頁碼、檔案名等定位信息，務必在回答中提及\n"
            "- 如果搜尋結果與問題不相關，應明確說明"
        )
        
        # 添加用戶問題特定的指導
        if user_query:
            tool_policy += f"\n\n【當前用戶問題分析】\n用戶問題: \"{user_query}\"\n"
            tool_policy += "請基於以上問題去分析搜尋結果，優先選擇最相關的內容進行回答。\n"
        
        if existing and existing.strip():
            return f"{existing.strip()}\n\n【工具使用規則】\n{tool_policy}"
        return tool_policy

    async def _prepare_context(self, state: ChatState) -> Dict[str, Any]:
        """準備上下文信息（合併系統提示與記憶）"""
        context_window = state.get("context_window", [])
        current_query = state.get("current_query", "")
        
        # 構建/合併系統提示
        messages = state.get("messages", [])
        # 提取第一條 system，再合併
        existing_sys = None
        other_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage) and existing_sys is None:
                existing_sys = m.content
            else:
                other_msgs.append(m)
        
        # 傳入用戶問題以便系統提示進行問題特定的指導
        system_prompt = self._merge_system_prompt(existing_sys, current_query)
        
        # 附加最近對話記錄
        if context_window:
            system_prompt += "\n\n【最近對話記錄】\n" + "\n".join(context_window)
        
        # 附加當前搜尋結果摘要（如果有的話），並進行問題-結果關聯性分析
        search_results = state.get("search_results")
        if search_results and len(search_results) > 0:
            system_prompt += "\n\n【當前搜尋結果 - 請基於用戶問題進行關聯性參考】"
            
            for result_group in search_results:
                if result_group.get("items"):
                    query = result_group.get("query", "N/A")
                    items = result_group.get("items", [])
                    
                    system_prompt += f"\n\n搜尋查詢: '{query}'"
                    system_prompt += f"\n找到 {len(items)} 個相關結果，按相似度排序如下："
                    
                    for idx, item in enumerate(items[:5], 1):
                        score = item.get("score", 0)
                        item_type = item.get("type", "unknown")
                        payload = item.get("payload", {})
                        
                        # 計算結果與用戶原始問題的關聯度
                        relevance_indicator = "✓ 高相關" if score > 0.6 else ("⊙ 中相關" if score > 0.4 else "○ 低相關")
                        
                        system_prompt += f"\n{idx}. {relevance_indicator} 【{item_type.upper()}】(相似度: {score:.2%})"
                        
                        # 根據類型提取關鍵信息
                        if item_type == "image":
                            filename = payload.get('filename', 'N/A')
                            text = payload.get('text', 'N/A')
                            system_prompt += f"\n   文件: {filename}"
                            system_prompt += f"\n   內容: {text[:100]}"
                        
                        elif item_type == "video":
                            filename = payload.get('filename', 'N/A')
                            start_time = payload.get('start_time', 'N/A')
                            end_time = payload.get('end_time', 'N/A')
                            speaker = payload.get('speaker', 'N/A')
                            text = payload.get('text', 'N/A')
                            system_prompt += f"\n   文件: {filename}"
                            system_prompt += f"\n   時間段: {start_time}s - {end_time}s"
                            system_prompt += f"\n   講者: {speaker}"
                            system_prompt += f"\n   內容: {text[:100]}"
                        
                        elif item_type == "audio":
                            filename = payload.get('filename', 'N/A')
                            start_time = payload.get('start_time', 'N/A')
                            end_time = payload.get('end_time', 'N/A')
                            speaker = payload.get('speaker', 'N/A')
                            text = payload.get('text', 'N/A')
                            system_prompt += f"\n   文件: {filename}"
                            system_prompt += f"\n   時間段: {start_time}s - {end_time}s"
                            system_prompt += f"\n   講者: {speaker}"
                            system_prompt += f"\n   內容: {text[:100]}"
                        
                        elif item_type == "document":
                            filename = payload.get('filename', 'N/A')
                            page_numbers = payload.get('page_numbers', 'N/A')
                            text = payload.get('text', 'N/A')
                            system_prompt += f"\n   文件: {filename}"
                            system_prompt += f"\n   頁碼: {page_numbers}"
                            system_prompt += f"\n   內容: {text[:100]}"
            
            # 添加關聯性指導
            system_prompt += (
                "\n\n【如何使用搜尋結果】"
                "\n- 根據相似度指標（✓ 高/⊙ 中/○ 低）優先選擇高相關度的結果"
                "\n- 在回答中明確說明選擇該結果的原因，特別是為什麼它與用戶問題相關"
                "\n- 使用搜尋結果中的具體定位信息（文件名、時間戳、頁碼、講者等）增強回答的可信度"
                "\n- 若多個結果都相關，按相似度順序組織回答"
            )
        
        # 重建消息，確保首條為我們合併後的 system
        messages = [SystemMessage(content=system_prompt)] + other_msgs
        
        return {"messages": messages}
    
    async def _call_model(self, state: ChatState) -> Dict[str, Any]:
        """調用 LLM 模型"""
        messages = state["messages"]
        response = await self.llm_with_tools.ainvoke(messages)
        
        _logger.debug(f"Model response: {response}")
        
        return {"messages": [response]}
    
    def _should_trigger_search(self, user_text: str, already_searched: bool) -> bool:
        """基於規則判斷是否需要觸發搜尋。"""
        if already_searched:
            return False
        if not user_text:
            return False
        text = user_text.lower()
        keywords = [
            "search", "find", "查", "找", "搜尋", "來源", "哪裡", "參考",
            "pdf", "document", "doc", "image", "圖片", "video", "影片", "audio", "音訊", "音頻"
        ]
        if any(k in text for k in keywords):
            return True
        if "?" in text or "？" in text:
            return True
        return False
    
    async def _save_memory(self, state: ChatState) -> Dict[str, Any]:
        """保存對話記憶"""
        user_id = state["user_id"]
        memory_key = f"chat_memory:{user_id}"
        
        try:
            # 提取最後一條 AI 消息
            final_response = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    final_response = strip_think_tags(msg.content)
                    break
            
            # 構建記憶項
            memory_item: ConversationMemory = {
                "timestamp": time.time(),
                "user_message": state["current_query"],
                "assistant_response": final_response,
                "search_results": state.get("search_results"),
            }
            
            # 取得現有記憶
            existing_data = await asyncio.wait_for(
                self.redis_client.get(memory_key), timeout=self.redis_timeout
            )
            memories = json.loads(existing_data) if existing_data else []
            
            # 添加新記憶並保持在 context_window 大小內
            memories.append(memory_item)
            memories = memories[-self.memory_context_window :]
            
            # 保存回 Redis
            await asyncio.wait_for(
                self.redis_client.setex(
                    memory_key,
                    self.memory_ttl,
                    json.dumps(memories, ensure_ascii=False, default=str)
                ),
                timeout=self.redis_timeout,
            )
            
            _logger.debug(f"Saved memory for user {user_id}, total items: {len(memories)}")
            
            return {"final_response": final_response}
            
        except Exception as e:
            _logger.error(f"Error saving memory: {str(e)}")
            return {"final_response": strip_think_tags(final_response)}
    
    async def chat(
        self,
        user_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        進行對話
        
        Args:
            user_id: 用戶 ID
            message: 用戶消息
            history: 對話歷史 [{"role": "user/assistant", "content": "..."}]
        
        Returns:
            包含回應、搜尋結果和工具調用日誌的字典
        """
        # 構建初始消息列表（不再預先插入 SystemMessage，交由 _prepare_context 合併）
        messages = []
        
        # 添加歷史消息
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # 添加當前消息
        messages.append(HumanMessage(content=message))
        
        # 初始化狀態
        initial_state: ChatState = {
            "user_id": user_id,
            "messages": messages,
            "current_query": message,
            "search_results": None,
            "search_triggered": False,
            "context_window": [],
            "final_response": "",
            "tool_calls_log": [],
        }
        
        # 執行工作流
        result = await self.graph.ainvoke(initial_state)
        
        # 提取最終回應
        final_response = strip_think_tags(result.get("final_response", ""))
        if not final_response:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_response = strip_think_tags(msg.content)
                    break
        
        # 構建回應
        return {
            "response": final_response,
            "user_id": user_id,
            "search_results": result.get("search_results"),
            "tool_calls": result.get("tool_calls_log", []),
            "search_triggered": result.get("search_triggered", False),
            "messages": result.get("messages"),
        }
    
    async def chat_stream(
        self,
        user_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        串流對話（生成器）
        
        Args:
            user_id: 用戶 ID
            message: 用戶消息
            history: 對話歷史
            
        Yields:
            對話事件
        """
        # 與 chat 一致：不預先加入 SystemMessage
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
            "messages": messages,
            "current_query": message,
            "search_results": None,
            "search_triggered": False,
            "context_window": [],
            "final_response": "",
            "tool_calls_log": [],
        }
        async for event in self.graph.astream(initial_state):
            yield event
