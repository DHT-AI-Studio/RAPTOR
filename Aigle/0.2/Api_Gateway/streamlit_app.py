"""
Streamlit Chatbot Application with File Upload
集成 UnifiedChatService 和文件上传功能的 Streamlit 应用
"""

import streamlit as st
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import httpx
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# ==================== 配置页面 ====================

st.set_page_config(
    page_title="智能对话助手",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #757575;
    }
    .search-result {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .tool-call {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 全局配置 ====================

API_BASE_URL = "http://localhost:8012"

# ==================== API 客戶端 ====================

class ChatClient:
    """API 客户端（同步版本，避免事件循環問題）"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
        self.token = None
    
    def login(self, username: str, password: str) -> bool:
        """登入"""
        try:
            login_url = f"{self.base_url}/api/v1/auth/login"
            response = self.client.post(
                login_url,
                data={"username": username, "password": password}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data.get("access_token")
            return bool(self.token)
        except Exception as e:
            _logger.error(f"Login error: {str(e)}")
            return False
    
    def chat(
        self,
        user_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """發送聊天消息"""
        try:
            if not self.token:
                raise ValueError("Not authenticated")
            
            chat_url = f"{self.base_url}/api/v1/chat/chat"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            payload = {
                "user_id": user_id,
                "message": message,
                "history": history or []
            }
            
            response = self.client.post(
                chat_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            _logger.error(f"Chat API error: HTTP {e.response.status_code}")
            return {
                "response": f"API 錯誤: {e.response.status_code}",
                "error": str(e)
            }
        except Exception as e:
            _logger.error(f"Chat error: {str(e)}")
            return {
                "response": f"發生錯誤: {str(e)}",
                "error": str(e)
            }
    
    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        processing_mode: str = "default"
    ) -> Dict[str, Any]:
        """上傳文件進行分析"""
        try:
            if not self.token:
                raise ValueError("Not authenticated")
            
            upload_url = f"{self.base_url}/api/v1/asset/fileupload_analysis"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            files = {"primary_file": (filename, file_content)}
            data = {"processing_mode": processing_mode}
            
            response = self.client.post(
                upload_url,
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            _logger.error(f"Upload error: {str(e)}")
            return {"error": str(e)}
    
    def get_redis_cache(self) -> Dict[str, Any]:
        """獲取 Redis 中的所有緩存"""
        try:
            if not self.token:
                raise ValueError("Not authenticated")
            
            cache_url = f"{self.base_url}/api/v1/processing/cache/all"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            response = self.client.get(
                cache_url,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            _logger.error(f"Get Redis cache error: {str(e)}")
            return {"error": str(e), "count": 0, "data": {}}
    
    def close(self):
        """關閉客戶端"""
        self.client.close()


# ==================== 會話狀態初始化 ====================

if "chat_client" not in st.session_state:
    st.session_state.chat_client = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "redis_cache" not in st.session_state:
    st.session_state.redis_cache = {"count": 0, "data": {}}

if "last_upload_time" not in st.session_state:
    st.session_state.last_upload_time = None


# ==================== 工具函數 ====================

def init_client() -> ChatClient:
    """初始化客戶端"""
    if st.session_state.chat_client is None:
        st.session_state.chat_client = ChatClient()
    return st.session_state.chat_client


def login_user(username: str, password: str) -> bool:
    """登入用戶"""
    client = init_client()
    success = client.login(username, password)
    if success:
        st.session_state.authenticated = True
        st.session_state.user_id = username
        st.session_state.username = username
    return success


def send_chat_message(message: str) -> Dict[str, Any]:
    """發送聊天消息"""
    client = init_client()
    
    # 構建歷史消息
    history = []
    for msg in st.session_state.messages:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    result = client.chat(
        user_id=st.session_state.user_id,
        message=message,
        history=history
    )
    
    return result


def refresh_redis_cache() -> None:
    """刷新 Redis 緩存"""
    client = init_client()
    cache_data = client.get_redis_cache()
    st.session_state.redis_cache = cache_data


def display_search_results(search_results: Optional[List[Dict]]):
    """展示搜尋結果，支援展開或收起"""
    if not search_results:
        return
    
    for result in search_results:
        collection = result.get("collection", "Unknown")
        items = result.get("items", [])
        query = result.get("query", "")
        stats = result.get("stats", {})
        total_count = result.get("total_results", len(items))
        
        # 使用展開組件
        with st.expander(f"🔍 搜尋結果 - '{query}' ({total_count} 條)", expanded=False):
            # 顯示統計信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("總結果數", total_count)
            with col2:
                st.metric("視頻", stats.get("video", {}).get("count", 0))
            with col3:
                st.metric("音頻", stats.get("audio", {}).get("count", 0))
            with col4:
                st.metric("文檔", stats.get("document", {}).get("count", 0))
            
            st.divider()
            
            if items:
                st.markdown("### 搜尋結果詳情")
                
                for idx, item in enumerate(items, 1):
                    if isinstance(item, dict):
                        item_type = item.get("type", "unknown")
                        payload = item.get("payload", {})
                        score = item.get("score", 0)
                        
                        # 創建帶顏色的邊框卡片
                        color_map = {
                            "image": "#FF9800",
                            "video": "#2196F3",
                            "audio": "#4CAF50",
                            "document": "#9C27B0"
                        }
                        border_color = color_map.get(item_type, "#757575")
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-left: 4px solid {border_color}; padding: 1rem; margin: 1rem 0; border-radius: 0.25rem; background-color: #f9f9f9;">
                        """, unsafe_allow_html=True)
                        
                        # 標題和評分
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"#### 📄 [{idx}] {item_type.upper()}")
                        with col2:
                            st.metric("相似度", f"{score:.2%}")
                        with col3:
                            st.write(f"**狀態:** {payload.get('status', 'N/A')}")
                        
                        # 根據類型顯示不同的內容
                        st.divider()
                        
                        if item_type == "image":
                            st.write(f"**📁 文件名:** `{payload.get('filename', 'N/A')}`")
                            st.write(f"**📍 資源路徑:** `{payload.get('asset_path', 'N/A')}`")
                            
                            text_content = payload.get("text", "")
                            if text_content and text_content != "NO_TEXT_FOUND":
                                st.markdown("**📝 識別文本:**")
                                st.info(text_content)
                            else:
                                st.warning("❌ 未檢測到文本內容")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**上傳時間:** {payload.get('upload_time', 'N/A')[:10]}")
                            with col2:
                                st.write(f"**文件類型:** {payload.get('source', 'N/A')}")
                            with col3:
                                st.write(f"**字符數:** {payload.get('char_count', 'N/A')}")
                        
                        elif item_type == "video":
                            st.write(f"**🎬 文件名:** `{payload.get('filename', 'N/A')}`")
                            st.write(f"**📍 資源路徑:** `{payload.get('asset_path', 'N/A')}`")
                            
                            # 視頻時間段
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**⏱️ 開始時間:** {payload.get('start_time', 'N/A')}s")
                            with col2:
                                st.write(f"**⏹️ 結束時間:** {payload.get('end_time', 'N/A')}s")
                            with col3:
                                st.write(f"**🎤 講者:** {payload.get('speaker', 'N/A')}")
                            
                            # 轉錄內容
                            st.markdown("**📝 音頻轉錄:**")
                            st.info(payload.get("text", "無內容"))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**上傳時間:** {payload.get('upload_time', 'N/A')[:10]}")
                            with col2:
                                st.write(f"**音頻標籤:** {', '.join(payload.get('audio_labels', []))}")
                        
                        elif item_type == "audio":
                            st.write(f"**🎵 文件名:** `{payload.get('filename', 'N/A')}`")
                            st.write(f"**📍 資源路徑:** `{payload.get('asset_path', 'N/A')}`")
                            
                            # 音頻時間段
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**⏱️ 開始時間:** {payload.get('start_time', 'N/A')}s")
                            with col2:
                                st.write(f"**⏹️ 結束時間:** {payload.get('end_time', 'N/A')}s")
                            with col3:
                                st.write(f"**🎤 講者:** {payload.get('speaker', 'N/A')}")
                            
                            # 轉錄內容
                            st.markdown("**📝 音頻轉錄:**")
                            st.info(payload.get("text", "無內容"))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**上傳時間:** {payload.get('upload_time', 'N/A')[:10]}")
                            with col2:
                                st.write(f"**音頻標籤:** {', '.join(payload.get('audio_labels', []))}")
                        
                        elif item_type == "document":
                            st.write(f"**📄 文件名:** `{payload.get('filename', 'N/A')}`")
                            st.write(f"**📍 資源路徑:** `{payload.get('asset_path', 'N/A')}`")
                            
                            # 頁碼信息
                            page_numbers = payload.get("page_numbers", [])
                            if page_numbers:
                                st.write(f"**📖 頁碼:** {', '.join(map(str, page_numbers))}")
                            
                            # 文檔內容
                            st.markdown("**📝 文檔內容:**")
                            content = payload.get("text", "")
                            
                            # 處理包含 <IMAGE> 標籤的內容
                            if "<IMAGE>" in content:
                                # 分割文本和圖片描述
                                parts = content.split("<IMAGE>")
                                for part in parts:
                                    if part.strip():
                                        if part.startswith("\n圖片"):
                                            st.info(part.replace("</IMAGE>", "").strip())
                                        else:
                                            text_part = part.replace("</IMAGE>", "").strip()
                                            if text_part:
                                                st.write(text_part[:500] + ("..." if len(text_part) > 500 else ""))
                            else:
                                st.write(content[:500] + ("..." if len(content) > 500 else ""))
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**上傳時間:** {payload.get('upload_time', 'N/A')[:10]}")
                            with col2:
                                st.write(f"**字符數:** {payload.get('char_count', 'N/A')}")
                            with col3:
                                st.write(f"**元素類型:** {', '.join(payload.get('element_types', []))}")
                        
                        # 其他通用信息
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**🆔 ID:** `{item.get('id', 'N/A')[:16]}...`")
                        with col2:
                            st.write(f"**📦 Document ID:** `{payload.get('document_id', 'N/A')[:20]}...`")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("❌ 未找到搜尋結果")


def format_tool_calls(tool_calls: Optional[List[Dict]]) -> str:
    """格式化工具調用日誌"""
    if not tool_calls:
        return ""
    
    formatted = []
    for call in tool_calls:
        tool = call.get("tool", "unknown")
        query = call.get("query", "")
        results_count = call.get("results_count", 0)
        exec_time = call.get("execution_time_ms", 0)
        
        formatted.append(
            f"🔧 **{tool}** - 查詢: '{query}' | "
            f"結果: {results_count} | 耗時: {exec_time:.0f}ms"
        )
    
    return "\n".join(formatted)


def display_redis_cache() -> None:
    """展示 Redis 緩存內容"""
    cache_data = st.session_state.redis_cache
    
    if "error" in cache_data and cache_data["error"]:
        st.error(f"無法獲取緩存: {cache_data['error']}")
        return
    
    count = cache_data.get("count", 0)
    data = cache_data.get("data", {})
    
    st.markdown(f"### 📊 Redis 緩存 (共 {count} 項)")
    
    if count == 0:
        st.info("Redis 中沒有緩存數據")
        return
    
    # 搜尋功能
    search_query = st.text_input(
        "🔍 搜尋緩存 (按 key 或 value 內容)",
        placeholder="輸入關鍵字過濾...",
        key="redis_cache_search"
    )
    
    # 過濾數據
    filtered_data = {}
    if search_query.strip():
        search_lower = search_query.lower()
        for key, value in data.items():
            # 按 key 搜尋
            if search_lower in key.lower():
                filtered_data[key] = value
                continue
            
            # 按 value 搜尋
            try:
                value_str = str(value).lower()
                if search_lower in value_str:
                    filtered_data[key] = value
            except Exception:
                pass
    else:
        filtered_data = data
    
    # 顯示過濾結果信息
    if search_query.strip():
        st.markdown(f"**找到 {len(filtered_data)} 項結果** (共 {count} 項)")
        if len(filtered_data) == 0:
            st.warning("❌ 未找到匹配的緩存項")
            return
    else:
        st.markdown(f"**顯示所有 {len(filtered_data)} 項**")
    
    st.divider()
    
    # 顯示所有的 key-value 對
    for idx, (key, value) in enumerate(filtered_data.items(), 1):
        with st.expander(f"🔑 [{idx}] {key}", expanded=False):
            st.markdown("**值:**")
            try:
                # 嘗試將值格式化為 JSON 顯示
                if isinstance(value, str):
                    try:
                        import json
                        parsed_value = json.loads(value)
                        st.json(parsed_value)
                    except (json.JSONDecodeError, ValueError):
                        st.code(value, language="text")
                else:
                    st.write(value)
            except Exception as e:
                st.write(f"無法格式化值: {str(e)}")
                st.write(value)


# ==================== 登入頁面 ====================

def show_login_page():
    """顯示登入頁面"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 🔐 智能對話助手")
        st.markdown("---")
        
        username = st.text_input(
            "用戶名",
            placeholder="輸入用戶名",
            key="login_username"
        )
        password = st.text_input(
            "密碼",
            type="password",
            placeholder="輸入密碼",
            key="login_password"
        )
        
        if st.button("登入", use_container_width=True):
            if not username or not password:
                st.error("請輸入用戶名和密碼")
            else:
                with st.spinner("登入中..."):
                    success = login_user(username, password)
                    if success:
                        st.success("登入成功！")
                        st.rerun()
                    else:
                        st.error("登入失敗，請檢查用戶名和密碼")


# ==================== 聊天頁面 ====================

def show_chat_page():
    """顯示聊天頁面"""
    # 侧边栏
    with st.sidebar:
        st.markdown(f"### 👤 用戶: {st.session_state.username}")
        
        if st.button("🚪 登出", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.chat_client = None
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💬 對話工具")
        
        if st.button("🔄 清空對話", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # 主界面
    st.markdown("# 💬 智能對話助手")
    
    # 消息顯示區域
    st.markdown("### 對話記錄")
    
    if not st.session_state.messages:
        st.info("👋 歡迎！開始與助手對話吧。")
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**👤 你:** {msg['content']}")
            else:
                # 助手消息
                st.markdown(f"**🤖 助手:** {msg['content']}")
                
                # 顯示搜尋結果（可展開）
                if msg.get("search_results"):
                    display_search_results(msg["search_results"])
                
                # 顯示工具調用日誌
                if msg.get("tool_calls"):
                    with st.expander("🔧 工具調用日誌"):
                        tool_text = format_tool_calls(msg["tool_calls"])
                        st.markdown(tool_text)
            
            st.markdown("---")
    
    # 輸入區域
    st.markdown("### 發送消息")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_message = st.text_input(
            "輸入你的消息",
            placeholder="請輸入你的問題...",
            key=f"chat_input_{len(st.session_state.messages)}"
        )
    
    with col2:
        send_button = st.button("發送 📤", use_container_width=True)
    
    if send_button and user_message:
        # 添加用戶消息
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 發送到 API
        with st.spinner("正在處理..."):
            result = send_chat_message(user_message)
        
        # 處理響應
        if "error" in result and result["error"]:
            st.error(f"錯誤: {result.get('response', '未知錯誤')}")
        else:
            # 添加助手消息
            assistant_msg = {
                "role": "assistant",
                "content": result.get("response", "無回應"),
                "search_results": result.get("search_results"),
                "tool_calls": result.get("tool_calls")
            }
            st.session_state.messages.append(assistant_msg)
            st.rerun()


# ==================== 文件上傳頁面 ====================

def show_upload_page():
    """顯示文件上傳頁面"""
    st.markdown("# 📁 文件分析")
    
    with st.sidebar:
        st.markdown("### 👤 用戶信息")
        st.markdown(f"用戶: **{st.session_state.username}**")
        
        if st.button("🚪 登出", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.chat_client = None
            st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 上傳文件")
        uploaded_file = st.file_uploader(
            "選擇要分析的文件",
            type=["pdf", "txt", "doc", "docx", "png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv", "wav", "mp3", "m4a"]
        )
        
        processing_mode = st.selectbox(
            "處理模式",
            ["default", "fast", "detailed"],
            help="選擇文件分析的處理模式"
        )
    
    with col2:
        st.markdown("### 信息")
        st.info("支持的文件類型:\n- 文檔: PDF, TXT, DOC, DOCX\n- 圖片: PNG, JPG, JPEG\n- 影片: MP4, AVI, MOV, MKV\n- 音頻: WAV, MP3, M4A")
    
    if uploaded_file and st.button("🔍 分析文件", use_container_width=True):
        with st.spinner("上傳並分析中..."):
            client = init_client()
            result = client.upload_file(
                uploaded_file.read(),
                uploaded_file.name,
                processing_mode
            )
        
        if "error" in result:
            st.error(f"上傳失敗: {result['error']}")
        else:
            st.success("文件已上傳並開始分析！")
            st.json(result)
            # 上傳成功後自動刷新 Redis 緩存
            refresh_redis_cache()
            st.session_state.last_upload_time = datetime.now()
    
    # 分隔線
    st.markdown("---")
    
    # Redis 緩存查看區塊
    st.markdown("### 🗂️ 查看緩存")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("查看 Redis 中的所有緩存數據")
    
    with col2:
        if st.button("🔄 刷新緩存", use_container_width=True, key="refresh_cache_btn"):
            refresh_redis_cache()
    
    # 顯示 Redis 緩存
    display_redis_cache()


# ==================== 主頁面 ====================

def main():
    """主應用程序"""
    if not st.session_state.authenticated:
        show_login_page()
    else:
        # 選擇頁面
        page = st.sidebar.radio(
            "選擇功能",
            ["💬 對話", "📁 文件上傳"],
            key="page_selector"
        )
        
        if page == "💬 對話":
            show_chat_page()
        elif page == "📁 文件上傳":
            show_upload_page()


if __name__ == "__main__":
    main()
