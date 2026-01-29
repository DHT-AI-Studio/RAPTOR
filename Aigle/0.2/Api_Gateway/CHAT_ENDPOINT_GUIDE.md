# 聊天端點使用指南

本指南說明如何使用新創建的聊天端點，該端點集成了 `UnifiedChatService`，提供基於 LangGraph 的對話功能。

## 概述

聊天端點位於 `/api/v1/chat`，提供以下功能：

- **對話管理**：基於 LangGraph 的多節點工作流
- **統一搜尋集成**：自動調用 Unified Search 工具
- **短期記憶管理**：使用 Redis 存儲用戶對話記憶（TTL=3600s，Context Window=5）
- **工具調用**：支持 `unified_search_tool`、`calculate` 和 `get_current_time`
- **認證保護**：所有端點需要 JWT Bearer token

## 端點列表

### 1. 發送聊天消息

**端點：** `POST /api/v1/chat/chat`

**認證：** ✅ 必須提供 Bearer token

**請求體：**
```json
{
  "user_id": "dht_admin",
  "message": "什麼是機器學習？",
  "history": [
    {
      "role": "user",
      "content": "你好"
    },
    {
      "role": "assistant",
      "content": "你好！有什麼我可以幫助的嗎？"
    }
  ]
}
```

**參數說明：**

| 參數 | 類型 | 必需 | 說明 |
|------|------|------|------|
| `user_id` | string | ✅ | 用戶 ID（必須與認證用戶相符） |
| `message` | string | ✅ | 用戶消息內容 |
| `history` | array | ❌ | 對話歷史記錄（包含 role 和 content） |

**成功回應（200）：**
```json
{
  "response": "機器學習是人工智能的一個分支，通過讓計算機從數據中學習模式...",
  "user_id": "dht_admin",
  "search_triggered": true,
  "search_results": [
    {
      "collection": "unified",
      "items": [
        {
          "title": "機器學習基礎",
          "type": "article",
          "content": "機器學習是...",
          "score": 0.95
        }
      ],
      "stats": {
        "total": 42
      },
      "query": "機器學習"
    }
  ],
  "tool_calls": [
    {
      "tool": "unified_search_tool",
      "query": "機器學習",
      "results_count": 42,
      "execution_time_ms": 245.3,
      "api_endpoint": "http://localhost:8012/api/v1/search/unified_search"
    }
  ]
}
```

**回應欄位說明：**

| 欄位 | 類型 | 說明 |
|------|------|------|
| `response` | string | 助手的文本回應 |
| `user_id` | string | 用戶 ID |
| `search_triggered` | boolean | 是否觸發了搜尋工具 |
| `search_results` | array | 搜尋結果列表（如果搜尋被觸發） |
| `tool_calls` | array | 所有工具調用的記錄和執行時間 |

**常見錯誤：**

| 狀態碼 | 錯誤 | 說明 |
|--------|------|------|
| 401 | Unauthorized | 缺少或無效的 Bearer token |
| 403 | Forbidden | 無權訪問該用戶的數據 |
| 500 | Internal Server Error | 聊天服務內部錯誤 |

**cURL 示例：**
```bash
# 1. 先登入獲取 token
TOKEN=$(curl -X POST "http://localhost:8012/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=dht_admin&password=dht00000" \
  | jq -r '.access_token')

# 2. 使用 token 發送聊天消息
curl -X POST "http://localhost:8012/api/v1/chat/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "dht_admin",
    "message": "什麼是機器學習？",
    "history": []
  }' | jq .
```

**Python 示例：**
```python
import httpx
import json

# 配置
BASE_URL = "http://localhost:8012"
USERNAME = "dht_admin"
PASSWORD = "dht00000"

# 創建客戶端
client = httpx.Client(base_url=BASE_URL)

# 登入獲取 token
login_response = client.post(
    "/api/v1/auth/login",
    data={"username": USERNAME, "password": PASSWORD}
)
token = login_response.json()["access_token"]

# 發送聊天消息
headers = {"Authorization": f"Bearer {token}"}
chat_response = client.post(
    "/api/v1/chat/chat",
    headers=headers,
    json={
        "user_id": USERNAME,
        "message": "什麼是機器學習？",
        "history": []
    }
)

result = chat_response.json()
print("回應：", result["response"])
print("搜尋已觸發：", result["search_triggered"])
print("工具調用：", result["tool_calls"])
```

**Node.js 示例：**
```javascript
const fetch = require('node-fetch');

const BASE_URL = "http://localhost:8012";
const USERNAME = "dht_admin";
const PASSWORD = "dht00000";

async function chat() {
  // 登入
  const loginRes = await fetch(`${BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `username=${USERNAME}&password=${PASSWORD}`
  });
  const { access_token } = await loginRes.json();

  // 發送聊天
  const chatRes = await fetch(`${BASE_URL}/api/v1/chat/chat`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${access_token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      user_id: USERNAME,
      message: "什麼是機器學習？",
      history: []
    })
  });

  const result = await chatRes.json();
  console.log("回應：", result.response);
  console.log("搜尋已觸發：", result.search_triggered);
}

chat().catch(console.error);
```

---

### 2. 取得用戶記憶

**端點：** `GET /api/v1/chat/memory/{user_id}`

**認證：** ✅ 必須提供 Bearer token

**參數：**

| 參數 | 類型 | 位置 | 說明 |
|------|------|------|------|
| `user_id` | string | URL | 用戶 ID（必須與認證用戶相符） |

**成功回應（200）：**
```json
{
  "user_id": "dht_admin",
  "memory_count": 3,
  "memories": [
    {
      "timestamp": 1704067200.0,
      "user_message": "什麼是機器學習？",
      "assistant_response": "機器學習是...",
      "search_results": [
        {
          "collection": "unified",
          "items": [...],
          "stats": {...}
        }
      ]
    },
    ...
  ]
}
```

**cURL 示例：**
```bash
curl -X GET "http://localhost:8012/api/v1/chat/memory/dht_admin" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

---

### 3. 清除用戶記憶

**端點：** `DELETE /api/v1/chat/memory/{user_id}`

**認證：** ✅ 必須提供 Bearer token

**成功回應（204）：** 無內容（記憶已清除）

**cURL 示例：**
```bash
curl -X DELETE "http://localhost:8012/api/v1/chat/memory/dht_admin" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 工作流程說明

聊天服務使用 LangGraph 實現以下 7 節點工作流：

```
┌─────────────┐
│ load_memory │ ← 載入用戶的短期記憶
└──────┬──────┘
       ↓
┌──────────────────┐
│ prepare_context  │ ← 準備系統提示和上下文
└──────┬───────────┘
       ↓
┌───────────────┐
│ call_model    │ ← 調用 LLM（mistral-small:latest）
└──────┬────────┘
       ↓
    [條件判斷] ← 檢查是否需要執行工具
   /    |    \
  /     |     \
search  tools  end
 │       │      │
 ↓       ↓      │
[search_node] [process_tool] │
 │       │      │
 └───┬───┘      │
     │          │
   ↓ [重新調用 LLM]
   save_memory  ← 保存記憶到 Redis
   │
   ↓
  [結束]
```

### 工作流節點詳解

1. **load_memory**: 從 Redis 載入用戶的短期記憶（最多保留 5 條）
2. **prepare_context**: 構建系統提示，包含記憶上下文
3. **call_model**: 調用 LLM 並檢查是否觸發工具調用
4. **_should_continue**: 條件判斷
   - 如果調用 `unified_search_tool` → 轉至 `search_node`
   - 如果調用其他工具 → 轉至 `process_tool_result`
   - 否則 → 保存記憶
5. **search_node**: 異步調用 `/api/v1/search/unified_search` 端點
6. **process_tool_result**: 處理其他工具調用
7. **save_memory**: 保存當前交互到 Redis（使用 SETEX，TTL=3600s）

---

## 記憶管理

### 短期記憶特性

- **存儲位置**: Redis
- **Key 格式**: `chat_memory:{user_id}`
- **Context Window**: 5（最多保留 5 條歷史交互）
- **TTL**: 3600 秒（1 小時）
- **內容**: 包含時間戳、用戶消息、助手回應和搜尋結果

### 記憶 JSON 結構

```json
{
  "timestamp": 1704067200.0,
  "user_message": "什麼是機器學習？",
  "assistant_response": "機器學習是人工智能的一個分支...",
  "search_results": [
    {
      "collection": "unified",
      "items": [...],
      "stats": {...},
      "query": "機器學習"
    }
  ]
}
```

### 記憶操作

```bash
# 查看記憶
curl -X GET "http://localhost:8012/api/v1/chat/memory/dht_admin" \
  -H "Authorization: Bearer $TOKEN"

# 清除記憶
curl -X DELETE "http://localhost:8012/api/v1/chat/memory/dht_admin" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 配置參數

所有配置參數均在環境變數中設置，前綴為 `GATEWAY_`：

| 環境變數 | 預設值 | 說明 |
|---------|--------|------|
| `GATEWAY_API_BASE_URL` | `http://localhost:8012` | API 基礎 URL |
| `GATEWAY_CHAT_MODEL_NAME` | `mistral-small:latest` | LLM 模型名稱 |
| `GATEWAY_CHAT_TEMPERATURE` | `0.7` | LLM 溫度參數（0.0-2.0） |
| `GATEWAY_OPENAI_API_KEY` | `None` | OpenAI API Key（可選） |
| `GATEWAY_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API 基礎 URL（Ollama） |
| `GATEWAY_CHAT_MEMORY_CONTEXT_WINDOW` | `5` | 記憶上下文窗口大小 |
| `GATEWAY_CHAT_MEMORY_TTL` | `3600` | 記憶 TTL（秒） |

### 設置示例（.env 文件）

```bash
# 聊天服務配置
GATEWAY_API_BASE_URL=http://localhost:8012
GATEWAY_CHAT_MODEL_NAME=mistral-small:latest
GATEWAY_CHAT_TEMPERATURE=0.7
GATEWAY_LLM_BASE_URL=http://localhost:11434/v1
GATEWAY_CHAT_MEMORY_CONTEXT_WINDOW=5
GATEWAY_CHAT_MEMORY_TTL=3600

# Redis 配置
GATEWAY_REDIS_HOST=localhost
GATEWAY_REDIS_PORT=6379

# 認證配置
GATEWAY_JWT_SECRET_KEY=your-secret-key-here
GATEWAY_JWT_ALGORITHM=HS256
```

---

## 工具調用詳解

### 1. Unified Search Tool

自動搜尋工具，當用戶提出需要搜尋的問題時觸發。

**觸發條件：**
- 用戶提出信息查詢相關問題
- LLM 判斷需要調用搜尋

**搜尋流程：**
1. 登入獲取 token
2. 調用 `/api/v1/search/unified_search` 端點
3. 傳遞查詢和搜尋參數
4. 返回最多 2 個集合的結果

**搜尋參數：**
```json
{
  "query_text": "用戶查詢",
  "embedding_type": "text",
  "limit_per_collection": 2
}
```

### 2. Calculate Tool

計算數學表達式。

**觸發條件：**
- 用戶要求計算
- LLM 判斷需要進行計算

**示例：**
```
用戶: 計算 2 + 2
LLM: 調用 calculate("2 + 2")
工具結果: 計算結果: 4
```

### 3. Get Current Time Tool

獲取當前時間。

**觸發條件：**
- 用戶詢問現在幾點
- LLM 需要時間信息

---

## 認證流程

1. **獲取 Token**
   ```bash
   curl -X POST "http://localhost:8012/api/v1/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=dht_admin&password=dht00000"
   ```

2. **使用 Token**
   ```bash
   curl -X POST "http://localhost:8012/api/v1/chat/chat" \
     -H "Authorization: Bearer <TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "dht_admin", "message": "..."}'
   ```

3. **Token 過期**
   - TTL: 30 分鐘
   - 過期後需重新登入

---

## 故障排除

### 問題 1: "401 Unauthorized"

**原因：** Token 無效或過期

**解決方案：**
```bash
# 重新登入
curl -X POST "http://localhost:8012/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=dht_admin&password=dht00000"
```

### 問題 2: "403 Forbidden - You can only access your own chat data"

**原因：** 試圖訪問其他用戶的數據

**解決方案：**
- 確保 `user_id` 與認證用戶相符
- 或使用具有管理員權限的帳戶

### 問題 3: 搜尋返回空結果

**原因：** 
- 查詢不匹配集合中的數據
- 搜尋服務未正常運行

**解決方案：**
```bash
# 檢查搜尋服務狀態
curl -X GET "http://localhost:8012/health"

# 檢查日誌
docker logs <fastapi-gateway-container>
```

### 問題 4: "HTTP client context not initialized"

**原因：** UnifiedChatService 初始化失敗

**解決方案：**
1. 檢查 Redis 連接
   ```bash
   redis-cli ping
   ```

2. 檢查 HTTP 客戶端配置
3. 查看應用日誌

---

## 性能優化建議

1. **批量對話**
   - 使用 `history` 參數提供完整對話上下文
   - 減少 LLM 的重新思考時間

2. **記憶管理**
   - 定期清除過期記憶
   - 監控 Redis 記憶使用量

3. **搜尋優化**
   - 避免過於寬泛的查詢
   - 使用特定的查詢關鍵詞

4. **連接池**
   - 應用已配置最優連接池參數
   - 默認最大連接數: 100
   - 最大保活連接數: 20

---

## API 文檔

自動生成的交互式 API 文檔：
- **Swagger UI**: http://localhost:8012/docs
- **ReDoc**: http://localhost:8012/redoc

---

## 範例應用

### 完整對話流程

```python
import httpx

BASE_URL = "http://localhost:8012"

async def main():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 1. 登入
        login_resp = await client.post(
            "/api/v1/auth/login",
            data={"username": "dht_admin", "password": "dht00000"}
        )
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. 第一條消息
        chat_resp = await client.post(
            "/api/v1/chat/chat",
            headers=headers,
            json={"user_id": "dht_admin", "message": "嗨"}
        )
        print("消息 1:", chat_resp.json()["response"])
        
        # 3. 後續消息（帶歷史）
        history = [
            {"role": "user", "content": "嗨"},
            {"role": "assistant", "content": chat_resp.json()["response"]}
        ]
        
        chat_resp = await client.post(
            "/api/v1/chat/chat",
            headers=headers,
            json={
                "user_id": "dht_admin",
                "message": "什麼是 Python？",
                "history": history
            }
        )
        print("消息 2:", chat_resp.json()["response"])
        
        # 4. 查看記憶
        memory_resp = await client.get(
            "/api/v1/chat/memory/dht_admin",
            headers=headers
        )
        print("記憶：", memory_resp.json()["memory_count"], "條")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 相關資源

- [LangGraph 文檔](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 文檔](https://platform.openai.com/docs/api-reference)
- [Redis 文檔](https://redis.io/documentation)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)
