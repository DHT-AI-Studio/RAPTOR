# Chat Service（Module 15）

以 LangGraph 實作的 RAG 聊天服務：組合短期記憶（Redis）、跨 session 記憶（Module 26）與個人資料搜尋（Module 25），再呼叫 LLM 產生回答。

LLM 透過 OpenAI 相容 API（`ChatOpenAI`）呼叫，位址是 `LLM_BASE_URL`。docker-compose 預設把它指向 **Module 23 的 OpenAI 相容 guardrail proxy**（`http://raptor-guardrail-service:8026/v1`），所以每次 LLM 呼叫的輸入與輸出都會先經過 guardrail 檢查，再由 Module 23 轉送到真正的 LLM（見 Module 23 README 的「`/v1/chat/completions`、`/v1/models`」一節）。

- 容器內部監聽 `8021`，對外發布於 `PORT_CHAT_SERVICE`（預設 `8021`）
- Swagger UI：`http://localhost:8021/docs`

## API 參考

| 端點 | 用途 |
|---|---|
| `POST /api/v1/chat` | 聊天：送出一則使用者訊息，取得 AI 回答 |
| `GET /api/v1/chat/memory/{user_id}` | 讀取使用者的短期對話記憶（Redis） |
| `DELETE /api/v1/chat/memory/{user_id}` | 清除使用者的短期對話記憶（Redis） |
| `GET /health` | 健康檢查，回傳 `{"status": "healthy", "service": "chat-service"}` |
| `GET /` | 服務資訊（名稱、版本、狀態） |
| `GET /metrics` | Prometheus 指標 |

### `POST /api/v1/chat`

**用途**：聊天主入口。依使用者的問題、對話記憶與（需要時）搜尋結果，產生一則回答。

**接收內容**（`Content-Type: application/json`）：

| 欄位 | 必填 | 型別 | 說明 |
|---|---|---|---|
| `user_id` | ✓ | string | 使用者 ID；短期記憶、跨 session 記憶與搜尋都以它區分 |
| `message` | ✓ | string | 本輪使用者訊息 |
| `history` | | `[{"role": "user"\|"assistant", "content": string}]` | 呼叫端自帶的對話歷史（選填），會放在本輪訊息之前 |
| `session_id` | | string | session ID；有帶時短期記憶以 `user_id` + `session_id` 區分 |
| `stream` | | bool | `true` 時改以 Server-Sent Events 回應（見下方），預設 `false` |

標頭 `X-Request-ID`（選填）：本次請求的 ID，會出現在回應、SSE 事件與 log；未帶時由服務產生。

```json
{
  "user_id": "user-1",
  "message": "幫我找義大利麵的筆記",
  "session_id": "s-001"
}
```

**回傳內容**（HTTP 200）：

| 欄位 | 說明 |
|---|---|
| `response` | AI 回答（已移除 `<think>` 標籤並轉為繁體中文） |
| `user_id`、`session_id` | 原樣回傳 |
| `request_id` | 本次請求 ID（`X-Request-ID` 或服務產生） |
| `search_triggered` | 本輪是否有搜尋到結果 |
| `search_results` | 搜尋結果（未搜尋時為 `null`） |
| `tool_calls` | 本輪執行過的工具與結果紀錄 |

處理過程中發生未預期的錯誤時回 HTTP `500`，`detail` 為錯誤訊息。

**串流（`"stream": true`）**：回應為 `text/event-stream`，標頭帶 `X-Request-ID`：

| event | data | 時機 |
|---|---|---|
| `start` | `{"request_id", "session_id"}` | 收到請求立即送出 |
| `delta` | `{"text"}` | 回答，逐句（依 `。！？!?` 與換行切） |
| `end` | `{"request_id", "response", "search_triggered", "tool_calls"}` | 完成；`response` 為完整回答，`tool_calls` 為次數 |
| `error` | `{"request_id", "message"}` | 取代 `delta`／`end`；SSE 的 HTTP 狀態仍為 200 |

LangGraph 執行期間每 10 秒送一個 `: ping` 註解行，避免代理因閒置斷線；客戶端斷線時停止這次執行。
LLM 經由 Module 23 的 guardrail proxy，它要檢查完整輸出、不支援串流，所以 `delta` 是在 guardrail 檢查過的完整回答
產生後逐句送出，而不是逐 token；事件格式已可承接逐 token 的串流，待 Module 23 支援串流輸出檢查後只需改這裡的來源。
Module 34 的語音對話以此接收回答並逐句合成語音。

**內部流程**（`src/services/chat_service.py` 的 LangGraph，依序執行）：

1. **`load_memory`**：從 Redis（key `chat_memory:{user_id}[:{session_id}]`）讀取最近 `MEMORY_CONTEXT_WINDOW` 輪對話，作為【最近對話記錄】。
2. **`retrieve_long_term`**：向 Module 26 查詢與本輪問題相關的跨 session 長期記憶（最多 3 筆），作為【長期記憶】。
3. **`retrieve_session_history`**：向 Module 26 以語意搜尋本 session 較早的對話（最多 5 筆），作為【相關歷史對話】。
4. **`compact_context`**：請 Module 26 判斷本 session 是否超過 `LLM_CONTEXT_WINDOW_TOKENS`，需要時壓縮並把摘要加到最前面；Module 26 不可用時改在本地依 `COMPACT_CONTEXT_BUDGET` 截斷較舊的對話。
5. **`prepare_context`**：組出 system prompt — 核心職責與工具使用規則、【當前用戶問題】、【長期記憶】、【相關歷史對話】、【最近對話記錄】。
6. **`call_model`**：以 `ChatOpenAI` 呼叫 `LLM_BASE_URL`（模型 `LLM_MODEL`），綁定 3 個工具：`unified_search_tool`（搜尋）、`calculate`（計算）、`get_current_time`（時間），並帶 `engine`、`num_ctx`（`LLM_NUM_CTX`）、`think`（`LLM_THINK`），以及 header `X-Raptor-Module: 15`（Module 23 稽核紀錄用）。
7. **`execute_tools` → `generate_response`**：模型要求呼叫工具，或問題含有搜尋關鍵字（search / find / 查 / 找 / 搜尋 / pdf / image / 影片…或問號）時，執行工具（搜尋走 Module 25 `PERSONAL_DB_SERVICE_URL`），再把結果交給模型產生最終回答。否則直接進入下一步。
8. **`save_memory`**：把本輪「使用者訊息＋回答」寫入 Redis，只保留最近 `MEMORY_CONTEXT_WINDOW` 輪，TTL 為 `MEMORY_TTL` 秒（每次寫入都會重設）。
9. **`archive_memory`**：把本輪對話送到 Module 26 封存（背景執行，不影響回應）。

Module 26 的步驟（2–4、9）一律 fail-open：Module 26 不可用時略過，聊天照常進行。

**Guardrail 行為**（第 6、7 步的 LLM 呼叫，`src/services/llm_guardrail.py`）：

| Module 23 的回應 | `/api/v1/chat` 的行為 |
|---|---|
| 正常回答 | 照常回答 |
| `400 policy_violation`（輸入或輸出違規） | HTTP 200，`response` 為「此請求違反內容政策（S1），無法回覆。」；不再觸發搜尋；**不會** fallback |
| 連不上 Module 23（連線失敗、逾時） | 記 error log；`GUARDRAIL_FALLBACK_PASSTHROUGH=true` 時改直連 `LLM_FALLBACK_BASE_URL` 照常回答，`false` 時回 `500` |
| `503 guardrail_unavailable`（Module 23 的檢查本身失敗） | 同上（Module 23 端也會記 error log） |
| 其他錯誤（例如 `502 upstream_unavailable`、上游的 `5xx`） | 不 fallback，回 `500` |

開關打開時，對 Module 23 的呼叫不做重試，連不上就立即 fallback。

**送出的訊息順序**：第 6、7 步送出前會把 system prompt 排到最前面，其餘訊息依對話順序排列：`[system, …history, 本輪 user]`，工具回合再接 `assistant(tool_calls)`、`tool`。LangGraph 的 `add_messages` 原本會把第 5 步新組的 system message 接到最後。Module 23 檢查的是 `messages[-1]`；遇到 `role=tool` 時，改查最後一則 `role=user`。因此它檢查的永遠是本輪使用者訊息，system prompt 裡的記憶不會被當成使用者輸入。某一輪被擋過的問題雖然會留在【最近對話記錄】中，但不會讓之後的回合也被擋。

```bash
curl -s -X POST http://localhost:8021/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","message":"你好"}'
curl -sN -X POST http://localhost:8021/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","message":"你好","stream":true}'
```

### `GET /api/v1/chat/memory/{user_id}`

**用途**：查看使用者目前的短期對話記憶（`load_memory` 讀到的內容）。

**接收內容**：path 參數 `user_id`；query 參數 `session_id`（選填）。

**內部流程**：讀取 Redis key `chat_memory:{user_id}[:{session_id}]`，回傳：

```json
{"user_id": "user-1", "session_id": null, "memory_count": 1,
 "memories": [{"timestamp": 1760000000.0, "user_message": "你好", "assistant_response": "你好！", "search_results": null}]}
```

Redis 錯誤時回 `500`。

### `DELETE /api/v1/chat/memory/{user_id}`

**用途**：清除使用者的短期對話記憶（例如重新開始對話）。

**接收內容**：path 參數 `user_id`；query 參數 `session_id`（選填，需與聊天時使用的相同）。

**內部流程**：刪除 Redis key `chat_memory:{user_id}[:{session_id}]`，成功回 `204`（無 body）；Redis 錯誤時回 `500`。只清除 Redis 的短期記憶，不影響 Module 26 已封存的內容。

## 環境變數設定

| 容器內變數 | `.env` 來源（docker-compose） | 預設值 | 用途 |
|---|---|---|---|
| `LLM_BASE_URL` | `CHAT_LLM_BASE_URL` | `http://raptor-guardrail-service:8026/v1` | LLM 呼叫位址（Module 23 guardrail proxy）。要回到直連：`CHAT_LLM_BASE_URL=${LLM_BASE_URL}` |
| `LLM_FALLBACK_BASE_URL` | `LLM_BASE_URL`（共用） | `http://raptor-ai-lifecycle-api:8010/v1` | Module 23 失效時直連的原上游；空字串表示不 fallback |
| `GUARDRAIL_FALLBACK_PASSTHROUGH` | `GUARDRAIL_FALLBACK_PASSTHROUGH` | `true` | Module 23 連不上或回 `guardrail_unavailable` 時是否改直連 |
| `LLM_MODEL` | `CHAT_LLM_MODEL` | `gemma4:12b` | 聊天模型 |
| `TEMPERATURE` | `TEMPERATURE` | `0.7` | 取樣溫度 |
| `LLM_NUM_CTX` | `LLM_NUM_CTX` | `262144` | 模型 context 長度（只有 Module 07 會套用，Ollama 的 `/v1` 會忽略） |
| `LLM_THINK` | `CHAT_LLM_THINK` | `false` | gemma4 系列模型的思考模式（模型預設會開啟，設 `false` 可跳過以大幅降低延遲） |
| `PERSONAL_DB_SERVICE_URL` | 同名 | `http://raptor-personal-db-service:8000` | Module 25 搜尋 |
| `CHAT_MEMORY_SERVICE_URL` | 同名 | `http://raptor-memory-service:8026` | Module 26 記憶服務 |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_DB` | 同名 | `raptor-redis-standalone` / `6379` / — / `0` | 短期記憶 |
| `MEMORY_CONTEXT_WINDOW` | 同名 | `5` | 短期記憶保留的輪數 |
| `MEMORY_TTL` | 同名 | `3600` | 短期記憶 TTL（秒） |
| `REQUEST_TIMEOUT` | 同名 | `60.0` | 對 Module 25／26 的 HTTP 逾時 |

## 測試方式

```bash
cd deployment/modules/15-chat-service
python -m pytest tests/unit/ -v
```

不需要任何外部服務：Module 23／LLM、Module 26 都以 `httpx.MockTransport` 模擬，Redis 與搜尋使用 in-memory 的假物件。
`test_chat_stream.py` 以假的聊天服務測 `request_id`（帶入／產生）、SSE 事件順序、`error` 事件與 `: ping`。
