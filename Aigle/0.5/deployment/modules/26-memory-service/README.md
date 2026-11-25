# Module 26 — Memory Service

Raptor 0.5 的持久化記憶層。每位用戶有獨立的 Session 對話紀錄（`.mv2`）及跨 Session 長期記憶（`.mv2`），以 BM25 + HNSW 向量索引提供語意搜尋。

---

## 目錄

- [架構](#架構)
- [環境設定](#環境設定)
- [啟動服務](#啟動服務)
- [API 參考](#api-參考)
  - [Session 記憶](#session-記憶-memorysessions)
  - [Long-term 記憶](#long-term-記憶-memorylongterm)
  - [Multimedia 記憶](#multimedia-記憶-memorymultimedia)
  - [搜尋 API](#搜尋-api)
  - [Timeline / 管理](#timeline--管理)
  - [記憶提取（自動）](#記憶提取自動)
  - [記憶壓縮](#記憶壓縮)
- [測試](#測試)
  - [Unit 測試](#unit-測試不需-ollama--redis--nfs)
  - [Latency Benchmark](#latency-benchmark)
  - [手動測試（curl）](#手動測試接真實-keycloak--redis--ollama)
- [GUI](#guidemo_guipy)
- [模組依賴](#模組依賴)

---

## 架構

```
用戶請求
    │
    ▼
Module 13 (API Gateway)  ← JWT 驗證
    │
    ▼
Module 26 (Memory Service :8026)
    │
    ├── /memory/sessions/*    → session_{id}.mv2          每輪對話紀錄
    ├── /memory/longterm/*    → long_term_N.mv2           跨 Session 事實 / 偏好（sharded）
    ├── /memory/multimedia/*  → media/{type}_{hash}.mv2   影片 / 音訊 / 圖片索引
    ├── /memory/search        → 全域搜尋（sessions + longterm + multimedia）
    ├── /memory/stats         → 使用者記憶統計
    └── /memory/export        → 完整記憶匯出（串流 JSON）
```

```
每輪對話輸入
    │
    ├── [Tier 1] Redis  最近 5 輪 · 1 小時 TTL · 毫秒級讀取
    │
    └── [Tier 2] MemVID .mv2
          ├── session_{id}.mv2    完整對話紀錄，BM25 + HNSW 全文搜尋
          └── long_term_N.mv2     重要事實 / 偏好 / 實體，BAAI/bge-m3 本地語意搜尋
```

---

## 環境設定

| 變數 | 預設值 | 說明 |
|---|---|---|
| `MEM_STORAGE_ROOT` | `/mnt/memvid-storage` | `.mv2` 根目錄（NFS 掛載點） |
| `MEM_EMBEDDING_MODEL` | `BAAI/bge-m3` | SentenceTransformer 本地 embedding 模型（HuggingFace 名稱） |
| `MEM_REDIS_HOST` | `raptor-redis-standalone` | Redis 主機名 |
| `MEM_REDIS_PORT` | `6379` | Redis 埠 |
| `MEM_REDIS_PASSWORD` | — | Redis 密碼 |
| `MEM_EXTRACTION_MODEL` | `gemma4:12b` | 提取用 LLM 名稱（需在 Module 07 / Ollama 中註冊） |
| `MEM_MODULE07_URL` | `http://raptor-ai-lifecycle-api:8010` | Module 07 AI Lifecycle API 位址（提取 + ASR） |
| `PORT_MEMORY_SERVICE` | `8026` | 對外服務埠 |
| `MEM_COMPACT_BUFFER_TOKENS` | `13000` | 壓縮安全緩衝，預留給 system prompt / token 計算誤差 |
| `MEM_COMPACT_RESERVED_OUTPUT_TOKENS` | `20000` | 保留給 LLM 輸出回覆的空間 |
| `MEM_COMPACT_MIN_TAIL_TOKENS` | `10000` | tail（保留區）最少 token 數，不足時向前擴展 |
| `MEM_COMPACT_MIN_TEXT_MESSAGES` | `5` | tail 最少保留的對話輪數 |
| `MEM_COMPACT_MAX_TAIL_TOKENS` | `40000` | tail 最多保留的 token 數，超出部分才會被摘要 |
| `MEM_COMPACT_SUMMARY_MAX_TOKENS` | `12000` | LLM 產生摘要時的最大 token 數 |
| `MEM_COMPACT_MAX_SECTION_TOKENS` | `2000` | 既有 summary 每個 `# heading` 段落的上限，重用於 incremental 摘要前超出才截斷 |
| `MEM_COMPACT_KEEP_TURNS` | `10` | 無論 token 預算為何，一定保留的最近完整輪數 |
| `MEM_COMPACT_TOOL_RESULT_RETENTION_HOURS` | `24.0` | 一定保留的 tool/result pair 時間窗（小時），超出才可能被摘要掉 |

---

## 啟動服務

### 正式部署

```bash
cd deployment/modules
python build.py -m 26 --build
```

### 本地直接啟動（開發用）

```bash
conda activate CIE
pip install -r deployment/modules/26-memory-service/requirements.txt
```

```bash
cd deployment/modules/26-memory-service

MEM_STORAGE_ROOT=/tmp/mv_demo \
MEM_REDIS_HOST=localhost \
MEM_REDIS_PORT=6381 \
MEM_REDIS_PASSWORD=dht888888 \
MEM_MODULE07_URL=http://localhost:8010 \
PYTHONPATH=app \
python -m uvicorn main:app --host 0.0.0.0 --port 8026
```

```bash
curl http://localhost:8026/health
# {"status":"ok","memvid_version":"2.0.160"}
```

`GET /metrics` 回傳 Prometheus 格式指標（`prometheus-fastapi-instrumentator` 提供的通用 HTTP
請求數/延遲，加上 compact 相關的自訂業務指標：`memory_compact_pre_tokens`、
`memory_compact_post_tokens`、`memory_compact_turns_compacted`、`memory_compact_turns_kept`、
`memory_compact_summary_write_seconds`、`memory_compact_llm_call_seconds{path=}`、
`memory_compact_llm_output_tokens{path=}`、`memory_compact_threshold_exceeded_total`）。

---

## API 參考

所有端點必須帶 `X-User-ID` header 帶入 `user_id`。

### Session 記憶（`/memory/sessions`）

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/memory/sessions/{session_id}/turns` | 寫入一輪對話 |
| `GET` | `/memory/sessions/{session_id}/recent?n=10` | 取最近 N 輪（時間升序） |
| `GET` | `/memory/sessions` | 列出當前用戶的所有 session |
| `DELETE` | `/memory/sessions/{session_id}` | 刪除 session（`404` 如不存在） |

**POST `/memory/sessions/{session_id}/turns`**

```json
{
  "user_message": "你好，我想查上次討論的項目",
  "assistant_response": "根據之前的記錄，上次討論的是...",
  "search_results": [],
  "tool_calls": [],
  "timestamp": 1718000000.0
}
```

---

### Long-term 記憶（`/memory/longterm`）

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/memory/longterm/facts` | 寫入事實 / 偏好 / 實體 |
| `GET` | `/memory/longterm/facts` | 列出所有 `preference` + `fact` + `entity`（時間降序） |
| `POST` | `/memory/longterm/search` | BM25 + HNSW 語意搜尋（支援 Unix timestamp 日期過濾） |
| `DELETE` | `/memory/longterm/facts/{frame_id}` | 刪除單筆事實 / 偏好 / 實體；不存在回 `404` |
| `DELETE` | `/memory/longterm` | 清空整個 long-term 記憶（`404` 如不存在） |

**POST `/memory/longterm/facts`**

```json
{
  "text": "用戶偏好繁體中文回覆",
  "frame_type": "preference",
  "session_id": "sess_abc"
}
```

`frame_type` 可選值：`conversation` | `preference` | `entity` | `fact`

**POST `/memory/longterm/search`**

```json
{
  "query": "用戶偏好的語言設定",
  "top_k": 5,
  "from_date": 1700000000.0,
  "to_date": 1750000000.0
}
```

`from_date` / `to_date` 為 Unix timestamp，可選。

---

### Multimedia 記憶（`/memory/multimedia`）

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/memory/multimedia/video` | 索引 video moment（transcript + 時間範圍） |
| `POST` | `/memory/multimedia/audio` | 索引 audio clip（省略 `transcription` 自動呼叫 Module 07 ASR） |
| `POST` | `/memory/multimedia/image` | 索引圖片（description + OCR text） |
| `POST` | `/memory/multimedia/search` | 跨媒體語意搜尋（可依 `media_type` 過濾） |

**POST `/memory/multimedia/video`**

```json
{
  "asset_path": "videos/cooling_system.mp4",
  "version_id": "abc123",
  "start_sec": 120.0,
  "end_sec": 145.0,
  "transcription": "冷卻系統透過循環流動冷卻液降低引擎溫度",
  "session_id": "sess_001",
  "context_query": "查一下冷卻系統的工作原理"
}
```

---

### 搜尋 API

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/memory/sessions/{session_id}/search` | 在單一 session 內做混合搜尋 |
| `POST` | `/memory/search` | 跨所有 sessions + long-term + multimedia 全域搜尋 |

**POST `/memory/sessions/{session_id}/search`**

```json
{
  "query": "冷卻系統",
  "top_k": 5,
  "from_date": "2026-01-01T00:00:00Z",
  "to_date": "2026-12-31T23:59:59Z"
}
```

`from_date` / `to_date` 為 ISO 8601 字串，可選。

**POST `/memory/search`**（全域）

```json
{
  "query": "引擎冷卻",
  "top_k": 5,
  "scope": ["sessions", "longterm", "multimedia"]
}
```

`scope` 預設全選，可縮小至任意子集。

---

### Timeline / 管理

| 方法 | 路徑 | 說明 |
|---|---|---|
| `GET` | `/memory/sessions/{session_id}/timeline` | 分頁瀏覽對話歷史；支援 time-travel |
| `GET` | `/memory/stats` | 記憶統計數字 |
| `GET` | `/memory/export` | 串流匯出完整記憶（JSON） |
| `DELETE` | `/memory` | GDPR 抹除：刪除當前用戶所有記憶 |

**GET `/memory/sessions/{session_id}/timeline`**

Query 參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `page` | `1` | 頁碼（1-based） |
| `page_size` | `20` | 每頁筆數（最大 100） |
| `at` | — | ISO 8601 timestamp；指定後只回傳該時間點**之前**的 frames（time-travel） |

**GET `/memory/stats`**

```json
{
  "session_count": 12,
  "total_turns": 347,
  "total_media_items": 23,
  "long_term_frame_count": 58,
  "storage_bytes_used": 4194304
}
```

**GET `/memory/export`**

回傳串流 JSON，格式穩定，帶版本標記：

```json
{
  "export_schema_version": "1.0",
  "user_id": "user_abc",
  "exported_at": "2026-07-02T10:00:00+00:00",
  "sessions": [
    {"session_id": "sess_001", "turns": [...]}
  ],
  "longterm": [...],
  "multimedia": [...]
}
```

`multimedia` 只含 metadata（asset_path、timestamps、text），不含原始媒體檔案。

**DELETE `/memory`**

永久刪除當前用戶所有 `.mv2`、`.meta.json` 及 Redis 索引鍵。回傳 `204 No Content`。

---

### 記憶提取（自動）

每次寫入對話（`POST /sessions/{id}/turns`）後，系統會在**背景自動**對該輪進行 LLM 提取，不阻塞回應。

#### 提取流程

```
append_turn() 寫入 .mv2
    │
    └─ asyncio.create_task (fire-and-forget)
         │
         ├─ 讀取用戶現有 long-term facts（最多 10 筆，含 frame_id）
         │
         ├─ 傳給 Module 07 /inference/infer（extraction_model，temperature=0）
         │     引擎：ollama；系統提示：輸出 ADD / DELETE / UPDATE JSON 操作列表
         │
         └─ 執行操作
               ADD    → 新增一筆 preference / entity / fact
               DELETE → soft-delete 舊 frame（從搜尋索引移除）
               UPDATE → DELETE 舊 frame + ADD 新版本
```

#### 跨 Session 持久化

提取結果儲存在 `user_{id}/long_term_N.mv2`，與 session 無關。
同一用戶在任意 session 說的偏好都會被累積，且互相覆蓋（UPDATE/DELETE 處理矛盾偏好）。

```
Session A (turn 3): 「我偏好英文回覆」
    └─ 提取 → ADD preference "user prefers English"

Session C (turn 1): 「改用繁體中文」
    └─ 提取 → DELETE frame_42 + ADD preference "user prefers Traditional Chinese"

Session D (任意): retrieve_long_term 搜尋
    └─ 找到「user prefers Traditional Chinese」→ 帶入 LLM context
```

#### 驗證提取是否走 Module 07

```bash
# 記錄基準值
curl -s http://localhost:8010/inference/stats | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('total_inferences:', d['stats']['total_inferences'])"

# 觸發一次提取
curl -s -X POST http://localhost:8026/memory/longterm/extract \
  -H "X-User-ID: $USER_ID" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-sess","turn":{"user_message":"我喜歡用繁體中文","assistant_response":"好的"}}'

# 再查一次，counter 應 +1
curl -s http://localhost:8010/inference/stats | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('total_inferences:', d['stats']['total_inferences'])"
```

memory service log 也會出現：
```
Module 07 extraction call succeeded (model=gemma4:12b, processing_time=X.XXXs)
```

#### 內部端點（不經 Gateway）

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/memory/longterm/extract` | 手動觸發提取（需 `X-User-ID` header；不接受 body 傳入 `user_id`） |

#### GDPR 內部端點（`X-Internal-Token`，只給模組 33 呼叫）

跟上面用 JWT 的端點不同——這些端點取明確傳入的 `user_id`，不經過使用者自己的 session（管理員發起的抹除沒有被抹除者自己的 JWT 可用），呼叫端身份靠共用的 `X-Internal-Token` 驗證。

| 方法 | 路徑 | 說明 |
|---|---|---|
| `DELETE` | `/internal/gdpr/user/{user_id}` | 全帳號抹除：刪除該用戶在本服務的所有資料 |
| `DELETE` | `/internal/gdpr/memory/user/{user_id}` | 記憶同意撤回（局部抹除）：今天跟上一列動作相同，刻意保留成獨立路由，讓兩種法律依據之後能分別調整 |
| `GET` | `/internal/gdpr/user/{user_id}/export` | 匯出用的 JSON 內容（sessions + long_term），餵進模組 33 匯出 ZIP 的 memory/ 區塊 |
| `GET` | `/internal/gdpr/user/{user_id}/summary` | 只回數量統計，給模組 33 的 DSAR 組裝器（不是完整匯出，避免超出 DSAR 的 10 秒預算） |
| `DELETE` | `/internal/gdpr/memory/sessions/expired?older_than_days=N&dry_run=` | 清除過期 session；`dry_run=true` 只回數量不刪除，供模組 33 的保留期 dry-run 端點使用 |

**POST `/memory/longterm/extract`**

```json
{
  "session_id": "sess_001",
  "turn": {
    "user_message": "我偏好繁體中文",
    "assistant_response": "好的，已記下您的偏好。"
  }
}
```

> `user_id` 由 Service 從 `X-User-ID` header 取得，**body 不需要也不接受 `user_id` 欄位**。

`stored` 為本次實際執行的 ADD/UPDATE 操作數（DELETE 也計入）。

---

### 記憶壓縮

當單一 session 的對話歷史超過 LLM context window 時，將較舊的對話輪次摘要成一筆 `summary` frame，並保留最近的高保真度 tail，避免 prompt 過長。

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/memory/compact/evaluate` | 估算請求體傳入 `messages` 的 token 用量，回傳是否需要壓縮（不寫入任何東西） |
| `POST` | `/memory/sessions/{session_id}/compact/evaluate` | 彙總該 session 的 turns + long-term facts + multimedia，估算實際 token 用量（不寫入任何東西） |
| `POST` | `/memory/sessions/{session_id}/compact` | 壓縮一個 session，寫入 summary frame |
| `GET` | `/memory/sessions/{session_id}/summaries` | 列出該 session 的所有 summary frames |
| `DELETE` | `/memory/sessions/{session_id}/summaries/{summary_id}` | 刪除一筆 summary frame（`404` 如不存在） |

#### 判斷邏輯

```
threshold = context_window - MEM_COMPACT_RESERVED_OUTPUT_TOKENS - MEM_COMPACT_BUFFER_TOKENS

session 總 token 數 < threshold → 不壓縮（source="under_budget"）
session 總 token 數 ≥ threshold → 觸發壓縮：
    1. 找出上次壓縮邊界（last_summarized_frame_id 或最新 summary frame 的 source_turn_range.to）
    2. 從邊界之後的 turns 開始，往前累積至滿足 MEM_COMPACT_MIN_TAIL_TOKENS /
       MEM_COMPACT_MIN_TEXT_MESSAGES，但不超過 MEM_COMPACT_MAX_TAIL_TOKENS → 這段留原文（tail）
    3. tail 以外、邊界以後的 turns → 交給 LLM 摘要
         - 若已有 summary frame 且內容非空 / 非純標題骨架（quality gate），優先用「增量摘要」
           （既有摘要按 MEM_COMPACT_MAX_SECTION_TOKENS 逐段截斷後 + 新 turns → 更新摘要）
         - 若既有 summary 是空字串或只有 `# heading` 沒有內容（例如 LLM 曾經回傳異常），
           視為不可重用，直接走全新 LLM 摘要，不嘗試增量
         - 若 LLM 呼叫失敗，fail open：回傳 compacted=false, source="llm_failed"，不影響現有對話
    4. 摘要結果寫回 session 的 .mv2 作為 frame_type="summary" 的 frame，並額外寫一筆
       frame_type="compact_boundary" 記錄本次壓縮的 preserved_segment（head/anchor/tail frame_id）
```

**POST `/memory/compact/evaluate`**

只評估請求體傳入的 `messages`，不會自己去查 Redis / session `.mv2` / long-term / multimedia 加總。這是給呼叫端自備訊息列表時使用的通用版本；要評估某個 session 的實際歸檔量，改用下面的 session-scoped 版本。

```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "context_window": 128000,
  "extra_tokens": 0
}
```

**POST `/memory/sessions/{session_id}/compact/evaluate`**

彙總該 session 已歸檔的 turns、該用戶的 long-term facts（`get_facts`，非 embedding 搜尋）、
multimedia 索引（`list_all`，非 embedding 搜尋）三項的 token 數，回傳格式與上面相同的
`CompactEvaluation`。`extra_tokens` 用來補上尚未寫入任何 store 的當前這輪內容（例如使用者這次的
提問）。Module 21 在組 grounded prompt 前會先呼叫這個 endpoint，`should_compact=false` 時直接跳過
`/compact` 呼叫；`should_compact=true` 或呼叫失敗（fail open）時才呼叫 `/compact`。

```json
{
  "context_window": 128000,
  "extra_tokens": 0
}
```

**POST `/memory/sessions/{session_id}/compact`**

```json
{
  "trigger": "auto",
  "context_window": 128000,
  "last_summarized_frame_id": null,
  "custom_instructions": null,
  "dry_run": false
}
```

`source` 可能為：`under_budget`（未達門檻）、`no_session`、`no_turns`、`nothing_to_compact`、`session_memory`（用既有摘要增量更新）、`llm_compact`（全新 LLM 摘要）、`llm_failed`（LLM 失敗，fail open，未壓縮）。

`dry_run=true` 時只回傳評估結果，不寫入 summary frame。

`threshold_exceeded`：壓縮後（含摘要裁切等縮減嘗試）token 數是否仍 ≥ threshold。`true` 代表 fail
open——已盡力把摘要裁到 `threshold - tail_tokens` 的剩餘預算內，但 tail 本身已經逼近或超過
threshold（例如 context_window 設得比 reserved_output + buffer 還小），無法完全達標；呼叫方可能
需要自行再裁切或調整 context_window。這不代表壓縮失敗，`compacted` 仍為 `true`、summary frame 仍會寫入。

#### Module 15 / Module 21 整合

兩邊都已接上 Module 26 的 `/memory/sessions/{session_id}/compact`（保留摘要、可稽核），不是只做本地裁切。本地裁切只作為 Module 26 不可用時的 fail-open fallback。

Module 15 直接呼叫 `/compact`，靠該端點內部的 threshold 判斷 no-op。Module 21 則先呼叫
`/memory/sessions/{session_id}/compact/evaluate`：`should_compact=false` 時跳過 `/compact`；
`should_compact=true` 或 evaluate 呼叫失敗（回傳 `None`，fail open）時才呼叫 `/compact`，一樣靠
`/compact` 內部的 threshold 判斷兜底，不會因為 evaluate 短暫失敗就漏掉真正需要的壓縮。

Module 15（Chat Service）：LangGraph 流程為 `load_memory → retrieve_long_term → retrieve_session_history → compact_context → prepare_context → ... → save_memory → archive_memory`。

- `retrieve_long_term`：回答前呼叫 `/memory/longterm/search`，撈跨 session 的偏好/事實併入 system prompt，fail open。
- `retrieve_session_history`：呼叫 `/memory/sessions/{id}/search`，語意搜尋比 Redis `context_window` 更早之前的對話。
- `archive_memory`：每輪對話結束後 fire-and-forget 寫入 `/memory/sessions/{id}/turns`。
- `compact_context`：先呼叫 `MemoryClient.compact_session()` → Module 26 `/memory/sessions/{session_id}/compact`；壓縮成功則把 `get_latest_summary()` 取回的摘要文字以 `[歷史摘要]` 前綴插到 `context_window` 最前面。Module 26 不可用、沒有 session 可壓縮、或回傳未壓縮時，fallback 成依 `COMPACT_CONTEXT_BUDGET`（預設 20000 tokens，`chars/4` 估算）從舊到新裁切 Redis 短期 `context_window`；全程 fail open。

Module 21（Agent Protocol）：`run_rag_pipeline` Step 7 在組 grounded prompt 前先呼叫
`MemoryClient.evaluate_compact()` → Module 26 `/memory/sessions/{session_id}/compact/evaluate`。
若回傳 `should_compact=false` 就跳過本次 `compact_session()` 呼叫；若回傳 `should_compact=true`，
或 evaluate 本身失敗（`None`，fail open），才呼叫 `MemoryClient.compact_session()` +
`get_latest_summary()`，壓縮成功則把摘要以 `[歷史摘要]` 前綴插進 `memory_lines` 最前面，與
long-term/multimedia 檢索結果一起併入 prompt；`session_id` 省略時退回用 `user_id` 當 session
key。呼叫失敗只 log warning，不影響回答（fail open）。`COMPACT_PROMPT_BUDGET_TOKENS`（預設
80000）是獨立的第二道防線，裁切的是檢索到的 chunks，不是 session 歷史。

各自的整合邏輯有獨立 unit test：

```bash
# Module 15：ChatService._compact_context()（含 Module 26 成功/declined/error 三種路徑）+ retrieve_long_term / MemoryClient.search_longterm、compact_session、get_latest_summary
cd deployment/modules/15-chat-service
python -m pytest test_compact_context.py test_retrieve_long_term.py -v

# Module 21：MemoryClient（含 compact_session/get_latest_summary fail-open）+ pipeline.py Step 5 的 chunk token 預算裁切
cd deployment/modules/21-agent-protocol
python -m pytest test_memory_integration.py test_compact_budget.py -v
```

---

## 測試

### Unit 測試（不需 Ollama / Redis / NFS）

```bash
cd deployment/modules/26-memory-service

# 全套快速測試
conda run -n CIE python -m pytest \
  test_memvid_store.py \
  test_session_memory.py \
  test_api.py \
  test_long_term_memory.py \
  test_multimedia_memory.py \
  test_search.py \
  test_timeline_export.py \
  test_extraction.py \
  test_compact_memory.py \
  -v -m "not slow"

# 單一模組
conda run -n CIE python -m pytest test_search.py -v -m "not slow"
conda run -n CIE python -m pytest test_timeline_export.py -v
conda run -n CIE python -m pytest test_extraction.py -v
conda run -n CIE python -m pytest test_compact_memory.py -v
```

| 測試檔 | 測試數 | 涵蓋範圍 |
|---|---|---|
| `test_memvid_store.py` | 9 | `.mv2` CRUD、BM25 搜尋、timeline |
| `test_session_memory.py` | 8 | Redis sorted set、turn index、delete |
| `test_api.py` | 8 | HTTP 狀態碼、response schema、404 |
| `test_long_term_memory.py` | 15 | 100 frames、型別過濾、日期範圍、刪除 |
| `test_multimedia_memory.py` | 12 | video/audio/image 索引、跨媒體搜尋、metadata、型別過濾、`list_all` |
| `test_search.py` | 18 (+2 slow) | Session 搜尋（日期過濾、user 隔離）、全域搜尋（scope、total_frames_searched）、API 整合 |
| `test_timeline_export.py` | 23 | Timeline 分頁、time-travel、Stats、Export JSON 格式、GDPR 刪除、API 整合 |
| `test_extraction.py` | 23 | `_parse_ops` 各操作類型、ADD/DELETE/UPDATE 執行、LLM 失敗不爆、背景觸發 |
| `test_compact_memory.py` | 11 | token 估算、tail 保留範圍計算、壓縮觸發門檻、summary frame 讀寫/刪除、LLM 失敗 fail open |

**總計：124 個 unit / integration tests**

---

### Latency Benchmark

與 unit tests 分開執行（seed 資料量大，耗時較長）：

```bash
# Long-term 搜尋 SLA（seed 10,000 frames，約 5 分鐘）
conda run -n CIE python -m pytest test_latency.py -v -m slow

# 搜尋 SLA（seed 50 sessions × 100 turns，約 12 分鐘）
conda run -n CIE python -m pytest test_search.py -v -m slow
```

| 測試 | 規模 | SLA |
|---|---|---|
| `test_search_latency_under_200ms` | 10,000 long-term frames，5 種 query | P95 < 200ms |
| `test_search_latency_worst_case_top50` | 10,000 frames，top_k=50 | < 200ms |
| `test_search_latency_with_date_filter` | 10,000 frames，含日期過濾 | < 200ms |
| `test_session_search_latency_50_sessions` | 50 sessions × 100 turns | P95 < 300ms |
| `test_global_search_latency_50_sessions` | 50 sessions + 50 long-term facts | P95 < 300ms |

---

### 手動測試 / Module 15 整合驗證

可透過服務的互動式 API 文件（FastAPI `GET /docs`）手動測試所有端點，並驗證 Module 15 是否已接上。

---

## GUI（demo_gui.py）

```bash
conda run -n CIE streamlit run demo_gui.py
```

瀏覽器開啟 `http://localhost:8501`

### Tab 說明

| Tab | 功能 |
|---|---|
| Session 記憶 | 寫入對話、查最近 N 輪、列出 sessions、刪除 session |
| Long-term 記憶 | 寫入 fact/preference、列出 facts、語意搜尋、清空 |
| Multimedia 記憶 | 索引 video/audio/image、跨媒體搜尋 |
| **搜尋** | Session 搜尋（ISO 8601 日期過濾）、全域搜尋（scope 多選） |
| **Timeline / 管理** | Timeline 分頁 + time-travel、Stats 統計卡片、Export 下載 JSON、GDPR 清除 |
| **記憶提取** | 手動觸發提取、查看現有 long-term facts（含 frame_id）、自動提取機制說明 |
| **記憶壓縮** | 快速產生測試資料、估算 token 用量、觸發壓縮並查看 summary frame |

### 端對端測試流程

1. Sidebar → Health Check（確認 ✅）→ Login（取得 token）
2. **Session 記憶** → 寫入 3 輪對話到 `sess_test_001`
3. **Long-term 記憶** → 寫入 `用戶偏好繁體中文回覆`（preference）、`用戶的車型是 Toyota Camry 2022`（fact）
4. **搜尋 → Session 搜尋** → query=`引擎冷卻`，預期 turn #1 score 最高，`total_frames_searched=3`
5. **搜尋 → 全域搜尋** → query=`引擎`，scope 全選，預期 sessions + longterm 各有結果
6. **Timeline / 管理 → Timeline** → `sess_test_001`，page_size=2，確認 `has_next=true`
7. **Timeline / 管理 → Timeline** → 勾選 Time-Travel，設時間為今天 `00:00:00`，確認 `total=0`
8. **Timeline / 管理 → 統計** → 確認 `session_count=1`、`total_turns=3`
9. **Timeline / 管理 → 匯出** → 下載 JSON，確認 `export_schema_version="1.0"`
10. **Timeline / 管理 → 清除** → 勾選確認後清除，回 Stats 確認全部歸零
11. **記憶提取 → 手動觸發** → 輸入對話內容，按「觸發提取」，確認 `stored >= 1`
12. **記憶提取 → 重新載入** → 查看提取出的 preference / fact，確認含 frame_id

---

## 模組依賴

| 模組 | 用途 |
|---|---|
| 01 NFS Server | `.mv2` + `.meta.json` 持久化儲存 |
| 02 Redis | Session sorted set 索引 |
| 07 AI/ML | `/inference/infer`（port 8010）呼叫 `gemma4:12b` 做偏好提取；`BAAI/bge-m3` SentenceTransformer 本地載入做 HNSW 向量 embedding |
| 13 API Gateway | JWT 驗證；`/api/0.3/sso/login` 取 token（port 8012） |
