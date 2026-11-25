# Raptor API Gateway

模組 13 — Raptor 平台的對外 API Gateway（`/api/{版本}/*`）。

## Guardrail 內容檢查

`app/middlewares/guardrail.py` 的 `GuardrailMiddleware` 會把使用者輸入（部分路由也包含回應）送到 Guardrail Service（模組 23）的 `POST /guardrail/policy/decide` 檢查。每次呼叫都帶 `module="13"` 與請求的 `X-Request-ID`（沒有時自動產生），非 allow 的結果由模組 23 寫入 audit trail。

### 檢查哪些路由

| 路由（`POST /api/{v}/...`） | 檢查內容 | 不安全時 |
|---|---|---|
| `/chat/completions` | 輸入：最後一則 user 訊息；輸出：`choices[0].message.content` | 一律阻擋（422） |
| `/a2a/query` | 輸入：`question`；輸出：`answer` | 一律阻擋（422） |
| `/chat/chat` | 輸入：`message`；輸出：`response` | 一律阻擋（422） |
| `/search/hybrid`、`/search/vector`、`/search/bm25` | 只檢查輸入：`query` | 依 `GATEWAY_GUARDRAIL_MODE` |
| `/docagent/process` | 只檢查輸入：`options` 表單欄位（JSON 字串）裡的 `question`，且只在 `task=qa` 時 | 依 `GATEWAY_GUARDRAIL_MODE` |

後兩列的路由表可用 `GATEWAY_GUARDRAIL_ROUTES` 調整（`{path_prefix: body_field}`，相對於 `/api/{v}`，只檢查 POST；`a.b` 表示巢狀欄位）。舊版路徑 `/api/0.3/...` 會先由 `LegacyApiAliasMiddleware` 改寫成正式路徑，所以同樣會被檢查。

### 模式（只套用在 search 與 DocAgent）

| `GATEWAY_GUARDRAIL_MODE` | 行為 |
|---|---|
| `audit`（預設） | 在背景檢查，不等待、不阻擋，請求照常處理；不安全內容記一筆 WARNING：`[guardrail] unsafe content: request_id=… path=… category=… decision=…` |
| `strict` | 等待檢查結果；`block` 時回 `422 {"error": "content_policy_violation", "policy_id": "<第一個類別，例如 S1>"}` |

`/chat/chat`、`/chat/completions`、`/a2a/query` 不受這個設定影響：`GR_ENABLED=true` 時一律阻擋，所以 `audit` 預設不會削弱既有的保護。

decide 的其他結果：`warn` 放行並記 WARNING；`redact`（只在前三個路由）把檢查的欄位換成 decide 回傳的 `redacted_content`。

### 設定

| 變數 | 預設 | 說明 |
|---|---|---|
| `GR_ENABLED` | `false` | 總開關。`false` 時完全略過，不呼叫模組 23、不讀取 body，沒有額外延遲 |
| `GUARDRAIL_URL` | `http://raptor-guardrail-service:8026` | 模組 23 的容器內部位址（內部 port 8026，不是對外的 8023） |
| `GUARDRAIL_TIMEOUT` | `20.0` | 呼叫模組 23 的逾時秒數 |
| `GATEWAY_GUARDRAIL_MODE` | `audit` | `audit` 或 `strict`，見上表 |
| `GATEWAY_GUARDRAIL_ROUTES` | search 三個路由 + `/docagent/process` | JSON 格式的路由表，見上方 |

### 失敗時與內部請求

- **Fail-open**：模組 23 連不到、逾時、回非 2xx 或格式錯誤時，一律當作 allow 放行，guardrail 故障不會讓 chat、search 等 API 掛掉。前三個路由會記 WARNING；search 與 DocAgent 只記 DEBUG，而且連線逾時只有 2 秒，模組 23 連不到時幾乎不增加延遲。
- **`X-Internal: true`**：帶這個 header 的請求完全略過檢查。外部請求的這個 header 會先被最外層的 `IngressHeaderStripMiddleware` 移除，所以 client 無法用它跳過檢查。
