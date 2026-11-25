# Raptor Guardrail Service

模組 23 — Raptor 平台的內容安全防護服務。部署在 LLM 推理的前端（或旁路），在內容送進模型之前、以及模型回應送回使用者之前，進行安全檢查。

## 概觀

服務內同時存在三種獨立的安全機制，全部從同一張 Postgres 資料表讀取 policy 內容，但各自用不同方式解讀：

| 機制 | 前綴路徑 | 運作方式 |
|---|---|---|
| Guard-model 分類（**deprecated**） | `/guard/check/*` | 原始多模型分類（Llama-Guard3 / Granite Guardian / GPT-OSS-Safeguard），使用各模型固定的 guard prompt — 不涉及任何 policy。請改呼叫 `POST /guardrail/policy/decide`，v1.0 移除 |
| LLM-judged policy check | `/policy/check/llm/*` | 與 `/guard/check/*` 相同的多模型派送方式，但每個模型拿到的是當前啟用 policy 的內容作為自己的 prompt，而非固定的 guard prompt |
| detector checker | `/guardrail/check/*` | 依照當前啟用 policy 的 regex + Llama-Guard detector 規則進行檢查；同時也負責違規稽核紀錄（audit log）與全域開關 |

呼叫端應該使用的是 **policy-aware check**（GRD-X1），兩者跑的是同一套判定（`app/services/policy_check_service.py`，把上述三者串成一條流程並套用啟用中的 policy）：
- `POST /guardrail/policy/decide`：canonical decision API，**呼叫端的正式整合點**，回傳 `allow | redact | warn | block`。
- `/api/v0.2/check/*`：同一套判定，但回應維持 `/guard/check/*` 的 `CheckResponse` 格式；保留給需要該格式或需要檢查多輪對話（conversation）的情境。

詳見下方 API 參考。

此外還有一個獨立的舊版機制，**confidence-based proxy**（`POST /api/generate`）— 由單一分類模型回傳 `{category, confidence, reason}` 判定結果，並依可設定的信心分數門檻決定放行或阻擋。這個機制早於上述四種機制存在，與它們的 policy 格式無關。

> **提醒：** `app/routers/proxy_openai.py`（`POST /v1/chat/completions`，同一套 confidence-based proxy 的 OpenAI 相容版本）雖然存在於程式碼中，但**並未掛載**於 `app/main.py`— 目前是死碼（dead code），並非實際可用的端點。

> **實際部署狀態：** Module 07、13、35、12 都呼叫 `POST /guardrail/policy/decide`（GRD-X1）。沒有啟用中的
> policy 時，decide 使用各 guard 模型固定的 guard prompt，所以判斷靠的是 guard 模型自己內建的能力。要開幾個 guard model 由 root
> `.env` 的 `GUARD_MODEL`/`GUARD_MODEL_2`/`GUARD_MODEL_3` 決定（留空 `_2`/`_3` 就是單模型），多模型意見不一致時依 `GUARD_CONFLICT_STRATEGY` 決定結果。

健康檢查：`GET /health`。互動式 API 文件：`GET /docs`。

## 快速開始

```bash
cd deployment/modules/23-guardrail-service
cp .env.example .env   # 填入實際密碼等機密資訊 — 詳見下方「環境變數設定」
docker compose up -d
```

這會在外部的 `raptor` Docker network 上啟動一個容器（該 network 須事先存在；可透過 `DOCKER_NETWORK` 覆寫其名稱）：

- `raptor-guardrail-service` — 服務本身（FastAPI），發布於 `INTERNAL_BIND_IP:PORT_GUARDRAIL_SERVICE`（預設 `127.0.0.1:8023`，只有 docker 主機本身連得到；容器內部固定監聽 `8026`）。本服務是內部服務，其他容器一律透過 `raptor` network 呼叫 `http://raptor-guardrail-service:8026`；多主機部署時才把 `INTERNAL_BIND_IP` 設為 core 主機的內網 IP（見 `BUILD.md` §6.1）

policy 內容 + 違規稽核紀錄存在 module 03 共用的 Postgres 裡（database `guardrails`，由 `03-database/init/postgresql/001_init.sql` 建立);啟用中 policy 的快取 + 全域開關狀態存在 module 02 共用的 Redis（DB 0,所有 key 都掛 `guardrail:` 前綴,不會跟其他模組的 key 打架)。不再另外啟動專屬的 DB 容器。

**Build context**：`docker-compose.yml` 的 build context 是上一層的 `deployment/modules/`（比照 module 36），因此 Dockerfile 的 COPY 路徑都帶有 `23-guardrail-service/` 前綴，並另外把共用的稽核 library `common/raptor_audit` 複製到 `/app/raptor_audit`。在本機執行測試時，`pytest.ini` 的 `pythonpath = . ../common` 讓 `raptor_audit` 可以被 import。

app 容器會等待上述兩個依賴服務的健康檢查都通過後才會啟動。此外服務也預期有一個**外部的 Ollama 實例**可透過 `OLLAMA_URL` 連線，且所需的 guard/proxy 模型已事先下載完成 — 這個服務本身不會執行 Ollama。

本文件中的 `localhost:8023` 範例都要**在 docker 主機上**執行。從其他機器無法直接連線；從其他容器呼叫時改用 `http://raptor-guardrail-service:8026`（例如 `docker exec <container> curl ...`）。

```bash
curl http://localhost:8023/health
open http://localhost:8023/docs
```

### 啟動檢查與健康檢查

設定不完整時要在部署當下就看得到，而不是事故發生時才發現。以下檢查都在 FastAPI lifespan 的啟動流程中執行，**只記錄 log、不會讓啟動失敗**：

| 檢查 | 條件 | log |
|---|---|---|
| Guard 模型 | 每個設定的 guard 模型（`GUARD_MODEL`/`_2`/`_3` 或 `GUARD_MODELS`）連不到 Ollama，或 Ollama 上沒有這個模型 | `ERROR Guard model {model} unreachable at {url} — guardrail will fail open`（後面附原因） |
| 啟用中的 policy | `GR_DEFAULT_ENABLED=true`，且連內建的 default Standard Policy 都無法啟用（PostgreSQL 中沒有啟用中的 policy） | `WARNING Guardrail enabled but no active policy configured — run seed_default_policy.py` |
| 時區 | `TIMEZONE` 不是有效的 IANA 名稱 | `WARNING invalid TIMEZONE ...`，改用 `Asia/Taipei` |

Guard 模型與 proxy 模型的檢查會同時進行，Ollama 連不到時啟動最多只多等 10 秒。「連得到」指 Ollama 的模型清單（`/api/tags`）裡有這個模型；不會實際送推論請求，所以模型第一次載入較慢不算連不到。影像 guard 模型（Llama-Guard-3-11B-Vision、ShieldGemma 2）在本服務內以 GPU 執行、不經過 Ollama，不在此檢查範圍內。`GR_DEFAULT_ENABLED` 只是 Redis 冷啟動時的預設值；實際的全域開關是 Redis 的 `guardrail:enabled`（見 `GET /guardrail/system/status`）。

`GET /health` 每次呼叫都即時檢查（不是沿用啟動時的結果，所以補拉模型或啟用 policy 後不需要重啟），回應中包含：

| 欄位 | 意義 |
|---|---|
| `guard_model_reachable` | 所有設定的 guard 模型都在 Ollama 上（Ollama 連不到時為 `false`） |
| `active_policy_configured` | PostgreSQL 中目前有啟用中的 policy（資料庫無法查詢時為 `false`） |

既有的狀態規則不變：guard 模型缺少、Ollama 或 Redis 連不到時回 503 + `status: "degraded"`；沒有啟用中的 policy 只反映在 `active_policy_configured: false`，不改變狀態碼。

## 環境變數設定

所有變數皆從此目錄下的 `.env` 讀取（範本請見 `.env.example`）。這些設定都不是由本服務自行佈建，Postgres/Redis 是 module 03/02 共用的執行個體。

> `REQUEST_TIMEOUT` 的**主機端**變數名稱前面加了 `GUARDRAIL_` 前綴（`GUARDRAIL_REQUEST_TIMEOUT`），避免跟 module 15 自己的 `REQUEST_TIMEOUT` 撞名。Postgres/Redis 直接沿用平台共用的 `POSTGRES_*`/`REDIS_*` 變數名稱（module 03/02 的統一 `.env` 定義），沒有另外的前綴。容器內部實際收到的環境變數名稱（app 讀取的名稱）見下表「容器內變數」欄。

| .env 變數（主機端） | 容器內變數 | 用途 |
|---|---|---|
| `OLLAMA_URL` | `OLLAMA_URL` | 所有 guard/proxy/signal-extraction 模型所在的 Ollama 伺服器位址 |
| `GUARDRAIL_REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | 呼叫 Ollama 的 HTTP 逾時秒數 |
| `LOG_LEVEL` | `LOG_LEVEL` | Python logging 等級 |
| `TIMEZONE` | `TZ` | log 時間戳記（含違規紀錄的 log）所用的時區，IANA 名稱，預設 `Asia/Taipei`。啟動時寫入 `os.environ["TZ"]` 並呼叫 `time.tzset()`；app 讀取 `TIMEZONE`，沒有時改讀 `TZ`。名稱無效時記錄 WARNING 並改用 `Asia/Taipei` |
| `PORT_GUARDRAIL_SERVICE` | — | app 容器發布的主機連接埠，只綁定在 `INTERNAL_BIND_IP`（根目錄 `.env`，預設 `127.0.0.1`；容器內部固定監聽 `8026`） |
| `DOCKER_NETWORK` | — | 此服務加入的外部 Docker network 名稱 |
| `PROXY_MODEL` | `PROXY_MODEL` | confidence-based proxy（`/api/generate`）所使用的模型 |
| `PROXY_MODE` | `PROXY_MODE` | confidence-based proxy 的模式：`monitor`（僅記錄）或 `enforce`（實際阻擋） |
| `PROXY_CONFIDENCE_THRESHOLD` | `PROXY_CONFIDENCE_THRESHOLD` | 0.00–1.00 — confidence-based proxy 中，信心分數超過此門檻即視為違規 |
| `DEFAULT_MODEL` | `DEFAULT_MODEL` | 保留/舊有欄位 — 目前沒有任何 router 或 service 實際讀取 |
| `PROXY_UPSTREAM_URL` | `PROXY_UPSTREAM_URL` | OpenAI 相容 proxy（`/v1/*`）檢查通過後轉送的上游，預設 module 07 `http://raptor-ai-lifecycle-api:8010` |
| `PROXY_UPSTREAM_TIMEOUT` | `PROXY_UPSTREAM_TIMEOUT` | 呼叫上游的 HTTP 逾時秒數（預設 `120.0`） |
| `GUARD_MODEL` | `GUARD_MODEL` | 主要 guard 模型 |
| `GUARD_MODEL_2`、`GUARD_MODEL_3` | `GUARD_MODEL_2`、`GUARD_MODEL_3` | 額外的 guard 模型 — 設定任一個即可啟用多模型模式（留空表示停用） |
| `GUARD_CONFLICT_STRATEGY` | `GUARD_CONFLICT_STRATEGY` | 多模型意見不一致時的判定方式：`strict`（任一判 unsafe 即 unsafe，預設）/ `majority`（過半）/ `permissive`（全部判 unsafe 才算） |
| `POSTGRES_HOST` / `_PORT` / `_USER` / `_PASSWORD`,`POSTGRES_DB` 固定寫死 `guardrails` | 同左 | module 03 共用 Postgres 的連線資訊,由統一 `.env` 提供 |
| `REDIS_HOST` / `_PORT` / `_PASSWORD`,`REDIS_DB` 固定寫死 `0` | 同左 | module 02 共用 Redis 的連線資訊,由統一 `.env` 提供 |
| `GR_DEFAULT_ENABLED` | `GR_DEFAULT_ENABLED` | 當 Redis 中的 `guardrail:enabled` 鍵值尚不存在時，冷啟動所寫入的預設值 |

## API 參考

### `/guard/check/*` — 原始 guard-model 分類（deprecated）

> **Deprecated（GRD-X1）**：這組端點只使用固定的 guard prompt，**不會套用啟用中的 policy**。
> - 請改呼叫 `POST /guardrail/policy/decide`：請求加上 `direction`，回應改為 `decision`（`allow | redact | warn | block`），不再是 `safe` 與 `categories`，呼叫端要改依 `decision` 處理。
> - decide 只接受單一訊息，所以 `conversation` 改用格式相同的 `/api/v0.2/check/conversation`。
> - `input`、`output`、`conversation` 的回應會帶 `X-Deprecated: true`，以及 `Link: <successor>; rel="successor-version"` header，OpenAPI 中也標記為 deprecated。
> - 預計在 v1.0 移除。
>
> | 舊 | 新 |
> |---|---|
> | `POST /guard/check/input` | `POST /guardrail/policy/decide`，`"direction": "input"` |
> | `POST /guard/check/output` | `POST /guardrail/policy/decide`，`"direction": "output"` |
> | `POST /guard/check/conversation` | `POST /api/v0.2/check/conversation` |
>
> `POST /guard/check/raw` 是除錯用的端點，沒有對應的 policy-aware 版本，因此不標記 deprecated。

| 端點 | 用途 |
|---|---|
| `POST /guard/check/input` | 檢查使用者輸入（role 固定為 `user`） |
| `POST /guard/check/output` | 檢查 AI 輸出（role 固定為 `assistant`） |
| `POST /guard/check/conversation` | 檢查多輪對話中的最後一輪 |
| `POST /guard/check/raw` | 呼叫所有啟用中的 guard 模型，回傳每個模型未經解析的原始輸出 — 供除錯/比較用 |

會將所有啟用中的模型結果合併為單一判定；若各模型結果不一致，回應中會包含 `conflict: true` 以及逐模型的 `results` 明細。

### `/policy/check/llm/*` 與 `/debug/policy/check/llm/*` — LLM-judged policy check

| 端點 | 用途 |
|---|---|
| `POST /policy/check/llm/input` | 每個啟用中的 guard 模型依照自己的 policy prompt 判斷使用者輸入 |
| `POST /policy/check/llm/output` | 同上，用於 AI 輸出 |
| `POST /policy/check/llm/conversation` | 同上，用於多輪對話的最後一輪 |
| `POST /debug/policy/check/llm/{input,output,conversation}` | 與上述三者完全相同，但回應中額外附上實際送給每個模型的完整 `prompt` |

回傳一份陣列，每個啟用中模型各佔一筆（絕不合併，因為不同模型可能是依照不同內容被判斷的）。這是理解 **Standard Policy Format** 的端點群組 — 詳見下方深入說明章節。

### `/api/v0.2/check/*` 與 `/api/v0.2/debug/check/*` — policy-aware check（`CheckResponse` 格式）

| 端點 | 用途 |
|---|---|
| `POST /api/v0.2/check/input` | 檢查使用者輸入 |
| `POST /api/v0.2/check/output` | 檢查 AI 輸出 |
| `POST /api/v0.2/check/conversation` | 檢查多輪對話 |
| `POST /api/v0.2/debug/check/{input,output,conversation}` | 與上述三者完全相同，但回應中額外附上實際送給每個模型的 `prompt` |

請求/回應格式與 `/guard/check/*` 相同（`CheckResponse` / debug 版為 `DebugCheckResponse`）。呼叫端要整合的是 `POST /guardrail/policy/decide`（見下一節）；本組端點保留給需要 `CheckResponse` 格式的情境，`/api/v0.2/check/conversation` 則是 `/guard/check/conversation` 的替代端點。

判定邏輯在 `app/services/policy_check_service.py`，與 `POST /guardrail/policy/decide` 共用；`app/routers/check_v0_2.py` 只負責把結果組成 `CheckResponse`。每次請求的流程如下：

1. 全域開關（`guardrail:enabled`）關閉時直接回 `safe=true`；policy 開關（`guardrail:policy_enabled`）只決定第 3 步走哪個分支，不會讓端點回 disabled。
2. **regex 前處理**：以 `app/services/regex_detector.py` 預設的 jailbreak 句型比對，命中的片段由 `redactor` 換成 `[REDACTED]`（只記 log，不會因此直接判 unsafe），之後只有處理過的內容會送給模型。
3. **多模型判斷**：有啟用中 policy（且 policy 開關開啟）時，與 `/policy/check/llm/*` 相同 — 每個模型拿各自的 policy prompt；否則與 `/guard/check/*` 相同 — 使用固定 guard prompt。
4. **最終判定**：啟用中 policy 的內容若符合 detector checker 的 `GuardrailPolicy` 格式，由 `checker.evaluate()` 依其 `input_guardrail`/`output_guardrail` 規則重新判定 `safe`（沿用第 3 步的模型結果，不會再呼叫模型）；否則（含 Standard Policy Format 的 policy）沿用第 3 步多模型合併的結果。回應只覆寫 `safe`，`categories`/`raw` 仍為第 3 步的合併結果。
5. 最終判定為 unsafe 時寫入 `guardrail_violations`：
   - `module`、`request_id` 取自請求本身；受檢文字只以 SHA-256 存在對應的 `audit_events.content_hash`（見「稽核紀錄」）。
   - 沒有啟用中 policy 時，`policy_id` 為 `NULL`。
   - `/conversation` 的 `direction` 一律記為 `conversation`。

### `POST /guardrail/policy/decide` — canonical policy-aware decision（GRD-X1）

**用途**：對一段輸入或輸出給出單一判定 `allow | redact | warn | block`，是呼叫端的正式整合點，取代 `/guard/check/input` 與 `/guard/check/output`。Module 07、13、35、12 都呼叫本端點。

**與 `/api/v0.2/check/*` 的關係**：兩者跑同一套判定（`policy_check_service`），不會互相矛盾；差別只在結果的表示方式。

**請求**：

| 欄位 | 必填 | 說明 |
|---|---|---|
| `content` | ✓ | 要判定的文字 |
| `direction` | | `input`（預設，以 user 身分判定）或 `output`（以 assistant 身分判定） |
| `module` | | 呼叫端模組，寫入稽核紀錄 |
| `request_id` | | 請求 ID；未提供時自動產生 UUID |
| `user_id`、`session_id` | | 只用於稽核：`user_id` 以 SHA-256 存成 `actor_id_hash`；`session_id` 記在 `metadata.session_id` |
| `modality` | | CM-1 內容類型：`text` 或 `transcript`。提供時回應多一個 `moderation` 欄位（見下方） |

**回應**：

```json
{"decision": "block", "reason": "S1: Violent Crimes", "policy_version": "1.0",
 "request_id": "…", "latency_ms": 132.5, "redacted_content": null, "categories": ["S1"]}
```

- `policy_version`：套用的啟用中 policy 版本。沒有 policy，或 `guardrail:policy_enabled` 關閉時為 `null`。
- `redacted_content`：只有 `decision` 為 `redact` 時才有值。
- `categories`：guard model 回報的類別代碼（例如 `S7`），與 `/guard/check/*` 的 `categories` 相同。
  - 不受 policy 影響：policy 把某個類別設為 `allow` 時，`decision` 是 `allow`，但 `categories` 仍會列出該類別。Module 12 的 PII 掃描就是用它判斷是否命中 S7。
  - guard 判定 safe，或 guardrail 關閉時為 `[]`。
- `moderation`：只在請求有 `modality` 時出現；沒有 `modality` 時回應就是上面七個欄位，與先前相同。
  - 把 `categories` 的 S-code 對應到 CM-1 的 moderation 分類（`nsfw` / `violence` / `hate` / `self_harm`），套用該 modality 的 `decision`。格式與圖片 guard 的 `moderation` 相同（見「圖片 guard 的 `moderation` 回應」），只是沒有 `score` / `threshold`。
  - 使用的 moderation 區塊來自本次判定所用的同一個 policy；該 policy 沒有 moderation 區塊時用內建預設值（`source: default`）。
  - 只是附加資訊：`decision` 仍由 policy 的 `categories` 決定，兩者可能不同（例如 moderation 把 `violence` 設為 `warn`，但 S1 的 decision 是 `block`）。
  - guardrail 全域開關關閉時為 `null`（沒有做任何檢查）。

**流程**（與 v0.2 相同）：
1. regex 偵測。
2. 讀取啟用中的 policy。
3. guard model 判定。
4. 衝突處理與 rule engine。
5. 依下表決定動作。
6. 非阻塞寫入稽核紀錄。

**動作對應**：

| 情況 | decision |
|---|---|
| 全域開關關閉 | `allow`（`reason: guardrail disabled`），不檢查、不寫稽核 |
| `GuardrailPolicy`（checker rules）格式的 policy | rule engine 的動作：`pass→allow`、`flag→warn`、`redact→redact`、`block→block` |
| Standard Policy Format 的 policy，guard 判定 unsafe | 依各 S 代碼對應類別的 `decision`：`allow`、`warn`、`review→warn`（reason 會註明 review）、`block`，取最嚴重者；沒有任何對應類別時 `block` |
| 沒有 policy，guard 判定 unsafe | `block` |
| regex 命中，但沒有更嚴重的判定 | `redact`（`redacted_content` 為遮蔽後的內容，`reason: regex: <pattern>`） |
| 其他 | `allow`（`reason: no violation`） |

**稽核**：`allow` 不寫；`warn`、`redact`、`block` 各寫一筆 `audit_events` 與 `guardrail_violations`（見「稽核紀錄」），`action_taken` 為 decision；`warn` 在 `audit_events` 中記為 `outcome=allow`、`metadata.decision="warn"`。

**錯誤**：判定本身失敗時（guard model、Redis、Postgres 無法使用）回 **503**，並記錄 error log：

```json
{"detail": {"type": "guardrail_unavailable", "message": "…"}}
```

```bash
curl -s -X POST http://raptor-guardrail-service:8026/guardrail/policy/decide \
  -H 'Content-Type: application/json' \
  -d '{"content":"hello","direction":"input","module":"07"}'
```

### `GET /guardrail/coverage` — coverage matrix（最小版本）

列出本服務所有內容檢查端點，每筆包含：
- 是否套用啟用中的 policy（`policy_aware`）。
- 是否 deprecated，以及替代端點（`successor`）。
- 對應的 Jira story。

清單定義在 `app/routers/coverage.py` 的 `ENFORCEMENT_POINTS`，之後會再補上各呼叫端的資訊。

### `/guardrail/*` — detector checker、policy 管理、違規紀錄、系統開關

| 端點 | 用途 |
|---|---|
| `POST /guardrail/policy/decide` | canonical policy-aware decision（GRD-X1），見上一節 |
| `GET /guardrail/coverage` | coverage matrix，見上一節 |
| `POST /guardrail/check/input`、`/guardrail/check/output` | 依照當前啟用 policy 的 `input_guardrail`/`output_guardrail` 規則，進行 regex + Llama-Guard detector 檢查 |
| `POST /guardrail/prompt_test` | 除錯工具 — 直接將 prompt 送給指定的 Ollama 模型，不涉及任何 guard prompt 或 policy |
| `GET /guardrail/violations` | 分頁列出違規稽核紀錄（可依 module/direction/category/時間範圍篩選） |
| `GET /guardrail/violations/summary` | 過去 24 小時內依 category + action 統計的違規次數 |
| `POST /guardrail/policies` / `POST /guardrail/policies/upload` | 建立新的 policy 版本（原始 body 或上傳檔案）；`target=original` 且內容為 JSON 陣列時，會以 Standard Policy Format 的 JSON Schema 嚴格驗證，不符則回傳 `422` |
| `PUT /guardrail/{policy_id}` | 部分更新既有 policy 的內容 |
| `GET /guardrail/policies/template?format=checker_rules\|standard` | 建議的 policy 範例格式：`checker_rules`（預設）為既有的 YAML/JSON 規則範例；`standard` 回傳 13 類別（S1–S13）的 Standard Policy Format JSON 範例 |
| `GET /guardrail/policies`、`GET /guardrail/policies/active`、`GET /guardrail/policies/{id}` | 列出全部 / 取得啟用中的 / 依 id 取得 |
| `PUT /guardrail/policies/{id}/activate`、`/deactivate` | 系統中同時只會有一份 policy 處於啟用狀態；停用 `default-standard-policy` 以外的 policy 時，會自動改為啟用 `default-standard-policy`（見下方「預設 policy」） |
| `DELETE /guardrail/policies/{id}` | 刪除（若該 policy 目前為啟用中，或為 `default-standard-policy`，則回傳 `409` 拒絕） |
| `POST /guardrail/system/enable`、`/disable` | 全域開關 — 以 Redis 儲存狀態，下一次請求即刻生效，不需重啟服務 |
| `GET /guardrail/system/status` | 目前開關狀態 + 啟用中 policy 的名稱/版本/格式（`active_policy_format`：`standard` / `checker_rules` / `null`）+ `conflict_strategy` + `proxy_mounted`（OpenAI 相容 proxy 是否已掛載） |

### `/moderation/flags/*` — 自動標記系統

模組 23 自己沒有 asset/moment/chunk 的身份概念，只分類被丟進來的 bytes；真正建立標記的呼叫端是模組 10/11/12,在自己的審核結果超過標記門檻時呼叫這裡。`GET`/`PATCH` 兩個端點是人工審核後台的後端,由模組 13 的 `/api/0.5/moderation/*` 代理(在那一層做 `moderation-admin` Keycloak 角色檢查,跟 `13-api-services/app/routers/gdpr.py` 的 `_is_gdpr_admin` 同一套模式),這裡本身信任呼叫端,跟本服務其他跨模組呼叫一致。

| 端點 | 用途 |
|---|---|
| `POST /moderation/flags` | 建立一筆標記(模組 10/11/12 呼叫) |
| `GET /moderation/flags` | 分頁列出標記,可依 `status`/`category`/`branch_id`/`version_id`/`asset_path`/`moment_id`/`chunk_id`/時間範圍篩選 |
| `GET /moderation/flags/{flag_id}` | 取得單筆標記,不存在回 `404` |
| `PATCH /moderation/flags/{flag_id}` | 更新單筆標記(人工審核後台用) |

### 稽核紀錄

每一筆稽核都在**同一個 transaction** 裡寫兩筆資料，以 `event_id` 關聯：

- **`audit_events`**：共用的 audit event（`common/raptor_audit`）。欄位如下：
  - `event_type`：固定為 `guardrail.decision`。
  - `source_module`：`23`。
  - `source_service`：`guardrail-service`。
  - `outcome`：`allow`、`block` 或 `redact`。`warn`、`flag` 這類沒有對應 outcome 的判定記為 `allow`，原始判定寫在 `metadata.decision`。
  - `resource_id`：`request_id`；沒有 `request_id` 時用 `session_id`。
  - `policy_version`。
  - `retention_days`：預設 `90`，由 `GUARDRAIL_AUDIT_RETENTION_DAYS` 設定（GRD-X4）。
  - `metadata`：`decision`、`direction`、`category`、`caller_module`、`policy_id`、`session_id`。
- **`guardrail_violations`**：`GET /guardrail/violations` 讀取的表，新增 `event_id` 欄位，API 回應中也會帶出。`event_id` 是指向 `audit_events` 的 foreign key（`ON DELETE SET NULL`，GRD-X4）：清掉 audit event 時，violation 那筆保留、只是不再關聯，兩張表仍可各自依保存期限清理。

**隱私最小化**：
- **不保存原文**：受檢文字（regex redact 之後送進模型的那份）只以 SHA-256 存在 `audit_events.content_hash`。`guardrail_violations` 沒有 `content` 欄位（GRD-X4）：要比對同一段文字，經 `event_id` 查 `audit_events.content_hash`。
- **服務日誌也不寫原文**：日誌只記受檢文字的長度與 SHA-256 前 12 碼（`len=42 sha=3f9a1c…`，見 `app/core/log_safe.py`），前 12 碼可對回 `audit_events.content_hash`。
- `user_id` 只以 SHA-256 存成 `actor_id_hash`。
- 舊資料：啟動時會刪除 `guardrail_violations.content` 欄位，之前留下的原文一併清除。
- `GET /guardrail/violations` 本來就不回傳 `content`，所以 API 回應不受影響（只多了 `event_id`）。

**寫入失敗**：維持 fail-open。寫入是背景執行的，失敗只記 warning，不影響判定結果，也不會讓請求失敗。

**資料表建立**：兩張表的 DDL 都在啟動時以可重複執行的方式套用（`app/db/schema.py`，`audit_events` 的 DDL 來自 `raptor_audit.AUDIT_EVENTS_DDL`）。

**尚未涵蓋（follow-up）**：
- Kafka `audit_event` topic（`acks=all`）。
- Module 33 的 retention engine 依 `retention_days` 清理 `audit_events`；目前它只清 `guardrail_violations`。
- Module 33／34 改用 `raptor_audit`。

### `/v1/chat/completions`、`/v1/models` — OpenAI 相容 guardrail proxy

**用途**：讓使用 OpenAI client 的服務（module 15 的 `ChatOpenAI`）不必改程式，只要把 `base_url` 改成 `http://raptor-guardrail-service:8026/v1`，LLM 流量就會先經過 guardrail 檢查，再到真正的 LLM。容器之間一律用 `8026`；從主機測試才用 `PORT_GUARDRAIL_SERVICE`（預設 `8023`）。Swagger UI（`/docs`）上有完整的 body schema 與範例。

#### `POST /v1/chat/completions`

**接收內容**：標準的 OpenAI chat completions request body（`Content-Type: application/json`）。

| 欄位 | 必填 | 說明 |
|---|---|---|
| `model` | ✓ | 上游的模型名稱，例如 `gemma4:12b` |
| `messages[]` | ✓ | 對話訊息。每則有 `role`（`system` / `user` / `assistant` / `tool`）與 `content`（字串，或 `{"type":"text"}` / `{"type":"image_url"}` parts 陣列）；工具呼叫的回合另有 `tool_calls` / `tool_call_id` |
| `temperature`、`max_tokens`、`tools`、`tool_choice` | | 一般 OpenAI 參數，原樣轉送 |
| `stream` | | 只能是 `false` 或省略；`true` 會被拒絕（`400 unsupported`），因為輸出檢查需要完整的回答 |
| 其他任何欄位 | | **原樣轉送**，例如 module 15 的 `engine`、`num_ctx`、`think` |

| Header（選填） | 說明 |
|---|---|
| `X-Raptor-Module` | 呼叫端模組代號，寫入違規稽核紀錄的 `module`（預設 `openai-proxy`；module 15 會帶 `15`） |
| `X-Request-ID` | 寫入違規稽核紀錄的 `request_id` |

```json
{
  "model": "gemma4:12b",
  "messages": [
    {"role": "system", "content": "你是一個有幫助的 AI 助手。"},
    {"role": "user", "content": "今天天氣好嗎"}
  ],
  "temperature": 0.7,
  "engine": "ollama",
  "num_ctx": 262144,
  "think": false
}
```

**回傳內容**：檢查都通過時，就是上游自己的 chat completion response，**完全不修改**（status、body 原樣回傳）。被擋或出錯時回傳 OpenAI 格式的錯誤 body `{"error": {"message", "type", "code"}}`，見下表。

**內部流程**（`app/routers/proxy_openai.py`）：

1. **驗證 request**：body 不可為空、必須是 JSON 物件，`stream` 不可為 `true`；否則回 `400`。
2. **讀全域開關**：Redis 的 `guardrail:enabled`（`/guardrail/system/enable`、`/disable` 控制）。關閉時跳過所有檢查，直接做第 6 步並原樣回傳；讀不到 Redis 時回 `503 guardrail_unavailable`。
3. **選出要檢查的輸入文字**：`messages[-1].content`（list 形式只取 `type=text` 的部分）。`messages[-1]` 是 `role=tool`（工具輸出，例如 module 15 的搜尋結果）時，改取最後一則 `role=user` 的訊息 — 檢索到的文件不是使用者輸入。內容為空（例如只有 `tool_calls` 的回合）就跳過輸入檢查。
4. **檢查**：先跑 guard 模型（與 `/guard/check/*` 相同的分類器，回傳 S1–S14 類別），再用同一個 guard 結果跑啟用中 policy 的規則（不會重複呼叫模型）。guard 判 unsafe，或 policy 規則為 `block` → 擋下。檢查本身失敗（guard 模型連不上、Redis／Postgres 錯誤等）→ 記 error log，回 `503 guardrail_unavailable`。
5. **稽核**：被擋（或輸出被 redact）時呼叫 `audit_log.record_violation()` 寫入 `guardrail_violations`（非同步，不拖慢回應），可用 `GET /guardrail/violations` 查詢。
6. **轉送**：把**原始 request body 的 bytes** 原封不動 POST 到 `{PROXY_UPSTREAM_URL}/v1/chat/completions`（逾時 `PROXY_UPSTREAM_TIMEOUT`）。連不上或逾時 → `502 upstream_unavailable`；上游回非 2xx → 原樣透傳。
7. **檢查輸出**：取 `choices[0].message.content`，用第 4 步同樣的方式檢查（`role=assistant`）。擋下 → `400 policy_violation`；policy 規則為 `redact` → 只把這個欄位換成遮蔽後的文字，其他欄位不動；通過 → 原樣回傳上游 response。內容為空（只有 `tool_calls`）就跳過。

| 情況 | HTTP | `error.type` |
|---|---|---|
| 輸入或輸出違規（guard 判 unsafe，或 policy 規則為 `block`） | `400` | `policy_violation`，`message` 為 `Policy violation: S1`（多個類別以逗號串接） |
| 輸出命中 policy 的 `redact` 規則 | `200` | —（只替換 `choices[0].message.content`） |
| guardrail 自身檢查失敗（guard 模型、Redis 等無法產生判定，會記 error log） | `503` | `guardrail_unavailable` |
| 上游連不上或逾時 | `502` | `upstream_unavailable` |
| 上游回傳非 2xx | 同上游 | 原樣透傳上游 body |
| `stream: true`（輸出檢查需要完整文字，不支援串流） | `400` | `unsupported` |
| body 為空、不是 JSON 或不是 JSON 物件 | `400` | `invalid_request` |

guardrail 失效時要不要改為直連上游，由呼叫端決定（module 15 的 `GUARDRAIL_FALLBACK_PASSTHROUGH`）；本服務只負責以 `503 guardrail_unavailable` 明確回報。`policy_violation` 不應觸發任何 fallback。

> **呼叫端須知**：第 3 步只檢查 `messages[-1]`（或工具回合中最後一則 `role=user`），所以呼叫端應把本輪使用者訊息放在最後、system prompt 放在最前面。
> - module 15 送出前會這樣排序：`[system, …history, 本輪 user]`。
> - 如果把含有對話記憶的 system prompt 放在最後，被檢查的就會是整段 prompt。這樣一來，記憶裡被擋過的問題會讓之後的回合也被擋。

#### `GET /v1/models`

**用途**：LangChain / OpenAI client 會呼叫它列出可用模型。**接收內容**：無。**內部流程**：直接轉送到 `{PROXY_UPSTREAM_URL}/v1/models`，原樣回傳上游的 status 與 body，不做任何檢查；上游連不上回 `502 upstream_unavailable`。

```bash
curl -s http://localhost:8023/v1/models
curl -s -X POST http://localhost:8023/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:12b","messages":[{"role":"user","content":"hello"}],"engine":"ollama"}'
curl -s http://localhost:8023/guardrail/system/status   # → "proxy_mounted": true
```

### `POST /api/generate` — confidence-based proxy（舊版）

輸入 `{"prompt": "..."}`，由單一分類模型回傳 `{allowed, category, confidence, reason}`。在 `PROXY_MODE=enforce` 時會實際阻擋，`monitor` 模式則僅記錄。

### `/infer/*` — 圖片安全分類（直接 GPU 推理）

這兩個端點直接載入 HuggingFace 模型（透過 `transformers` 套件），**不經 Ollama**，需要 NVIDIA GPU。Docker image 內的 `torch`／`torchvision` 是從 PyTorch 的 CUDA 12.6 index 安裝（版本與 `requirements.txt` 相同），主機的 NVIDIA driver 必須是 525 以上。PyPI 預設的 torch 2.14 是 CUDA 13 版，需要 driver 580 以上；在 image 外直接 `pip install -r requirements.txt` 時要注意。

| 端點 | 模型 | 用途 |
|---|---|---|
| `POST /infer/Llama-Guard-3-11B-Vision` | `meta-llama/Llama-Guard-3-11B-Vision` | 圖片安全分類，可選附帶文字 prompt 一起判斷 |
| `POST /infer/google-shieldgemma-2` | `google/shieldgemma-2-4b-it` | 圖片安全分類，同時對三項內建 policy 評分 |

**`POST /infer/Llama-Guard-3-11B-Vision`**

請求（multipart/form-data）：

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `image` | 檔案 | 必填 | 圖片（JPEG / PNG / WEBP 等） |
| `prompt` | 文字 | 選填 | 圖片附帶的使用者文字；省略時預設為 `"Check this image for unsafe content."` |
| `modality` | 文字 | 選填 | `image` 或 `video_frame`（CM-1）。有帶時回應多一個 `moderation` 欄位，見〈內容審核分類〉 |

回應：

```json
{
  "label": "safe",
  "categories": [],
  "raw": "safe"
}
```

`label` 為 `"safe"` 或 `"unsafe"`；`categories` 為違規類別代碼清單（S1–S13，安全時為空陣列）；`raw` 為模型原始輸出。

---

**`POST /infer/google-shieldgemma-2`**

請求（multipart/form-data）：

| 欄位 | 型別 | 必填 | 說明 |
|---|---|---|---|
| `image` | 檔案 | 必填 | 圖片（JPEG / PNG / WEBP 等） |
| `modality` | 文字 | 選填 | `image` 或 `video_frame`（CM-1）。有帶時回應多一個 `moderation` 欄位，見〈內容審核分類〉 |

回應：

```json
{
  "overall_verdict": "safe",
  "policies": {
    "dangerous_content": {
      "unsafe_probability": 0.03,
      "safe_probability": 0.97,
      "verdict": "safe",
      "policy_definition": "..."
    }
  }
}
```

`overall_verdict` 為 `"safe"` 或 `"unsafe"`（任一 policy 判 unsafe 即為 unsafe）；`policies` 包含三個內建 policy 各自的機率分數與判定結果。

**環境變數（`.env` 中設定）：**

| 變數 | 預設值 | 說明 |
|---|---|---|
| `HF_TOKEN` | —（必填） | HuggingFace access token。兩個模型都是 gated model，需在 huggingface.co 申請存取後填入 |
| `HF_CACHE_PATH` | `~/.cache/huggingface` | 主機端 HF 模型快取目錄，掛載至容器內 `/hf_cache`（與 module 10 共用同一份快取即可） |
| `LLAMA_GUARD_3_MODEL_PATH` | `meta-llama/Llama-Guard-3-11B-Vision` | HF repo ID 或本地路徑。首次執行自動下載並快取至 `HF_CACHE_PATH` |
| `SHIELDGEMMA2_MODEL_PATH` | `google/shieldgemma-2-4b-it` | HF repo ID 或本地路徑。首次執行自動下載並快取至 `HF_CACHE_PATH` |
| `GUARDRAIL_GPU_ID` | `0` | 分配給此服務的 GPU 編號（`docker-compose.yml` 的 device_ids）。若 GPU 0 已被其他服務佔用請改成空閒 GPU 編號 |
| `LLAMA_GUARD_LOAD_IN_4BIT` | `true` | `true`：NF4 4-bit 量化（實測 ≈ 6.85 GB，載入峰值 ≈ 7.75 GB）；`false`：bfloat16（≈ 22 GB，超出所有 ≤24 GB GPU 容量，請勿設 false 除非有 A100/H100） |
| `SHIELDGEMMA2_LOAD_IN_8BIT` | `true` | `true`：8-bit 量化（實測 ≈ 4.8 GB，預設）；`false`：bfloat16（≈ 8.2 GB）。兩種模式皆以 bfloat16 執行非量化模組並將 `lm_head` 綁回 embedding（否則分數來自隨機初始化的 `lm_head`） |
| `GUARDRAIL_CUDA_DEVICE` | `0` | 容器內可見 GPU 的索引（實體 GPU 由 `GUARDRAIL_GPU_ID` 決定） |
| `GUARDRAIL_ALLOW_MODEL_COEXIST` | `true` | 剩餘 VRAM 足夠時兩個模型同時常駐；`false`：切換模型前一律先卸載另一個 |
| `GUARDRAIL_VRAM_SAFETY_MARGIN_MB` | `256` | 每次載入／推論前，在模型估計值與推論預留之上額外保留的 VRAM |
| `GUARDRAIL_MODEL_ACQUIRE_TIMEOUT_S` | `300` | 等待 GPU（其他圖片推論／載入進行中）的上限，逾時回 503 |
| `LLAMA_GUARD_VRAM_ESTIMATE_MB` / `SHIELDGEMMA2_VRAM_ESTIMATE_MB` | `7000` / `4900` | 首次載入前使用的權重估計值（bf16 模式為 `21500` / `8300`）；載入後改用實測值。另有 `*_INFERENCE_RESERVE_MB`（`1024` / `1536`）與 `*_RAM_ESTIMATE_MB`（`4096`） |

> **啟動前準備（首次部署）：** 在執行 `docker compose up` 之前，先在**本機**執行下載腳本，將模型權重下載到 `HF_CACHE_PATH`：
> ```bash
> python3 scripts/preload_models.py
> ```
> 腳本會：(1) 若 `HF_CACHE_PATH` 不存在則自動建立並通知；(2) 逐一檢查兩個模型的 pinned snapshot 是否已在 `${HF_CACHE_PATH}/hub/` 中；(3) 缺少的才下載（Llama Guard 11B 首次約需 10–30 分鐘）。需先在 huggingface.co 申請 gated model 存取並在 `.env` 設定 `HF_TOKEN`。
>
> **兩層快取機制：**
>
> **1. HF 檔案快取（跨容器重啟持久化）**
> Docker volume 將主機的 `${HF_CACHE_PATH}` 掛載到容器內的 `/hf_cache`（`HF_HOME=/hf_cache`）。只要主機端目錄未被刪除，容器重啟或映像重建後不需重新下載。若 module 10 與 module 23 指向同一個 `HF_CACHE_PATH`，兩者共用已下載的權重。
>
> **2. 進程記憶體快取（跨請求持久化，由 Dynamic Model Resource Manager 管理）**
> `app/services/model_resource_manager.py` 是已載入模型的唯一持有者。模型首次被呼叫時，管理器即時探測該 GPU 剩餘 VRAM（含其他程序佔用，不假設 GPU 由本服務獨占）與系統 RAM：足夠就載入並常駐，後續請求直接重用；不足時等待進行中的推論結束，依 LRU 卸載閒置模型後重新探測，仍不足則回 **503**（附需求量／剩餘量），不會重試。同一張 GPU 同時只有一個請求在載入／卸載／推論（FIFO 排隊），因此不會重複載入、也不會卸載正在使用的模型。卸載只釋放程序內的模型／processor 引用並清理 CUDA cache，**不會刪除** `HF_CACHE_PATH` 中已下載的權重。
>
> GPU 需求與 Ollama-based 的 guard 模型分開，可同時共存於同一容器中。

## Policy Schema 比較 — 該用哪一種？

所有 policy 內容都以不透明文字的形式儲存在 `guardrail_policies` 資料表中（`raw_content`，外加每個 guard 模型各自的覆寫欄位）。服務在上傳時**不會**強制檢查格式 — 每個端點群組會各自嘗試以自己需要的格式去解讀這段文字（best-effort）。如果上傳的格式與想使用的端點不相符，上傳當下並不會出錯；而是等到實際呼叫該端點時才會解析失敗（或悄悄退回成把內容當成純文字 prompt 使用）。

| Schema | 格式 | 使用端點 | 結構 |
|---|---|---|---|
| **Standard Policy Format** | JSON 陣列 | `/policy/check/llm/*`、`/debug/policy/check/llm/*` | 一組 category 物件的陣列（`id`、`name`、`criteria` ...）— 與模型無關，會自動轉譯成每個 guard 模型各自的原生 prompt 格式 |
| **GuardrailPolicy checker rules** | YAML | `/guardrail/check/*` | 一個物件，內含 `input_guardrail`/`output_guardrail` 區塊，每條規則以 `category` + `detector`（`llama_guard` 或 `regex`）+ `action` 定義 |

這兩種 schema 結構並不相同，**彼此並不通用**。以下章節聚焦於 Standard Policy Format，因為它是設計上用來「撰寫一次、就能在每個 guard 模型上正確運作」的主要、可攜格式。

<!--
維護者附註：程式碼中還存在第三種 schema——Policy Engine 的 rules 格式（app/engine/models.py 的
PolicyRule，供 /policy/check/* 使用），本文件刻意不予記載。原因：/policy/check/* 要求當前啟用
policy 的內容能被解析成「頂層有 rules: 的 YAML/JSON 物件」，但 Standard Policy Format 是純 JSON
陣列，兩者結構互不相容。只要啟用中的 policy 是以 Standard Policy Format 撰寫（本文件教學的主要
格式），/policy/check/* 一律會解析失敗並回傳 409，因此在目前的部署方式下這組端點形同不可用，不
應被當作正式功能介紹。若日後兩者相容性有變化，請一併更新這份 README。
-->

## Standard Policy Format

Standard Policy Format 是一種可攜、與模型無關的方式，用來定義內容安全類別。透過 `POST /guardrail/policies?target=original` 上傳一份 JSON 陣列後，每個啟用中的 guard 模型都會自動從同一份來源產生自己格式正確的原生 prompt — 不需要為每個模型分別撰寫 prompt（不過仍可針對個別模型的用詞進行覆寫，詳見下方〈上傳並啟用一份 policy〉）。

### 格式

一份 Standard Policy 文件是一個 **JSON 陣列**（不是 YAML，也不是單一物件），陣列中每個元素定義一個類別：

```json
[
  {
    "id": "...",
    "name": "...",
    "description": "...",
    "severity": "low | medium | high | critical",
    "decision": "allow | warn | review | block",
    "criteria": ["...", "..."],
    "exceptions": ["...", "..."],
    "examples": {
      "violation": ["...", "..."],
      "allowed": ["...", "..."]
    }
  }
]
```

| 欄位 | 型別 | 是否必填 | 說明 |
|---|---|---|---|
| `id` | 字串 | 必填 | 類別代碼，例如 `"M1"`、`"F2"` — guard 模型判定違規時，會以此代碼回報 |
| `name` | 字串 | 必填 | 人類可讀的類別名稱 |
| `description` | 字串 | 必填 | 說明這個類別涵蓋的範圍 |
| `severity` | `low` \| `medium` \| `high` \| `critical` | 必填 | 固定列舉值 — 其他任何值都會驗證失敗 |
| `decision` | `allow` \| `warn` \| `review` \| `block` | 必填 | 固定列舉值，代表這個類別預期的處理方式（僅供參考；實際路由邏輯是依照各 guard 模型自身回傳的 safe/unsafe 判定 + category id 來動作，並非直接讀取這個欄位） |
| `criteria` | 字串陣列 | 必填 | 構成違規的條件 — 依模型不同會被轉譯成「Violation criteria:」/「Violates if:」/`## VIOLATES` |
| `exceptions` | 字串陣列 | 選填（預設 `[]`） | 明確**不算**違規的情況 — 會被轉譯成「Allowed:」/「Exceptions (do NOT flag):」/`## SAFE` |
| `examples.violation` | 字串陣列 | 選填（預設 `[]`） | 違規範例句子 |
| `examples.allowed` | 字串陣列 | 選填（預設 `[]`） | 非違規範例句子 |

### 範例 — 完整的一個類別

```json
[
  {
    "id": "M1",
    "name": "Medical Misinformation",
    "description": "Providing false, misleading, or scientifically unsupported medical information, including claiming the effectiveness of unverified treatments, providing incorrect disease information, or spreading health claims not supported by medical consensus.",
    "severity": "high",
    "decision": "block",
    "criteria": [
      "Claiming that scientifically unverified treatments can cure diseases",
      "Spreading false health information that contradicts current medical consensus",
      "Describing unverified medical effects with a definitive tone",
      "Claiming non-existent medical technologies or false efficacy",
      "Denying medical knowledge supported by established scientific evidence"
    ],
    "exceptions": [
      "Medical research, clinical trials, or academic discussions",
      "News reports or medical information fact-checking",
      "Discussing medical myths while explicitly pointing out their errors",
      "Fictional contexts such as novels, movies, and games",
      "General health education information"
    ],
    "examples": {
      "violation": [
        "Drinking bleach can cure COVID-19.",
        "Cancer does not need treatment; simply changing your diet will make it completely disappear.",
        "Vaccines will implant microchips to control human minds."
      ],
      "allowed": [
        "Currently, there is no scientific evidence supporting that bleach can cure viral infections.",
        "Cancer treatment methods need to be evaluated by a physician based on the disease type.",
        "Vaccine safety needs to be confirmed through clinical research."
      ]
    }
  }
]
```

一份實際的 policy 通常會在同一個陣列中定義多個類別（可參考 `policy_example/medical_policy.md` 的完整 5 類別醫療安全 policy，以及 `policy_example/financial_policy.md` 的 6 類別金融安全 policy — 兩者皆可直接使用）。

### 最小範例（僅必填欄位）

```json
[
  {
    "id": "H1",
    "name": "Hate",
    "description": "Hateful or discriminatory content targeting a protected group.",
    "severity": "medium",
    "decision": "warn",
    "criteria": ["Targets a group based on a protected characteristic."]
  }
]
```

### 上傳並啟用一份 policy

```bash
# 1. 將 JSON 陣列以「original」共用內容的形式上傳
curl -X POST "http://localhost:8023/guardrail/policies?name=medical-safety&version=1.0&target=original" \
  -H "Content-Type: application/json" \
  --data-binary @policy_example/medical_policy.md   # （取出其中的 JSON 陣列本身）

# 2. 啟用它 — 會將先前啟用中的 policy 停用
curl -X PUT "http://localhost:8023/guardrail/policies/{id}/activate"

# 3. 之後任何呼叫 /policy/check/llm/* 或 /debug/policy/check/llm/* 都會套用這份 policy
curl -X POST http://localhost:8023/policy/check/llm/input \
  -H "Content-Type: application/json" \
  -d '{"content": "Drinking bleach can cure COVID-19."}'
```

`target` 參數也接受 `llama-guard`、`granite-guardian`、`gpt-oss-safeguard` — 以其中之一上傳時，會為該模型附加一份手動撰寫、逐字使用的 prompt 覆寫，其優先權高於 Standard Policy Format 自動轉譯的結果，且僅對該模型生效。以相同的 `name`+`version` 但不同的 `target` 重複上傳，會把內容累加到同一筆 policy 上，而不是建立新版本 — 這正是一份 policy 能同時累積「共用 original 內容」與「各模型覆寫內容」的方式（透過多次呼叫）。

### 各 guard 模型如何各自轉譯同一份 policy

每個啟用中的 guard 模型拿到的都是**相同**一份 `StandardPolicyCategory` 類別清單，但各自的 adapter 會將其轉換成該模型自己官方認可的 prompt 結構：

- **Llama Guard 3** — 每個類別各自產生一個區塊（`Description:` / `Violation criteria:` / `Allowed:` / `Examples of violations:` / `Examples of allowed:`），包裹在 Llama Guard 真正的 `<BEGIN UNSAFE CONTENT CATEGORIES>...<END UNSAFE CONTENT CATEGORIES>` 範本中；違規時會回報具體是哪個類別 id。
- **Granite Guardian** — 將所有類別合併成一段散文式的「criteria」區塊，附加在 Granite 自己的 `<guardian>...` 格式中；由於 Granite 的原生格式沒有逐類別的 id 欄位，因此只能回報單純的 safe/unsafe 分數，無法指出具體是哪個類別違規。
- **GPT-OSS-Safeguard** — 每個類別各自產生一個完整的 `# Policy {id}: {name}` 區段，依照 OpenAI 官方多重 policy cookbook 範本撰寫（`## INSTRUCTIONS` / `## DEFINITIONS` / `## VIOLATES` / `## SAFE` / `## EXAMPLES`）；違規時會回報具體是哪個類別 id。

### 啟用時預先轉譯（activation time）

Standard Policy Format 的 policy 在**啟用時**（`PUT /guardrail/policies/{id}/activate`）就由 `app/services/policy_translator.py` 依 `app/prompt_templates/*.jinja2` 轉譯成三個 guard 模型各自的 prompt，並以 `{"placement": ..., "template": ...}` 的形式寫入該 policy 的 `content_llama_guard` / `content_granite_guardian` / `content_gpt_oss_safeguard` 欄位，之後每次 check 直接使用，不再逐次重新組裝：

- 只會寫入**仍為空**的欄位 — 以 `target=llama-guard` 等手動上傳的覆寫內容永遠不會被覆蓋。
- Llama Guard 3 的 prompt 需要同時包含類別與實際對話，因此預先轉譯的範本仍保留 `check_role` / `conversation` 兩個變數，在 check 時再代入；類別文字本身以 `{% raw %}` 包住，不會被二次解析。
- Granite Guardian 的範本依 IBM 官方格式額外加上 `<no-think>` 指示行。
- 不是 Standard Policy Format 的內容（checker_rules YAML、純文字）不做轉譯，行為與以往相同。
- 注意：已寫入的轉譯結果不會在之後修改 `raw_content` 時自動更新；若修改了某份已啟用過的 Standard policy 的內容，需先清空（`PUT /guardrail/{policy_id}`）對應的 `content_*` 欄位，再重新啟用。

### 預設 policy（`default-standard-policy`）

服務保證任何時候都有一份啟用中的 policy：

- **啟動時**：若沒有任何啟用中的 policy，會自動啟用 `default-standard-policy` v1.0（內容為 `app/services/default_policy.py` 的 13 類別 S1–S13）；若該筆 policy 尚不存在則先建立。已存在時直接重新啟用，不會覆寫管理者修改過的內容。若這一步失敗且 `GR_DEFAULT_ENABLED=true`，會記錄 WARNING：`Guardrail enabled but no active policy configured — run seed_default_policy.py`，服務照常啟動（見「啟動檢查與健康檢查」）。
- **停用時**：停用其他 policy 後會自動改為啟用 `default-standard-policy`；只有停用 `default-standard-policy` 本身，才會刻意留下「沒有啟用中 policy」的狀態（直到下次重啟）。
- **刪除**：`default-standard-policy`（任何版本）不能刪除，回傳 `409`。
- **手動 seed**：`scripts/seed_default_policy.py` 會把內建的 13 類別內容 POST 到執行中的服務並啟用（可重複執行，會覆寫該 policy 的 `raw_content`）。此腳本需在 host 上執行（Docker image 只包含 `app/`），需要 `httpx` 與本模組的 `app` 套件：

```bash
cd deployment/modules/23-guardrail-service
python scripts/seed_default_policy.py --url http://localhost:8023
```

### 解析行為

`parse_standard_policies()` 是一個 **best-effort、絕不拋出例外**的解析器：只要內容不是合法 JSON、不是非空陣列，或陣列中任一元素驗證失敗（缺少必填欄位、`severity`/`decision` 值不合法），就會回傳 `None` 而非拋出錯誤。呼叫端會將 `None` 視為「不是 Standard Policy Format」，進而退回成把內容當成一般文字/YAML system prompt 逐字使用 — 因此既有的純文字 policy 仍能維持原本行為不受影響，只有真正通過驗證、符合此格式的內容，才會觸發上述的逐模型轉譯。

## 內容審核分類（Moderation Taxonomy，CM-1）

v0.5 的內容審核類別，以及每種內容（modality）各自的判定方式。定義在 Standard Policy 的 `moderation` 區塊裡，與一般 policy 一樣上傳、啟用、分版本。內建預設值在 `app/services/default_moderation.py`。

### 類別與對應

| 類別 | 內容 | Llama Guard 3（S-code） | ShieldGemma 2（policy） |
|---|---|---|---|
| `nsfw` | 色情、性暴力、任何涉及未成年人的性內容 | S12、S3、S4 | `sexual` |
| `violence` | 暴力、血腥、武器（含大規模殺傷性武器）、恐怖主義 | S1、S9 | `violence`、`dangerous` |
| `hate` | 仇恨言論（針對種族、宗教、性別、性傾向等受保護特徵） | S10 | —（ShieldGemma 2 沒有對應 policy） |
| `self_harm` | 鼓勵或教唆自殺、自殘、飲食失調 | S11 | `dangerous`（部分涵蓋：其定義含 "instructions for suicide"） |

- 不屬於審核類別的 S-code（S2、S5、S6、S7、S8、S13、S14）照舊交由 `/guardrail/policy/decide` 依各 S-code 的 `decision` 處理。
- `dangerous` 同時對應 `violence` 與 `self_harm`：一張 `dangerous` 分數高的圖會兩類都被標記。
- **已知缺口**：圖片的 `hate` 只能由 Llama Guard 3 Vision 判斷（ShieldGemma 2 判不了）；CM-2 實測 Llama Guard 3 Vision 用在入庫圖片並不可靠。判不了的類別會列在回應的 `not_covered`，不會被當成「安全」。

### 各 modality 的判定方式

| Modality | 用的模型 | 設定 | 預設值 |
|---|---|---|---|
| `text` | Llama Guard 3（文字） | `decision` | `block` |
| `transcript` | Llama Guard 3（文字） | `decision` | `block` |
| `image` | ShieldGemma 2（或 Llama Guard 3 Vision） | `decision` + `threshold`（0–1） | `block`，0.5 |
| `video_frame` | ShieldGemma 2（或 Llama Guard 3 Vision） | `decision` + `threshold`（0–1） | `block`，0.5 |

- ShieldGemma 2 每個 policy 給一個機率：某類別的分數是它對應 policy 中最高的那個，**分數 ≥ threshold** 即標記。
- Llama Guard 3 只給 S-code、不給分數，所以 `text` / `transcript` 沒有 threshold；Llama Guard 3 Vision 也一樣，回報到對應 S-code 即標記。
- 預設 threshold 0.5 等同 CM-1 之前圖片 guard 的判定（`unsafe_probability > safe_probability`）。這是起始值，尚未以實際資料調校。
- `decision` 沿用 Standard Policy 的 `allow | warn | review | block`。多個類別被標記時取最嚴重者。入庫流程（CM-2 圖片、CM-3 影片）只記錄結果、不擋。
- 預設的 `text` decision 與預設 S1–S13 policy 一致，因此啟用內建的 moderation policy 不會改變文字判定結果（有 unit test 檢查）。
- 圖片 guard（`/infer/*`）帶 `modality=image|video_frame` 時套用 `image` / `video_frame` 的設定；`/guardrail/policy/decide` 帶 `modality=text|transcript` 時套用 `text` / `transcript` 的設定（回應的 `moderation` 欄位）。兩者都只附加判定結果，不改變原本的回應。

### Policy 格式（object form）

Standard Policy 除了原本的 JSON 陣列，也可以寫成物件，只允許 `categories` 與 `moderation` 兩個 key：

```json
{
  "categories": [ { "id": "S1", "...": "..." } ],
  "moderation": {
    "version": "1.0",
    "categories": {
      "nsfw": {
        "description": "...",
        "llama_guard": ["S12", "S3", "S4"],
        "shieldgemma2": ["sexual"],
        "modalities": {
          "text":        {"decision": "block"},
          "transcript":  {"decision": "block"},
          "image":       {"decision": "block", "threshold": 0.5},
          "video_frame": {"decision": "block", "threshold": 0.5}
        }
      }
    }
  }
}
```

- `categories` 的用法與純陣列完全相同（逐模型 prompt 轉譯、`/guardrail/policy/decide`）。
- 上傳時驗證 `moderation`：S-code 必須是 S1–S14、ShieldGemma 2 policy 必須是 `dangerous` / `sexual` / `violence`、四個 modality 都要有、`image` / `video_frame` 必須有 0–1 的 `threshold`、`text` / `transcript` 不可有 threshold。不符合回 `422`。
- 物件若有其他 key，就不視為 Standard Policy（不會誤判 checker_rules 文件）。

### 上傳並啟用內建的 moderation policy

```bash
# 1. 取得內建內容（預設 S1–S13 + moderation 區塊）
curl -s "http://localhost:8023/guardrail/policies/template?format=moderation" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['example_json'])" > moderation_policy.json
# 2. 上傳（target 預設 original）並啟用
curl -s -X POST "http://localhost:8023/guardrail/policies?name=moderation-policy&version=1.0" \
  --data-binary @moderation_policy.json
curl -s -X PUT "http://localhost:8023/guardrail/policies/<id>/activate"
```

沒有啟用中的 moderation 區塊時（啟用的 policy 沒有這個區塊、policy 開關 `guardrail:policy_enabled` 關閉，或查詢 Redis/Postgres 失敗），圖片 guard 使用內建預設值，回應的 `source` 為 `default`。

### 圖片 guard 的 `moderation` 回應

`POST /infer/google-shieldgemma-2` 帶 `modality=video_frame`，原有欄位不變，另加：

```json
"moderation": {
  "modality": "video_frame",
  "decision": "block",
  "flagged": ["nsfw"],
  "not_covered": ["hate"],
  "categories": {
    "nsfw": {"covered": true, "flagged": true, "decision": "block",
             "score": 0.97, "threshold": 0.5, "matched": ["sexual"]},
    "hate": {"covered": false, "flagged": false, "decision": "allow",
             "score": null, "threshold": 0.5, "matched": []}
  },
  "source": "active_policy",
  "policy_name": "moderation-policy",
  "policy_version": "1.0",
  "moderation_version": "1.0"
}
```

`POST /infer/Llama-Guard-3-11B-Vision` 相同，但類別裡沒有 `score` / `threshold`，`matched` 是對應到的 S-code。不帶 `modality` 時兩個端點的回應與以往完全相同；`modality` 只接受 `image` / `video_frame`，其他值回 `422`。

## 測試方式

```bash
cd deployment/modules/23-guardrail-service
pip install -r requirements.txt
python -m pytest tests/unit/
```

`pytest.ini` 設定 `testpaths = tests/unit` 與 `asyncio_mode = auto`。

- 測試套件（`tests/unit/`）皆針對純函式與手寫的假物件（fake）撰寫，不需要真實的 Postgres、Redis 或 Ollama。
- policy-aware check 的共用假物件（開關、啟用中 policy、guard model、稽核寫入）在 `tests/unit/conftest.py` 的 `policy_env` fixture。
- CI 會對有 `tests/unit/` 目錄的模組自動執行 `python -m pytest tests/unit/`（見 `.github/workflows/ci.yml`）。

**Load test**（GRD-X1：100 並發、p95 ≤ 200 ms）：`tests/load/locustfile.py`。它需要實際運行中的服務與 guard model，**不在 CI 中執行**；執行方式見 `tests/load/README.md`。

## 專案目錄結構

```
23-guardrail-service/
├── app/
│   ├── main.py                # FastAPI app，啟動/關閉流程，掛載各 router
│   ├── adapters/               # Guard-model adapter 框架
│   │   └── models/             #   每個模型家族各自的 adapter（Llama Guard 3、Granite Guardian、GPT-OSS-Safeguard）
│   ├── core/                   # 設定（config.py）、Prometheus 指標
│   ├── db/                     # Postgres（schema.py、repo.py）+ Redis 單例連線
│   ├── engine/                 # signal.py、policy_engine.py、policy_store.py
│   ├── models/                 # Pydantic 請求/回應 schema（policy.py 含 GuardrailPolicy 與 Standard Policy Format 解析器）
│   ├── routers/                # HTTP 層 — 每個端點群組各一個檔案
│   ├── services/                # 邏輯協調：policy_check_service（v0.2 與 decide 共用的判定）、audit_sink（稽核）、
│   │                            #   guard_classifier、checker、audit_log、redactor、state、
│   │                            #   policy_translator（啟用時轉譯）、default_policy（13 類別預設 policy）
│   ├── prompt_templates/         # Standard Policy 轉譯用的 Jinja2 範本（各 guard 模型一份）
│   ├── guard_prompts/            # 實際執行時使用的固定 guard 模型 prompt
│   └── official_prompts/         # 各廠商官方 prompt 格式的參考副本
├── policy_example/               # 可直接使用的 Standard Policy Format 範例
│   ├── medical_policy.md
│   └── financial_policy.md
├── scripts/seed_default_policy.py # 手動 seed 並啟用 default-standard-policy
├── tests/unit/                   # pytest — 不需要真實外部服務（CI 執行）
├── tests/load/                   # locust load test — 需要運行中的服務（手動執行，不進 CI）
├── docker-compose.yml, Dockerfile, requirements.txt, pytest.ini, .env.example
```

## 參考連結

- [Llama Guard 4 — model card 與 prompt 格式](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama-guard-4/)
- [Granite Guardian — 模型文件](https://www.ibm.com/granite/docs/models/guardian)
- [GPT-OSS-Safeguard — OpenAI cookbook 指南](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide)
