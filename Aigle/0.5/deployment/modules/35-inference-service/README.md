# Raptor LLM Inference Service

模組 35 — 任務層級的文字推論 API（翻譯、摘要、補全、改寫），以及模型發現與各任務預設模型。
35 負責提示、任務選項、模型解析、可選 Guardrail 檢查與對外回應／錯誤契約；文字模型管理、runtime、
GPU 排程、token 計數與輸出上限留在 Module 07（契約見
[07 INFERENCE_TEXT_CONTRACT.md](../07-ai-ml-services/docs/INFERENCE_TEXT_CONTRACT.md)）。

健康檢查：`GET /health` → `{"status":"ok","module":35}`。互動式 API 文件：`GET /docs`。

## 快速開始

```bash
cd deployment/modules
python3 build.py 35            # 或：docker compose --env-file .env -f 35-inference-service/docker-compose.yml up -d
curl localhost:8012/api/0.5/inference/models -H "Authorization: Bearer $ACCESS_TOKEN"
curl -X POST localhost:8012/api/0.5/inference/text/translate \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H 'content-type: application/json' \
  -d '{"input":"今天天氣很好","options":{"target_language":"en"}}'
```

依賴：Module 02（Redis，預設模型）、Module 07（registry 與推論）、Ollama（07 使用的同一個 daemon）。
本服務屬內部服務，對外經 Module 13 的 `/api/0.5/inference` JWT／UMA 代理。
上述呼叫需要已啟動的 Gateway、有效的 ACCESS_TOKEN 與文字模型預設；35 不開 host port。

預設主機埠為 **8036 → 容器 8035**，避免與 Module 34（PR #172）的主機埠 8035 衝突。
既有部署若指定 `PORT_INFERENCE_SERVICE=8035`，請選擇未占用的主機埠並同步更新 client／reverse-proxy upstream；
容器間仍使用 `raptor-inference-service:8035`。

## API

| 路由 | 說明 |
| --- | --- |
| `GET /models` | 合併 07 MLflow registry 與 Ollama 的模型清單（60 秒快取） |
| `GET /models/{name}` | 單一模型；名稱可含 `:`、`/`；不存在回 404 |
| `GET /models/default` | 各任務目前預設：`{"text","stt","tts","clone","diarize"}` |
| `POST /models/default` | `{"task","model"}` 寫入 Redis `inf:model:default:{task}`，回 204 |
| `POST /inference/text/translate` | 翻譯 |
| `POST /inference/text/summarize` | 摘要 |
| `POST /inference/text/complete` | 補全 |
| `POST /inference/text/transform` | 改寫 |
| `POST /inference/text` | 通用分派：`{"task": "<translate、summarize、complete 或 transform>", ...該任務欄位}` |
| `POST /inference/voice/stt` | 轉發到 Module 34（見下方「語音批次任務」） |
| `POST /inference/voice/tts` | 轉發到 Module 34（見下方「語音批次任務」） |

### 語音批次任務

`/inference/voice/stt`、`/inference/voice/tts` 本身不載入模型——轉發到 **Module 34**
（realtime-audio）已經為即時語音 session 載入好的 Whisper／TTS 模型，走它新增的
`POST /audio/transcribe`、`POST /audio/synthesize`（一次性、非 session，見 34 自己的
README）。這樣不用在 35 這邊多載一份模型，但也代表這條批次呼叫會跟 34 上正在進行的即時語音
session 搶同一組 GPU 並發槽位（34 自己的 `RT_STT_MAX_CONCURRENT`／`RT_TTS_MAX_CONCURRENT`），
批次量大時可能讓即時對話多一點延遲——目前流量小（主要是 E2E 測試）可以接受，之後批次量大時
再考慮另開一組槽位。

- `/inference/voice/stt`：`multipart/form-data`，欄位 `audio`（WAV/MP3/FLAC，沿用
  `read_audio()` 的限制，先在 35 這邊解碼正規化成 16 kHz mono WAV 再送到 34）、`language`
  （可選）。回傳 `{"text", "language", "language_probability", "confidence"}`。
- `/inference/voice/tts`：`multipart/form-data`，欄位 `text`、`language`（可選）。回傳
  `audio/wav`。
- Module 34 的這兩個新端點本身是對外公開的（跟 34 其他端點一樣沒有 gateway 路由，內部威脅模型
  文件裡的 B2 項有記錄），所以用 `X-Internal-Token`（跟 Module 33 既有的
  GDPR 內部呼叫共用同一組 `GD_INTERNAL_TOKEN`，不是新密鑰）擋掉非 35 的呼叫方。
- 35 會把已驗證過的 `x-user-id`（gateway 轉來的）當成 `X-User-Id` 一起轉給 34，34 用它
  查 `RT_MAX_SESSIONS_PER_USER` 配額——這批 batch 呼叫跟同一使用者的即時語音 session
  共用同一個每人配額，不是另開一套，超過時回 429（`too_many_sessions_for_user`）。

### 文字任務共用格式

請求：`{"model": "<選用>", "input": "<文字>", "options": {...}}`。未知欄位一律 400。

回應：

```json
{"task": "translate", "model": "gemma4:12b", "model_version": null, "output": "Good morning",
 "usage": {"prompt_tokens": 81, "completion_tokens": 3, "total_tokens": 84}, "latency_ms": 235}
```

- `model_version`：MLflow 模型為 registry 選定的版本；Ollama-only 模型為 `null`。
- `usage` 直接取自 07；07 無真實計數時為 `null`（不以 0 冒充）。摘要分段時為各次呼叫加總，任一段缺計數則為 `null`。
- `latency_ms`：35 收到請求到回應的總時間。
- 回應開頭的 `<think>…</think>` 區塊會移除。

| 任務 | options | 額外回應欄位 |
| --- | --- | --- |
| translate | `target_language`（BCP-47，必填）、`source_language`（選用，省略時自動偵測）、`formality`（`formal`/`casual`） | `target_language`；`source_language` 或 `detected_language` |
| summarize | `style`（`bullet`/`paragraph`/`headline`，預設 bullet）、`max_sentences`（≥1）、`language`（BCP-47，省略時保留來源語言） | `style`、`chunks` |
| complete | `system_prompt`、`max_tokens`（預設 1024）、`temperature`（0～2，預設 0.7）、`stop_sequences` | — |
| transform | `instruction`（必填）、`preserve_structure`（預設 false） | `preserve_structure=true` 時有 `structure` |

### 錯誤

錯誤 body 一律為扁平格式 `{"error": "<code>", "message": "...", ...}`。

| HTTP | error | 情況 |
| --- | --- | --- |
| 400 | `invalid_request` | 欄位缺漏／格式錯誤（含缺 `target_language`、`instruction`），`details` 列出欄位 |
| 400 | `unsupported_task` | `/inference/text` 的 task 不在四種之內 |
| 400 | `model_required` | 沒帶 `model` 且該任務沒有預設 |
| 400 | 07 的 code | 07 回 400（參數衝突、`policy_violation` 等），code 原樣帶出 |
| 422 | `model_not_found` | 模型不在 `GET /models` 或型別不符任務；帶 `model`、`available`（該任務可用的模型名） |
| 422 | `max_tokens_exceeded` | complete 的 `max_tokens` 超過模型上限；帶 `model_limit`、`requested`，在生成前拒絕 |
| 413 | `input_too_long` | 摘要輸入切段後超過 `SUMMARIZE_MAX_CHUNKS` |
| 503 | `model_limits_not_configured` | complete 使用的模型在 07 沒設定 `max_output_tokens` |
| 503 | `module07_unavailable` / `redis_unavailable` / 07 的 code | 07 或 Redis 無法連線；07 的 `resource_exhausted` 會帶 `Retry-After` |
| 504 | `upstream_timeout` | 等待 07 超過 `MODULE07_TIMEOUT_S`，或 07 回 `inference_timeout` |
| 502 | `upstream_error` | 07 其他錯誤（載入／執行失敗），`upstream_code` 保留 07 的 code |

## 規格細節

**模型清單**
- MLflow 來源改用 07 `GET /models/registered_in_mlflow?show_all=true`，不直連 MLflow REST
  （`registered-models/list` 在新版 MLflow 已不存在）。因此設定是 `MODULE07_API_URL`，沒有 `MODULE07_MLFLOW_URL`。
- 選版：每個註冊模型取最高 production 版本；沒有則取最高 staging 版本；兩者皆無則不列出。
  推論與 model-info 一律送出這個**明確版本**，清單顯示的就是實際執行的版本。
- 狀態：`loaded`（Ollama `/api/ps` 中有其 daemon tag，或 07 `/inference/loaded-models` 有該 HF 版本）>
  `staged`（選定版本為 staging）> `available`。`/api/tags` 只代表可用，不代表已載入。
- 型別：07 `task_family` `text-generation→llm`、`asr→stt`、`tts→tts`；其餘 family（vlm、embedding…）不列出。
  07 沒有 clone／diarize family，註冊版本可用 tag `inf_model_type=clone|diarize` 指定。Ollama 的 embedding 模型（名稱含 embed 或 bert 系 family）不列出。
- `size_gb`：tag `size_gb` > `ollama_model_size_gb` > Ollama `/api/tags` 大小 > `est_weights_vram_gb`；都沒有為 `null`。
- 去重只比名稱，同名時 MLflow 項目優先；註冊名與其 Ollama tag 不同時兩者都會列出（`runtime_model` 標示對應關係）。
- 07 無法連線（連線錯誤、逾時、非 2xx）時只列 Ollama 來源並記錄 warning，`sources.mlflow` 為 `unavailable`；不回 5xx。
  這只保證模型發現，07 停機時推論仍會回 503。
- 查無模型時，若快取已超過 5 秒會強制刷新一次，新註冊的模型不必等 TTL 到期。

**預設模型**
- Redis 為 Module 02（`raptor-redis-standalone`，DB 0）。解析順序：`request.model` > Redis > `DEFAULT_<TASK>_MODEL`。
- `POST /models/default` 與推論都會檢查模型存在且型別符合任務（text 任務需 `llm`）。
- Redis 讀取失敗時退回出廠預設並記錄 warning；寫入失敗回 503。
- 驗證是共用的 resolver（dependency），不是 HTTP middleware。

**翻譯**：來源語言省略時，提示要求模型第一行輸出 `DETECTED_LANGUAGE: <code>`，35 解析後移除；
模型沒照格式時 `detected_language` 為 `null`、全文作為譯文。

**摘要**
- headline 的「20 字」：CJK 每字算 1，其他以空白分隔的詞算 1，純標點不算；超過時截斷。
- `max_sentences`：bullet 限條列數；paragraph 限全文句數（段落上限 3）；headline 固定一句。以提示要求並在後處理裁切。
- 長文：超過 `SUMMARIZE_CHUNK_CHARS` 字元時依段落／行／句切段，各段先摘要再合併做最終摘要（同時最多 2 段並行）；
  超過 `SUMMARIZE_MAX_CHUNKS` 段回 413。

**補全**
- `max_tokens`、`temperature`、`stop_sequences` 以相同欄位名與值轉給 07（07 負責 runtime 映射與實際套用）。
- 上限檢查以 07 `GET /inference/model-info` 的 `limits.max_output_tokens` 為準（同一選定版本，快取 TTL 同清單）；
  上限未設定時回 503，不把未知上限當成超限。預設 `max_tokens=1024` 也會檢查，上限較小的模型需明確帶 `max_tokens`。

**改寫**：`preserve_structure` 是提示約束，不保證模型遵守；回應的 `structure` 回報輸入／輸出段落與標題數及是否一致，由呼叫端決定如何處理。

翻譯、摘要、改寫由 35 固定 `temperature=0.2`，並在 07 已知上限時以該上限為 `max_tokens`，避免長文被 runtime 預設（HF 512 tokens）截斷。

## 環境變數

統一 `.env` 使用 `INF_` 前綴，compose 轉成服務內的名稱：

| 服務內名稱 | 統一 .env | 預設 | 說明 |
| --- | --- | --- | --- |
| `MODULE07_API_URL` | `INF_MODULE07_API_URL` | `http://raptor-ai-lifecycle-api:8010` | 07 API（registry、model-info、推論） |
| `MODULE07_OLLAMA_HOST` | `INF_MODULE07_OLLAMA_HOST` | `OLLAMA_HOST` | 07 使用的 Ollama daemon；清單與降級用 |
| `MODULE07_TIMEOUT_S` | `INF_MODULE07_TIMEOUT_S` | 120 | 單次推論逾時（秒）→ 504 |
| `MODULE07_REGISTRY_TIMEOUT_S` | `INF_MODULE07_REGISTRY_TIMEOUT_S` | 5 | 清單、model-info 查詢逾時（秒） |
| `MODEL_CACHE_TTL_S` | `INF_MODEL_CACHE_TTL_S` | 60 | 清單與 model-info 快取（秒） |
| `MODULE34_API_URL` | `INF_MODULE34_API_URL` | `http://raptor-realtime-audio:8034` | `/inference/voice/stt`、`/tts` 轉發目標（Module 34 內部 port，不是對外的 `PORT_REALTIME_AUDIO`） |
| `MODULE34_TIMEOUT_S` | `INF_MODULE34_TIMEOUT_S` | 60 | 轉發 Module 34 的逾時（秒）→ 504 |
| `DEFAULT_{TEXT,STT,TTS,CLONE,DIARIZE}_MODEL` | `INF_DEFAULT_*_MODEL` | 空 | 出廠預設模型 |
| `REDIS_URL` 或 `REDIS_HOST/PORT/PASSWORD/DB` | 平台共用 | `raptor-redis-standalone:6379/0` | Module 02 Redis |
| `TRANSLATE_SYSTEM_PROMPT` | `INF_TRANSLATE_SYSTEM_PROMPT` | 內建 | 佔位符：`{target_language}` `{source_instruction}` `{formality_instruction}` `{output_instruction}` |
| `SUMMARIZE_SYSTEM_PROMPT` | `INF_SUMMARIZE_SYSTEM_PROMPT` | 內建 | `{style_instruction}` `{length_instruction}` `{language_instruction}`（依 style 產生） |
| `TRANSFORM_SYSTEM_PROMPT` | `INF_TRANSFORM_SYSTEM_PROMPT` | 內建 | `{instruction}` `{structure_instruction}` |
| `SUMMARIZE_CHUNK_CHARS` / `SUMMARIZE_MAX_CHUNKS` | `INF_SUMMARIZE_CHUNK_CHARS` / `INF_SUMMARIZE_MAX_CHUNKS` | 12000 / 8 | 長文分段摘要 |
| `GD_INTERNAL_TOKEN` | 平台共用 `GD_INTERNAL_TOKEN` | 空（拒絕所有呼叫） | 全帳號抹除：Module 33 呼叫 `DELETE /internal/gdpr/user/{user_id}` 時帶的 `X-Internal-Token`；查詢 consent 時也帶這個；35 呼叫 Module 34 的 `/audio/transcribe`、`/audio/synthesize` 時也帶這個（34 沒有 gateway 路由，直接公開給外部，靠這個 token 擋非 35 的呼叫） |
| `MODULE33_URL` | `INF_MODULE33_URL` | `http://raptor-privacy-compliance-service:8033` | `/voice/clone` 每次呼叫前向 Module 33 `GET /consent` 確認使用者有 `voice_clone` 同意；沒有回 403，查不到回 503 |

自訂範本若漏掉任務必需的佔位符（目標語言、偵測格式、風格、改寫指示等），35 會把該段附加在範本後面，不會靜默遺失。
範本以純字串取代，可含其他大括號。

## 測試

```bash
cd deployment/modules/35-inference-service
pip install -r requirements.txt pytest pytest-asyncio
python -m pytest tests/unit -v
```

單元測試（`tests/unit/`，CI 會自動納入）以 `httpx.MockTransport` 假造 07／Ollama、以 fakeredis 取代 Redis，
不需網路或 GPU。涵蓋：來源合併／去重／選版／狀態、07 斷線與逾時降級、TTL 快取、預設值與逐次覆寫、
422／400／413／503／504 映射、complete 參數原樣轉送與上限、摘要格式與分段、改寫結構回報、分派器。

**舊分支整合紀錄（2026-09-29，本期未重跑）**：35 本機執行，接 07 測試環境（`aiml-test`，07 修正分支，MLflow 3.16.1）、
Ollama 192.168.157.135 與臨時 Redis：

- `GET /models` 列出 MLflow 6 個（production/staging 選版正確）＋ Ollama 28 個，daemon 已載入的模型顯示 `loaded`；
  07 位址錯誤時 `sources.mlflow=unavailable`、只列 Ollama、HTTP 200。
- 翻譯（qwen3.5:9b、HF Qwen3-0.6B）：中英互譯、自動偵測 `detected_language`、`formality`。
- 摘要三種風格（qwen3.5:9b，輸出繁中）：bullet 3 條、paragraph 2 段、headline ≤20 字。
- 補全（HF Qwen3-0.6B，上限 256）：預設 1024 → 422 `max_tokens_exceeded`；`max_tokens=40`、`temperature=0`、
  `stop_sequences` 生效；Ollama-only 模型未設上限 → 503 `model_limits_not_configured`。
- 改寫（`/inference/text` 分派）：正式語氣改寫，標題與 3 段結構保留，`structure.preserved=true`。
- 預設模型：POST 204 寫入 Redis、覆寫不寫入 Redis、未知模型／型別不符 422、無效 task 400；Redis 斷線時 GET 退回出廠預設、POST 503。
- `MODULE07_TIMEOUT_S=0.3` 時回 504 `upstream_timeout`；07 無法連線時推論回 503 `module07_unavailable`。

Sprint 28 gateway/guardrail and voice contracts: [API.md](docs/API.md).
CPU text image is private behind Module 13; optional GPU voice build uses
`Dockerfile.voice` via `docker compose --profile voice`. See docs/API.md for gated
model build instructions and acceptance limits.

## 授權注意事項

`/voice/clone` 使用的 XTTS-v2 模型權重是 **Coqui Public Model License**，
僅限非商業用途（`coqui-tts` 套件本身是 MIT，不受影響）。因此這個端點**預設關閉**，
即使 `VOICE_ENABLED=true` 也一樣，必須另外設定 `CLONE_ENABLED=true`
（統一 `.env`：`INF_CLONE_ENABLED`）才會載入模型；未設定時呼叫會回
403 `clone_disabled`。
