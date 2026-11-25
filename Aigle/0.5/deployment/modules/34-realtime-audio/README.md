# Raptor Real-time Audio

模組 34 — 即時音訊串流。瀏覽器／客戶端以 WebSocket 傳送音訊，服務以一次性票證驗證、在 Redis
記錄工作階段，以 silero-VAD 即時切出語句，並以 faster-whisper 做串流語音辨識：說話中送出暫定／穩定逐字稿，
語句結束送出最終逐字稿。每句最終逐字稿是一輪 Module 15 對話，回答以可插拔的 TTS（預設 Kokoro）逐句合成，
以客戶端的編碼串流回傳。

客戶端整合（序號、續傳、插話與回音處理、JavaScript 與 gRPC 範例）見 `docs/CLIENT_GUIDE.md`。

## 流程

```text
client ──POST /audio/session/token (Bearer JWT)──▶ {"token": "<ticket>", "expires_in": 30}
client ──WS /audio/stream?token=<ticket>──▶ 驗證票證 → 檢查並行上限 → 消耗票證 → Upgrade
       ──CONFIG──▶ READY
       ──AUDIO (binary) …──▶ SPEECH_START / TRANSCRIPT_PROVISIONAL / TRANSCRIPT_STABLE（說話中）
                              UTTERANCE / TRANSCRIPT_PARTIAL / TRANSCRIPT_FINAL（語句結束）
                              RESPONSE_START / RESPONSE_TEXT / AUDIO (binary) … / RESPONSE_END（語音回答）
                              BARGE_IN + RESPONSE_END{interrupted}（回答中插話）
                              DISCARDED / STATE / PING（客戶端回 PONG）
       ──END──▶ ENDED，close 1000
（斷線）
client ──WS /audio/stream?token=<新票證>&session_id=<id>──▶ ──RESUME{session_id,last_seq}──▶
       重送 last_seq 之後的 frame → RESUMED → 工作階段繼續
```

## HTTP

| 路由 | 說明 |
| --- | --- |
| `GET /health` | `{"status":"ok","module":34}` |
| `POST /audio/session/token` | 需 `Authorization: Bearer <JWT>`。核發一次性票證，存於 Redis `rt:ws_ticket:{token}` = `{user_id, issued_at}`，TTL 30 秒 |
| `GET /tts/models` | 內部健康檢查：已註冊的 TTS 後端與狀態（`loaded`／`available`／`unavailable` 與原因）、`active`、使用中的聲音與聲音清單 |
| `GET /audio/status` | 目前並行工作階段數、上限、silero-VAD 版本與 STT 模型（`stt.model`／`device`／`compute_type`） |
| `POST /audio/transcribe` | 只供 Module 35（`X-Internal-Token` = `GD_INTERNAL_TOKEN`，同下面 GDPR 兩個端點；另需 `X-User-Id`，35 轉發它已驗證過的 JWT `sub`）：單次語音轉文字，不開 session、不跑 VAD/rolling decode。輸入 16 kHz mono PCM16 WAV（Module 35 已正規化好），回 `{text, language, language_probability, confidence}`。沿用本模組即時 session 同一組 `SttService`（含其並發名額）跟同一個 `RT_MAX_SESSIONS_PER_USER` 配額（`SessionStore.acquire_slot`/`release_slot`）——超過配額回 429，批次呼叫也會跟正在進行的即時語音搶 GPU 槽位 |
| `POST /audio/synthesize` | 同上權限與配額：單次文字轉語音，回整段 `audio/wav`（不是逐段串流）。沿用本模組即時 session 同一組 `TtsService`，並發/配額限制同上 |
| `GET /metrics` | Prometheus：佇列深度、GPU 名額、背壓計數（見「工作階段控制」） |
| `DELETE /internal/gdpr/user/{user_id}` | 只供 Module 33（`X-Internal-Token` = `GD_INTERNAL_TOKEN`）：刪除該使用者的錄音，回 `{erased_sessions, bytes_freed}` |
| `GET /internal/gdpr/user/{user_id}/recordings` | 只供 Module 33（同上 token）：GDPR 匯出用的錄音清單，回 `{recordings, pending_sessions}`，不含音檔與 fid |

JWT 以 RS256 驗證：預設從 `KEYCLOAK_URL/realms/<iss 中的 realm>/protocol/openid-connect/certs` 取 JWKS
（與 Module 13 相同做法），`user_id` 取 `sub`。設定 `RT_JWT_PUBLIC_KEY`（PEM）時改用固定公鑰。
缺少／錯誤／過期的 token 回 401（`missing_token` / `invalid_token` / `token_expired`），Keycloak 無法連線回 503。

## WebSocket `/audio/stream?token=<ticket>`

**Upgrade 前檢查**（以 HTTP 狀態拒絕，不會建立 WS）：

| 情況 | HTTP | body |
| --- | --- | --- |
| 票證缺少、過期或已使用 | 401 | `{"error":"invalid_ticket"}` |
| 超過 `RT_MAX_SESSIONS` | 503 | `{"error":"too_many_sessions","max_sessions":N}` |
| 超過 `RT_MAX_SESSIONS_PER_USER`（同一 JWT `sub`） | 503 | `{"error":"too_many_sessions_for_user","max_sessions_per_user":N}` |

順序是「確認票證存在 → 取得並行名額 → 以 `GETDEL` 原子消耗票證」：名額已滿時回 503 且**不消耗票證**，
客戶端可在 30 秒內重試；同一張票證同時被使用兩次時，只有一個連線會成功，另一個 401。
需要 ASGI denial-response 擴充（uvicorn 支援）；不支援的伺服器只能回 403。

**Client → server**

| frame | 內容 |
| --- | --- |
| text（第一個 frame） | `{"type":"CONFIG","codec":"pcm","sample_rate":16000,"language":"zh-TW","framing":"raw"}`（codec 可為 `pcm`、`opus`、`pcmu`、`pcma`；`framing` 可為 `raw`（預設）或 `seq`），`type` 可省略 |
| text（續傳時的第一個 frame） | `{"type":"RESUME","session_id":"…","last_seq":41}`，見「工作階段控制」 |
| text（外部大腦） | `{"type":"SPEAK","utterance_id":"…","text":"…","final":false}`、`{"type":"SPEAK_CANCEL","utterance_id":"…"}`，見「外部大腦」 |
| binary | AUDIO。`pcm`：mono、s16le，取樣率為 CONFIG 的 `sample_rate`，frame 可任意切（跨 frame 的半個 sample 會接上）。`opus`：每個 frame 一個原始 Opus packet（無 Ogg/WebM 容器，例如 WebCodecs `AudioEncoder` 的輸出）。`pcmu`／`pcma`：G.711 μ-law／A-law 8 kHz，每 byte 一個 sample，frame 大小不限（電話通常為 20 ms = 160 bytes） |
| text | `{"type":"PONG"}` 回應 `PING`（`RT_PONG_TIMEOUT_S` 內）；`{"type":"PING"}` 會得到 `PONG` |
| text | `{"type":"END"}` 或字串 `END` |

CONFIG 另可帶 `brain`（`module15` 預設／`external`）、`subject_id`、`record`、`index_transcripts`、`store_memory`，見「外部大腦與代理主體」。
`framing: "seq"` 時雙向的 binary frame 開頭都是 4-byte little-endian 序號；客戶端的 JSON frame 可帶 `seq`
（與 binary 共用一個計數）。伺服器以 `RESUMED.last_client_seq` 回報收到的最大序號，`seq` 模式下重複的音訊會丟掉。

CONFIG 規則：`pcm` 的 `sample_rate` 為 8000–48000 任意整數，`opus` 為編碼端取樣率（8000／12000／16000／24000／48000），
`pcmu`／`pcma` 必須是 8000；解碼後一律重取樣為 16 kHz mono int16 交給 VAD／STT。不支援的 codec（或
`RT_CODEC_RESAMPLE=false` 時 16 kHz `pcm` 以外的任何格式）回 ERROR `UNSUPPORTED_CODEC` 並 close 4400；
其他 CONFIG 錯誤仍為 `invalid_config`。`language` 為 BCP-47 時強制 STT 使用該語言，
`null` 為自動偵測，省略時使用 `RT_STT_LANGUAGE`（預設 `zh-TW`；設為 `auto` 則省略即自動偵測）。
不支援的語言回 ERROR `invalid_config` 並 close 4400。

**Server → client**（text JSON；每個事件都帶 `seq` 與 `session_id`。`seq` 是伺服器送出的所有 frame（含 binary）共用、
從 1 開始連續遞增的序號，`raw` 模式下客戶端也能自己數出 binary frame 的序號）

| type | 欄位 | 時機 |
| --- | --- | --- |
| `READY` | `session_id`、`state`、`config`（含 `brain`）、`vad`、`stt`（`model`、`language`、`streaming`、`stride_s`；停用 STT 時為 null）、`tts`、`output`、`barge_in`、`recording`、`subject_id`、`side_effects`、`expires_in`、`resume_window_s`、`ping_interval_s` | CONFIG 通過 |
| `STATE` | `state` | 狀態改變 |
| `SPEECH_START` | `start_ms` | 偵測到語音開始 |
| `UTTERANCE` | `index`、`start_ms`、`end_ms`、`duration_ms`、`speech_ms`、`reason` | 語句切出，已送入 STT 佇列 |
| `DISCARDED` | `start_ms`、`end_ms`、`speech_ms`、`reason`=`too_short`／`during_response` | 短於最短語句的聲音；或在回答播放中開始、未觸發插話的語句（多半是回音） |
| `TRANSCRIPT_PROVISIONAL` | `utterance`、`text`、`stable_text`、`unstable_text`、`audio_ms` | 說話中每 `RT_STT_STRIDE_S`；`text` 是目前完整假設，可能變動 |
| `TRANSCRIPT_STABLE` | `utterance`、`text`（新確認的部分）、`stable_text`、`word_index`、`words` | 連續兩次解碼一致的字詞，只送一次、不再變動 |
| `TRANSCRIPT_PARTIAL` | `utterance`、`text`、`confidence`、`word_index`、`start_ms`、`end_ms` | 語句結束後的最終解碼，逐詞送出 |
| `TRANSCRIPT_CORRECTION` | `utterance`、`start`、`end`、`old_text`、`new_text`、`stable_text` | 最終結果與已送出的穩定文字不符：`stable_text[start:end]` 改為 `new_text`（`stable_text` 為修正後全文） |
| `TRANSCRIPT_FINAL` | `utterance`、`text`、`language`、`confidence`、`duration_ms`、`latency_ms`、`first_word_ms` | 語句的最終逐字稿（可能為空字串）；`utterance` 對應 `UTTERANCE.index` |
| `RESPONSE_START` | `request_id`、`output`（`codec`、`sample_rate`、`frame_ms`） | 回答的第一個音訊幀之前 |
| `RESPONSE_TEXT` | `request_id`、`index`、`text` | 每一句要念的文字（在該句音訊之前） |
| （binary） | 回答音訊，格式見下方「語音回覆」 | `RESPONSE_START` 與 `RESPONSE_END` 之間 |
| `RESPONSE_END` | `request_id`、`duration_ms`（含結尾靜音）、`sentences`；插話時為 `interrupted: true`、`duration_ms` = 停止位置 | 最後一個音訊幀之後；插話時緊接 `BARGE_IN` |
| `BARGE_IN` | `request_id`、`tts_position_ms` | 使用者在回答播放中開口：在該位置停止播放並丟棄其餘已收到的回答音訊 |
| `GUARDRAIL_BLOCK` | `request_id`、`utterance`、`direction`（`input`／`output`）、`reason` | Module 23 擋下逐字稿（不問 Module 15，改念拒答語句）或回答中的一句（其餘回答改為拒答語句） |
| `PING` / `PONG` | `ts`（伺服器時間，ms） | 每 `RT_PING_INTERVAL_S` 送 `PING`；客戶端送 `PING` 時回 `PONG` |
| `RESUMED` | `session_id`、`replayed`、`gap`、`last_client_seq`、`state` | 續傳成功，排在重送的 frame 之後 |
| `WARNING` | `code`（`invalid_audio`／`unexpected_frame`／`QUEUE_OVERFLOW`／`stt_failed`／`response_failed`）、`message` | 不中斷串流的問題 |
| `ERROR` | `code`、`message`；`fatal: false` 時另有 `request_id` 或 `utterance` | 隨後關閉連線；`fatal: false`（`LLM_TIMEOUT`／`LLM_UNAVAILABLE`／`STT_OVERLOADED`）不關閉，工作階段繼續 |
| `ENDED` | `session_id`、`utterances`、`audio_ms`、`dropped_chunks`、`responses`、`barge_ins`、`resumes` | 處理完 END、回答播完之後 |

時間（`*_ms`）以該工作階段收到的音訊時間計算，從 0 開始，不是牆鐘時間。

**Close code**：1000 正常結束（END）；4400 CONFIG 無效或第一個 frame 不是 CONFIG／RESUME；
4408 等不到 CONFIG（`RT_CONFIG_TIMEOUT_S`）、工作階段達到 `RT_SESSION_TTL_S` 或沒有回 PONG；
4403 `BRAIN_MODE_FORBIDDEN`；4409 工作階段已由另一條連線 RESUME；4410 `SESSION_EXPIRED`（沒有可續傳的工作階段）；1011 內部錯誤。

## gRPC `AudioService.AudioChat`

給原生 gRPC 客戶端（行動裝置、嵌入式、CLI）的雙向串流。合約為 `proto/raptor/audio/v1/audio.proto`
（package `raptor.audio.v1`），由 Module 34 自己在 `RT_GRPC_PORT`（預設 8044，主機 `PORT_REALTIME_AUDIO_GRPC`）
提供，不經 Module 36 gateway。一次呼叫就是一個工作階段：`app/grpc_audio.py` 把呼叫轉成 WebSocket 的 frame
交給 `app/stream.py`，所以 VAD → STT → Module 23／15 → TTS、插話、`RT_MAX_SESSIONS` 名額、Redis 紀錄、
Module 25／26 與錄音都和 WebSocket 相同。

- **驗證**：call metadata `authorization: Bearer <JWT>`，interceptor 以與 `POST /audio/session/token`
  相同的方式驗證（Keycloak JWKS 或 `RT_JWT_PUBLIC_KEY`、`RT_JWT_AUDIENCE`、必須有 `sub`），不需要一次性票證。
  realm role `rt-delegate`／`rt-external-brain` 與票證一樣帶進工作階段；gRPC 沒有 `subject_id` 與 SPEAK，固定 `brain=module15`。
- **請求**：第一個 `AudioChunk` 決定 `codec`（空 = `pcm`）、`sample_rate`（0 = pcm 16000、opus 48000、G.711 8000）
  與 `language`（空 = `RT_STT_LANGUAGE`、`auto` = 自動偵測），之後的 chunk 只需 `payload`（重複相同值可以，改變 → `INVALID_ARGUMENT`）。
  `session_id` 是呼叫端自己的識別，只寫進 log。音訊依即時速度送（每 20–100 ms 一塊），佇列上限與丟棄規則同 WebSocket。
- **結束**：half-close（request stream 結束）= END，剩下的語句辨識、回答、播完後以 OK 結束；取消呼叫 = 斷線，
  工作階段立即結束。gRPC 沒有 RESUME 與應用層 PING（由 HTTP/2 keep-alive 處理）。

| WebSocket 事件 | `AudioEvent`（皆帶伺服器的 `session_id`） |
| --- | --- |
| `TRANSCRIPT_PROVISIONAL` | `transcript{is_final: false, text, utterance}` |
| `TRANSCRIPT_FINAL` | `transcript{is_final: true, text, confidence, utterance, language}` |
| 回答音訊（binary） | `audio{payload, codec, sample_rate, request_id, is_last: false}`（呼叫端的編碼；pcm 為 16 kHz） |
| `RESPONSE_END` | `audio{is_last: true, payload 空, request_id, interrupted}`，每個回答剛好一個 |
| `ERROR` | `error{code, message, fatal: true}`，隨後呼叫以非 OK 狀態結束 |
| `WARNING`、`fatal: false` 的 `ERROR`、`GUARDRAIL_BLOCK` | `error{fatal: false}`，工作階段繼續 |
| 其他（`READY`、`STATE`、`UTTERANCE`、`ENDED`…） | 不送 |

**狀態碼**：`UNAUTHENTICATED`（token 缺少、錯誤、過期或沒有 `sub`）、`UNAVAILABLE`（取不到 Keycloak JWKS）、
`RESOURCE_EXHAUSTED`（名額滿；`QUEUE_OVERFLOW`）、`INVALID_ARGUMENT`（codec／取樣率／語言錯誤或中途改變）、
`PERMISSION_DENIED`、`DEADLINE_EXCEEDED`（`RT_SESSION_TTL_S`）、`INTERNAL`。以非 OK 狀態結束前，伺服器先讀掉呼叫端
已送出的音訊，等到呼叫端停頓 50 ms、half-close，或 1 秒後緊接在一個訊息之後才送狀態：grpc.aio 客戶端若在寫入途中
收到狀態，會回報 `INTERNAL "Internal error from Core"` 而不是實際狀態。

連線未加密（與 HTTP 埠相同），對外需由 gateway／proxy 做 TLS。Python stub：`python scripts/gen_proto.py [輸出目錄]`
（需 `grpcio-tools`；Docker build 產生到 `/opt/realtime-audio/generated`，不提交）。客戶端範例見 `docs/CLIENT_GUIDE.md` §9。

## 工作階段

- Redis `rt:session:{session_id}`（hash）：`user_id`、`codec`、`sample_rate`、`language`、`framing`、`started_at`、`state`、
  `connection`（`connected`／`suspended`／`closed`）、`resume_until`、`recording_enabled`、`recording_skipped_reason`、
  `utterances`、`updated_at`；結束時加上 `end_reason`
  （`end`／`disconnect`／`disconnect_after_end`／`expired`／`session_expired`／`config_timeout`／`invalid_config`／`error`／`cancelled`）、
  `ended_at`、`audio_ms`、`dropped_chunks`。
- TTL 是 `RT_SESSION_TTL_S`，註冊時設定一次，代表**最長存活時間**：連線達到此時間會以 4408 關閉；
  結束後的紀錄保留到 TTL 到期，供查詢最終狀態。
- 狀態：`INIT → LISTENING → PROCESSING → RESPONDING → LISTENING`，任何狀態可到 `ENDED`（END、錯誤、存活時間到）
  或 `EXPIRED`（斷線後未在時限內續傳）。語句做最終辨識、等 Module 15 回答時為 `PROCESSING`；回答音訊在客戶端播放期間
  （第一個音訊幀送出到依播放時間估計播完）為 `RESPONDING`；播完時若下一輪已在準備則回 `PROCESSING`，否則 `LISTENING`；
  插話時立即 `RESPONDING → LISTENING`。沒有回答（空逐字稿、少於 `RT_MIN_UTTERANCE_CHARS` 個字）時直接回 `LISTENING`。
  回答期間音訊照常接收、照常辨識下一句；一次只進行一輪對話，後續的輪次依序排隊。
- 並行上限 `RT_MAX_SESSIONS` 是**全服務**的上限：以 Redis sorted set `rt:sessions:active` 記錄，
  共用同一 Redis 的多個 instance 一起計算。`RT_MAX_SESSIONS_PER_USER`（per-user quota）
  是**同一個使用者**（JWT `sub`，即 `rt:sessions:user:{user_id}` 這個獨立的 sorted set，
  跟代理主體模式下的 `subject_id` 無關——quota 管的是誰在開連線,不是這個 session 內容算誰的）
  的上限,兩個名額在同一個 Lua script 裡原子檢查,任一滿了都回 503,分別回報 `too_many_sessions`／
  `too_many_sessions_for_user`。每個連線每 `RT_HEARTBEAT_S`（10 秒）更新心跳（全域與該使用者兩個
  sorted set 一起更新）；instance 當機留下的名額在心跳逾期（3 倍間隔）後的下一次取得名額時回收。
- END：把 VAD 中尚未結束的語句送出（`reason=end`，仍需達最短長度），等待剩下的辨識與回答送完、播完，
  寫入最終狀態、送 `ENDED` 後關閉。這段期間仍處理 PING／PONG；客戶端在此之前離開則放棄剩下的回答
  （`end_reason=disconnect_after_end`，不可續傳）。
- 沒有 END 就斷線（或沒回 PONG）：工作階段暫停 `RT_SESSION_RESUME_WINDOW_S`（60 秒）等待續傳，見「工作階段控制」；
  設為 0 則立即結束（`end_reason=disconnect`）。

## VAD

- `app/vad.py` 在啟動時載入 silero-VAD（pip 套件 `silero-vad==6.2.3` 內附的 TorchScript 權重，不需下載；
  授權為 MIT）。預載 `RT_VAD_POOL_SIZE` 個實例，每個工作階段各用一個
  （silero 保有 RNN 狀態，不能共用），不足時再載入；VAD 在 CPU 上執行。
- **傳輸切塊與推論窗不同**：音訊先切成 `RT_VAD_CHUNK_MS`（30 ms = 480 samples）的塊排入佇列；silero 在 16 kHz
  只接受 512 samples（32 ms）的窗，Segmenter 會重新組窗。下列時間規則因此以 32 ms 為解析度。
- 規則：
  - 語音機率 ≥ `RT_VAD_THRESHOLD`（0.5）的窗算語音。開始與結束用同一門檻，沒有 silero `VADIterator` 的 −0.15 遲滯。
  - 語音後連續 ≥ `RT_VAD_MIN_SILENCE_MS`（300 ms，即 10 個窗）非語音 → 語句結束（`reason=silence`）。
  - 語音長度 < `RT_VAD_MIN_SPEECH_MS`（200 ms）→ 丟棄（`DISCARDED`）。
  - 緩衝到 `RT_MAX_UTTERANCE_S`（10 秒）→ 立即送出（`reason=max_length`）並清空；若仍在說話則開始下一句。
  - 語句前後各保留 `RT_VAD_SPEECH_PAD_MS`（96 ms）音訊，避免切掉字頭字尾。
- 非同步：接收迴圈只負責解碼與排隊，不等 VAD；VAD 在獨立 `asyncio.Task` 中批次取出佇列、在執行緒跑模型。
  佇列上限 `RT_AUDIO_QUEUE_MAX`（200 塊 ≈ 6 秒），滿了就丟棄**最舊**的音訊並回 `QUEUE_OVERFLOW` 警告（每秒至多一次），
  不會讓接收端卡住。
- 語句佇列（STT 佇列）上限 `RT_STT_QUEUE_MAX`（10；舊名 `RT_UTTERANCE_QUEUE_MAX` 仍可用），滿了丟棄新語句並回
  `ERROR {code: "STT_OVERLOADED", fatal: false}`。
- 送給 STT 的格式：16 kHz、mono、int16 PCM（`Utterance.pcm`），`Utterance.wav()` 可包成 WAV，
  直接給 Module 07 `POST /v1/audio/transcriptions`。除錯時設定 `RT_UTTERANCE_DUMP_DIR` 會把每句存成 WAV。

## 語音回覆

- **對話（`app/llm_bridge.py`）**：非空的 `TRANSCRIPT_FINAL` 以 `POST {RT_CHAT_SERVICE_URL}/chat`
  （預設 `http://raptor-chat-service:8021/api/v1/chat`）送
  `{"message", "session_id": 34 的 session_id, "user_id": JWT sub, "stream": true}` 與 `X-Request-ID`。同一個語音工作階段
  的每一輪都用同一個 `session_id`，Module 15 的 LangGraph 對話記憶因此延續。Module 15 以 SSE（`start`／`delta`／`end`／`error`）
  回答；舊版只回 JSON 時改用整段回答。log 以 `request_id` 記錄每個 `TRANSCRIPT_FINAL` 對應的 Module 15 請求（不記錄逐字稿內容）。
- **切句**：在 `。！？!?`、後接空白或結尾的 `.`、`…` 處切；不足 2 個字的片段併入下一句；超過 60 字仍無句號時在最後一個逗號切，
  讓語音早點開始。第一句一湊齊就開始合成，後面的句子邊送音訊邊合成。
- **逾時與失敗**：第一段回答文字須在 `RT_LLM_FIRST_TOKEN_TIMEOUT_S`（30 秒；Module 15 常需 8–29 秒）內到達，`RT_LLM_TIMEOUT_S`（30 秒）涵蓋整個回答
  （只計等待 Module 15 的時間；送 TTS 的句子佇列 `RT_TTS_QUEUE_MAX` 滿時不讀也不計時）；逾時送 `ERROR {code: "LLM_TIMEOUT", fatal: false}`，
  Module 15 無法連線或回錯誤送 `LLM_UNAVAILABLE`，兩者都改念 `RT_LLM_FALLBACK_TEXT`（「系統暫時無法回應，請稍後再試。」），
  工作階段繼續、下一句照常處理。
- **TTS 後端（`app/tts_backends/`）**：介面 `base.py`（`synthesize(text, voice, language, speed) → np.ndarray`、`stream()`、
  `list_voices()`、`name()`），`__init__.py` 為註冊表；派發器 `app/tts.py` 在啟動時依 `RT_TTS_MODEL` 載入一次。
  未知名稱或後端無法使用時服務不會啟動（訊息列出可用後端）。`RT_TTS_VOICE` 不是該後端的聲音時記警告、改用後端預設。
  新增後端只需在 `app/tts_backends/` 放模組並在 `__init__.py` 註冊名稱。

| 後端 | 授權 | 說明 |
| --- | --- | --- |
| `kokoro-82m`（預設） | Apache-2.0 | 本程序內 GPU 推論，權重（`hexgrad/Kokoro-82M-v1.1-zh`，100 個普通話聲音 `zf_*`／`zm_*`，預設 `zf_001`；另有英文 `af_maple` 等）由 `scripts/prefetch_kokoro.py` 在 build time 下載進 image，不需主機掛載，runtime 不連網。繁體輸入先以 OpenCC t2s 轉簡體給 G2P（發音不變），中文句中的英文字走英文 G2P。24 kHz |
| `breeze-tts-2` | 權重與輸出**僅限研究／非商業** | `RT_TTS_BREEZE_ENABLED=true` 才可用，否則 `/tts/models` 顯示 `unavailable` 與授權說明。需要 torch 2.9／transformers 4.57／numpy 2，與 media-worker 底層不相容，所以以 compose profile `breeze` 另起容器（`breeze/Dockerfile`，上游 `breeze_infer.api`），34 透過 HTTP 串流取用。聲音為 `RT_TTS_BREEZE_VOICES` 的預設組（voice design 描述或參考音檔＋逐字稿） |

  原先評估的備援 `speecht5`（Microsoft SpeechT5）只支援英文，無法作為 zh-TW 備援，因此未實作，以 `breeze-tts-2` 為第二個後端。
- **音訊回傳（`app/response.py`、`app/codec.py`）**：`RESPONSE_START` → 逐句 `RESPONSE_TEXT` 與音訊 → 結尾補
  `RT_TTS_SILENCE_PAD_MS`（200 ms）靜音 → `RESPONSE_END`。音訊格式依 CONFIG 的 codec：`pcm` 為 16 kHz s16le、
  每個 binary frame ≤ 4096 bytes；`pcmu`／`pcma` 直接從 TTS 取樣率轉 8 kHz、每幀 20 ms（160 bytes）；`opus` 為每幀一個 20 ms packet。
  `READY` 的 `tts` 與 `output` 欄位說明使用中的後端、聲音與回傳格式。

`TRANSCRIPT_FINAL` → `RESPONSE_START` 幾乎全花在 Module 15（LangGraph 記憶與搜尋 + LLM，非逐 token 串流）；
Module 15 的回答是給螢幕看的長文（最長合成 93 秒語音），語音情境建議 Module 15 另提供口語、簡短的回答模式（後續項目）。
測試中 Module 15 有一次超過 30 秒，客戶端收到 `ERROR LLM_TIMEOUT`（`fatal: false`）與罐頭語句，工作階段照常繼續。

## 工作階段控制

所有送給客戶端的東西都經過每個工作階段一個的 outbox（`app/outbox.py`）：編號、依播放速度送出回答音訊、保留已送出的
frame 供續傳重送。

**有界佇列與背壓**

| 位置 | 上限 | 滿了的時候 |
| --- | --- | --- |
| 音訊接收佇列（每工作階段） | `RT_AUDIO_QUEUE_MAX` = 200 塊（30 ms） | 丟最舊的一塊，`WARNING QUEUE_OVERFLOW`（每秒至多一次） |
| STT 佇列（每工作階段） | `RT_STT_QUEUE_MAX` = 10 句 | 丟新語句，`ERROR STT_OVERLOADED`（`fatal: false`） |
| GPU 上的最終辨識（全服務） | `RT_STT_MAX_CONCURRENT` = 1 | 排隊；說話中的暫定辨識直接略過 |
| 回答句子 → TTS（每工作階段） | `RT_TTS_QUEUE_MAX` = 5 句 | 暫停讀取 Module 15 |
| GPU 上的合成（全服務） | `RT_TTS_MAX_CONCURRENT` = 2 | 排隊；名額只在合成期間占用，不含播放 |
| 尚未送出的回答音訊（每工作階段） | `RT_OUTPUT_QUEUE_MS` = 10 秒，回答 lane 至多 256 frames | 暫停合成。工作階段暫停（等續傳）時一直等到續傳或逾期；連線中卻 `RT_OUTPUT_STALL_S`（10 秒）都沒有送出任何東西時中斷回答並送出 `WARNING response_failed` |
| 控制事件（每工作階段） | 至多 256 frames | `ERROR QUEUE_OVERFLOW` 並關閉連線 |
| 對話輪次（每工作階段） | 進行中 1 輪 + 排隊 2 輪 | 暫停處理下一句的結果 |
| Module 15 | 第一段文字 `RT_LLM_FIRST_TOKEN_TIMEOUT_S` = 30 秒、整個回答 `RT_LLM_TIMEOUT_S` = 30 秒 | `ERROR LLM_TIMEOUT`（`fatal: false`）＋罐頭語句 |

回答音訊依播放速度送出，最多領先客戶端播放位置 `RT_OUTPUT_LEAD_MS`（500 ms）；`RT_OUTPUT_PACE=false` 則合成多快送多快。
`GET /metrics`（Prometheus）：`rt_sessions{connection}`、`rt_audio_queue_chunks`、`rt_stt_queue_utterances`、
`rt_tts_queue_sentences`、`rt_output_queue_ms`、`rt_stt_slots`／`_busy`／`_waiting`、`rt_tts_slots`／`_busy`／`_waiting`，
以及 `rt_audio_dropped_total`、`rt_stt_overloaded_total`、`rt_stt_failed_total`、`rt_llm_timeouts_total`、
`rt_response_failed_total`、`rt_barge_ins_total`、`rt_resumes_total`、`rt_resume_rejected_total`、
`rt_sessions_expired_total`、`rt_pong_timeouts_total`、`rt_guardrail_blocks_total`。佇列量是本程序所有工作階段的總和。
`rt_stt_finalize_latency_seconds`（histogram）：STT `finalize()` 的延遲分布，從一句話開始解碼到送出
`TRANSCRIPT_FINAL` 為止（跟該事件自己帶的 `latency_ms` 同一段時間）；解碼失敗（`stt_failed`）的那次不計入。

**插話**

- 回答播放期間 VAD 照常執行；語音機率 > `RT_BARGE_IN_THRESHOLD`（0.7）持續 ≥ `RT_BARGE_IN_MIN_MS`（300 ms）即插話：
  取消這一輪（Module 15 串流、合成、尚未送出的音訊），送 `BARGE_IN {request_id, tts_position_ms}` 與
  `RESPONSE_END {interrupted: true}`，狀態立即回 `LISTENING`。這段話（從語音開始、含前置保留）成為下一句，照常辨識與回答。
- `tts_position_ms` 是估計的播放位置（從第一個音訊幀送出起算、扣除斷線時間、不超過已送出的長度）；
  客戶端在該位置停止並丟棄其餘已收到的回答音訊。因為音訊依播放速度送出，客戶端手上最多只有約 `RT_OUTPUT_LEAD_MS` 要丟。
- 在回答播放中**開始**、且未觸發插話的語句不處理（`DISCARDED reason=during_response`）：沒有回音消除時那是回答本身。
  在回答開始之前就開始說的話一律照常處理（使用者的話，不是回音）。
- 誤觸發抑制：語句至少 `RT_VAD_MIN_SPEECH_MS`（200 ms）語音，且最終逐字稿至少 `RT_MIN_UTTERANCE_CHARS`（2）個字才開始一輪對話。
- 34 **不做**伺服器端回音消除：客戶端必須避免把播放的回答收進麥克風（瀏覽器 `echoCancellation: true` 或硬體 AEC）；
  做不到時把 `RT_BARGE_IN_THRESHOLD` 調到 0.85。詳見 `docs/CLIENT_GUIDE.md`。

**序號、續傳與 PING／PONG**

- 伺服器送出的每個 frame 都有 `seq`（JSON 欄位；`framing: "seq"` 時 binary 前 4 bytes），從 1 開始、連續、不跳號；
  JSON 事件另帶 `session_id`。已送出的 frame 保留最近 `RT_RESUME_BUFFER_KB`（1024 KB）。
- 沒有 END 就斷線（或沒回 PONG）時，工作階段**暫停** `RT_SESSION_RESUME_WINDOW_S`（60 秒）：保留 VAD 緩衝、
  逐字稿、待播音訊與 Module 15 對話；歸還 VAD 模型，續傳時重新取得。輸出滿載等待超過 1 秒會停止該回答並保留已排隊的音訊，也繼續占用並行名額。
  Redis 紀錄 `connection=suspended`、`resume_until`。
- 續傳：取得新票證（同一使用者），連線 `/audio/stream?token=<票證>&session_id=<id>`，第一個 frame 送
  `{"type":"RESUME","session_id":…,"last_seq":<最後收到的 seq>}`。伺服器重送 `last_seq` 之後保留的 frame（原 seq），
  接著 `RESUMED`，之後照常；回答從中斷處繼續播。URL 帶 `session_id` 時不需空的並行名額；不帶也可以，但需要一個空名額。
  舊連線若還沒斷，會以 4409 關閉（新連線接手）。
- 時限內沒有續傳：狀態 `EXPIRED`，資源與名額釋放；之後以該 `session_id` 續傳（或其他使用者嘗試續傳）都得到
  `ERROR SESSION_EXPIRED` 並以 4410 關閉。
- 管線在建立工作階段的程序內，續傳必須連到同一個 instance（目前部署只有一個）。
- 伺服器每 `RT_PING_INTERVAL_S`（30 秒）送 `PING`，客戶端須在 `RT_PONG_TIMEOUT_S`（10 秒）內回 `PONG`，否則送
  `ERROR pong_timeout`、以 4408 關閉並進入暫停（可續傳）。`RT_PING_INTERVAL_S=0` 關閉。

以真實 Module 15 測試時，第一句回答常超過 10 秒，因此
`RT_LLM_FIRST_TOKEN_TIMEOUT_S` 預設為 30（Module 15 提供較快的語音模式後可再調低）。
問句中有停頓時 VAD 會切成數句、各成一輪；Module 15 若很快回答第一個片段，回答開始時使用者還在說，會立即被插話取消。

## 護欄、個人索引與錄音

**護欄（Module 23，`app/guardrail.py`）**：每個 `TRANSCRIPT_FINAL`（至少 `RT_MIN_UTTERANCE_CHARS` 個字）在送 Module 15 前，
以及每一句回答在合成前，都呼叫 `POST {RT_GUARDRAIL_URL}/guardrail/policy/decide`
`{content, direction: input|output, module: "34", request_id, user_id, session_id}`。

| 判定 | 逐字稿（input） | 回答句（output） |
| --- | --- | --- |
| `allow` | 照常 | 照常 |
| `redact` | 以遮蔽後的文字送 Module 15、寫入索引 | 念遮蔽後的文字 |
| `warn` | 照常，記 log；Module 15 請求帶 `X-Guardrail-Warning: true` | 照常，記 log |
| `block` | `GUARDRAIL_BLOCK`，念 `RT_GUARDRAIL_BLOCK_TEXT`，不呼叫 Module 15、不寫索引 | `GUARDRAIL_BLOCK`，該句起改念拒答語句，其餘回答丟棄 |

Module 23 無法使用（503、逾時）時 `RT_GUARDRAIL_FAIL_OPEN=true`（預設，與 Module 15 的 `GUARDRAIL_FALLBACK_PASSTHROUGH` 相同）放行並記 log；
false 視同 block。回答逐句檢查與 TTS 並行（在送 TTS 的句子佇列中先做），`RT_GUARDRAIL_OUTPUT=false` 可關閉。
Module 15 自己的 LLM 呼叫另外經 Module 23 proxy 檢查；34 的檢查涵蓋 34 自己送出與念出的內容（外部大腦模式只剩這一層）。

**個人索引（Module 25）**：通過護欄（allow／warn／redact）的逐字稿以 Kafka 發到 `personal-index-requests`，格式與 Module 09–12 相同
（`schema_version: personal-index-v1`、`event_id = sha256(asset_path␟version_id␟user_id)`）：一個 `type: "audios"` 的 chunk，
asset `realtime-audio/<session_id>`、version `<session_id>:<utterance>`，payload 另含 `user_id`、`session_id`、`transcript`、`timestamp`。
沒有語音回覆（未設 TTS／LLM）時一樣會檢查並寫入。`RT_INDEX_TRANSCRIPTS=false` 關閉。

**對話記憶（Module 26）**：Module 15 每輪都會自己寫入 Module 26（`append_turn`），34 再寫會重複，所以 Module 15 的輪次 34 不寫；
`app/integrations.MemoryArchive`（`POST /memory/sessions/{session_id}/turns`，`X-User-ID`）留給不自行保存的大腦（外部大腦）。

**錄音（`app/recording.py`、`app/recording_consumer.py`）**

- 只有 `RT_RECORDING_ENABLED=true`（預設 false）**且**使用者同意 `voice_recording` 才錄。工作階段開始時先看 Redis
  `gdpr:voice_recording_disabled:{user_id}`（Module 33 撤回同意時設定），再以 `GET {RT_MODULE33_URL}/consent`（`X-User-ID`、`X-Internal-Token`）確認；
  Module 33 無法連線、使用者不存在或沒有回答時一律不錄（fail closed）。Redis 工作階段紀錄 `recording_enabled` 與
  `recording_skipped_reason`（`disabled_by_config`／`consent_withdrawn`／`no_consent`／`module33_unavailable`／`storage_unavailable`）；`READY.recording` 同步告知客戶端。
- 錄的是使用者的音訊（解碼後 16 kHz mono int16）。每 `RT_RECORDING_CHUNK_S`（10 秒）上傳一段到 SeaweedFS（master `/dir/assign`），
  fid 記在 Redis `rt:rec:{session_id}:chunks`，程序內最多只留 10 秒。工作階段結束時發 Kafka `audio-recording-finalize`
  `{"event": "session_ended", "session_id", "wav_chunks_key", "user_id", ...}`。
- 獨立容器 `raptor-realtime-audio-recording`（同一映像，`python -m app.recording_consumer`）把各段接成一個 WAV 存回 SeaweedFS，
  寫入 PostgreSQL `realtime_audio.rt_audio_recordings`（`id, user_id, session_id, fid, duration_s, size_bytes, created_at, expires_at, erased_at`；
  `expires_at = created_at + RT_RECORDING_RETENTION_DAYS`（90 天）），再刪除各段。處理完才 commit offset；34 程序當掉而沒有送出
  `session_ended` 的錄音，consumer 每分鐘掃描一次，在工作階段心跳過期且 `RT_RECORDING_ORPHAN_IDLE_S`（300 秒）沒有新音訊後補做；
  每小時刪除過期的錄音。`RT_RECORDING_ENABLED=false` 時 consumer 只待命。
- 資料庫：新部署由 `03-database/init/postgresql/009_realtime_audio.sql` 建立；既有部署需手動套用一次（同其他 GDPR SQL）。
- 抹除：`DELETE /internal/gdpr/user/{user_id}`（Module 33 的全帳號抹除與撤回 `voice_recording` 同意都會呼叫；compose 已把
  Module 33 的 `GD_MODULE_34_URL` 指向 `http://raptor-realtime-audio:8034`）立即停止該使用者進行中的錄音、刪除尚未合併的段落與所有 WAV，
  設定 `erased_at`，回 `{"erased_sessions": N, "bytes_freed": N}`（`bytes_freed` 為 SeaweedFS 回報的實際刪除量）；沒有錄音時回 0。
  合併時若使用者已被抹除或撤回同意，直接丟棄不存。
- 匯出：`GET /internal/gdpr/user/{user_id}/recordings` 列出尚未抹除的錄音
  （`session_id`、`created_at`、`expires_at`、`duration_s`、`size_bytes`）與仍在錄音中的工作階段數 `pending_sessions`；
  Module 33 的匯出功能把它寫成 `voice/recordings.json`。只有中繼資料，不含音檔，也不含 SeaweedFS fid。

## 外部大腦與代理主體

**外部大腦**：CONFIG `"brain": "external"` 只有票證來自帶 realm role `rt-external-brain` 的 token 時才允許，否則
`ERROR BRAIN_MODE_FORBIDDEN` 並以 4403 關閉。這種工作階段 34 **不呼叫 Module 15**：只送出 `TRANSCRIPT_FINAL`，由客戶端決定要說什麼。

- `SPEAK {utterance_id, text, final}`：把文字合成為語音。同一 `utterance_id` 的多個 SPEAK 依序無縫接成一段回答
  （`RESPONSE_START`（`request_id` = `utterance_id`）→ 每句 `RESPONSE_TEXT` 與音訊 → `final: true` 之後 `RESPONSE_END`，只在最後補靜音）；
  不同 `utterance_id` 依到達順序播放。這段回答的 binary 音訊即 `AUDIO_RESPONSE`（工作階段的 codec，電話為 20 ms G.711）。
  整個工作階段都保留已結束的 `utterance_id`；重送 SPEAK 回 `WARNING speak_after_final`。
- `SPEAK_CANCEL {utterance_id}`：丟掉還沒念的文字；正在念的會停止、未送出的音訊丟棄，並送 `RESPONSE_END {cancelled: true}`；其他回答不受影響。
- 插話會取消正在念的與所有排隊中的 SPEAK，`BARGE_IN.request_id` 為被打斷的 `utterance_id`，客戶端據此取消自己的生成。
- 護欄：`RT_EXTERNAL_GUARDRAIL=off|input|output|both`（預設 `output`：每句 SPEAK 文字合成前經 Module 23；`input` 時逐字稿被擋只送
  `GUARDRAIL_BLOCK`，由客戶端處理）。
- SPEAK 文字也會依句切開，Module 23 檢查與合成重疊進行。

**代理主體**：帶 realm role `rt-delegate` 的 token 可在 CONFIG 帶 `subject_id`（`^[A-Za-z0-9_-]{1,128}$`，去識別化代號），
表示這個工作階段實際是為哪位終端使用者服務。同意查詢、錄音（`rt_audio_recordings.user_id`）、Module 33 抹除、Module 25 索引、
Module 26 記憶與 Module 15 的 `user_id` 都改用 `subject_id`；Redis 工作階段紀錄同時保留 `actor_sub`（token 的 sub）與 `subject_id`，log 也兩者都記。
沒有該角色時忽略 `subject_id`，記 log 並在 `READY` 後送 `WARNING subject_ignored`。

**每個工作階段的副作用**：CONFIG `record`、`index_transcripts`、`store_memory`。

| | `brain: "module15"` 預設 | `brain: "external"` 預設 |
| --- | --- | --- |
| `record` | 開（仍需 `RT_RECORDING_ENABLED` 與主體同意） | 關 |
| `index_transcripts` | `RT_INDEX_TRANSCRIPTS` | 關 |
| `store_memory` | `RT_MEMORY_ENABLED`（Module 15 自行寫入 Module 26） | 關；開啟時 34 以 `POST /memory/sessions/{sid}/turns`、`X-User-ID: subject_id` 寫入「聽到的話＋SPEAK 回答」 |

全域設定為上限：旗標只能在全域允許的範圍內開啟。帶 `rt-external-brain`／`rt-delegate` 角色的服務帳號沒有有效的 `subject_id` 時，
除非明確 `index_transcripts: true`，逐字稿絕不寫入服務帳號自己的 Module 25 分支。`READY.side_effects` 回報實際生效的值。
`brain: "module15"` 時 Module 15 會自行把對話寫入 Module 26，`store_memory: false` 無法阻止（需 Module 15 支援）。

## 編解碼器

- `app/codec.py`。G.711 μ-law／A-law 以 ITU-T G.711 的分段規則用 NumPy 建表（解碼 256 項、編碼 65536 項），
  與 CPython `audioop` 逐位元相同（單元測試對 65536 個值逐一比對），不依賴 3.13 已移除的 `audioop`。
- 重取樣用 libswresample（PyAV，Opus 原本就用它），跨 frame 保留濾波狀態，20 ms 小幀與整段一次轉換的結果相同；
  END 時取出濾波延遲留下的尾端。`librosa`／`scipy.resample_poly` 是整段處理，用在串流會在每個 frame 邊界產生不連續，因此改用前者。
- 送回客戶端的音訊（給 TTS 使用）：`make_encoder(codec, sample_rate)`。`pcm` 為 16 kHz s16le、每塊 ≤ 4096 bytes；
  `pcmu`／`pcma` 直接把 TTS 原生取樣率（如 24 kHz）重取樣到 8 kHz 再編碼（不經 16 kHz），每幀 20 ms = 160 bytes，
  最後一幀補靜音；`opus` 為 CONFIG 的取樣率、每幀一個 20 ms packet。

## 語音辨識

- `app/stt.py`。啟動時載入 faster-whisper（CTranslate2）`RT_STT_MODEL`（預設 `large-v3`）：先找
  `RT_STT_MODEL_DIR` 下的同名目錄或 `faster-whisper-<名稱>`，否則交給 faster-whisper 從 HF cache 取得。
  `RT_STT_DEVICE=auto` 有 GPU 時用 CUDA float16；沒有 GPU 時改用 `RT_STT_CPU_MODEL`（`tiny`，int8），
  供只有 CPU 的 CI 使用。模型載入失敗時服務不會啟動；`RT_STT_ENABLED=false` 只做 VAD。
- **最終辨識**：每個 VAD 語句以 `word_timestamps=True`、beam 5 解碼；解出的詞逐一送 `TRANSCRIPT_PARTIAL`，
  再送 `TRANSCRIPT_FINAL`。
- **說話中**：語音開始 `RT_STT_FIRST_DECODE_MS`（300 ms）後第一次解碼，之後每 `RT_STT_STRIDE_S`
  （0.5 秒音訊）解碼一次正在進行的整段語句（greedy、不做溫度回退），送 `TRANSCRIPT_PROVISIONAL`；
  與上一次解碼結果一致的前綴（以字元比對，最新一個詞除外）升為 `TRANSCRIPT_STABLE`，只送一次。
  最終結果與已送出的穩定文字不符時送 `TRANSCRIPT_CORRECTION`。
- **設計取捨**：原本設想的是「2 秒滑動視窗」。Whisper 每次都把輸入補到 30 秒再編碼，
  解碼整段進行中的語句（最長 `RT_MAX_UTTERANCE_S`）與解碼 2 秒的成本相近；而用詞時間戳拼接 2 秒視窗，
  實測會在視窗邊界掉字（20 句中 8 句，如「有空[嗎]我們」），所以改為每次解碼整段語句，不設 `RT_STT_WINDOW_S`。
- **GPU 名額**：`RT_STT_MAX_CONCURRENT`（預設 1）限制同時解碼數。最終辨識會排隊等名額；說話中的解碼在名額
  被占用（或有最終辨識在等）時直接略過該次，不會拖慢最終結果。每次解碼的 token 上限依音訊長度計算，
  避免 Whisper 陷入重複迴圈時長時間占用 GPU；超過預期時間的解碼會記錄 `slow ... decode` 警告。
- **繁體中文**：Whisper 的中文輸出為簡體。zh-TW／zh-HK／zh-Hant、單純 `zh` 與自動偵測為中文的工作階段，
  以 OpenCC `RT_STT_ZH_CONVERT`（預設 `s2tw`，只轉字）轉換；zh-CN／zh-SG／zh-Hans 不轉。
  `s2twp` 會連用詞一起換（数据→資料、设备→裝置），等於改了說話者的用詞，實測使 CER 變差，所以不作為預設。

## 部署

```bash
cd deployment/modules
python3 build.py 34            # 依賴 02、06、08；--build 會先重建 08 的 media-worker 再建 34
```

- Compose 的 image 為 `raptor/realtime-audio:${RAPTOR_VERSION}`，Dockerfile `FROM ${BASE_IMAGE}`
  使用 module 08 建出的 `raptor/media-worker:${RAPTOR_VERSION}`。
  只有舊 tag 的主機可在 `.env` 覆寫 `RT_BASE_IMAGE`。直接使用 `docker build` 時應傳入
  `--build-arg BASE_IMAGE=raptor/media-worker:<版本>`；Dockerfile 保留 0.4 作為舊版獨立建置的 fallback。
- STT 與 TTS 需要 GPU：compose 預留 `REALTIME_AUDIO_GPU`（預設 0），build.py 將 34 標為 GPU 模組；large-v3 float16 約占 4 GB，
  Kokoro 約 0.7 GB。映像另裝 `requirements-tts.txt`（kokoro、misaki、spaCy 英文小模型）與 espeak-ng，執行時不下載任何東西。
- Breeze TTS 2（選用）：`docker compose --profile breeze up -d` 另起 `raptor-breeze-tts`，GPU 為 `REALTIME_AUDIO_BREEZE_GPU`（約 8 GB），
  權重掛 `RT_TTS_BREEZE_MODEL_DIR`；並設定 `RT_TTS_MODEL=breeze-tts-2`、`RT_TTS_BREEZE_ENABLED=true`。
  CTranslate2 格式的 Whisper 放在主機 `RT_STT_MODEL_DIR`（唯讀掛到 `/models/stt`，例如內含 `faster-whisper-large-v3/`），
  可用 `benchmarks/convert_whisper.py` 從 Transformers 格式離線轉換；未設定時使用 `HF_CACHE_PATH` 的 HF cache。
- gRPC：容器內 8044（`RT_GRPC_PORT`），主機 `PORT_REALTIME_AUDIO_GRPC`（預設 8044），已記在 `deployment/service-catalog.yaml`。
- 主機預設埠為 `PORT_REALTIME_AUDIO=8035`，容器內埠仍為 8034；避免與 Module 11 的 video-face 主機埠 8034 衝突。
  舊設定若仍使用 8034，請改為未被其他模組占用的主機埠，並更新 client／反向代理的 upstream。
- 要經 Module 13 gateway 或其他反向代理對外時，代理需轉送 `Upgrade`／`Connection` header；13 目前沒有此路由。

## 環境變數

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `RT_TICKET_TTL_S` | 30 | 票證有效秒數 |
| `RT_SESSION_TTL_S` | 1800 | 工作階段最長存活秒數 |
| `RT_MAX_SESSIONS` | 20 | 全服務並行上限 |
| `RT_MAX_SESSIONS_PER_USER` | 3 | 同一使用者（JWT `sub`）並行上限 |
| `RT_CONFIG_TIMEOUT_S` | 10 | 等待 CONFIG 秒數 |
| `RT_HEARTBEAT_S` | 10 | 名額心跳間隔 |
| `RT_VAD_CHUNK_MS` | 30 | 傳給 VAD 的切塊大小 |
| `RT_VAD_THRESHOLD` | 0.5 | 語音機率門檻 |
| `RT_VAD_MIN_SILENCE_MS` | 300 | 語句結束所需靜音 |
| `RT_VAD_MIN_SPEECH_MS` | 200 | 最短語句 |
| `RT_MAX_UTTERANCE_S` | 10 | 語句緩衝上限 |
| `RT_VAD_SPEECH_PAD_MS` | 96 | 語句前後保留 |
| `RT_VAD_POOL_SIZE` | 4 | 啟動時預載的 silero 實例數 |
| `RT_AUDIO_QUEUE_MAX` / `RT_STT_QUEUE_MAX` | 200 / 10 | 每個工作階段的音訊／STT 佇列上限（`RT_STT_QUEUE_MAX` 舊名 `RT_UTTERANCE_QUEUE_MAX`） |
| `RT_TTS_QUEUE_MAX` | 5 | 每個工作階段等待合成的回答句數 |
| `RT_OUTPUT_PACE` / `RT_OUTPUT_LEAD_MS` / `RT_OUTPUT_QUEUE_MS` / `RT_OUTPUT_STALL_S` | true / 500 / 10000 / 10 | 回答音訊依播放速度送出、最多領先、未送出的上限、連線中沒有進度多久算卡住 |
| `RT_BARGE_IN_THRESHOLD` / `RT_BARGE_IN_MIN_MS` / `RT_MIN_UTTERANCE_CHARS` | 0.7 / 300 / 2 | 插話門檻與持續時間（無回音消除時建議 0.85）、開始一輪對話的最少字數 |
| `RT_SESSION_RESUME_WINDOW_S` / `RT_RESUME_BUFFER_KB` | 60 / 1024 | 斷線後等待續傳的秒數（0 = 不續傳）、保留供重送的 frame |
| `RT_PING_INTERVAL_S` / `RT_PONG_TIMEOUT_S` | 30 / 10 | PING 間隔（0 = 不送）、等待 PONG |
| `RT_EXTERNAL_GUARDRAIL` | output | 外部大腦工作階段的 Module 23 檢查：off／input／output／both；其他值會阻止啟動 |
| `RT_GUARDRAIL_ENABLED` / `RT_GUARDRAIL_URL` / `RT_GUARDRAIL_OUTPUT` / `RT_GUARDRAIL_FAIL_OPEN` / `RT_GUARDRAIL_BLOCK_TEXT` | true / `http://raptor-guardrail-service:8026` / true / true / 「抱歉，這個問題我無法協助處理。」 | Module 23 檢查逐字稿與回答 |
| `RT_INDEX_TRANSCRIPTS` / `RT_KAFKA_BOOTSTRAP`（或 `KAFKA_BOOTSTRAP_SERVERS`） / `RT_KAFKA_PERSONAL_INDEX_TOPIC` | true / `kafka-broker1:9092,…` / `personal-index-requests` | 逐字稿寫入 Module 25 |
| `RT_MEMORY_ENABLED` / `RT_MEMORY_SERVICE_URL` | true / `http://raptor-memory-service:8026` | Module 26（僅不自行保存的大腦） |
| `RT_RECORDING_ENABLED` / `RT_RECORDING_CHUNK_S` / `RT_RECORDING_RETENTION_DAYS` / `RT_RECORDING_COLLECTION` | false / 10 / 90 / 空 | 錄音開關、上傳間隔、保存天數、SeaweedFS collection |
| `RT_MODULE33_URL` / `GD_INTERNAL_TOKEN` | `http://raptor-privacy-compliance-service:8033` / 空 | 同意查詢；與 Module 33 共用的內部 token（同時驗證抹除請求） |
| `RT_KAFKA_RECORDING_TOPIC` / `RT_SEAWEEDFS_MASTER` | `audio-recording-finalize` / `http://raptor-seaweedfs-master1:9333` | 錄音合併訊息、SeaweedFS |
| `RT_POSTGRES_DSN` 或 `POSTGRES_HOST/PORT/USER/PASSWORD` + `RT_POSTGRES_DB` | `realtime_audio` | `rt_audio_recordings` 所在資料庫 |
| `RT_RECORDING_SWEEP_S` / `RT_RECORDING_ORPHAN_IDLE_S` | 60 / 300 | consumer 的掃描間隔、無新音訊多久視為遺留 |
| `KEYCLOAK_URL` / `RT_JWT_PUBLIC_KEY` / `RT_JWT_AUDIENCE` | | JWT 驗證（HTTP 與 gRPC 共用） |
| `RT_GRPC_ENABLED` / `RT_GRPC_PORT` | true / 8044 | gRPC `AudioService` |
| `REDIS_URL` 或 `REDIS_HOST/PORT/PASSWORD/DB` | `raptor-redis-standalone:6379/0` | Module 02 Redis |
| `RT_UTTERANCE_DUMP_DIR` | 空 | 除錯：把語句寫成 WAV |
| `RT_CODEC_RESAMPLE` | true | false = 只接受 16 kHz PCM |
| `RT_TTS_ENABLED` / `RT_TTS_MODEL` / `RT_TTS_VOICE` | true / kokoro-82m / 空 | TTS 開關、後端、聲音（空 = 後端預設） |
| `RT_TTS_SPEED` / `RT_TTS_DEVICE` / `RT_TTS_MAX_CONCURRENT` / `RT_TTS_SILENCE_PAD_MS` | 1.0 / auto / 2 / 200 | 語速、裝置、同時合成數、結尾靜音 |
| `RT_TTS_BREEZE_ENABLED` / `RT_TTS_BREEZE_URL` / `RT_TTS_BREEZE_VOICES` / `RT_TTS_BREEZE_MODEL_DIR` | false / `http://raptor-breeze-tts:7860` / 空 / 空 | Breeze TTS 2（非商業授權，需明確開啟） |
| `RT_LLM_ENABLED` / `RT_CHAT_SERVICE_URL` / `RT_CHAT_STREAM` | true / `http://raptor-chat-service:8021/api/v1` / true | 語音對話、Module 15 API 前綴、SSE |
| `RT_LLM_TIMEOUT_S` / `RT_LLM_FIRST_TOKEN_TIMEOUT_S` / `RT_LLM_FALLBACK_TEXT` | 30 / 30 / 「系統暫時無法回應，請稍後再試。」 | 回答逾時、第一段文字逾時、逾時或失敗時念的語句 |
| `RT_STT_ENABLED` | true | false = 只做 VAD |
| `RT_STT_MODEL` / `RT_STT_MODEL_DIR` | large-v3 / 空 | 模型名稱或目錄；主機上的 CTranslate2 模型目錄 |
| `RT_STT_PROVISIONAL_MODEL` | 空 | 說話中解碼用的模型（空 = 同 `RT_STT_MODEL`） |
| `RT_STT_DEVICE` / `RT_STT_COMPUTE_TYPE` / `RT_STT_CPU_MODEL` | auto / 空 / tiny | 裝置、精度（cuda 預設 float16、cpu 預設 int8）、無 GPU 時的模型 |
| `RT_STT_LANGUAGE` | zh-TW | CONFIG 省略 `language` 時的語言；`auto` = 自動偵測 |
| `RT_STT_BEAM_SIZE` | 5 | 最終辨識的 beam |
| `RT_STT_MAX_CONCURRENT` | 1 | 同時在 GPU 上的解碼數 |
| `RT_STT_ZH_CONVERT` | s2tw | OpenCC 轉換設定；空 = 不轉 |
| `RT_STT_STREAMING` / `RT_STT_STRIDE_S` / `RT_STT_FIRST_DECODE_MS` | true / 0.5 / 300 | 說話中的暫定辨識 |

## 測試

```bash
cd deployment/modules/34-realtime-audio
pip install -r requirements.txt pytest pytest-asyncio
python -m pytest tests/unit -v
```

單元測試不需 torch、silero、faster-whisper、Keycloak 或 Redis：以振幅判斷的假 VAD 模型、假 Whisper 後端、
fakeredis（含 Lua）與自簽 RS256 金鑰執行。STT 部分涵蓋：最終辨識的 PARTIAL／FINAL、說話中的 PROVISIONAL／STABLE
（兩次一致才升級、重複字保留、不同斷詞仍能對齊、插入的字不會遺失）、CORRECTION 範圍、語言 null／省略／auto、
簡轉繁（整句轉換再切回詞）、GPU 名額滿時略過暫定辨識、STT 失敗時只警告不斷線。
編解碼器部分：G.711 參考向量與 audioop 逐位元比對、量化 SNR、串流重取樣無接縫、G.711 8 kHz／任意取樣率 PCM／
Opus 48 kHz → 16 kHz、TTS 24 kHz → 8 kHz G.711 20 ms 幀、PCM 4096-byte 分塊、Opus 編碼可解回、
`UNSUPPORTED_CODEC`、`RT_CODEC_RESAMPLE=false`，以及 G.711 電話串流的 VAD 斷句。
語音回覆部分（假 TTS 後端、以 `httpx.MockTransport` 模擬 Module 15）：註冊表與未知後端、Breeze 授權開關、聲音驗證、
`/tts/models`、`RESPONSE_START`／`TEXT`／`END` 與 200 ms 靜音、PCM 分塊與 G.711 20 ms 幀、切句規則、SSE／JSON／錯誤事件／
逾時（含 SSE 開始後停滯）、整輪對話的狀態轉換與 `session_id`／`user_id`／`request_id`、逾時念罐頭語句且工作階段繼續。另涵蓋票證核發／單次使用／過期、Upgrade 前 401／503（名額滿不消耗票證、當機名額回收）、
CONFIG 驗證與逾時、Redis 工作階段紀錄與 TTL、語句切分／END flush／短雜訊丟棄、斷線結束狀態、
PCM 奇數位元組 frame、Opus 串流與壞 packet、工作階段到期、VAD 落後時丟棄而不阻塞、各工作階段獨立 VAD 實例，
以及 Segmenter 的 300 ms／200 ms／10 秒／門檻／前後保留規則。
gRPC（`test_grpc_service.py`，真正的 grpc.aio server，埠 0）：語音一輪的 transcript／回答 chunk／`is_last`、暫定逐字稿、
PCMU 進出、空呼叫、token 缺少／偽造／過期／無 `sub` → `UNAUTHENTICATED`、不支援的 codec／取樣率／語言與中途改 codec →
`INVALID_ARGUMENT`、名額滿 → `RESOURCE_EXHAUSTED`、取消呼叫即結束工作階段（不等續傳）、角色傳遞、事件轉換，
以及被拒絕的呼叫端連續寫入時仍收到正確狀態。

### 整合測試

```bash
# 在 Module 34 映像內（raptor/media-worker 為底），掛入原始碼與模型
pip install pytest pytest-asyncio
python -m pytest tests/ -m integration -v -s
```

`tests/integration/` 在測試行程內以 uvicorn 啟動真正的服務：app 自己的 lifespan 載入 silero-VAD、faster-whisper、
TTS 後端並啟動 gRPC server，HTTP／WS／gRPC 都在本機空閒埠。Module 15 預設由 `benchmarks/fake_chat.py` 代替
（回答 `RT_IT_ANSWER`，延遲 `RT_IT_CHAT_DELAY_S` = 0.2 秒，量到的是 Module 34 本身）；`RT_IT_CHAT_URL` 可改打真正的 Module 15。
Redis 預設 fakeredis（`RT_IT_REDIS_URL` 指定真的 Redis）；Module 23／25／26 與錄音預設關閉。沒有 GPU 時設 `RT_STT_MODEL=tiny`
（CI 的 CPU runner；`RT_STT_DEVICE=auto` 偵測不到 GPU 時，服務本身也會改用 `RT_STT_CPU_MODEL`）。
一般的 `pytest` 只跑單元測試（`pytest.ini` 的 `-m "not integration"`）；其他變數見 `tests/integration/conftest.py`。

| 檔案 | 內容 |
| --- | --- |
| `test_ws_integration.py` | CONFIG → 5 秒 zh-TW PCM（即時速度）→ END：`TRANSCRIPT_FINAL` 在 END 後 3 秒內、`RESPONSE_START` + ≥ 1 個 TTS binary + `RESPONSE_END` 在 END 後 8 秒內（`RT_IT_RESPONSE_TIMEOUT_S`），VAD 切出語句；只送 END → READY／ENDED、close 1000、沒有 ERROR／WARNING；只有底噪 → 沒有語句；G.711 PCMU 通話（回 160-byte 20 ms 幀）；Upgrade 帶偽造、過期或有效 JWT（應帶票證）或沒有憑證 → HTTP 401，不建立 WS；`POST /audio/session/token` 的無效 JWT → 401 |
| `test_grpc_audio.py` | 以產生的 stub、`grpc.aio.insecure_channel` 與測試 JWT metadata 即時串流 10 秒 zh-TW WAV：≥ 1 個 `TranscriptEvent(is_final=true)`、≥ 1 個 `AudioResponseChunk(is_last=false)`、每個回答恰好一個 `is_last=true`；無效 JWT → `UNAUTHENTICATED` |
| `test_codec.py` | libavcodec 編碼的 PCMU（8 kHz μ-law WAV）與 Opus 48 kHz（Ogg）fixture 逐幀經 `app.audio.make_decoder(codec, rate).decode()` → 16 kHz int16：長度、響度包絡相關 ≥ 0.95（Opus ≥ 0.9）、音量差 ≤ 3 dB；另測 PCMA 與 8／22.05／44.1／48 kHz PCM |

Fixture（`tests/fixtures/`）以 Kokoro-82M-v1.1-zh（Apache-2.0）`zf_001` 合成，不含第三方錄音；文字見 `manifest.json`，
`make_fixtures.py` 可重新產生（G.711／Opus 由 libavcodec 編碼，不用 Module 34 自己的編碼器）。
