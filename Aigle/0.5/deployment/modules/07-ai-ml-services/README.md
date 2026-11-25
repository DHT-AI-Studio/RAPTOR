# AI 模型生命週期管理平台

這是一個面向 MLOps 的 AI 模型生命週期管理平台，整合模型下載、版本控制、註冊、階段管理與線上推理。架構以 **MLflow Model Registry**（模型註冊與生命週期）+ **LakeFS**（模型/資料原子化版本控制）為核心，外加一個 **spec-driven 推理閘道** 統一所有 runtime（Ollama / HuggingFace Transformers，可擴 vLLM 等）。

---

## 快速開始 — Ollama 模型 5 分鐘部署

```bash
# 1. 確認本地 Ollama 模型
ollama list
# 或：curl "http://localhost:8010/models/local?model_source=ollama"

# 2. 啟動服務
docker compose up --build -d

# 3. 註冊 Ollama 模型到 MLflow
curl -X POST "http://localhost:8010/models/register_ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "local_model_name": "gemma4:12b",
    "task": "text-generation",
    "registered_name": "gemma4-12b-ollama",
    "stage": "production"
  }'

# 4. 推理
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation",
    "model_name": "gemma4-12b-ollama",
    "data": {"inputs": "你好，請介紹一下你自己。"}
  }'
```

完整流程見 [工作流 A](#工作流-aollama-模型快速註冊與推理)。

---

## 目錄

1. [專案概覽](#專案概覽)
2. [核心功能](#核心功能)
3. [目錄結構](#目錄結構)
4. [推理架構](#推理架構)
5. [ModelSpec：模型推理規格](#modelspec模型推理規格)
6. [OpenAI 相容 API](#openai-相容-api)
7. [閒置卸載與 GPU 釋放](#閒置卸載與-gpu-釋放)
8. [部署與設定](#部署與設定)
9. [範例工作流](#範例工作流)
10. [API 端點清單](#api-端點清單)
11. [常見問題](#常見問題)

---

## 專案概覽

平台核心元件：

- **MLflow Tracking Server** — 實驗追蹤與成品版本記錄。
- **MLflow Model Registry** — 集中管理模型生命週期（透過 stage tag：`production` / `staging` / `archived` / `none`），同時記錄 spec-driven 推理規格。
- **LakeFS** — 模型與資料的 Git-like 原子化版本控制；MLflow 中每個版本都對應一個 LakeFS commit。
- **InferenceService（推理閘道）** — 接收統一推理請求，解析 MLflow tag 為 `ModelSpec`，分派到對應 runtime adapter。
- **Models / Datasets / GPU / Config 管理 API** — RESTful 介面控制下載、上傳、註冊、階段切換、資源查詢。

---

## 核心功能

- **多來源整合**：HuggingFace 直接下載、Ollama 本地納管、未來可擴 vLLM/Triton。
- **Git-like 模型版本**：每個 MLflow 版本以 `lakefs_commit_id` 標籤指向 LakeFS 不可變 commit。
- **Spec-driven 動態調用**：註冊時把「該模型該怎麼跑」寫進 MLflow tag（`runtime` / `model_class` / `processor_class` / `pipeline_task` / `custom_handler`），推理端不需改程式碼即可支援新模型架構。
- **統一推理 API**：所有任務（text-gen / VLM / ASR / OCR / audio-cls / video / document / TTS / embedding / rerank）都走 `POST /inference/infer`；`task` 可省略（由模型註冊的 task_family 決定）。TTS 另有便捷端點 `POST /inference/tts`。
- **文字推論契約**：`model_version` / `model_alias` 選版、共用生成參數正規化（`max_tokens`、`stop`、`temperature=0` 在 Ollama 與 HF 都實際生效）、頂層真實 `usage`、`GET /inference/model-info` 能力／輸出上限查詢、穩定錯誤碼（`detail.code`）。
- **OpenAI 相容 API（xinference 式）**：`/v1/models`、`/v1/chat/completions`（LLM 與 VLM 通用）、`/v1/completions`、`/v1/audio/transcriptions`、`/v1/audio/speech`（TTS）、`/v1/embeddings`、`/v1/rerank` — 任何 OpenAI SDK 換個 `base_url` 即可調用已註冊模型。
- **內建多模態分支**：VLM handler 依 spec 自動分派 Qwen2.5-VL（chat template + qwen_vl_utils）與 InternVL（動態切圖 + `.chat()`）專屬路線；其餘走通用 processor→generate。
- **閒置自動卸載（ollama 式）**：模型閒置超過 `idle_timeout` 自動卸載並釋放 GPU；逐請求可用 `keep_alive` 覆寫（`0` = 用完即卸、負值 = 常駐）。
- **Stage-based 生命週期**：透過 `/models/transition_stage` 切換 production / staging / archived 並可自動封存舊版本。
- **資源管理**：每個 adapter 內建 LRU 快取，支援按需載入/卸載；提供 GPU 狀態 API。

---

## 目錄結構

```
ai-lifecycle-api/
├── src/
│   ├── api/
│   │   ├── models_api.py        # 模型管理 API（下載/上傳/註冊/階段）
│   │   ├── datasets_api.py
│   │   ├── inference_api.py     # 統一推理 API
│   │   ├── openai_api.py        # OpenAI 相容 API（/v1/*，xinference 式）
│   │   ├── gpu_api.py
│   │   └── config_api.py
│   ├── core/
│   │   ├── configs/
│   │   │   ├── base.yaml
│   │   │   └── inference.yaml   # adapter 配置（engines.ollama / transformers）
│   │   ├── config.py
│   │   ├── model_manager.py     # 下載/上傳 LakeFS/註冊 MLflow（寫入 spec tag）
│   │   ├── dataset_manager.py
│   │   └── gpu_manager.py
│   ├── inference/
│   │   ├── spec.py              # ModelSpec 與 MLflow tag 解析
│   │   ├── service.py           # InferenceService（取代舊 manager+router+executor）
│   │   ├── adapters/
│   │   │   ├── base.py          # BaseAdapter ABC（含 LRU）
│   │   │   ├── ollama.py
│   │   │   └── hf_transformers.py  # spec-driven dispatch
│   │   ├── handlers/
│   │   │   ├── base.py
│   │   │   ├── text_generation.py
│   │   │   ├── vlm.py           # 含 Qwen2.5-VL 分支（由 spec.model_class 判斷）
│   │   │   ├── asr.py
│   │   │   ├── ocr.py
│   │   │   ├── audio_classification.py
│   │   │   ├── video_analysis.py
│   │   │   └── document_analysis.py
│   │   ├── exceptions.py
│   │   ├── vram_estimator.py
│   │   ├── README.md            # 推理模組說明
│   │   ├── USAGE_GUIDE.md       # 接入流程示範
│   │   └── REFACTORING_SUMMARY.md
│   └── main.py                  # FastAPI 入口
├── tmp/                         # HF 下載 / LakeFS dummy artifact 暫存
├── data/                        # 從 LakeFS 下載到本機的推理用模型
├── docker-compose.yaml
├── Dockerfile
└── pyproject.toml / uv.lock / requirements.lock.txt
```

---

## 推理架構

兩層、spec-driven 設計：

```
┌─────────────────────────────────────────────────────────┐
│   API     POST /inference/infer  (FastAPI)              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│   InferenceService                                      │
│     1. canonicalize task name                           │
│     2. validate data 必填欄位                            │
│     3. resolve_spec(model_name) ← MLflow tag            │
│     4. get_adapter(spec.runtime).run(spec, data, opts)  │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
        runtime=ollama          runtime=hf-transformers
               │                          │
┌──────────────▼──────────┐    ┌──────────▼───────────────┐
│   OllamaAdapter         │    │   HFTransformersAdapter  │
│   • POST /api/generate  │    │   • spec-driven dispatch │
│   • daemon 管模型載入    │    │   • LRU cache            │
└─────────────────────────┘    └──────────┬───────────────┘
                                          │
                              spec.task_family / custom_handler
                                          ▼
                          ┌──────────────────────────────┐
                          │   handlers/*                 │
                          │   encode → generate → decode │
                          └──────────────────────────────┘
```

刪除的舊類：`InferenceManager` / `TaskRouter` / `ModelExecutor` / `ModelRegistry` / `ModelCache` / `BaseEngine` / `BaseModelHandler`（都被上面兩層吸收）。

詳細說明見 [src/inference/README.md](./src/inference/README.md)。

---

## ModelSpec：模型推理規格

每個 MLflow 模型版本的 tag 中保存一份「該模型該怎麼跑」的規格。註冊時寫入，推論時讀回。

### Schema 摘要

| Tag | 必填 | 說明 |
|-----|-----|------|
| `runtime` | ✅ | `ollama` / `hf-transformers` |
| `task_family` | ✅ | canonical 任務（`text-generation`/`vlm`/`asr`/`ocr`/`audio-classification`/`video-analysis`/`document-analysis`/`image-captioning`/`tts`/`embedding`/`rerank`） |
| `physical_path` | ✅ | `lakefs://repo/commit/` 或本機路徑或 HF repo-id |
| `lakefs_commit_id` | – | 用於溯源 |
| `model_class` | (HF, a) | transformers 模型 class 名（如 `Qwen2_5_VLForConditionalGeneration`） |
| `processor_class` | (HF) | `AutoProcessor` / `AutoTokenizer` / `AutoFeatureExtractor` |
| `pipeline_task` | (HF, a) | 走 `transformers.pipeline` 捷徑（如 `text-generation`） |
| `trust_remote_code` | – | 預設 `true` |
| `quantization` | – | `4bit` / `8bit` |
| `custom_handler` | – | dotted 路徑（如 `myorg.handlers.x:Cls`）或內建短名（`internvl` / `vibevoice`） |
| `ollama_model_name` | (Ollama) | daemon 上的真名（允許與 MLflow 註冊名不同） |

(a) HF runtime：`model_class` 與 `pipeline_task` 至少要填一個。

### 向下相容

`ModelSpec.from_mlflow_tags` 在**讀取**時仍相容舊 schema：`inference_engine` / `inference_task` / `ollama_model_name` 會被自動映射到新 schema，舊註冊不需要遷移。

**寫入端**自 v3.1 起改為單一來源 — `register_*_to_mlflow` 只寫 v3 schema（`runtime` / `task_family` / `model_class` / `processor_class` / `pipeline_task` / `physical_path` / `trust_remote_code` / `custom_handler` / `quantization`），不再回寫 `inference_engine` / `inference_task`。

唯一需要手動補 tag 的情境：**v2 註冊的 HF 模型完全沒有 `model_class`/`pipeline_task`**（v2 是靠路徑字串匹配判斷模型架構），新 service 第一次推論會回 `ValidationError` 提示補 tag。範例腳本見 [USAGE_GUIDE — 從 v2.x 遷移](./src/inference/USAGE_GUIDE.md#從-v2x-遷移)。

---

## OpenAI 相容 API

`/v1/*` 是 xinference 式的相容層：任何 OpenAI SDK / client 把 `base_url` 指到本服務，就能直接調用 MLflow 中已註冊的模型（LLM、VLM、ASR），呼叫端不需要知道 task 或 runtime。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8010/v1", api_key="not-needed")

# 列出可註冊模型（metadata 帶出 task_family / runtime / stage）
client.models.list()

# LLM（Ollama 或 HF transformers 模型皆可 — runtime 由 MLflow tag 決定）
resp = client.chat.completions.create(
    model="gemma4-12b-ollama",
    messages=[{"role": "user", "content": "請介紹 MLOps"}],
)

# VLM：同一個端點，image_url part 自動轉為 vlm 推理
resp = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "詳細描述這張圖片"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ],
    }],
)

# ASR（Whisper 式）
with open("clip.wav", "rb") as f:
    resp = client.audio.transcriptions.create(model="whisper-large-v3", file=f)
```

各模型類型 → 端點對應：

| 模型 task_family | 端點 |
|-----|-----|
| `text-generation` | `POST /v1/chat/completions`、`POST /v1/completions` |
| `vlm` / `image-captioning` | `POST /v1/chat/completions`（帶 `image_url` part） |
| `asr` | `POST /v1/audio/transcriptions` |
| `tts` | `POST /v1/audio/speech`（回二進位 WAV）、`POST /inference/tts`（回 base64 JSON） |
| `embedding` | `POST /v1/embeddings` |
| `rerank` | `POST /v1/rerank`（jina/xinference 式） |
| 其他（ocr / audio-classification / video / document） | 走原生 `POST /inference/infer` |

Chat messages 的處理是 runtime-aware 的：Ollama 模型走 daemon 的 `/api/chat`（由 daemon 套 chat template），HF 模型用 tokenizer 的 `apply_chat_template`。

> `stream: true` 對 **text-generation 模型**（ollama / hf-transformers）為真·逐 token SSE 串流；
> 其他 task（vlm 等）自動 fallback 為單 chunk 偽串流以維持 client 相容。

---

## 閒置卸載與 GPU 釋放

ollama 式的資源回收：模型閒置超過存活時間就自動卸載、釋放 GPU，下次請求再自動載回。

三個控制層（優先序由高到低）：

1. **逐請求 `keep_alive`**（`options.keep_alive`，或 `/v1` 請求的頂層 `keep_alive` 欄位）
   - 秒數或字串（`"30s"` / `"5m"` / `"1h"`）
   - `0` = 推理完立即卸載；負值 = 常駐不回收
2. **adapter 預設**（`inference.yaml`）
   - `engines.transformers.idle_timeout`（預設 300 秒；`<=0` 停用自動卸載）
   - `engines.ollama.keep_alive`（透傳給 Ollama daemon，讓 daemon 的 GPU 釋放節奏一致）
3. **LRU 容量**（`max_cached_models`）— 快取滿時仍會淘汰最舊模型（跳過推理中的模型）

背景 reaper（`idle_reaper.interval`，預設 30 秒掃一次）負責回收過期模型並 `torch.cuda.empty_cache()`。推理中的模型（in-flight > 0）不會被回收。

```bash
# 這次推理完 10 分鐘後才卸載
curl -X POST localhost:8010/inference/infer -H "Content-Type: application/json" -d '{
  "model_name": "gemma-2-2b-it",
  "data": {"inputs": "hi"},
  "options": {"keep_alive": "10m"}
}'

# 觀察快取狀態（idle_seconds / expires_in_seconds / in_flight）
curl localhost:8010/inference/loaded-models
```

對 Ollama 模型，`keep_alive` 會一併透傳給 daemon 的 `/api/generate`、`/api/chat`；手動 `POST /inference/unload-model` 也會要求 daemon 立即釋放該模型的 VRAM。

---

## 部署與設定

### 前置準備

- **Docker** + Docker Compose
- **NVIDIA GPU + nvidia-container-toolkit**（推理用，可選但建議）
- **LakeFS** 實例（或共用平台級的 LakeFS）
- **Ollama** daemon（若要跑 Ollama 模型；可同 host）
- 開發本機：Python 3.10+，建議用 [UV](https://github.com/astral-sh/uv)

### 設定環境

1. 複製 `.env.example` 為 `.env`，至少填入：

   ```ini
   POSTGRES_USER=mlflow
   POSTGRES_PASSWORD=mlflow
   POSTGRES_DB=mlflow

   MLFLOW_S3_ENDPOINT_URL=http://<LAKEFS_HOST>:8001
   AWS_ACCESS_KEY_ID=<LAKEFS_ACCESS_KEY>
   AWS_SECRET_ACCESS_KEY=<LAKEFS_SECRET_KEY>

   MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/
   MLFLOW_BACKEND_STORE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@mlflow-postgres:5432/${POSTGRES_DB}

   OLLAMA_API_BASE=http://host.docker.internal:11434
   HF_TOKEN=<HuggingFace token>

   AI_LIFECYCLE_API_HOST=0.0.0.0
   PORT_AI_LIFECYCLE_API=8010
   ```

2. 確認 `src/core/configs/base.yaml` 與 `inference.yaml` 中 `mlflow.tracking_uri`、`lakefs.endpoint`、`ollama.api_base` 指向對的位址。容器內訪問 host 服務通常用 `host.docker.internal`。

### 啟動

```bash
docker compose up --build -d
```

### 服務介面

- FastAPI Swagger UI：<http://localhost:8010/docs>
- MLflow UI：<http://localhost:5000>

---

## 範例工作流

### 工作流 A：Ollama 模型快速註冊與推理

#### 1. 確認本地 Ollama 模型

```bash
ollama list
# 或
curl "http://localhost:8010/models/local?model_source=ollama"
```

#### 2. 註冊到 MLflow

`POST /models/register_ollama`

```json
{
  "local_model_name": "gemma4:12b",
  "task": "text-generation",
  "registered_name": "gemma4-12b-ollama",
  "stage": "production"
}
```

註冊端會自動寫入 spec tag：

```
runtime              = ollama
task_family          = text-generation
ollama_model_name    = gemma4:12b
physical_path        = gemma4:12b
trust_remote_code    = true
```

> v3.1 起新註冊端**只寫 v3 schema**（不再回寫 `inference_engine` / `inference_task` 舊 tag）。讀取端對舊註冊仍相容，舊模型不需要遷移。

#### 3. 推理

```json
POST /inference/infer
{
  "task": "text-generation",
  "model_name": "gemma4-12b-ollama",
  "data": {"inputs": "請用繁體中文寫一段對 MLOps 的介紹"},
  "options": {"temperature": 0.7, "max_length": 500}
}
```

回應：

```json
{
  "success": true,
  "result": {
    "response": "MLOps 是一套...",
    "model": "gemma4:12b",
    "metadata": {"total_duration": 3850000000, "eval_count": 245}
  },
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "gemma4-12b-ollama",
  "model_version": "1",
  "processing_time": 3.85,
  "timestamp": 1714195200.0
}
```

### 工作流 B：HuggingFace → LakeFS → MLflow（標準 LLM）

#### 1. 下載到本地暫存

`POST /models/download`

```json
{"model_source": "huggingface", "model_name": "google/gemma-2-2b-it"}
```

#### 2. 上傳至 LakeFS（取得 commit id）

`POST /models/upload_to_lakefs`

```json
{"repo_name": "gemma-2-2b-it", "local_model_name": "google_gemma-2-2b-it"}
```

#### 3. 註冊到 MLflow（**填 spec 欄位**）

`POST /models/register_from_lakefs`

```json
{
  "registered_name": "gemma-2-2b-it",
  "task": "text-generation",
  "engine": "transformers",
  "model_params": 2,
  "lakefs_repo": "gemma-2-2b-it",
  "stage": "staging",
  "pipeline_task": "text-generation",
  "trust_remote_code": true
}
```

#### 4. 推理

```json
POST /inference/infer
{
  "task": "text-generation",
  "model_name": "gemma-2-2b-it",
  "data": {"inputs": "Explain MLOps in 3 bullet points."},
  "options": {"max_new_tokens": 200, "temperature": 0.7}
}
```

### 工作流 C：HuggingFace → LakeFS → MLflow（VLM 例：Qwen2.5-VL）

差別只在註冊時填 `model_class` + `processor_class`。

`POST /models/register_from_lakefs`

```json
{
  "registered_name": "qwen2.5-vl-7b",
  "task": "vlm",
  "engine": "transformers",
  "model_params": 7,
  "lakefs_repo": "qwen-vl-models",
  "stage": "production",
  "model_class": "Qwen2_5_VLForConditionalGeneration",
  "processor_class": "AutoProcessor",
  "trust_remote_code": true,
  "quantization": "4bit"
}
```

推理：

```json
POST /inference/infer
{
  "task": "vlm",
  "model_name": "qwen2.5-vl-7b",
  "data": {
    "image": "/data/images/scene.jpg",
    "prompt": "詳細描述這張圖片"
  },
  "options": {"max_new_tokens": 256}
}
```

`image` 欄位接受本機路徑、PIL.Image、base64、`data:image/...` URL。

### 工作流 C2：InternVL（10/11/12 模組共用的 VLM）

InternVL 走 `.chat()` 介面 + 動態切圖，07 已內建專屬 handler，註冊時不需寫任何程式碼：

`POST /models/register_from_lakefs`

```json
{
  "registered_name": "internvl3.5-1b",
  "task": "vlm",
  "engine": "transformers",
  "model_params": 1,
  "lakefs_repo": "internvl-models",
  "stage": "production",
  "model_class": "AutoModel",
  "processor_class": "AutoTokenizer",
  "torch_dtype": "bf16",
  "trust_remote_code": true
}
```

handler 由 `model_class`/`physical_path` 中的 `InternVL` 字樣自動偵測；
也可顯式指定 `"custom_handler": "internvl"`（內建短名）。推理同工作流 C（task=vlm）。

### 工作流 F：embedding / rerank / TTS 服務化

```json
// 註冊 bge-m3（17-hybrid-search 使用的 embedding 模型）
POST /models/register_from_lakefs
{
  "registered_name": "bge-m3",
  "task": "embedding",
  "engine": "transformers",
  "model_params": 0.6,
  "lakefs_repo": "bge-m3",
  "stage": "production",
  "model_class": "AutoModel",
  "processor_class": "AutoTokenizer"
}

// 註冊 bge-reranker-v2-m3
POST /models/register_from_lakefs
{
  "registered_name": "bge-reranker-v2-m3",
  "task": "rerank",
  "engine": "transformers",
  "model_params": 0.6,
  "lakefs_repo": "bge-reranker-v2-m3",
  "stage": "production",
  "model_class": "AutoModelForSequenceClassification",
  "processor_class": "AutoTokenizer"
}
```

調用（OpenAI / jina 相容）：

```bash
# embedding
curl -X POST localhost:8010/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["什麼是 MLOps？", "向量檢索"]}'

# rerank
curl -X POST localhost:8010/v1/rerank -H "Content-Type: application/json" \
  -d '{"model": "bge-reranker-v2-m3", "query": "什麼是 MLOps",
       "documents": ["MLOps 是機器學習運維", "今天天氣很好"], "top_n": 1}'

# TTS（09-audio-processing 的 audio_tts_service 走這個端點）
curl -X POST localhost:8010/inference/tts -H "Content-Type: application/json" \
  -d '{"text": "歡迎使用 Raptor", "output_format": "wav"}'
```

TTS 端點的模型名可省略（預設取環境變數 `DEFAULT_TTS_MODEL`，預設值 `vibevoice-tts`）。
標準 HF TTS 模型（SpeechT5 / VITS / Bark）註冊時填 `pipeline_task="text-to-speech"` 即可；
VibeVoice 等非標準介面需另寫 custom_handler（見工作流 D）。

### 工作流 D：客製模型（custom_handler）

當預設 handler 不夠用（例如自家訓練的 multi-head 模型、特殊 chat template、非標準 generate loop）：

1. 寫一個自訂 handler 類，繼承 `src.inference.handlers.base.BaseHandler`，實作 `run(loaded, spec, data, options)`。
2. 註冊時帶 `custom_handler="myorg.handlers.x:CustomHandler"`。
3. 第一次推論時 service 會 lazy import。

詳見 [USAGE_GUIDE — 場景 G](./src/inference/USAGE_GUIDE.md#場景-g--完全特殊的模型custom_handler)。

### 工作流 G：直接註冊 NAS 上已下載的模型（測試用）

`MODELS_TMP_ROOT` 指向掛載的 NAS 模型庫（例如 `/models`）後：

```bash
curl "http://localhost:8010/models/local"      # name 例："google/gemma-4-12B"
curl -X POST "http://localhost:8010/models/register_local" -H "Content-Type: application/json" -d '{
  "registered_name": "gemma4-12b-local", "local_model_name": "google/gemma-4-12B",
  "task": "text-generation", "model_class": "AutoModelForCausalLM", "processor_class": "AutoTokenizer",
  "torch_dtype": "bf16", "model_params": 8, "stage": "staging", "max_output_tokens": 2048
}'
```

`physical_path` 是 07 容器內的路徑，只有掛載同一模型庫的 07 實例能載入，且本地權重可被就地修改；
正式模型仍建議走工作流 B（lakeFS commit 不可變）。整套隔離測試環境（MLflow 新版 + 全新 postgres +
NAS 模型庫）見 `docker-compose.m07-test.yml` 檔頭說明。

### 工作流 E：階段切換

`POST /models/transition_stage`

```json
{
  "model_name": "gemma4-12b-ollama",
  "version": "2",
  "stage": "production",
  "archive_existing_versions": true
}
```

`archive_existing_versions=true` 時會把該名字下其他 production 版本封存為 archived。

---

## API 端點清單

完整互動式文檔在 <http://localhost:8010/docs>。

### 模型管理 `/models`

| 端點 | 方法 | 說明 |
|-----|-----|------|
| `/download` | POST | 從 HuggingFace 或 Ollama 拉取模型到 tmp |
| `/local` | GET | 列出本地模型庫（`MODELS_TMP_ROOT`，可指向 NAS；支援 `<org>/<model>` 兩層）；`?model_source=ollama` 篩 Ollama |
| `/upload_to_lakefs` | POST | 上傳本地模型到 LakeFS |
| `/lakefs_repos` | GET | 列出 LakeFS 儲存庫 |
| `/register_from_lakefs` | POST | 從 LakeFS commit 註冊到 MLflow（**支援 spec 欄位**） |
| `/register_ollama` | POST | 註冊本地 Ollama 模型到 MLflow |
| `/register_local` | POST | 把本地模型庫（NAS）中已下載的 HF 模型直接註冊（`physical_path` 為本地路徑，不經 lakeFS） |
| `/set_model_limits` | POST | 設定某版本的 `max_output_tokens` / `context_window_tokens` |
| `/registered_in_mlflow` | GET | 列出 MLflow 已註冊模型 |
| `/registered_in_mlflow/{model_name}` | GET | 取得單一模型詳情 |
| `/transition_stage` | POST | 切換階段（production/staging/archived/none） |
| `/batch_download` | POST | 批量下載 |
| `/stats` | GET | 模型統計 |
| `/health` | GET | 服務健康檢查 |

### OpenAI 相容 `/v1`

| 端點 | 方法 | 說明 |
|-----|-----|------|
| `/models` | GET | 列出已註冊模型（OpenAI 格式，metadata 帶 task_family/runtime/stage） |
| `/chat/completions` | POST | Chat（text-generation 與 vlm 模型通用；支援 `keep_alive` 擴充欄位） |
| `/completions` | POST | 傳統 prompt completion（text-generation 模型） |
| `/audio/transcriptions` | POST | 語音轉文字（asr 模型，multipart 上傳） |
| `/audio/speech` | POST | 文字轉語音（tts 模型；回二進位 WAV） |
| `/embeddings` | POST | 文本向量化（embedding 模型；OpenAI 格式） |
| `/rerank` | POST | 文檔重排（rerank 模型；jina/xinference 格式） |

### 推理服務 `/inference`

| 端點 | 方法 | 說明 |
|-----|-----|------|
| `/infer` | POST | **統一推理端點**（所有任務；`task` 可省略，`model_version`/`model_alias` 選版，`options.keep_alive` 控制卸載） |
| `/model-info` | GET | 只讀的模型能力與輸出上限查詢（不載入模型） |
| `/tts` | POST | 文字轉語音便捷端點（`{"text", "voice", "speed", "output_format"}`；模型可省略 — 用 `DEFAULT_TTS_MODEL`） |
| `/health` | GET | service + adapter 狀態 |
| `/stats` | GET | 推理計數、成功率 |
| `/supported-tasks` | GET | task family 與必填欄位 |
| `/loaded-models` | GET | 列出每個 adapter LRU 中的模型（含 idle/expiry/in-flight 狀態） |
| `/cache/clear` | POST | 卸載所有模型並清 GPU cache |
| `/unload-model?model_name=X` | POST | 卸載指定模型的所有快取版本 |
| `/unload-all-models` | POST | 卸載全部 |

### 註冊端 `/models/register_from_lakefs` 主要欄位（spec-driven）

| 欄位 | 必填 | 說明 |
|-----|-----|------|
| `registered_name` | ✅ | MLflow 中的註冊名 |
| `task` | ✅ | 任務（舊名仍可用，會 canonicalize） |
| `engine` | ✅ | `transformers` / `ollama`（→ 寫 `runtime`） |
| `model_params` | ✅ | 參數量（B 為單位，用於估 VRAM） |
| `lakefs_repo` | ✅ | LakeFS 儲存庫名 |
| `commit_id` | – | 不填則取分支最新 commit |
| `stage` | – | 註冊後直接切到的階段 |
| `quantization` | – | `4bit` / `8bit` |
| `model_class` | (HF, a) | transformers class 名 |
| `processor_class` | – | processor / tokenizer class |
| `pipeline_task` | (HF, a) | 走 pipeline 捷徑 |
| `trust_remote_code` | – | 預設 `true` |
| `custom_handler` | – | dotted 路徑 |

(a) HF runtime：`model_class` 與 `pipeline_task` 至少要填一個。

---

## 常見問題

**Q1: 我用舊 schema 註冊的 Ollama 模型還能推理嗎？**
能。舊 tag（`inference_engine=ollama` + `ollama_model_name=...`）會被自動讀進新 spec。

**Q2: 我用舊 schema 註冊的 HF 模型呢？**
若 v2 註冊時走的是路徑字串匹配（沒 `model_class`、沒 `pipeline_task`），新 service 第一次推論會回 `ValidationError`，提示要補 tag。可用 `MlflowClient.set_model_version_tag` 補；範例腳本見 [USAGE_GUIDE — 從 v2.x 遷移](./src/inference/USAGE_GUIDE.md#從-v2x-遷移)。

**Q3: `task` 欄位可以填什麼？**
canonical 名（`text-generation` / `vlm` / `asr` / `ocr` / `audio-classification` / `video-analysis` / `document-analysis` / `image-captioning` / `tts` / `embedding` / `rerank`）或舊名/別名（`text-generation-ollama` / `text-generation-hf` / `asr-hf` / `vad-hf` / `ocr-hf` / `audio-transcription` / `scene-detection` / `video-summary` / `text-to-speech` / `feature-extraction` / `reranker`）。後者會自動轉成前者。

**Q4: 推理時還要傳 `engine` 嗎？**
不需要。從 v3.0 起 runtime 由 MLflow tag 決定，request 中的 `engine` 欄位被忽略（保留是為了不破壞既有 client）。

**Q5: GPU OOM 怎麼辦？**
立即 `POST /inference/cache/clear`。然後考慮：(a) 註冊時加 `quantization: "4bit"`；(b) 在 `inference.yaml.engines.transformers` 調小 `max_cached_models` 或 `idle_timeout`；(c) 給該模型較小的 `max_new_tokens`；(d) 對一次性請求帶 `options.keep_alive: 0`（用完即卸）。

**Q5b: 模型閒置時 GPU 什麼時候會被釋放？**
HF 模型：閒置超過 `idle_timeout`（預設 300 秒）後，背景 reaper（每 `idle_reaper.interval` 秒掃一次）自動卸載並 `torch.cuda.empty_cache()`。Ollama 模型：由 daemon 依 `keep_alive`（預設同為 300 秒）自行釋放。兩者都能用逐請求 `keep_alive` 覆寫；`GET /inference/loaded-models` 可看每個模型的 `idle_seconds` 與 `expires_in_seconds`。

**Q6: 我有一個 transformers 不直接支援的模型架構（自家訓練的）怎麼接？**
寫 `custom_handler`：繼承 `BaseHandler`、實作 `run()`、註冊時把 dotted 路徑塞到 `custom_handler` tag。詳見 [USAGE_GUIDE — 場景 G](./src/inference/USAGE_GUIDE.md#場景-g--完全特殊的模型custom_handler)。

**Q7: 服務啟動順序？**
`docker compose up` 會以 `mlflow-postgres → mlflow → api` 的依賴順序啟動。若 LakeFS 是外部服務，請先確認可用。

**Q8: 健康檢查端點？**
- 整體：`GET http://localhost:8010/health`
- 模型管理：`GET /models/health`
- 推理：`GET /inference/health`
- MLflow：`GET http://localhost:5000/health`

---

## 延伸文件

- [推理模組設計](./src/inference/README.md)
- [接入流程示範](./src/inference/USAGE_GUIDE.md)
- [v2 → v3 重構紀要](./src/inference/REFACTORING_SUMMARY.md)
- [UV 依賴管理](./docs/UV_DEPENDENCY_MANAGEMENT.md)
