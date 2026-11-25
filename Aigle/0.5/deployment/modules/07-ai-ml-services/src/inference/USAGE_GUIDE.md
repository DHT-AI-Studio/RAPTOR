# Inference 模組 — 使用指南

> v3.2 — Spec-driven 二層架構 + OpenAI 相容層（/v1）+ 閒置自動卸載（keep_alive）

本指南示範如何把不同類型的模型（Ollama / HuggingFace / 客製模型）接入推理服務、設定 MLflow tag、以及進行推理。架構與設計思想請見 [README.md](./README.md)。

## 目錄

1. [心智模型](#心智模型)
2. [註冊端：把「怎麼跑」寫進 MLflow](#註冊端把怎麼跑寫進-mlflow)
3. [推理端：呼叫 InferenceService](#推理端呼叫-inferenceservice)
4. [模型接入流程（按類型）](#模型接入流程按類型)
5. [推理選項](#推理選項)
6. [模型資源管理](#模型資源管理)
7. [監控與健康檢查](#監控與健康檢查)
8. [擴展](#擴展)
9. [從 v2.x 遷移](#從-v2x-遷移)

---

## 心智模型

```
        註冊時                                     推理時
   ┌────────────┐                          ┌──────────────────┐
   │ 模型來源    │                          │ /inference/infer │
   │ HF / Ollama│                          │ model_name(+task)│
   └─────┬──────┘                          └────────┬─────────┘
         │ POST /models/register_*                  │
         │  + model_class / processor_class /       │  resolve ModelSpec
         │    pipeline_task / custom_handler        │  from MLflow tags
         ▼                                          ▼
   ┌──────────────────────────────────────────────────────┐
   │             MLflow Model Registry                    │
   │  tags: runtime / task_family / model_class / ...     │
   └──────────────────────────────────────────────────────┘
                             │
                             ▼
              Adapter dispatch by spec.runtime
              Handler dispatch by spec.task_family / custom_handler
```

**設計原則**：模型「怎麼跑」的決策只在註冊時做一次，推論端永遠照 spec 執行。新增模型架構不需要改推論層程式碼。

---

## 註冊端：把「怎麼跑」寫進 MLflow

`register_lakefs_to_mlflow` 與 `register_ollama_to_mlflow` 已支援 spec-driven schema。共通必填：

| 參數 | 說明 |
|-----|-----|
| `task` | 任務名（舊名仍可用，會自動 canonicalize） |
| `engine` | `transformers` / `ollama`（→ 寫 `runtime` tag） |

### HF Transformers 模型額外欄位

| 參數 | 必填 | 範例 |
|-----|-----|------|
| `model_class` | (a) | `Qwen2_5_VLForConditionalGeneration`、`AutoModelForCausalLM`、`AutoModelForSpeechSeq2Seq` |
| `processor_class` | – | `AutoProcessor`、`AutoTokenizer`、`AutoFeatureExtractor` |
| `pipeline_task` | (a) | `text-generation`、`automatic-speech-recognition`、`audio-classification`、`image-to-text` |
| `trust_remote_code` | – | 預設 `true` |
| `custom_handler` | – | dotted 路徑，覆寫預設 handler |

(a) `model_class` 與 `pipeline_task` 至少要填一個。

### Ollama 模型

不需要填上述 HF 欄位；adapter 從 `ollama_model_name` tag（既有）拉模型。

---

## 推理端：呼叫 InferenceService

### Python in-process

```python
from src.inference import inference_service

result = inference_service.infer(
    task=None,  # 可省略 — 用模型註冊的 task_family；有填會與註冊值核對
    model_name="gemma4-12b-ollama",
    data={"inputs": "請寫一首關於春天的詩"},
    options={"max_length": 200, "temperature": 0.7},
)

print(result["result"]["response"])
print(f"runtime={result['engine']} took {result['processing_time']:.2f}s")
```

注意：模型已註冊時 runtime 由 spec 決定，`engine` 不影響；只有模型未註冊時 `engine="ollama"` 才會
把 `model_name` 當 Ollama daemon tag 直接呼叫。需要固定版本時帶 `model_version="3"` 或
`model_alias="production"`（互斥；未帶時使用最大版本號）。

### HTTP

```bash
# task 可省略（xinference 式：只給模型名即可調用）
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gemma4-12b-ollama",
    "data": {"inputs": "請寫一首關於春天的詩"},
    "options": {"max_tokens": 200, "temperature": 0.7}
  }'
```

回應頂層的 `usage`（`prompt_tokens` / `completion_tokens` / `total_tokens`）是實際 token 計數；
查詢某版本的能力與輸出上限用 `GET /inference/model-info?model_name=...&model_alias=production`
（不載入模型）。

`text-generation` 的 `data` 也接受 OpenAI 式 `messages`（取代 `inputs`）：
Ollama 走 daemon 的 `/api/chat`、HF 用 tokenizer 的 `apply_chat_template`，
由各 runtime 套用模型自己的 chat template：

```json
{
  "model_name": "gemma4-12b-ollama",
  "data": {"messages": [
    {"role": "system", "content": "你是一位詩人"},
    {"role": "user", "content": "寫一首關於春天的詩"}
  ]}
}
```

### OpenAI 相容層（/v1）

外部應用（或任何 OpenAI SDK）建議直接走 `/v1`，不需要知道 task / runtime：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8010/v1", api_key="not-needed")
client.chat.completions.create(model="gemma4-12b-ollama",
                               messages=[{"role": "user", "content": "你好"}])
```

端點清單與 VLM / ASR 用法見 [專案 README — OpenAI 相容 API](../../README.md#openai-相容-api)。

### 例外處理

```python
from src.inference.exceptions import (
    InferenceError, ValidationError, ModelNotFoundError, MaxTokensExceededError,
    RegistryUnavailableError, RuntimeUnavailableError, InferenceTimeoutError,
    ResourceExhaustedError, error_detail,
)

try:
    result = inference_service.infer(...)
except ValidationError as e:
    # 400 — data 缺欄位、參數不合法或衝突；e.code 區分 invalid_parameter / conflicting_parameters ...
    ...
except ModelNotFoundError as e:
    # 404 — e.code: model_not_found / model_version_not_found / model_alias_not_found
    ...
except MaxTokensExceededError as e:
    # 422 — e.model_limit 為該版本的輸出上限
    ...
except InferenceTimeoutError as e:
    # 504 — runtime 逾時
    ...
except (RegistryUnavailableError, RuntimeUnavailableError, ResourceExhaustedError) as e:
    # 503 — registry / runtime 無法連線，或 GPU/RAM 不足
    ...
except InferenceError as e:
    # 500 — 其他載入／執行錯誤；error_detail(e) 可取得 (HTTP status, detail dict)
    ...
```

---

## 模型接入流程（按類型）

### 場景 A — Ollama 模型（最快路徑）

只要本機 Ollama daemon 上已有模型，下面三步搞定：

```bash
# 1. 確認 daemon 上有
ollama list

# 2. 註冊到 MLflow（仍用既有的 register_ollama 端點）
curl -X POST "http://localhost:8010/models/register_ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "local_model_name": "gemma4:12b",
    "task": "text-generation",
    "registered_name": "gemma4-12b-ollama",
    "stage": "production"
  }'

# 3. 推理
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation",
    "model_name": "gemma4-12b-ollama",
    "data": {"inputs": "你好"}
  }'
```

寫入的 spec tag 自動包含 `runtime=ollama` / `task_family=text-generation` / `ollama_model_name=gemma4:12b`。

### 場景 B — 標準 HF LLM（如 Llama / Gemma / Qwen 純文字版）

走 `pipeline_task="text-generation"` 即可，不需指定 `model_class`：

```json
POST /models/register_from_lakefs
{
  "registered_name": "gemma-2b-it",
  "task": "text-generation",
  "engine": "transformers",
  "model_params": 2,
  "lakefs_repo": "gemma-2b-it",
  "stage": "staging",
  "pipeline_task": "text-generation",
  "trust_remote_code": true
}
```

推理請求與 Ollama 完全相同（task + model_name + data）；service 會從 spec 知道要走 HF runtime。

### 場景 C — 視覺語言模型 Qwen2.5-VL

關鍵：`model_class` 填正確的 class 名，handler 會自動走 Qwen-VL 路徑（內含 `apply_chat_template` + `qwen_vl_utils.process_vision_info`）：

```json
POST /models/register_from_lakefs
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

```bash
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "vlm",
    "model_name": "qwen2.5-vl-7b",
    "data": {
      "image": "/data/images/scene.jpg",
      "prompt": "詳細描述這張圖片"
    },
    "options": {"max_new_tokens": 256}
  }'
```

`image` 欄位接受：本機路徑、PIL.Image、base64 字串、`data:image/...` URL。

### 場景 D — 通用 VLM（LLaVA / BLIP-2 等）

通用 VLMHandler 走 `processor(images, text, return_tensors="pt")` → `model.generate` → `processor.batch_decode`：

```json
{
  "registered_name": "llava-1.5-7b",
  "task": "vlm",
  "engine": "transformers",
  "model_params": 7,
  "lakefs_repo": "llava-models",
  "model_class": "LlavaForConditionalGeneration",
  "processor_class": "AutoProcessor"
}
```

### 場景 E — Whisper / 通用 ASR

最簡：走 pipeline 路徑。

```json
{
  "registered_name": "whisper-large-v3",
  "task": "asr",
  "engine": "transformers",
  "model_params": 1.5,
  "lakefs_repo": "whisper-models",
  "pipeline_task": "automatic-speech-recognition"
}
```

推理：

```json
{
  "task": "asr",
  "model_name": "whisper-large-v3",
  "data": {"audio": "/data/audio/clip.wav"},
  "options": {"language": "zh", "task": "transcribe"}
}
```

如果需要更細的 generate 參數控制（例如 batched）就改用 raw 路徑：填 `model_class="WhisperForConditionalGeneration"` + `processor_class="AutoProcessor"`。

### 場景 F — OCR (TrOCR)

```json
{
  "registered_name": "trocr-large-printed",
  "task": "ocr",
  "engine": "transformers",
  "model_params": 0.5,
  "lakefs_repo": "trocr-models",
  "pipeline_task": "image-to-text"
}
```

### 場景 G — 完全特殊的模型（custom_handler）

當預設 handler 不夠用時（例如自家訓練的 multi-head 模型、需要特殊 chat template、非標準 generate loop），寫一個自訂 handler：

```python
# myorg/handlers/custom_vlm.py
from src.inference.handlers.base import BaseHandler

class CustomVLMHandler(BaseHandler):
    def run(self, loaded, spec, data, options):
        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        # 你的 encode 邏輯
        # 你的 generate 邏輯
        # 你的 decode 邏輯

        return {
            "response": decoded_text,
            "metadata": {"output_length": int(outputs.shape[-1])},
        }
```

註冊時加上 `custom_handler`：

```json
{
  "registered_name": "my-custom-vlm",
  "task": "vlm",
  "engine": "transformers",
  "model_params": 13,
  "lakefs_repo": "my-vlm",
  "model_class": "MyCustomVLMForConditionalGeneration",
  "processor_class": "AutoProcessor",
  "custom_handler": "myorg.handlers.custom_vlm:CustomVLMHandler"
}
```

第一次推論時 service 會 lazy import 該類；handler 必須繼承 `BaseHandler`。

---

## 推理選項

### 通用（HF + Ollama 共通可用）

| 選項 | 說明 |
|-----|------|
| `max_new_tokens` | 新生成 token 數（推薦這個） |
| `max_length` | 總長度（含 prompt） |
| `temperature` | 取樣溫度 |
| `top_p` | nucleus sampling |
| `top_k` | top-k sampling |
| `do_sample` | 預設 `true` |
| `keep_alive` | 本次推理後模型存活時間：秒數或 `"30s"`/`"5m"`/`"1h"`；`0` = 用完立即卸載、負值 = 常駐。未填用 `idle_timeout` 預設（見下方[閒置自動卸載](#閒置自動卸載keep_alive)） |

### HF 專屬

| 選項 | 說明 |
|-----|------|
| `num_beams` | beam search |
| `repetition_penalty` | 重複懲罰 |
| `length_penalty` | 長度懲罰 |
| `early_stopping` | beam 提前停止 |

### Ollama 專屬

| 選項 | 說明 |
|-----|------|
| `repeat_penalty` | 重複懲罰 |
| `stop` | 停止序列（list） |
| `mirostat` / `mirostat_eta` / `mirostat_tau` | mirostat 取樣 |
| `num_ctx` / `num_batch` / `num_gpu` / `num_thread` | 引擎層參數 |

### 任務特定

| 任務 | 選項 |
|-----|------|
| ASR | `language`、`task`（`transcribe` / `translate`） |
| audio-classification | `top_k` |

---

## 模型資源管理

每個 adapter 內部維護一個 LRU cache，並帶閒置逾時。以下端點操作這個 cache。

### 閒置自動卸載（keep_alive）

ollama 式資源回收：模型閒置超過存活時間，背景 reaper（預設每 30 秒掃一次）自動卸載並清 GPU cache；下次請求再自動載回。優先序：

1. 逐請求 `options.keep_alive`（`0` = 用完即卸、負值 = 常駐）
2. `inference.yaml` 的 `engines.transformers.idle_timeout`（預設 300 秒；`<=0` 停用）
   / `engines.ollama.keep_alive`（透傳給 daemon）
3. LRU 容量 `max_cached_models`（快取滿時淘汰最舊；跳過推理中的模型）

```bash
# 一次性請求：推理完立即釋放 GPU
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen2.5-vl-7b",
       "data": {"image": "/data/x.jpg", "prompt": "描述"},
       "options": {"keep_alive": 0}}'
```

推理中的模型（`in_flight > 0`）不會被 reaper 或 LRU 回收。

### 卸載指定模型

```bash
curl -X POST "http://localhost:8010/inference/unload-model?model_name=qwen2.5-vl-7b"
```

回應包含 `unloaded_runtimes`（被卸載的 adapter 名）與 `gpu_memory_freed`。

### 卸載全部

```bash
curl -X POST "http://localhost:8010/inference/unload-all-models"
# 或同義
curl -X POST "http://localhost:8010/inference/cache/clear"
```

### 列出已載入

```bash
curl "http://localhost:8010/inference/loaded-models"
```

```json
{
  "total": 2,
  "by_runtime": {
    "ollama": ["gemma4-12b-ollama"],
    "hf-transformers": ["qwen2.5-vl-7b"]
  },
  "details": {
    "hf-transformers": [{
      "model_name": "qwen2.5-vl-7b",
      "loaded_at": 1714195100.0,
      "last_used": 1714195180.0,
      "idle_seconds": 20.0,
      "expires_at": 1714195480.0,
      "expires_in_seconds": 280.0,
      "use_count": 3,
      "in_flight": 0
    }]
  },
  "timestamp": 1714195200.0
}
```

`expires_in_seconds` 為 `null` 表示常駐（`keep_alive` 負值或 `idle_timeout <= 0`）。

### LRU 大小調整

`src/core/configs/inference.yaml`：

```yaml
engines:
  transformers:
    max_cached_models: 2     # HF adapter 最多同時駐留幾個模型
    idle_timeout: 300        # 閒置幾秒後自動卸載（<=0 = 不自動卸載）

idle_reaper:
  enabled: true
  interval: 30               # reaper 掃描間隔（秒）
```

Ollama adapter 預設 LRU 大（64）且不做本地 idle 回收，因為模型不佔本地 RAM —
GPU 釋放由 daemon 依 `engines.ollama.keep_alive` 自行處理。

---

## 監控與健康檢查

```bash
curl "http://localhost:8010/inference/health"
# {"status": "healthy", "adapters_initialized": ["ollama", "hf-transformers"], ...}

curl "http://localhost:8010/inference/stats"
# {"stats": {"total": 1024, "success": 1018, "failed": 6, "success_rate": 0.994, ...}}

curl "http://localhost:8010/inference/supported-tasks"
# {"tasks": {"text-generation": {"required_fields": ["inputs | messages"]}, ...}}
```

---

## 擴展

### 新增 task family

1. 在 `spec.py` 的 `TASK_FAMILIES` 加新名稱
2. （若需要）在 `_TASK_ALIAS` 加舊名 → 新名映射
3. 在 `service.py` 的 `_REQUIRED_DATA_FIELDS` 加必填欄位
4. 在 `handlers/` 新增對應 handler，並在 `handlers/__init__.py` 的 `_DEFAULT` 表註冊

### 新增 runtime（例如 vLLM）

詳見 [README.md 擴展開發](./README.md#擴展開發)。

### 新增 handler 但不換預設

走 `custom_handler` tag，無需動 `_DEFAULT`。適合「同一個 task family 但這個模型要特殊處理」的情境。

---

## 從 v2.x 遷移

對外 API 形狀向下相容；以下是行為差異：

| 變化 | v2.x 行為 | v3.0 行為 | 影響 |
|-----|----------|----------|-----|
| `engine` 欄位 | 必填，與 task 嚴格綁定 | 可省略，由 MLflow tag 決定 | 既有 client 可繼續傳，會被忽略 |
| Task 名 | `text-generation-ollama`、`asr-hf`、`vad-hf`、`scene-detection`、`video-summary`、`audio-transcription` 各自獨立 | 上述名稱會被映射到 canonical task family | 既有呼叫照常運作；新呼叫建議用 canonical 名 |
| 回應欄位 | 無 `model_version` | 多了 `model_version` | 加值，不影響既有解析 |
| HF VLM 推理 | 靠路徑字串匹配 Qwen-VL | 靠 `model_class` tag 判斷 | **舊 HF 註冊需補 `model_class` tag** |
| 模型快取 | 全局 `ModelCache` 類 | 每個 adapter 內部 OrderedDict | 對外 API 不變 |
| 異常型別 | 同樣的 `ValidationError` / `ModelNotFoundError` / ... | 不變 | client 不需改 |

### 補 tag 的腳本範例

對於只有 v2 schema 的舊註冊模型：

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
# 例：把所有「以 transformers 註冊但沒有 model_class」的 HF text-gen 補上 pipeline_task
for rm in client.search_registered_models():
    for mv in client.search_model_versions(f"name='{rm.name}'"):
        tags = mv.tags
        if tags.get("inference_engine") == "transformers" and not tags.get("model_class") and not tags.get("pipeline_task"):
            if tags.get("inference_task", "").startswith("text-generation"):
                client.set_model_version_tag(rm.name, mv.version, "pipeline_task", "text-generation")
                print(f"patched {rm.name} v{mv.version}")
```

跑完後新版 service 就能 resolve spec。
