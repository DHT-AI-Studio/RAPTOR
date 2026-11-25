# Inference 模組

> Spec-driven 二層推理框架

版本: **v3.0.0**

## 目錄

- [概述](#概述)
- [核心設計](#核心設計)
- [架構](#架構)
- [ModelSpec Schema](#modelspec-schema)
- [快速開始](#快速開始)
- [API 文檔](#api-文檔)
- [支援任務](#支援任務)
- [配置說明](#配置說明)
- [擴展開發](#擴展開發)
- [故障排除](#故障排除)

---

## 概述

Inference 模組是 AI Model Lifecycle 平台的推理執行層。設計目標是讓「該模型該怎麼跑」的決策完全由 MLflow 註冊端決定，推論端只負責照規格執行；新增新模型架構不需要改推論層程式碼。

兩個關鍵抽象：

- **`ModelSpec`** — 從 MLflow tag 解析出的推理規格（`runtime` / `task_family` / `model_class` / `processor_class` / ...），是 service 與 adapter 之間的合約。
- **`BaseAdapter`** — 每個 runtime（Ollama、HF Transformers、未來的 vLLM）一個 adapter，吃 spec、回結果。

---

## 核心設計

| 設計原則 | 落實點 |
|---------|------|
| 註冊時記「怎麼跑」，推論時照做 | MLflow tag 寫入 `model_class` / `processor_class` / `pipeline_task` |
| 無路徑字串匹配 | adapter 透過 `getattr(transformers, spec.model_class)` 反射加載 |
| 預設行為與逃生口並存 | 預設走 task-family handler；非標準模型可透過 `custom_handler` 載入自訂類 |
| 兩層而非五層 | `service.py` + `adapters/*` + `handlers/*`；移除 router/executor/registry/cache 類 |
| 舊註冊不需重註 | `ModelSpec.from_mlflow_tags` 同時讀新舊 tag |

---

## 架構

```
┌──────────────────────────────────────────────────┐
│                    API Layer                     │
│            POST /inference/infer                 │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                InferenceService                  │
│   • 驗證 task / data 必填欄位                     │
│   • resolve_spec(model_name) ← MLflow tag        │
│   • dispatch by spec.runtime                     │
└──────────┬─────────────────────────┬─────────────┘
           │                         │
   spec.runtime=ollama       spec.runtime=hf-transformers
           │                         │
┌──────────▼──────────┐    ┌─────────▼────────────────────┐
│   OllamaAdapter     │    │   HFTransformersAdapter      │
│  • HTTP /api/generate│    │  • getattr(tx, model_class) │
│  • daemon 管模型     │    │  • LRU 快取載入結果          │
└─────────────────────┘    └─────────┬────────────────────┘
                                     │
                          spec.task_family / spec.custom_handler
                                     │
┌────────────────────────────────────▼─────────────┐
│                  Handlers                        │
│  TextGen / VLM (含 Qwen-VL 分支) / ASR / OCR /   │
│  AudioCls / VideoAnalysis / DocAnalysis / ...    │
│  • encode → generate → decode → 統一格式 dict     │
└──────────────────────────────────────────────────┘
```

### 對應檔案

```
src/inference/
├── __init__.py            # 對外 export
├── spec.py                # ModelSpec + canonicalize_task
├── service.py             # InferenceService 單例
├── exceptions.py          # 異常層次
├── vram_estimator.py      # 註冊時的 VRAM 估算
├── adapters/
│   ├── __init__.py        # get_adapter(runtime) 工廠
│   ├── base.py            # BaseAdapter ABC（含 LRU）
│   ├── ollama.py          # OllamaAdapter
│   └── hf_transformers.py # HFTransformersAdapter
└── handlers/
    ├── __init__.py        # resolve_handler(spec)
    ├── base.py            # BaseHandler ABC
    ├── text_generation.py
    ├── vlm.py             # 含 Qwen-VL 路徑（透過 spec 判斷）
    ├── asr.py
    ├── ocr.py
    ├── audio_classification.py
    ├── video_analysis.py
    └── document_analysis.py
```

---

## ModelSpec Schema

ModelSpec 在註冊時寫入 MLflow tag，推論時讀回。新舊 tag 同時存在以保持向下相容。

### 通用欄位

| 欄位 | 必填 | 說明 |
|-----|-----|------|
| `runtime` | ✅ | `ollama` 或 `hf-transformers` |
| `task_family` | ✅ | canonical 任務名（`text-generation`/`vlm`/`asr`/...） |
| `physical_path` | ✅ | `lakefs://repo/commit/` 或本機路徑或 HF repo-id |
| `lakefs_commit_id` | – | 用於溯源 |
| `trust_remote_code` | – | 預設 `true` |
| `quantization` | – | `4bit` / `8bit` / 不填 |
| `custom_handler` | – | dotted 路徑，覆寫預設 handler |

### HF Transformers 專屬

| 欄位 | 必填 | 說明 |
|-----|-----|------|
| `model_class` | (a) | transformers 中的模型 class 名稱（如 `AutoModelForCausalLM`、`Qwen2_5_VLForConditionalGeneration`） |
| `processor_class` | – | `AutoProcessor` / `AutoTokenizer` / `AutoFeatureExtractor` |
| `pipeline_task` | (a) | 若設了，走 `transformers.pipeline` 捷徑（如 `text-generation`、`automatic-speech-recognition`） |
| `torch_dtype` | – | `auto` / `fp16` / `bf16` / `fp32` |

(a) `model_class` 與 `pipeline_task` 至少要填一個。

### Ollama 專屬

| 欄位 | 必填 | 說明 |
|-----|-----|------|
| `ollama_model_name` | ✅ | daemon 內的真名（如 `qwen3:1.7b`），允許與 MLflow 註冊名不同 |

### 範例 — Qwen2.5-VL 7B 註冊

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
  "trust_remote_code": true
}
```

寫入 MLflow 後的 tag（節錄）：

```
runtime              = hf-transformers
task_family          = vlm
model_class          = Qwen2_5_VLForConditionalGeneration
processor_class      = AutoProcessor
trust_remote_code    = true
physical_path        = lakefs://qwen-vl-models/<commit>/
inference_engine     = transformers   # 舊 tag, 保留
inference_task       = vlm            # 舊 tag, 保留
```

---

## 快速開始

### Python 內呼叫

```python
from src.inference import inference_service

result = inference_service.infer(
    task="text-generation",
    model_name="qwen3-1.7b-ollama",
    data={"inputs": "請解釋什麼是深度學習"},
    options={"max_length": 200, "temperature": 0.7},
)
print(result["result"]["response"])
```

### HTTP

```bash
curl -X POST "http://localhost:8009/inference/infer" \
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

`engine` 欄位可省略：模型已註冊時 runtime 一律由 MLflow tag 決定（傳了也不影響）；
只有模型**未註冊**、且未指定 `model_version` / `model_alias` 時，`engine="ollama"` 才會把 `model_name`
當成 Ollama daemon 上的 tag 直接呼叫。MLflow 無法連線時不會觸發這個 fallback（回 503 `registry_unavailable`）。

---

## API 文檔

### POST `/inference/infer`

#### Request

```json
{
  "task": "text-generation",
  "model_name": "qwen3-1.7b-ollama",
  "model_alias": "production",
  "data": {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]},
  "options": {"max_tokens": 200, "temperature": 0, "stop_sequences": ["\n\n"]}
}
```

| 欄位 | 必填 | 說明 |
|-----|-----|------|
| `task` | – | 任務類型；省略時用模型註冊的 task_family（有填會核對）。舊名（`text-generation-ollama`/`asr-hf` 等）會自動 canonicalize |
| `model_name` | ✅ | 在 MLflow 註冊的模型名 |
| `model_version` | – | 指定版本號；與 `model_alias` 互斥。不存在回 404 `model_version_not_found`，不回退 |
| `model_alias` | – | 指定 alias（目前只有 production 轉換會建立 `production` alias）；請求開始時解析為固定版本。未設定回 404 `model_alias_not_found` |
| `data` | ✅ | 輸入資料；欄位依任務而異（見下表） |
| `options` | – | 推理選項（見[推理選項](#推理選項每次請求-options)） |
| `engine` | – | 模型未註冊時的 fallback 依據（`ollama`）；模型已註冊時不影響 runtime |

未帶 `model_version` / `model_alias` 時沿用「最大版本號」政策（不會默默改用 production）。
需要「列表、限制查詢、執行」三者同一版本的呼叫端（如 Module 35），應帶明確 selector。

#### data 欄位需求

| task | 必填 data 欄位 |
|-----|---------------|
| `text-generation` | `inputs` (str) 或 `messages` (OpenAI 式 list) |
| `vlm` / `image-captioning` | `image`, `prompt` |
| `asr` | `audio` |
| `ocr` | `image` |
| `audio-classification` | `audio` |
| `video-analysis` | `video` (`prompt` 可選) |
| `document-analysis` | `document` (`query` 可選) |

#### Response

```json
{
  "success": true,
  "result": {
    "response": "深度學習是...",
    "metadata": {"prompt_eval_count": 12, "eval_count": 245, "prompt_tokens": 12, "completion_tokens": 245}
  },
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "qwen3-1.7b-ollama",
  "model_version": "1",
  "usage": {"prompt_tokens": 12, "completion_tokens": 245, "total_tokens": 257},
  "processing_time": 1.83,
  "api_processing_time": 1.85,
  "timestamp": 1714195200.0
}
```

- `result.response` 只含新生成的文字（HF 不再回傳 prompt），並在第一個停止字串處截斷。
- `usage`：text-generation 的真實 token 計數。HF 以實際送入模型的 prompt token IDs 與新生成的 token IDs
  計數（含 EOS 等 runtime 原始統計；停止字串截斷不改寫計數）；Ollama 採 daemon 的 `prompt_eval_count` /
  `eval_count`。不同 runtime 的計數方式不保證完全一致。非文字 task、custom handler、或 runtime 沒回報
  計數時為 `null`（不以 0 冒充）。`/v1/*` 使用同一份 usage。
- 原有 `result.metadata` 欄位保留（HF raw 的 `output_length` 仍是含 prompt 的總長度，請改用 `usage`）。

#### 錯誤碼

`detail` 保留 `error_type` / `message`，新增穩定的 `code`（`/v1/*` 在 OpenAI envelope 的 `error.code` 放相同值）。

| HTTP | code | 條件 |
|-----|------|------|
| 400 | `invalid_request` / `unsupported_task` | data 缺欄位、task 未知或不符 |
| 400 | `invalid_parameter` | 共用生成參數值不合法（如 temperature 超出 0～2） |
| 400 | `conflicting_parameters` | 同義欄位值不同、`max_tokens` 與 `max_length` 混用、`model_version` 與 `model_alias` 同時提供、`temperature=0` 配 `do_sample=true`（含 `parameters`） |
| 400 | `policy_violation` | guardrail 阻擋 |
| 404 | `model_not_found` / `model_version_not_found` / `model_alias_not_found` | 模型／指定版本／alias 不存在（含 `model_name`） |
| 422 | `max_tokens_exceeded` | `max_tokens` 超過該版本已配置的 `max_output_tokens`（含 `model_limit`、`requested`）；生成前拒絕 |
| 500 | `model_load_failed` / `inference_failed` / `engine_error` | 其他載入／執行錯誤 |
| 503 | `registry_unavailable` | MLflow 無法連線（不當作模型不存在） |
| 503 | `runtime_unavailable` | Ollama daemon 無法連線（含連線逾時） |
| 503 | `resource_exhausted` | GPU/RAM 不足（含 `retry_after`=60） |
| 504 | `inference_timeout` | runtime 在 `timeout` 內未回應 |

串流（`/v1/*` `stream=true`）在 SSE headers 送出前發生的錯誤回上表的 HTTP 錯誤；送出後只能發送
`{"error": {"message", "type", "code"}}` 事件並以 `[DONE]` 結束。

### 管理端點

| 方法 | 路徑 | 用途 |
|-----|------|-----|
| GET | `/inference/health` | service + adapter 狀態 |
| GET | `/inference/stats` | 推理計數、成功率、已載入模型 |
| GET | `/inference/supported-tasks` | task family 與必填欄位 |
| GET | `/inference/model-info?model_name=X[&model_version=N\|&model_alias=A]` | 只讀的能力／限制查詢（不載入模型），與推論共用版本解析 |
| POST | `/models/set_model_limits` | 設定某版本的 `max_output_tokens` / `context_window_tokens` tag |
| GET | `/inference/loaded-models` | 列出每個 adapter 內 LRU cache 的模型（`by_runtime` 為名稱；`details` 含 `model_version`、idle/expiry/in-flight、`pending_unload`） |
| POST | `/inference/cache/clear` | 卸載所有模型並清 GPU cache |
| POST | `/inference/unload-model?model_name=X` | 卸載該名稱的**所有快取版本**；仍在推論中的實例等請求結束才釋放 |
| POST | `/inference/unload-all-models` | 卸載全部 |

---

## 支援任務

| task_family | 預設 handler | 引擎建議 | 輸入 | 輸出主欄位 |
|------------|-------------|---------|-----|----------|
| `text-generation` | `TextGenerationHandler` | ollama / hf | `{"inputs"}` 或 `{"messages"}` | `response` |
| `vlm` | `VLMHandler`（含 Qwen-VL 分支） | hf | `{"image","prompt"}` | `response` |
| `image-captioning` | `VLMHandler` | hf | `{"image"}` | `response` |
| `asr` | `ASRHandler` | hf | `{"audio"}` | `text` |
| `ocr` | `OCRHandler` | hf | `{"image"}` | `text` |
| `audio-classification` | `AudioClassificationHandler` | hf | `{"audio"}` | `classifications` |
| `video-analysis` | `VideoAnalysisHandler` | hf | `{"video"}` | `response` |
| `document-analysis` | `DocumentAnalysisHandler` | hf | `{"document"}` | `response` |

舊 task 名（`text-generation-ollama` / `text-generation-hf` / `asr-hf` / `ocr-hf` / `vad-hf` / `audio-transcription` / `scene-detection` / `video-summary`）皆會被 `canonicalize_task` 映射到上表，呼叫端不需修改。

---

## 配置說明

### 引擎配置（`src/core/configs/inference.yaml`）

```yaml
engines:
  ollama:
    base_url: "http://localhost:11434"
    timeout: 300
    auto_pull: true
  transformers:
    device: "auto"           # auto | cuda | cpu
    torch_dtype: "auto"      # auto | fp16 | bf16 | fp32
    trust_remote_code: true
    max_cached_models: 2     # adapter 內 LRU 大小
```

引擎 key 也可寫成 `hf-transformers`，效果等同 `transformers`。

### 推理選項（每次請求 `options`）

text-generation 的共用欄位由 `src/inference/options.py` 先正規化，Ollama、HF raw、HF pipeline、
串流路徑都會實際套用（不是只轉送 HTTP body），或明確回 400：

| 選項 | 適用 | 說明 |
|-----|------|-----|
| `max_tokens` | 兩者 | 新生成 token 上限（Ollama → `num_predict`、HF → `max_new_tokens`）；未指定時 HF 預設 512，模型有 `max_output_tokens` 時以其為上限 |
| `max_new_tokens` | 兩者 | `max_tokens` 的同義欄位；兩者都給且不同 → 400 |
| `stop` / `stop_sequences` | 兩者 | str 或 list[str]，同義；回應不含停止字串及其後文字。HF 以 StoppingCriteria 實際停止生成（跨 token、只比對生成部分）；串流扣住可能是停止字串前綴的尾端 |
| `temperature` | 兩者 | 0～2；0 = greedy（HF 改用 `do_sample=False`） |
| `max_length` | 兩者 | 保留既有語意：HF = 含 prompt 的總長度、Ollama = 生成上限；不可與 `max_tokens`/`max_new_tokens` 混用 |
| `top_p` / `top_k` / `do_sample` | 兩者 | 取樣參數（greedy 時 HF 不傳 sampling 參數） |
| `think` | 兩者 | 預設 false；Ollama 為 API 頂層欄位，HF 以 chat template 的 `enable_thinking` 套用（如 gemma4） |
| `keep_alive` | 兩者 | 見[閒置卸載](../../README.md#閒置卸載與-gpu-釋放) |
| `num_beams` / `repetition_penalty` / `length_penalty` | hf | 原樣傳給 `generate()` |
| `repeat_penalty` / `num_ctx` / `mirostat*` 等 | ollama | 原樣傳給 daemon options |
| `language` / `task` | ASR | Whisper 風格參數（非文字 task 不經正規化） |

### 模型限制 metadata

| 來源 | 適用 | 設定方式 |
|-----|------|---------|
| model-version tag `max_output_tokens` / `context_window_tokens` | 已註冊模型（HF / Ollama） | 註冊時帶同名欄位，或 `POST /models/set_model_limits` |
| `inference.yaml` `engines.ollama.model_limits.<daemon tag>` | 未註冊、以 `engine="ollama"` 呼叫的 Ollama-only 模型 | 同 schema |

未配置 = 上限未知：`model-info` 回 `null`、推論不做 422 判定。`context_window_tokens` 只是 metadata，
不代替輸出上限，也不會拿來縮小 `max_tokens`。

### 量化

註冊時 `quantization: "4bit"` 或 `"8bit"`，HF adapter 會用 BitsAndBytes 載入：

```python
# spec 上看到 quantization=="4bit" 時
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# 並自動加 device_map="auto"
```

---

## 擴展開發

### 案例 1 — 新增標準 architecture 的 LLM

只要 transformers 原生支援，註冊時填 `model_class="AutoModelForCausalLM"` 即可，不寫一行程式碼。

### 案例 2 — 新增需要客製 chat template / processor 的 VLM

例如類似 LLaVA、InternVL：判斷 `VLMHandler._infer_generic` 是否夠用：

- 夠用 → 直接註冊 `model_class` 即可
- 需要客製 chat template → 寫 `custom_handler`：

```python
# myorg/handlers/internvl.py
from src.inference.handlers.base import BaseHandler

class InternVLHandler(BaseHandler):
    def run(self, loaded, spec, data, options):
        model = loaded["model"]
        processor = loaded["processor"]
        messages = self._build_messages(data["image"], data["prompt"])
        # ... encode + generate + decode
        return {"response": text, "metadata": {...}}
```

註冊時設 `custom_handler="myorg.handlers.internvl:InternVLHandler"`，service 會於推論時 lazy import。

### 案例 3 — 新增整個 runtime（例如 vLLM）

1. 新增 `adapters/vllm.py`：

```python
from .base import BaseAdapter

class VLLMAdapter(BaseAdapter):
    runtime = "vllm"

    def load_model(self, spec):
        from vllm import LLM
        return LLM(model=spec.physical_path, ...)

    def infer(self, model, spec, data, options):
        outputs = model.generate([data["inputs"]], ...)
        return {"response": outputs[0].outputs[0].text, "metadata": {}}
```

2. 在 `adapters/__init__.py` 的 `get_adapter` 加一條 elif，並在 `spec.py` 的 `SUPPORTED_RUNTIMES` 加 `"vllm"`。
3. 註冊時設 `runtime="vllm"`（目前要從 model_manager 端把 `vllm` 加入 `_ENGINE_TO_RUNTIME` 映射）。

---

## 故障排除

**Q: 註冊新 HF 模型後推理報 `ValidationError: spec for 'X' is missing both 'pipeline_task' and 'model_class'`**

A: 註冊時沒填 `model_class`。重新註冊一次新版本，或直接用 MLflow client 對舊版本補上 tag：

```python
mlflow_client.set_model_version_tag("X", "1", "model_class", "AutoModelForCausalLM")
mlflow_client.set_model_version_tag("X", "1", "processor_class", "AutoTokenizer")
```

**Q: Ollama daemon 上的模型名稱與 MLflow 註冊名不同**

A: 沒問題。`ollama_model_name` tag 就是用來解這個的；推論時 adapter 會把 `model_name` 映射到 `ollama_model_name`。

**Q: GPU OOM**

A: 立即 `POST /inference/cache/clear` 卸載所有模型；然後考慮：

- 註冊時加 `quantization: "4bit"`
- 調小 `max_cached_models`（在 `inference.yaml.engines.transformers`）
- 給該模型較小的 `max_new_tokens`

**Q: 我有舊版註冊的模型，能直接用嗎？**

A: 能。`ModelSpec.from_mlflow_tags` 會把 `inference_engine` → `runtime`、`inference_task` → `task_family`、`ollama_model_name` 等舊 tag 讀進來。但對 HF 模型，若沒有 `model_class` 也沒有 `pipeline_task`，第一次推論會在載入時拋 `ValidationError`，請補上 tag。

**Q: 怎麼確認某模型現在是用哪個 handler？**

A: 推理回應中沒有 handler 資訊，但可以從 `task_family` 與 `custom_handler` tag 推斷：handler 解析順序是「`spec.custom_handler` 優先，否則查預設表」。詳見 `handlers/__init__.py:resolve_handler`。

---

## 版本歷史

- **v3.0.0** — Spec-driven 二層架構；移除 `manager`/`router`/`executor`/`registry`/`cache`/`engines`/`models` 七個檔案；改成 `service` + `adapters/*` + `handlers/*` + `spec.py`。
- **v2.0.0** — 5 層壓到 3 層；統一 `/infer` 端點。
- **v1.0.0** — 初版多引擎支援。
