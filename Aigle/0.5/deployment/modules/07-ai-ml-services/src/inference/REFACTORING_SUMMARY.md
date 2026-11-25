# Inference 模組重構紀要 (v3.0.0)

## 重構動機

v2.0 雖然把 Engine/Manager/Selector/Pipeline 從 5 層壓到 3 層，但仍有結構性痛點導致「非 Ollama 模型無法被動態調用」：

1. **單一 TransformersEngine 撐起所有模態**：text-generation / VLM / ASR / OCR / video / document 共用一個類，每加一個模型架構就得加一條 `if 'qwen' in path.lower()` 之類的字串匹配分支。
2. **MLflow tag 缺「怎麼跑」這層描述**：HF 模型只記 `physical_path`、`repo_id`，沒有 `model_class`、`processor_class` 等 runtime 提示，推論端只能靠路徑字串猜模型結構。
3. **層級職責糊掉**：`InferenceManager._validate_parameters` 與 `TaskRouter._task_engine_mapping` 重複驗證；`ModelExecutor` 只是 4 行膠水；`ModelHandler.preprocess` 與 Engine 的 `_infer_vlm` 都做客製分支但職責切不乾淨。
4. **沒有 servable 抽象**：所有 transformers 模型都直接 `from_pretrained` 到 API 同 process，OOM、依賴衝突、熱重載會直接打掛 API。

## 解決方向

**Spec-driven dispatch**：把「該模型該怎麼跑」的決策權從推論端搬到註冊端，存進 MLflow tag 變成 `ModelSpec`，推論端只負責照 spec 執行。

## 架構變化

```
v2.0（5 個類，3 層）              v3.0（2 個類，2 層）
─────────────────────────         ─────────────────────────
API                               API
 ↓                                 ↓
InferenceManager                  InferenceService
 ↓                                 ↓ (resolve ModelSpec from MLflow)
TaskRouter ──> ModelRegistry      Adapter (Ollama / HF)
 ↓                                 ↓ (dispatch via spec.model_class)
ModelExecutor ──> ModelCache      Handler (task-family specific)
 ↓
Engine (Ollama / HF)
 ↓
ModelHandler
```

刪除的類：`InferenceManager` / `TaskRouter` / `ModelExecutor` / `ModelRegistry` / `ModelCache` / `BaseEngine` / `BaseModelHandler`。
新增的類：`ModelSpec` / `InferenceService` / `BaseAdapter` / `BaseHandler`。

## 檔案層級對照

| v2.0 路徑 | v3.0 路徑 | 動作 |
|----------|----------|------|
| `inference/manager.py` | `inference/service.py` | 重寫並合併 router/executor 邏輯 |
| `inference/router.py` | （移除） | 任務驗證合併到 service，runtime 由 spec.runtime 決定 |
| `inference/executor.py` | （移除） | adapter.run() 直接做 load + infer |
| `inference/registry.py` | `inference/handlers/__init__.py` | 改用簡單的 dict registry |
| `inference/cache.py` | `inference/adapters/base.py` | LRU 直接內建在 BaseAdapter |
| `inference/engines/base.py` | `inference/adapters/base.py` | 重新設計介面 |
| `inference/engines/ollama.py` | `inference/adapters/ollama.py` | spec-aware 改寫 |
| `inference/engines/transformers.py` | `inference/adapters/hf_transformers.py` | spec-driven 完全改寫 |
| `inference/models/*.py` | `inference/handlers/*.py` | rename + 介面收斂為 `run(loaded, spec, data, options)` |
| —（新增） | `inference/spec.py` | `ModelSpec` 與 MLflow tag 解析 |

## 對外行為變化

### API（`POST /inference/infer`）

請求格式向下相容；`engine` 欄位降級為可選欄位（保留以避免破壞既有呼叫端，service 內部忽略）。

```json
{
  "task": "vlm",
  "model_name": "qwen2.5-vl-7b",
  "data": {"image": "...", "prompt": "..."},
  "options": {}
}
```

回應新增 `model_version` 欄位：

```json
{
  "success": true,
  "result": {...},
  "task": "vlm",
  "engine": "hf-transformers",
  "model_name": "qwen2.5-vl-7b",
  "model_version": "3",
  "processing_time": 4.21,
  "timestamp": 1714195200.0
}
```

### 註冊（`POST /models/register_from_lakefs`）

新增 5 個可選欄位，用來填 spec：

| 欄位 | 用途 | 必填情境 |
|-----|-----|---------|
| `model_class` | transformers 模型 class 名 | 不走 pipeline 時必填 |
| `processor_class` | processor / tokenizer class | 多數情況需要 |
| `pipeline_task` | 走 `transformers.pipeline` 捷徑時填 | 與 `model_class` 二擇一 |
| `trust_remote_code` | 預設 `true` | 通常不用動 |
| `custom_handler` | dotted 路徑 (`pkg.mod:Class`) | 需自訂 pre/post-processing 時填 |

### MLflow Tag Schema

v3.1 開始 spec tag 是**唯一寫入端**；舊 `inference_engine` / `inference_task` 不再回寫。
讀取端仍能解析舊 tag（`ModelSpec.from_mlflow_tags` 內含相容路徑），舊註冊不需遷移。

| 新 tag | 取代/補充 | 範例 |
|-------|----------|------|
| `runtime` | 取代 `inference_engine`（不再寫） | `ollama` / `hf-transformers` |
| `task_family` | 取代 `inference_task`（不再寫） | `text-generation` / `vlm` / `asr` |
| `model_class` | （新） | `Qwen2_5_VLForConditionalGeneration` |
| `processor_class` | （新） | `AutoProcessor` |
| `pipeline_task` | （新） | `text-generation` |
| `trust_remote_code` | （新） | `true` |
| `custom_handler` | （新） | `myorg.qwen_vl:CustomHandler` |

## 解決了什麼

1. **非 Ollama 模型可動態調用**：註冊時填 `model_class` + `processor_class`，HFTransformersAdapter 透過 `getattr(transformers, model_class)` 反射加載，不再有路徑字串 if 分支。
2. **新模型零程式碼接入**：標準 architecture（如 LLaMA、Whisper、TrOCR）只要走 `pipeline_task`；非標準（如 Qwen-VL）填 `model_class` + 既有 `VLMHandler`；完全特殊（如自家訓練的多 head 模型）走 `custom_handler` 逃生口。
3. **舊註冊的模型不需重註**：`ModelSpec.from_mlflow_tags` 會把 `inference_engine`/`inference_task`/`ollama_model_name` 等舊 tag 自動映射到新 schema。
4. **層級單一職責**：service 管請求與 spec 解析；adapter 管 runtime 與模型載入；handler 管 task-family 的 encode/generate/decode。沒有重複驗證、沒有 4 行膠水類。

## 沒解決、留給後續

- **同 process 推論**：Adapter 仍把 transformers 模型載入到 API 同一個 Python process。OOM/依賴衝突風險未解，這是方案 B（多 runtime 子程序化、引入 vLLM/TGI/Triton）要處理的。
- **多 GPU 排程**：`gpu_manager` 仍只在註冊時用來估 VRAM，沒進入推理路徑做 placement。
- **熱卸載觸發策略**：目前 LRU 只在 cache 滿時才 evict；沒有依空閒時間或全域記憶體壓力觸發。
