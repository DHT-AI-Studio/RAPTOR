# Inference Module

> A spec-driven, two-layer inference framework

Version: **v3.0.0**

## Contents

- [Overview](#overview)
- [Core design](#core-design)
- [Architecture](#architecture)
- [ModelSpec Schema](#modelspec-schema)
- [Quick start](#quick-start)
- [API docs](#api-docs)
- [Supported tasks](#supported-tasks)
- [Configuration](#configuration)
- [Extending](#extending)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Inference module is the inference execution layer of the AI Model Lifecycle platform. It's designed so that the decision of "how this model should run" is made entirely at MLflow registration time, and the inference side just executes according to the spec; adding a new model architecture requires no changes to the inference layer's code.

Two key abstractions:

- **`ModelSpec`** — the inference spec parsed from MLflow tags (`runtime` / `task_family` / `model_class` / `processor_class` / ...), the contract between the service and the adapters.
- **`BaseAdapter`** — one adapter per runtime (Ollama, HF Transformers, and vLLM in the future), taking a spec and returning a result.

---

## Core design

| Design principle | Where it's applied |
|---------|------|
| Record "how to run it" at registration, execute accordingly at inference | MLflow tags carry `model_class` / `processor_class` / `pipeline_task` |
| No path-string matching | Adapters load via reflection, `getattr(transformers, spec.model_class)` |
| Default behavior alongside an escape hatch | Defaults to the task-family handler; non-standard models can load a custom class via `custom_handler` |
| Two layers, not five | `service.py` + `adapters/*` + `handlers/*`; the router/executor/registry/cache classes are gone |
| Old registrations need no re-registration | `ModelSpec.from_mlflow_tags` reads both old and new tags |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    API Layer                     │
│            POST /inference/infer                 │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                InferenceService                  │
│   • Validates task / required data fields        │
│   • resolve_spec(model_name) ← MLflow tag        │
│   • dispatch by spec.runtime                     │
└──────────┬─────────────────────────┬─────────────┘
           │                         │
   spec.runtime=ollama       spec.runtime=hf-transformers
           │                         │
┌──────────▼──────────┐    ┌─────────▼────────────────────┐
│   OllamaAdapter     │    │   HFTransformersAdapter      │
│  • HTTP /api/generate│    │  • getattr(tx, model_class) │
│  • daemon owns models│    │  • LRU-cached load result   │
└─────────────────────┘    └─────────┬────────────────────┘
                                     │
                          spec.task_family / spec.custom_handler
                                     │
┌────────────────────────────────────▼─────────────┐
│                  Handlers                        │
│  TextGen / VLM (incl. Qwen-VL branch) / ASR /    │
│  OCR / AudioCls / VideoAnalysis / DocAnalysis /..│
│  • encode → generate → decode → unified dict     │
└──────────────────────────────────────────────────┘
```

### File layout

```
src/inference/
├── __init__.py            # public exports
├── spec.py                # ModelSpec + canonicalize_task
├── service.py              # InferenceService singleton
├── exceptions.py          # exception hierarchy
├── vram_estimator.py      # VRAM estimation at registration time
├── adapters/
│   ├── __init__.py        # get_adapter(runtime) factory
│   ├── base.py            # BaseAdapter ABC (includes LRU)
│   ├── ollama.py          # OllamaAdapter
│   └── hf_transformers.py # HFTransformersAdapter
└── handlers/
    ├── __init__.py        # resolve_handler(spec)
    ├── base.py            # BaseHandler ABC
    ├── text_generation.py
    ├── vlm.py             # includes the Qwen-VL path (determined via spec)
    ├── asr.py
    ├── ocr.py
    ├── audio_classification.py
    ├── video_analysis.py
    └── document_analysis.py
```

---

## ModelSpec Schema

ModelSpec is written into an MLflow tag at registration time and read back at inference time. Old and new tags coexist for backward compatibility.

### Common fields

| Field | Required | Description |
|-----|-----|------|
| `runtime` | ✅ | `ollama` or `hf-transformers` |
| `task_family` | ✅ | canonical task name (`text-generation`/`vlm`/`asr`/...) |
| `physical_path` | ✅ | `lakefs://repo/commit/`, a local path, or an HF repo-id |
| `lakefs_commit_id` | – | for provenance |
| `trust_remote_code` | – | defaults to `true` |
| `quantization` | – | `4bit` / `8bit` / unset |
| `custom_handler` | – | a dotted path overriding the default handler |

### HF Transformers–specific

| Field | Required | Description |
|-----|-----|------|
| `model_class` | (a) | the model class name in transformers (e.g. `AutoModelForCausalLM`, `Qwen2_5_VLForConditionalGeneration`) |
| `processor_class` | – | `AutoProcessor` / `AutoTokenizer` / `AutoFeatureExtractor` |
| `pipeline_task` | (a) | if set, takes the `transformers.pipeline` shortcut (e.g. `text-generation`, `automatic-speech-recognition`) |
| `torch_dtype` | – | `auto` / `fp16` / `bf16` / `fp32` |

(a) At least one of `model_class` or `pipeline_task` must be set.

### Ollama–specific

| Field | Required | Description |
|-----|-----|------|
| `ollama_model_name` | ✅ | the real name on the daemon (e.g. `qwen3:1.7b`), which may differ from the MLflow registered name |

### Example — registering Qwen2.5-VL 7B

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

The resulting MLflow tags (excerpt):

```
runtime              = hf-transformers
task_family          = vlm
model_class          = Qwen2_5_VLForConditionalGeneration
processor_class      = AutoProcessor
trust_remote_code    = true
physical_path        = lakefs://qwen-vl-models/<commit>/
inference_engine     = transformers   # legacy tag, kept
inference_task       = vlm            # legacy tag, kept
```

---

## Quick start

### Calling from Python

```python
from src.inference import inference_service

result = inference_service.infer(
    task="text-generation",
    model_name="qwen3-1.7b-ollama",
    data={"inputs": "Please explain what deep learning is"},
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
      "prompt": "Describe this image in detail"
    },
    "options": {"max_new_tokens": 256}
  }'
```

The `engine` field can now be omitted — the runtime is decided by the MLflow tag. If passed anyway, it's simply ignored, with a warning logged.

---

## API docs

### POST `/inference/infer`

#### Request

```json
{
  "task": "text-generation",
  "model_name": "qwen3-1.7b-ollama",
  "data": {"inputs": "..."},
  "options": {"max_length": 200, "temperature": 0.7},
  "engine": null
}
```

| Field | Required | Description |
|-----|-----|------|
| `task` | – | the task type; if omitted, uses the model's registered task_family (if provided, it's checked against it). Legacy names (`text-generation-ollama`/`asr-hf` etc.) are auto-canonicalized |
| `model_name` | ✅ | the model name registered in MLflow |
| `data` | ✅ | input data; the required fields vary by task (see the table below) |
| `options` | – | inference options (`max_new_tokens`/`temperature`/`keep_alive`/...) |
| `engine` | – | deprecated; kept for backward compatibility |

#### Required `data` fields per task

| task | Required data fields |
|-----|---------------|
| `text-generation` | `inputs` (str) or `messages` (an OpenAI-style list) |
| `vlm` / `image-captioning` | `image`, `prompt` |
| `asr` | `audio` |
| `ocr` | `image` |
| `audio-classification` | `audio` |
| `video-analysis` | `video` (`prompt` optional) |
| `document-analysis` | `document` (`query` optional) |

#### Response

```json
{
  "success": true,
  "result": {
    "response": "Deep learning is...",
    "metadata": {"input_length": 12, "output_length": 245}
  },
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "qwen3-1.7b-ollama",
  "model_version": "1",
  "processing_time": 1.83,
  "api_processing_time": 1.85,
  "timestamp": 1714195200.0
}
```

#### Error codes

| HTTP | Condition |
|-----|------|
| 400 | `ValidationError` / `UnsupportedTaskError` — unknown task, or a missing data field |
| 404 | `ModelNotFoundError` — the model can't be found in MLflow |
| 500 | `ModelLoadError` / `InferenceExecutionError` / `EngineError` |
| 503 | `ResourceExhaustedError` — insufficient GPU/memory, retry later |

### Admin endpoints

| Method | Path | Purpose |
|-----|-----|------|
| GET | `/inference/health` | service + adapter status |
| GET | `/inference/stats` | inference count, success rate, loaded models |
| GET | `/inference/supported-tasks` | task families and their required fields |
| GET | `/inference/loaded-models` | lists the models in each adapter's LRU cache (with idle/expiry/in-flight status) |
| POST | `/inference/cache/clear` | unload every model and clear the GPU cache |
| POST | `/inference/unload-model?model_name=X` | unload one specific model |
| POST | `/inference/unload-all-models` | unload everything |

---

## Supported tasks

| task_family | default handler | recommended engine | input | main output field |
|------------|-------------|---------|-----|----------|
| `text-generation` | `TextGenerationHandler` | ollama / hf | `{"inputs"}` or `{"messages"}` | `response` |
| `vlm` | `VLMHandler` (includes the Qwen-VL branch) | hf | `{"image","prompt"}` | `response` |
| `image-captioning` | `VLMHandler` | hf | `{"image"}` | `response` |
| `asr` | `ASRHandler` | hf | `{"audio"}` | `text` |
| `ocr` | `OCRHandler` | hf | `{"image"}` | `text` |
| `audio-classification` | `AudioClassificationHandler` | hf | `{"audio"}` | `classifications` |
| `video-analysis` | `VideoAnalysisHandler` | hf | `{"video"}` | `response` |
| `document-analysis` | `DocumentAnalysisHandler` | hf | `{"document"}` | `response` |

Legacy task names (`text-generation-ollama` / `text-generation-hf` / `asr-hf` / `ocr-hf` / `vad-hf` / `audio-transcription` / `scene-detection` / `video-summary`) are all mapped onto the table above by `canonicalize_task` — no caller-side changes needed.

---

## Configuration

### Engine configuration (`src/core/configs/inference.yaml`)

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
    max_cached_models: 2     # size of the adapter's internal LRU
```

The engine key can also be written as `hf-transformers` — equivalent to `transformers`.

### Inference options (per-request `options`)

| Option | Applies to | Description |
|-----|------|------|
| `max_new_tokens` | both | number of newly generated tokens |
| `max_length` | both | overall length cap |
| `temperature` | both | sampling temperature |
| `top_p` / `top_k` | both | sampling parameters |
| `num_beams` | hf | beam search |
| `repeat_penalty` | ollama | repetition penalty |
| `stop` | ollama | stop sequences |
| `language` / `task` | ASR | Whisper-style parameters |

### Quantization

Setting `quantization: "4bit"` or `"8bit"` at registration makes the HF adapter load via BitsAndBytes:

```python
# when the spec shows quantization=="4bit"
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# and device_map="auto" is added automatically
```

---

## Extending

### Case 1 — Adding an LLM with a standard architecture

As long as transformers natively supports it, just register with `model_class="AutoModelForCausalLM"` — no code to write.

### Case 2 — Adding a VLM that needs a custom chat template / processor

For something like LLaVA or InternVL, first check whether `VLMHandler._infer_generic` is sufficient:

- Sufficient → just register `model_class` directly
- Needs a custom chat template → write a `custom_handler`:

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

Set `custom_handler="myorg.handlers.internvl:InternVLHandler"` at registration — the service will lazy-import it at inference time.

### Case 3 — Adding an entirely new runtime (e.g. vLLM)

1. Add `adapters/vllm.py`:

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

2. Add an elif branch to `get_adapter` in `adapters/__init__.py`, and add `"vllm"` to `SUPPORTED_RUNTIMES` in `spec.py`.
3. Set `runtime="vllm"` at registration (for now, `vllm` also needs to be added to `_ENGINE_TO_RUNTIME`'s mapping on the model_manager side).

---

## Troubleshooting

**Q: After registering a new HF model, inference throws `ValidationError: spec for 'X' is missing both 'pipeline_task' and 'model_class'`**

A: `model_class` wasn't set at registration. Either re-register a new version, or use the MLflow client to patch the tag onto the existing version:

```python
mlflow_client.set_model_version_tag("X", "1", "model_class", "AutoModelForCausalLM")
mlflow_client.set_model_version_tag("X", "1", "processor_class", "AutoTokenizer")
```

**Q: The model name on the Ollama daemon differs from its MLflow registered name**

A: That's fine. The `ollama_model_name` tag exists exactly to handle this; at inference time, the adapter maps `model_name` to `ollama_model_name`.

**Q: GPU OOM**

A: Immediately call `POST /inference/cache/clear` to unload every model; then consider:

- Adding `quantization: "4bit"` at registration
- Lowering `max_cached_models` (under `inference.yaml.engines.transformers`)
- Giving that model a smaller `max_new_tokens`

**Q: I have a model registered under the old scheme — can I use it as-is?**

A: Yes. `ModelSpec.from_mlflow_tags` reads in the old tags — `inference_engine` → `runtime`, `inference_task` → `task_family`, `ollama_model_name`, etc. But for an HF model, if neither `model_class` nor `pipeline_task` is set, the first inference call will throw a `ValidationError` at load time — add the tag to fix it.

**Q: How do I check which handler a given model is currently using?**

A: The inference response doesn't include handler info, but you can infer it from the `task_family` and `custom_handler` tags: handler resolution order is "`spec.custom_handler` first, otherwise look up the default table." See `handlers/__init__.py:resolve_handler` for details.

---

## Version history

- **v3.0.0** — Spec-driven two-layer architecture; removed the seven files `manager`/`router`/`executor`/`registry`/`cache`/`engines`/`models`; replaced with `service` + `adapters/*` + `handlers/*` + `spec.py`.
- **v2.0.0** — Collapsed 5 layers into 3; unified the `/infer` endpoint.
- **v1.0.0** — Initial multi-engine support.
