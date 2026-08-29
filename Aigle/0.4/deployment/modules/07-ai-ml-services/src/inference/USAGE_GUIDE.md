# Inference Module — Usage Guide

> v3.2 — spec-driven two-layer architecture + the OpenAI-compatible layer (/v1) + automatic idle unload (keep_alive)

This guide walks through how to bring different model types (Ollama / HuggingFace / custom models) onto the inference service, set MLflow tags, and run inference. For the architecture and design rationale, see [README.md](./README.md).

## Contents

1. [Mental model](#mental-model)
2. [Registration side: writing "how to run it" into MLflow](#registration-side-writing-how-to-run-it-into-mlflow)
3. [Inference side: calling InferenceService](#inference-side-calling-inferenceservice)
4. [Onboarding flow (by model type)](#onboarding-flow-by-model-type)
5. [Inference options](#inference-options)
6. [Model resource management](#model-resource-management)
7. [Monitoring & health checks](#monitoring--health-checks)
8. [Extending](#extending)
9. [Migrating from v2.x](#migrating-from-v2x)

---

## Mental model

```
     At registration                           At inference
   ┌────────────┐                          ┌──────────────────┐
   │ Model source│                          │ /inference/infer │
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

**Design principle**: the decision of "how this model should run" is made exactly once, at registration; the inference side always just executes the spec. Adding a new model architecture needs no changes to the inference layer's code.

---

## Registration side: writing "how to run it" into MLflow

`register_lakefs_to_mlflow` and `register_ollama_to_mlflow` already support the spec-driven schema. Common required fields:

| Parameter | Description |
|-----|-----|
| `task` | the task name (legacy names still work, auto-canonicalized) |
| `engine` | `transformers` / `ollama` (→ written to the `runtime` tag) |

### Extra fields for HF Transformers models

| Parameter | Required | Example |
|-----|-----|------|
| `model_class` | (a) | `Qwen2_5_VLForConditionalGeneration`, `AutoModelForCausalLM`, `AutoModelForSpeechSeq2Seq` |
| `processor_class` | – | `AutoProcessor`, `AutoTokenizer`, `AutoFeatureExtractor` |
| `pipeline_task` | (a) | `text-generation`, `automatic-speech-recognition`, `audio-classification`, `image-to-text` |
| `trust_remote_code` | – | defaults to `true` |
| `custom_handler` | – | a dotted path overriding the default handler |

(a) At least one of `model_class` or `pipeline_task` must be set.

### Ollama models

None of the HF fields above are needed; the adapter pulls the model from the (existing) `ollama_model_name` tag.

---

## Inference side: calling InferenceService

### Python, in-process

```python
from src.inference import inference_service

result = inference_service.infer(
    task=None,  # can be omitted — uses the model's registered task_family; if given, checked against it
    model_name="qwen3-1.7b-ollama",
    data={"inputs": "Write a poem about spring"},
    options={"max_length": 200, "temperature": 0.7},
)

print(result["result"]["response"])
print(f"runtime={result['engine']} took {result['processing_time']:.2f}s")
```

Note: the `engine` parameter is deprecated; the service decides the runtime from the spec itself. If passed, it's ignored.

### HTTP

```bash
# task can be omitted (xinference-style: just give the model name to call it)
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3-1.7b-ollama",
    "data": {"inputs": "Write a poem about spring"},
    "options": {"max_length": 200, "temperature": 0.7}
  }'
```

For `text-generation`, `data` also accepts OpenAI-style `messages` (in place of `inputs`):
Ollama goes through the daemon's `/api/chat`, HF uses the tokenizer's `apply_chat_template` —
each runtime applies the model's own chat template:

```json
{
  "model_name": "qwen3-1.7b-ollama",
  "data": {"messages": [
    {"role": "system", "content": "You are a poet"},
    {"role": "user", "content": "Write a poem about spring"}
  ]}
}
```

### The OpenAI-compatible layer (/v1)

For external apps (or any OpenAI SDK), we recommend going straight through `/v1` — no need to know about task / runtime at all:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8010/v1", api_key="not-needed")
client.chat.completions.create(model="qwen3-1.7b-ollama",
                               messages=[{"role": "user", "content": "Hello"}])
```

See [the project README — OpenAI-compatible API](../../README.md#openai-compatible-api) for the endpoint list and VLM / ASR usage.

### Exception handling

```python
from src.inference.exceptions import (
    ValidationError, ModelNotFoundError, ModelLoadError,
    ResourceExhaustedError, InferenceExecutionError,
)

try:
    result = inference_service.infer(...)
except ValidationError as e:
    # 400 — unknown task, or a missing data field; fix client-side and retry
    ...
except ModelNotFoundError as e:
    # 404 — MLflow has no such model
    ...
except ResourceExhaustedError as e:
    # 503 — insufficient GPU/RAM; retry later, or call /cache/clear first
    ...
except (ModelLoadError, InferenceExecutionError) as e:
    # 500 — usually an incorrect spec, or a problem with the model itself
    ...
```

---

## Onboarding flow (by model type)

### Scenario A — An Ollama model (the fastest path)

As long as the model is already on the local Ollama daemon, three steps get you there:

```bash
# 1. Confirm it's on the daemon
ollama list

# 2. Register it with MLflow (still using the existing register_ollama endpoint)
curl -X POST "http://localhost:8010/models/register_ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "local_model_name": "qwen3:1.7b",
    "task": "text-generation",
    "registered_name": "qwen3-1.7b-ollama",
    "stage": "production"
  }'

# 3. Infer
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation",
    "model_name": "qwen3-1.7b-ollama",
    "data": {"inputs": "Hello"}
  }'
```

The spec tag that gets written automatically includes `runtime=ollama` / `task_family=text-generation` / `ollama_model_name=qwen3:1.7b`.

### Scenario B — A standard HF LLM (e.g. the plain-text version of Llama / Gemma / Qwen)

Just use `pipeline_task="text-generation"` — no need to specify `model_class`:

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

The inference request is identical to the Ollama case (task + model_name + data); the service knows to use the HF runtime from the spec.

### Scenario C — the vision-language model Qwen2.5-VL

The key: set `model_class` to the correct class name, and the handler automatically takes the Qwen-VL path (which internally uses `apply_chat_template` + `qwen_vl_utils.process_vision_info`):

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

Inference:

```bash
curl -X POST "http://localhost:8010/inference/infer" \
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

The `image` field accepts: a local path, a PIL.Image, a base64 string, or a `data:image/...` URL.

### Scenario D — A generic VLM (LLaVA / BLIP-2, etc.)

The generic VLMHandler goes through `processor(images, text, return_tensors="pt")` → `model.generate` → `processor.batch_decode`:

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

### Scenario E — Whisper / generic ASR

Simplest: take the pipeline path.

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

Inference:

```json
{
  "task": "asr",
  "model_name": "whisper-large-v3",
  "data": {"audio": "/data/audio/clip.wav"},
  "options": {"language": "zh", "task": "transcribe"}
}
```

If finer-grained control over the generate parameters is needed (e.g. batching), use the raw path instead: set `model_class="WhisperForConditionalGeneration"` + `processor_class="AutoProcessor"`.

### Scenario F — OCR (TrOCR)

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

### Scenario G — A fully bespoke model (custom_handler)

When the default handler isn't enough (e.g. an in-house multi-head model, a special chat template, a non-standard generate loop), write a custom handler:

```python
# myorg/handlers/custom_vlm.py
from src.inference.handlers.base import BaseHandler

class CustomVLMHandler(BaseHandler):
    def run(self, loaded, spec, data, options):
        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        # your encode logic
        # your generate logic
        # your decode logic

        return {
            "response": decoded_text,
            "metadata": {"output_length": int(outputs.shape[-1])},
        }
```

Add `custom_handler` at registration:

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

On the first inference call, the service lazy-imports this class; the handler must subclass `BaseHandler`.

---

## Inference options

### Common (usable across both HF and Ollama)

| Option | Description |
|-----|-----|
| `max_new_tokens` | number of newly generated tokens (this one is recommended) |
| `max_length` | total length (including the prompt) |
| `temperature` | sampling temperature |
| `top_p` | nucleus sampling |
| `top_k` | top-k sampling |
| `do_sample` | defaults to `true` |
| `keep_alive` | how long the model stays loaded after this inference: seconds, or `"30s"`/`"5m"`/`"1h"`; `0` = unload immediately after use, negative = keep resident. If unset, uses the `idle_timeout` default (see [automatic idle unload](#automatic-idle-unload-keep_alive) below) |

### HF-specific

| Option | Description |
|-----|-----|
| `num_beams` | beam search |
| `repetition_penalty` | repetition penalty |
| `length_penalty` | length penalty |
| `early_stopping` | early-stop beam search |

### Ollama-specific

| Option | Description |
|-----|-----|
| `repeat_penalty` | repetition penalty |
| `stop` | stop sequences (a list) |
| `mirostat` / `mirostat_eta` / `mirostat_tau` | mirostat sampling |
| `num_ctx` / `num_batch` / `num_gpu` / `num_thread` | engine-level parameters |

### Task-specific

| Task | Options |
|-----|-----|
| ASR | `language`, `task` (`transcribe` / `translate`) |
| audio-classification | `top_k` |

---

## Model resource management

Each adapter maintains its own internal LRU cache, with an idle timeout. The endpoints below operate on that cache.

### Automatic idle unload (keep_alive)

Ollama-style resource reclamation: once a model has been idle longer than its lifetime, a background reaper (scanning every 30 seconds by default) automatically unloads it and clears the GPU cache; the next request loads it back automatically. Priority order:

1. Per-request `options.keep_alive` (`0` = unload right after use, negative = keep resident)
2. `inference.yaml`'s `engines.transformers.idle_timeout` (defaults to 300 seconds; `<=0` disables it)
   / `engines.ollama.keep_alive` (passed straight through to the daemon)
3. The LRU capacity `max_cached_models` (evicts the oldest when the cache is full; skips models currently in flight)

```bash
# A one-off request: free the GPU immediately after inference
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen2.5-vl-7b",
       "data": {"image": "/data/x.jpg", "prompt": "Describe"},
       "options": {"keep_alive": 0}}'
```

A model currently in flight (`in_flight > 0`) is never reclaimed by the reaper or the LRU.

### Unloading a specific model

```bash
curl -X POST "http://localhost:8010/inference/unload-model?model_name=qwen2.5-vl-7b"
```

The response includes `unloaded_runtimes` (which adapters were unloaded) and `gpu_memory_freed`.

### Unloading everything

```bash
curl -X POST "http://localhost:8010/inference/unload-all-models"
# or equivalently
curl -X POST "http://localhost:8010/inference/cache/clear"
```

### Listing what's loaded

```bash
curl "http://localhost:8010/inference/loaded-models"
```

```json
{
  "total": 2,
  "by_runtime": {
    "ollama": ["qwen3-1.7b-ollama"],
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

`expires_in_seconds` of `null` means resident (a negative `keep_alive`, or `idle_timeout <= 0`).

### Adjusting the LRU size

`src/core/configs/inference.yaml`:

```yaml
engines:
  transformers:
    max_cached_models: 2     # how many models the HF adapter keeps resident at once
    idle_timeout: 300        # auto-unload after this many idle seconds (<=0 = never auto-unload)

idle_reaper:
  enabled: true
  interval: 30               # reaper scan interval (seconds)
```

The Ollama adapter defaults to a large LRU (64) and does no local idle reclamation, since the model doesn't occupy local RAM —
GPU release is handled by the daemon itself, according to `engines.ollama.keep_alive`.

---

## Monitoring & health checks

```bash
curl "http://localhost:8010/inference/health"
# {"status": "healthy", "adapters_initialized": ["ollama", "hf-transformers"], ...}

curl "http://localhost:8010/inference/stats"
# {"stats": {"total": 1024, "success": 1018, "failed": 6, "success_rate": 0.994, ...}}

curl "http://localhost:8010/inference/supported-tasks"
# {"tasks": {"text-generation": {"required_fields": ["inputs | messages"]}, ...}}
```

---

## Extending

### Adding a new task family

1. Add the new name to `TASK_FAMILIES` in `spec.py`
2. (If needed) add an old-name → new-name mapping in `_TASK_ALIAS`
3. Add the required fields to `_REQUIRED_DATA_FIELDS` in `service.py`
4. Add the corresponding handler under `handlers/`, and register it in the `_DEFAULT` table in `handlers/__init__.py`

### Adding a new runtime (e.g. vLLM)

See [README.md — Extending](./README.md#extending).

### Adding a handler without changing the default

Use the `custom_handler` tag — no need to touch `_DEFAULT`. Good for "same task family, but this particular model needs special handling."

---

## Migrating from v2.x

The external API shape is backward-compatible; here are the behavioral differences:

| Change | v2.x behavior | v3.0 behavior | Impact |
|-----|----------|----------|-----|
| `engine` field | Required, tightly bound to task | Optional, decided by the MLflow tag | Existing clients can keep passing it — it's just ignored |
| Task names | `text-generation-ollama`, `asr-hf`, `vad-hf`, `scene-detection`, `video-summary`, `audio-transcription` were each independent | These names now map onto the canonical task family | Existing calls keep working as-is; new calls should use the canonical name |
| Response field | No `model_version` | Gains `model_version` | Additive, doesn't affect existing parsing |
| HF VLM inference | Relied on path-string matching for Qwen-VL | Relies on the `model_class` tag | **Old HF registrations need the `model_class` tag added** |
| Model caching | A global `ModelCache` class | An `OrderedDict` internal to each adapter | The external API is unchanged |
| Exception types | Same `ValidationError` / `ModelNotFoundError` / ... | Unchanged | No client changes needed |

### Example script for patching in tags

For old models registered only under the v2 schema:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
# Example: add pipeline_task to every HF text-gen model registered as "transformers" but missing model_class
for rm in client.search_registered_models():
    for mv in client.search_model_versions(f"name='{rm.name}'"):
        tags = mv.tags
        if tags.get("inference_engine") == "transformers" and not tags.get("model_class") and not tags.get("pipeline_task"):
            if tags.get("inference_task", "").startswith("text-generation"):
                client.set_model_version_tag(rm.name, mv.version, "pipeline_task", "text-generation")
                print(f"patched {rm.name} v{mv.version}")
```

Once this has run, the new service can resolve the spec.
