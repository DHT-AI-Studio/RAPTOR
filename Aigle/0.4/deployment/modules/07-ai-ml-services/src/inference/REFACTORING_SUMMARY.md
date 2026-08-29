# Inference Module Refactoring Summary (v3.0.0)

## Motivation

v2.0 collapsed Engine/Manager/Selector/Pipeline from 5 layers down to 3, but structural pain points remained that meant "non-Ollama models couldn't be dispatched dynamically":

1. **A single TransformersEngine carrying every modality**: text-generation / VLM / ASR / OCR / video / document all shared one class — every new model architecture meant adding another string-matching branch like `if 'qwen' in path.lower()`.
2. **MLflow tags lacked a "how to run it" layer**: HF models recorded only `physical_path` and `repo_id`, with no `model_class`, `processor_class`, or other runtime hints — the inference side had to guess the model's structure from the path string.
3. **Blurred layer responsibilities**: `InferenceManager._validate_parameters` and `TaskRouter._task_engine_mapping` duplicated validation; `ModelExecutor` was 4 lines of glue; `ModelHandler.preprocess` and the Engine's `_infer_vlm` both had custom branches with no clean separation of responsibility.
4. **No "servable" abstraction**: every transformers model was loaded via `from_pretrained` directly in the API's own process — an OOM, a dependency conflict, or a hot-reload could take the whole API down.

## Direction taken

**Spec-driven dispatch**: move the decision of "how this model should run" from the inference side to the registration side, store it in an MLflow tag as a `ModelSpec`, and have the inference side just execute according to the spec.

## Architectural change

```
v2.0 (5 classes, 3 layers)         v3.0 (2 classes, 2 layers)
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

Classes removed: `InferenceManager` / `TaskRouter` / `ModelExecutor` / `ModelRegistry` / `ModelCache` / `BaseEngine` / `BaseModelHandler`.
Classes added: `ModelSpec` / `InferenceService` / `BaseAdapter` / `BaseHandler`.

## File-level mapping

| v2.0 path | v3.0 path | Action |
|----------|----------|------|
| `inference/manager.py` | `inference/service.py` | Rewritten, merging in the router/executor logic |
| `inference/router.py` | (removed) | Task validation merged into service; runtime now decided by spec.runtime |
| `inference/executor.py` | (removed) | adapter.run() does load + infer directly |
| `inference/registry.py` | `inference/handlers/__init__.py` | Switched to a simple dict registry |
| `inference/cache.py` | `inference/adapters/base.py` | LRU now built directly into BaseAdapter |
| `inference/engines/base.py` | `inference/adapters/base.py` | Interface redesigned |
| `inference/engines/ollama.py` | `inference/adapters/ollama.py` | Rewritten to be spec-aware |
| `inference/engines/transformers.py` | `inference/adapters/hf_transformers.py` | Fully rewritten, spec-driven |
| `inference/models/*.py` | `inference/handlers/*.py` | Renamed + interface consolidated to `run(loaded, spec, data, options)` |
| — (new) | `inference/spec.py` | `ModelSpec` and MLflow tag parsing |

## External behavior changes

### API (`POST /inference/infer`)

The request format is backward-compatible; the `engine` field is downgraded to optional (kept so existing callers don't break, but ignored internally by the service).

```json
{
  "task": "vlm",
  "model_name": "qwen2.5-vl-7b",
  "data": {"image": "...", "prompt": "..."},
  "options": {}
}
```

The response gains a `model_version` field:

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

### Registration (`POST /models/register_from_lakefs`)

5 new optional fields, used to fill in the spec:

| Field | Purpose | When required |
|-----|-----|---------|
| `model_class` | The transformers model class name | Required when not going through a pipeline |
| `processor_class` | The processor / tokenizer class | Needed in most cases |
| `pipeline_task` | Set when taking the `transformers.pipeline` shortcut | Either this or `model_class`, not both |
| `trust_remote_code` | Defaults to `true` | Usually left as-is |
| `custom_handler` | A dotted path (`pkg.mod:Class`) | Set when custom pre/post-processing is needed |

### MLflow Tag Schema

Starting in v3.1, the spec tag is the **only write path**; the old `inference_engine` / `inference_task` are no longer written back.
The read path can still parse the old tags (`ModelSpec.from_mlflow_tags` has a compatibility path built in) — existing registrations need no migration.

| New tag | Replaces/supplements | Example |
|-------|----------|------|
| `runtime` | Replaces `inference_engine` (no longer written) | `ollama` / `hf-transformers` |
| `task_family` | Replaces `inference_task` (no longer written) | `text-generation` / `vlm` / `asr` |
| `model_class` | (new) | `Qwen2_5_VLForConditionalGeneration` |
| `processor_class` | (new) | `AutoProcessor` |
| `pipeline_task` | (new) | `text-generation` |
| `trust_remote_code` | (new) | `true` |
| `custom_handler` | (new) | `myorg.qwen_vl:CustomHandler` |

## What this fixed

1. **Non-Ollama models can now be dispatched dynamically**: fill in `model_class` + `processor_class` at registration, and HFTransformersAdapter loads via reflection (`getattr(transformers, model_class)`) — no more path-string if-branches.
2. **Zero-code onboarding for new models**: standard architectures (LLaMA, Whisper, TrOCR, etc.) just use `pipeline_task`; non-standard ones (Qwen-VL) set `model_class` + the existing `VLMHandler`; fully bespoke ones (an in-house multi-head model) use the `custom_handler` escape hatch.
3. **Previously-registered models need no re-registration**: `ModelSpec.from_mlflow_tags` auto-maps old tags like `inference_engine`/`inference_task`/`ollama_model_name` onto the new schema.
4. **Single responsibility per layer**: service handles the request and spec resolution; adapter handles the runtime and model loading; handler handles task-family-specific encode/generate/decode. No duplicated validation, no 4-line glue classes.

## Not solved, left for later

- **In-process inference**: the Adapter still loads transformers models into the API's own Python process. The OOM/dependency-conflict risk is unresolved — that's for a follow-up plan B (moving multiple runtimes into subprocesses, introducing vLLM/TGI/Triton).
- **Multi-GPU scheduling**: `gpu_manager` is still only used at registration time to estimate VRAM — it doesn't participate in placement during the inference path.
- **Hot-unload trigger policy**: LRU currently only evicts when the cache is full — there's no eviction triggered by idle time or overall memory pressure.
