# AI Model Lifecycle Management Platform

An MLOps-oriented AI model lifecycle management platform, integrating model download, version control, registration, stage management, and online inference. The architecture centers on the **MLflow Model Registry** (model registration and lifecycle) + **LakeFS** (atomic model/data version control), plus a **spec-driven inference gateway** that unifies every runtime (Ollama / HuggingFace Transformers, extensible to vLLM and others).

---

## Quick start — deploy an Ollama model in 5 minutes

```bash
# 1. Confirm the local Ollama model
ollama list
# or: curl "http://localhost:8010/models/local?model_source=ollama"

# 2. Start the service
docker compose up --build -d

# 3. Register the Ollama model with MLflow
curl -X POST "http://localhost:8010/models/register_ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "local_model_name": "qwen3:1.7b",
    "task": "text-generation",
    "registered_name": "qwen3-1.7b-ollama",
    "stage": "production"
  }'

# 4. Infer
curl -X POST "http://localhost:8010/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation",
    "model_name": "qwen3-1.7b-ollama",
    "data": {"inputs": "Hello, please introduce yourself."}
  }'
```

See [Workflow A](#workflow-a-quick-registration--inference-for-an-ollama-model) for the full flow.

---

## Contents

1. [Project overview](#project-overview)
2. [Core features](#core-features)
3. [Directory structure](#directory-structure)
4. [Inference architecture](#inference-architecture)
5. [ModelSpec: the model inference spec](#modelspec-the-model-inference-spec)
6. [OpenAI-Compatible API](#openai-compatible-api)
7. [Idle unload & GPU release](#idle-unload--gpu-release)
8. [Deployment & configuration](#deployment--configuration)
9. [Example workflows](#example-workflows)
10. [API endpoint list](#api-endpoint-list)
11. [FAQ](#faq)

---

## Project overview

Core platform components:

- **MLflow Tracking Server** — experiment tracking and artifact-version recording.
- **MLflow Model Registry** — centrally manages the model lifecycle (via the stage tag: `production` / `staging` / `archived` / `none`), and also records the spec-driven inference spec.
- **LakeFS** — git-like atomic version control for models and data; every MLflow version corresponds to one LakeFS commit.
- **InferenceService (the inference gateway)** — accepts unified inference requests, resolves the MLflow tag into a `ModelSpec`, and dispatches to the matching runtime adapter.
- **Models / Datasets / GPU / Config management APIs** — RESTful interfaces for download, upload, registration, stage transitions, and resource queries.

---

## Core features

- **Multi-source integration**: direct downloads from HuggingFace, local Ollama management, extensible to vLLM/Triton in the future.
- **Git-like model versioning**: every MLflow version points to an immutable LakeFS commit via a `lakefs_commit_id` tag.
- **Spec-driven dynamic dispatch**: at registration time, "how this model should run" is written into MLflow tags (`runtime` / `model_class` / `processor_class` / `pipeline_task` / `custom_handler`) — the inference side needs no code changes to support a new model architecture.
- **A unified inference API**: every task (text-gen / VLM / ASR / OCR / audio-cls / video / document / TTS / embedding / rerank) goes through `POST /inference/infer`; `task` can be omitted (decided by the model's registered task_family). TTS also has a convenience endpoint, `POST /inference/tts`.
- **OpenAI-compatible API (xinference-style)**: `/v1/models`, `/v1/chat/completions` (works for both LLM and VLM), `/v1/completions`, `/v1/audio/transcriptions`, `/v1/audio/speech` (TTS), `/v1/embeddings`, `/v1/rerank` — any OpenAI SDK can call a registered model just by pointing its `base_url` here.
- **Built-in multimodal branches**: the VLM handler dispatches by spec automatically to dedicated Qwen2.5-VL (chat template + qwen_vl_utils) and InternVL (dynamic tiling + `.chat()`) paths; everything else takes the generic processor→generate path.
- **Automatic idle unload (Ollama-style)**: a model idle past `idle_timeout` is automatically unloaded and its GPU released; overridable per request via `keep_alive` (`0` = unload right after use, negative = keep resident).
- **Stage-based lifecycle**: switch between production / staging / archived via `/models/transition_stage`, with optional auto-archiving of old versions.
- **Resource management**: each adapter has a built-in LRU cache supporting on-demand load/unload; a GPU status API is provided.

---

## Directory structure

```
ai-lifecycle-api/
├── src/
│   ├── api/
│   │   ├── models_api.py        # model management API (download/upload/register/stage)
│   │   ├── datasets_api.py
│   │   ├── inference_api.py     # unified inference API
│   │   ├── openai_api.py        # OpenAI-compatible API (/v1/*, xinference-style)
│   │   ├── gpu_api.py
│   │   └── config_api.py
│   ├── core/
│   │   ├── configs/
│   │   │   ├── base.yaml
│   │   │   └── inference.yaml   # adapter config (engines.ollama / transformers)
│   │   ├── config.py
│   │   ├── model_manager.py     # download / upload to LakeFS / register with MLflow (writes the spec tag)
│   │   ├── dataset_manager.py
│   │   └── gpu_manager.py
│   ├── inference/
│   │   ├── spec.py              # ModelSpec and MLflow tag parsing
│   │   ├── service.py           # InferenceService (replaces the old manager+router+executor)
│   │   ├── adapters/
│   │   │   ├── base.py          # BaseAdapter ABC (includes LRU)
│   │   │   ├── ollama.py
│   │   │   └── hf_transformers.py  # spec-driven dispatch
│   │   ├── handlers/
│   │   │   ├── base.py
│   │   │   ├── text_generation.py
│   │   │   ├── vlm.py           # includes the Qwen2.5-VL branch (determined via spec.model_class)
│   │   │   ├── asr.py
│   │   │   ├── ocr.py
│   │   │   ├── audio_classification.py
│   │   │   ├── video_analysis.py
│   │   │   └── document_analysis.py
│   │   ├── exceptions.py
│   │   ├── vram_estimator.py
│   │   ├── README.md            # inference module docs
│   │   ├── USAGE_GUIDE.md       # onboarding-flow walkthrough
│   │   └── REFACTORING_SUMMARY.md
│   └── main.py                  # FastAPI entry point
├── tmp/                         # scratch space for HF downloads / LakeFS dummy artifacts
├── data/                        # models downloaded from LakeFS for local inference
├── docker-compose.yaml
├── Dockerfile
└── pyproject.toml / uv.lock / requirements.lock.txt
```

---

## Inference architecture

A two-layer, spec-driven design:

```
┌─────────────────────────────────────────────────────────┐
│   API     POST /inference/infer  (FastAPI)              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│   InferenceService                                      │
│     1. canonicalize the task name                       │
│     2. validate required data fields                     │
│     3. resolve_spec(model_name) ← MLflow tag            │
│     4. get_adapter(spec.runtime).run(spec, data, opts)  │
└──────────────┬──────────────────────────┴───────────────┘
               │                          │
        runtime=ollama          runtime=hf-transformers
               │                          │
┌──────────────▼──────────┐    ┌──────────▼───────────────┐
│   OllamaAdapter         │    │   HFTransformersAdapter  │
│   • POST /api/generate  │    │   • spec-driven dispatch │
│   • daemon owns loading │    │   • LRU cache            │
└─────────────────────────┘    └──────────┬───────────────┘
                                          │
                              spec.task_family / custom_handler
                                          ▼
                          ┌──────────────────────────────┐
                          │   handlers/*                 │
                          │   encode → generate → decode │
                          └──────────────────────────────┘
```

Old classes removed: `InferenceManager` / `TaskRouter` / `ModelExecutor` / `ModelRegistry` / `ModelCache` / `BaseEngine` / `BaseModelHandler` (all absorbed into the two layers above).

See [src/inference/README.md](./src/inference/README.md) for the full write-up.

---

## ModelSpec: the model inference spec

Every MLflow model version's tags hold a spec describing "how this model should run" — written at registration, read back at inference.

### Schema summary

| Tag | Required | Description |
|-----|-----|------|
| `runtime` | ✅ | `ollama` / `hf-transformers` |
| `task_family` | ✅ | the canonical task (`text-generation`/`vlm`/`asr`/`ocr`/`audio-classification`/`video-analysis`/`document-analysis`/`image-captioning`/`tts`/`embedding`/`rerank`) |
| `physical_path` | ✅ | `lakefs://repo/commit/`, a local path, or an HF repo-id |
| `lakefs_commit_id` | – | for provenance |
| `model_class` | (HF, a) | the transformers model class name (e.g. `Qwen2_5_VLForConditionalGeneration`) |
| `processor_class` | (HF) | `AutoProcessor` / `AutoTokenizer` / `AutoFeatureExtractor` |
| `pipeline_task` | (HF, a) | takes the `transformers.pipeline` shortcut (e.g. `text-generation`) |
| `trust_remote_code` | – | defaults to `true` |
| `quantization` | – | `4bit` / `8bit` |
| `custom_handler` | – | a dotted path (e.g. `myorg.handlers.x:Cls`) or a built-in short name (`internvl` / `vibevoice`) |
| `ollama_model_name` | (Ollama) | the real name on the daemon (may differ from the MLflow registered name) |

(a) For the HF runtime: at least one of `model_class` or `pipeline_task` must be set.

### Backward compatibility

On **read**, `ModelSpec.from_mlflow_tags` still supports the old schema: `inference_engine` / `inference_task` / `ollama_model_name` are automatically mapped onto the new schema — old registrations need no migration.

On the **write** side, as of v3.1 there's a single source of truth — `register_*_to_mlflow` writes only the v3 schema (`runtime` / `task_family` / `model_class` / `processor_class` / `pipeline_task` / `physical_path` / `trust_remote_code` / `custom_handler` / `quantization`), and no longer writes back `inference_engine` / `inference_task`.

The only case that needs a manual tag patch: **an HF model registered under v2 with neither `model_class` nor `pipeline_task` set at all** (v2 relied on path-string matching to determine the model architecture) — the new service's first inference call returns a `ValidationError` prompting you to add the tag. See the example script at [USAGE_GUIDE — Migrating from v2.x](./src/inference/USAGE_GUIDE.md#migrating-from-v2x).

---

## OpenAI-Compatible API

`/v1/*` is an xinference-style compatibility layer: any OpenAI SDK/client pointing its `base_url` at this service can directly call a model already registered in MLflow (LLM, VLM, ASR) — the caller doesn't need to know the task or runtime.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8010/v1", api_key="not-needed")

# List registerable models (metadata carries task_family / runtime / stage)
client.models.list()

# LLM (either Ollama or HF transformers models work — runtime is decided by the MLflow tag)
resp = client.chat.completions.create(
    model="qwen3-1.7b-ollama",
    messages=[{"role": "user", "content": "Please introduce MLOps"}],
)

# VLM: same endpoint — an image_url part is automatically converted into a vlm inference
resp = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ],
    }],
)

# ASR (Whisper-style)
with open("clip.wav", "rb") as f:
    resp = client.audio.transcriptions.create(model="whisper-large-v3", file=f)
```

Model type → endpoint mapping:

| Model task_family | Endpoint |
|-----|-----|
| `text-generation` | `POST /v1/chat/completions`, `POST /v1/completions` |
| `vlm` / `image-captioning` | `POST /v1/chat/completions` (with an `image_url` part) |
| `asr` | `POST /v1/audio/transcriptions` |
| `tts` | `POST /v1/audio/speech` (returns a binary WAV), `POST /inference/tts` (returns base64 in JSON) |
| `embedding` | `POST /v1/embeddings` |
| `rerank` | `POST /v1/rerank` (jina/xinference-style) |
| others (ocr / audio-classification / video / document) | use the native `POST /inference/infer` |

Chat message handling is runtime-aware: Ollama models go through the daemon's `/api/chat` (the daemon applies the chat template), HF models use the tokenizer's `apply_chat_template`.

> `stream: true` is genuine token-by-token SSE streaming for **text-generation models** (ollama / hf-transformers);
> other tasks (vlm etc.) automatically fall back to a single-chunk pseudo-stream to keep clients compatible.

---

## Idle unload & GPU release

Ollama-style resource reclamation: a model idle past its lifetime is automatically unloaded and its GPU released, and automatically loaded back on the next request.

Three control layers (highest to lowest priority):

1. **Per-request `keep_alive`** (`options.keep_alive`, or the top-level `keep_alive` field on a `/v1` request)
   - Seconds, or a string (`"30s"` / `"5m"` / `"1h"`)
   - `0` = unload immediately after inference; negative = keep resident, never reclaimed
2. **Adapter defaults** (`inference.yaml`)
   - `engines.transformers.idle_timeout` (defaults to 300 seconds; `<=0` disables auto-unload)
   - `engines.ollama.keep_alive` (passed straight through to the Ollama daemon, keeping the daemon's own GPU-release cadence consistent)
3. **LRU capacity** (`max_cached_models`) — even so, the oldest model is still evicted when the cache is full (skipping models currently in flight)

A background reaper (`idle_reaper.interval`, scanning every 30 seconds by default) reclaims expired models and calls `torch.cuda.empty_cache()`. A model currently in flight (in-flight > 0) is never reclaimed.

```bash
# Don't unload this one until 10 minutes after this inference finishes
curl -X POST localhost:8010/inference/infer -H "Content-Type: application/json" -d '{
  "model_name": "gemma-2-2b-it",
  "data": {"inputs": "hi"},
  "options": {"keep_alive": "10m"}
}'

# Check the cache state (idle_seconds / expires_in_seconds / in_flight)
curl localhost:8010/inference/loaded-models
```

For Ollama models, `keep_alive` is also passed through to the daemon's `/api/generate`/`/api/chat`; a manual `POST /inference/unload-model` also tells the daemon to release that model's VRAM immediately.

---

## Deployment & configuration

### Prerequisites

- **Docker** + Docker Compose
- **An NVIDIA GPU + nvidia-container-toolkit** (for inference — optional but recommended)
- A **LakeFS** instance (or a shared platform-level LakeFS)
- An **Ollama** daemon (if running Ollama models; can be on the same host)
- For local development: Python 3.10+, [UV](https://github.com/astral-sh/uv) recommended

### Configuring the environment

1. Copy `.env.example` to `.env`, and at minimum fill in:

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

2. Confirm `mlflow.tracking_uri`, `lakefs.endpoint`, and `ollama.api_base` in `src/core/configs/base.yaml` and `inference.yaml` point at the right addresses. From inside a container, reaching a host service usually means using `host.docker.internal`.

### Starting up

```bash
docker compose up --build -d
```

### Service interfaces

- FastAPI Swagger UI: <http://localhost:8010/docs>
- MLflow UI: <http://localhost:5000>

---

## Example workflows

### Workflow A: quick registration & inference for an Ollama model

#### 1. Confirm the local Ollama model

```bash
ollama list
# or
curl "http://localhost:8010/models/local?model_source=ollama"
```

#### 2. Register with MLflow

`POST /models/register_ollama`

```json
{
  "local_model_name": "qwen3:1.7b",
  "task": "text-generation",
  "registered_name": "qwen3-1.7b-ollama",
  "stage": "production"
}
```

The registration call automatically writes the spec tag:

```
runtime              = ollama
task_family          = text-generation
ollama_model_name    = qwen3:1.7b
physical_path        = qwen3:1.7b
trust_remote_code    = true
```

> As of v3.1, new registrations **write only the v3 schema** (no longer writing back the old `inference_engine` / `inference_task` tags). The read side still supports old registrations — old models need no migration.

#### 3. Infer

```json
POST /inference/infer
{
  "task": "text-generation",
  "model_name": "qwen3-1.7b-ollama",
  "data": {"inputs": "Write a short introduction to MLOps in Traditional Chinese"},
  "options": {"temperature": 0.7, "max_length": 500}
}
```

Response:

```json
{
  "success": true,
  "result": {
    "response": "MLOps is a set of...",
    "model": "qwen3:1.7b",
    "metadata": {"total_duration": 3850000000, "eval_count": 245}
  },
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "qwen3-1.7b-ollama",
  "model_version": "1",
  "processing_time": 3.85,
  "timestamp": 1714195200.0
}
```

### Workflow B: HuggingFace → LakeFS → MLflow (a standard LLM)

#### 1. Download to local scratch space

`POST /models/download`

```json
{"model_source": "huggingface", "model_name": "google/gemma-2-2b-it"}
```

#### 2. Upload to LakeFS (get a commit id)

`POST /models/upload_to_lakefs`

```json
{"repo_name": "gemma-2-2b-it", "local_model_name": "google_gemma-2-2b-it"}
```

#### 3. Register with MLflow (**fill in the spec fields**)

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

#### 4. Infer

```json
POST /inference/infer
{
  "task": "text-generation",
  "model_name": "gemma-2-2b-it",
  "data": {"inputs": "Explain MLOps in 3 bullet points."},
  "options": {"max_new_tokens": 200, "temperature": 0.7}
}
```

### Workflow C: HuggingFace → LakeFS → MLflow (a VLM example: Qwen2.5-VL)

The only difference is setting `model_class` + `processor_class` at registration.

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

Inference:

```json
POST /inference/infer
{
  "task": "vlm",
  "model_name": "qwen2.5-vl-7b",
  "data": {
    "image": "/data/images/scene.jpg",
    "prompt": "Describe this image in detail"
  },
  "options": {"max_new_tokens": 256}
}
```

The `image` field accepts a local path, a PIL.Image, base64, or a `data:image/...` URL.

### Workflow C2: InternVL (the VLM shared by modules 10/11/12)

InternVL uses the `.chat()` interface + dynamic tiling; Module 07 already has a dedicated built-in handler, so no code is needed at registration:

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

The handler is auto-detected from the string `InternVL` appearing in `model_class`/`physical_path`;
it can also be set explicitly with `"custom_handler": "internvl"` (a built-in short name). Inference is the same as Workflow C (task=vlm).

### Workflow F: serving embedding / rerank / TTS

```json
// Register bge-m3 (the embedding model used by 17-hybrid-search)
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

// Register bge-reranker-v2-m3
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

Calling them (OpenAI / jina compatible):

```bash
# embedding
curl -X POST localhost:8010/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["What is MLOps?", "vector retrieval"]}'

# rerank
curl -X POST localhost:8010/v1/rerank -H "Content-Type: application/json" \
  -d '{"model": "bge-reranker-v2-m3", "query": "What is MLOps",
       "documents": ["MLOps is machine learning operations", "The weather is nice today"], "top_n": 1}'

# TTS (module 09-audio-processing's audio_tts_service uses this endpoint)
curl -X POST localhost:8010/inference/tts -H "Content-Type: application/json" \
  -d '{"text": "Welcome to Raptor", "output_format": "wav"}'
```

The model name can be omitted for the TTS endpoint (defaults to the `DEFAULT_TTS_MODEL` env var, default `vibevoice-tts`).
A standard HF TTS model (SpeechT5 / VITS / Bark) only needs `pipeline_task="text-to-speech"` at registration;
non-standard interfaces like VibeVoice need a separate custom_handler (see Workflow D).

### Workflow D: a custom model (custom_handler)

When the default handler isn't enough (e.g. an in-house multi-head model, a special chat template, a non-standard generate loop):

1. Write a custom handler class, subclassing `src.inference.handlers.base.BaseHandler`, implementing `run(loaded, spec, data, options)`.
2. Pass `custom_handler="myorg.handlers.x:CustomHandler"` at registration.
3. The service lazy-imports it on the first inference call.

See [USAGE_GUIDE — Scenario G](./src/inference/USAGE_GUIDE.md#scenario-g--a-fully-bespoke-model-custom_handler) for details.

### Workflow E: switching stages

`POST /models/transition_stage`

```json
{
  "model_name": "qwen3-1.7b-ollama",
  "version": "2",
  "stage": "production",
  "archive_existing_versions": true
}
```

With `archive_existing_versions=true`, any other production version under that name gets archived.

---

## API endpoint list

Full interactive docs at <http://localhost:8010/docs>.

### Model management `/models`

| Endpoint | Method | Description |
|-----|-----|------|
| `/download` | POST | Pull a model from HuggingFace or Ollama into tmp |
| `/local` | GET | List locally-staged models (`?model_source=ollama` filters to Ollama) |
| `/upload_to_lakefs` | POST | Upload a local model to LakeFS |
| `/lakefs_repos` | GET | List LakeFS repositories |
| `/register_from_lakefs` | POST | Register from a LakeFS commit into MLflow (**supports spec fields**) |
| `/register_ollama` | POST | Register a local Ollama model into MLflow |
| `/registered_in_mlflow` | GET | List models registered in MLflow |
| `/registered_in_mlflow/{model_name}` | GET | Get details for a single model |
| `/transition_stage` | POST | Switch stage (production/staging/archived/none) |
| `/batch_download` | POST | Bulk download |
| `/stats` | GET | Model statistics |
| `/health` | GET | Service health check |

### OpenAI-compatible `/v1`

| Endpoint | Method | Description |
|-----|-----|------|
| `/models` | GET | List registered models (OpenAI format, metadata carries task_family/runtime/stage) |
| `/chat/completions` | POST | Chat (works for both text-generation and vlm models; supports the extended `keep_alive` field) |
| `/completions` | POST | Classic prompt completion (text-generation models) |
| `/audio/transcriptions` | POST | Speech-to-text (asr models, multipart upload) |
| `/audio/speech` | POST | Text-to-speech (tts models; returns binary WAV) |
| `/embeddings` | POST | Text embedding (embedding models; OpenAI format) |
| `/rerank` | POST | Document reranking (rerank models; jina/xinference format) |

### Inference service `/inference`

| Endpoint | Method | Description |
|-----|-----|------|
| `/infer` | POST | **The unified inference endpoint** (every task; `task` can be omitted; `options.keep_alive` controls unloading) |
| `/tts` | POST | Text-to-speech convenience endpoint (`{"text", "voice", "speed", "output_format"}`; the model can be omitted — uses `DEFAULT_TTS_MODEL`) |
| `/health` | GET | service + adapter status |
| `/stats` | GET | inference count, success rate |
| `/supported-tasks` | GET | task families and their required fields |
| `/loaded-models` | GET | lists the models in each adapter's LRU (with idle/expiry/in-flight status) |
| `/cache/clear` | POST | unload every model and clear the GPU cache |
| `/unload-model?model_name=X` | POST | unload one specific model |
| `/unload-all-models` | POST | unload everything |

### Main fields for `/models/register_from_lakefs` (spec-driven)

| Field | Required | Description |
|-----|-----|------|
| `registered_name` | ✅ | the registered name in MLflow |
| `task` | ✅ | the task (legacy names still work, auto-canonicalized) |
| `engine` | ✅ | `transformers` / `ollama` (→ written to `runtime`) |
| `model_params` | ✅ | parameter count (in billions, used to estimate VRAM) |
| `lakefs_repo` | ✅ | the LakeFS repository name |
| `commit_id` | – | if omitted, uses the branch's latest commit |
| `stage` | – | the stage to switch to right after registration |
| `quantization` | – | `4bit` / `8bit` |
| `model_class` | (HF, a) | the transformers class name |
| `processor_class` | – | the processor / tokenizer class |
| `pipeline_task` | (HF, a) | takes the pipeline shortcut |
| `trust_remote_code` | – | defaults to `true` |
| `custom_handler` | – | a dotted path |

(a) For the HF runtime: at least one of `model_class` or `pipeline_task` must be set.

---

## FAQ

**Q1: Can I still run inference on an Ollama model I registered under the old schema?**
Yes. The old tags (`inference_engine=ollama` + `ollama_model_name=...`) are automatically read into the new spec.

**Q2: What about an HF model I registered under the old schema?**
If it was registered under v2 via path-string matching (no `model_class`, no `pipeline_task`), the new service's first inference call returns a `ValidationError` prompting you to add the tag. Use `MlflowClient.set_model_version_tag` to patch it; see the example script at [USAGE_GUIDE — Migrating from v2.x](./src/inference/USAGE_GUIDE.md#migrating-from-v2x).

**Q3: What values can the `task` field take?**
The canonical name (`text-generation` / `vlm` / `asr` / `ocr` / `audio-classification` / `video-analysis` / `document-analysis` / `image-captioning` / `tts` / `embedding` / `rerank`), or a legacy name/alias (`text-generation-ollama` / `text-generation-hf` / `asr-hf` / `vad-hf` / `ocr-hf` / `audio-transcription` / `scene-detection` / `video-summary` / `text-to-speech` / `feature-extraction` / `reranker`). The latter are auto-converted to the former.

**Q4: Do I still need to pass `engine` at inference time?**
No. As of v3.0, the runtime is decided by the MLflow tag — the `engine` field on a request is ignored (kept only so existing clients don't break).

**Q5: What do I do about GPU OOM?**
Immediately call `POST /inference/cache/clear`. Then consider: (a) adding `quantization: "4bit"` at registration; (b) lowering `max_cached_models` or `idle_timeout` in `inference.yaml.engines.transformers`; (c) giving that model a smaller `max_new_tokens`; (d) passing `options.keep_alive: 0` for one-off requests (unload right after use).

**Q5b: When does the GPU get released once a model goes idle?**
HF models: once idle past `idle_timeout` (default 300 seconds), a background reaper (scanning every `idle_reaper.interval` seconds) automatically unloads it and calls `torch.cuda.empty_cache()`. Ollama models: released by the daemon itself, according to `keep_alive` (also defaulting to 300 seconds). Both can be overridden per request via `keep_alive`; `GET /inference/loaded-models` shows each model's `idle_seconds` and `expires_in_seconds`.

**Q6: How do I onboard a model architecture transformers doesn't support directly (an in-house trained one)?**
Write a `custom_handler`: subclass `BaseHandler`, implement `run()`, and put the dotted path into the `custom_handler` tag at registration. See [USAGE_GUIDE — Scenario G](./src/inference/USAGE_GUIDE.md#scenario-g--a-fully-bespoke-model-custom_handler) for details.

**Q7: What's the service startup order?**
`docker compose up` starts things in dependency order: `mlflow-postgres → mlflow → api`. If LakeFS is an external service, make sure it's reachable first.

**Q8: What are the health-check endpoints?**
- Overall: `GET http://localhost:8010/health`
- Model management: `GET /models/health`
- Inference: `GET /inference/health`
- MLflow: `GET http://localhost:5000/health`

---

## Further reading

- [Inference module design](./src/inference/README.md)
- [Onboarding-flow walkthrough](./src/inference/USAGE_GUIDE.md)
- [v2 → v3 refactoring summary](./src/inference/REFACTORING_SUMMARY.md)
- [The original strategy document (design strategy)](./docs/design-strategy.md)
- [UV dependency management](./docs/UV_DEPENDENCY_MANAGEMENT.md)
