# Model Usage Scripts

Each script demonstrates the full lifecycle of one model category: **download → upload to lakeFS → register with MLflow → invoke inference**.
All operate through Module 07's HTTP API (curl) — none touch the MLflow / lakeFS SDKs directly.

## Prerequisites

- Module 07 is running (`docker compose up -d`), MLflow / lakeFS reachable
- Ollama scripts require the model already present on the daemon; HF scripts require `HF_TOKEN` (for gated models like gemma)
- `curl` and `python3` (for pretty-printed output) available locally

## Shared environment variables (all scripts)

| Variable | Default | Description |
|------|------|------|
| `API_BASE` | `http://localhost:8010` | Module 07 API address; use `http://localhost:9997` for the dev stack |
| `STAGE` | `staging` | Stage to switch to after registration |
| `SKIP_DOWNLOAD=1` | – | Skip download/upload when the model is already in lakeFS |
| `SKIP_REGISTER=1` | – | Skip registration when the model is already in MLflow — go straight to inference |
| `HF_MODEL` / `MODEL_PARAMS` / `LAKEFS_REPO` / `REGISTERED_NAME` | per-script defaults | Override when swapping models |

## Script overview

| Script | Model (default) | task | Registration mode | Endpoint invoked |
|------|--------------|------|----------|----------|
| `01_ollama_llm.sh` | qwen2.5:7b | text-generation | `register_ollama` (bypasses lakeFS) | `/inference/infer`, `/v1/chat/completions` |
| `02_hf_text_generation.sh` | google/gemma-3-270m-it | text-generation | `pipeline_task=text-generation` | `/inference/infer`, `/v1/completions` |
| `03_vlm_qwen2.5_vl.sh` | Qwen/Qwen2.5-VL-7B-Instruct | vlm | `model_class=Qwen2_5_VLForConditionalGeneration` + 4bit | `/inference/infer` |
| `04_vlm_internvl.sh` | OpenGVLab/InternVL3_5-1B | vlm | `custom_handler=internvl` + `torch_dtype=bf16` | `/inference/infer` |
| `05_asr_whisper.sh` | openai/whisper-large-v3 | asr | `pipeline_task=automatic-speech-recognition` | `/inference/infer`, `/v1/audio/transcriptions` |
| `06_embedding_bge_m3.sh` | BAAI/bge-m3 | embedding | `model_class=AutoModel` | `/v1/embeddings`, `/inference/infer` |
| `07_rerank_bge.sh` | BAAI/bge-reranker-v2-m3 | rerank | `model_class=AutoModelForSequenceClassification` | `/v1/rerank`, `/inference/infer` |
| `08_tts.sh` | facebook/mms-tts-eng | tts | `pipeline_task=text-to-speech` | `/inference/tts`, `/v1/audio/speech` |
| `09_tts_vibevoice.sh` ⚠️ | VibeVoice (requires mirrored weights + package) | tts | `custom_handler=vibevoice` + dotted class | `/inference/tts` |

> ⚠️ `09_tts_vibevoice.sh` is experimental: the official repo/weights have been taken down, so a community
> fork package must be installed manually and the weights obtained from a mirror (see the prerequisites note
> at the top of the script). Once working, Module 09-audio-processing's TTS is fully wired up.

## Usage examples

```bash
# On the dev stack, validate the whole flow first with a small model (only ~90MB download)
API_BASE=http://localhost:9997 \
HF_MODEL=sentence-transformers/all-MiniLM-L6-v2 MODEL_PARAMS=0.02 \
./06_embedding_bge_m3.sh

# Run bge-m3 for real
API_BASE=http://localhost:9997 ./06_embedding_bge_m3.sh

# Model already registered — just re-run inference
SKIP_DOWNLOAD=1 SKIP_REGISTER=1 ./06_embedding_bge_m3.sh

# InternVL image captioning (a local file path in IMAGE is auto-converted to base64)
IMAGE=./sample.jpg ./04_vlm_internvl.sh

# TTS synthesis, saved to a file
TEXT="Welcome to Raptor" OUT=./hello.wav ./08_tts.sh
```

## How other modules plug in

- **09 audio_tts_service** → `08_tts.sh`'s `/inference/tts` (09's payload carries no `model_name`;
  the model is decided by Module 07's `DEFAULT_TTS_MODEL` env var; default `vibevoice-tts`)
- **10/11/12's InternVL** → `04_vlm_internvl.sh` (register once, shared by all three modules)
- **17 hybrid-search** → `06_embedding_bge_m3.sh` + `07_rerank_bge.sh`
- **21 reranker fallback** → `07_rerank_bge.sh` (`HF_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Each module's Ollama LLM** → `01_ollama_llm.sh` (after registering, just point `base_url` at Module 07's `/v1`)

## Troubleshooting

- Registration/inference errors print the API's `detail` JSON directly (`error_type` / `message`)
- GPU OOM: `curl -X POST $API_BASE/inference/cache/clear`, or add `"quantization": "4bit"` at registration
- View loaded models and idle status: `curl $API_BASE/inference/loaded-models`
- To unload immediately after a one-off request: add `"keep_alive": 0` to `options`
