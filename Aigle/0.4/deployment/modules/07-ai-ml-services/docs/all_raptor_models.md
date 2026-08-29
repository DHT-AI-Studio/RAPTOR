# Raptor Model Inventory Across All Modules & Module 07 Inference Coverage Analysis

> Survey date: 2026-07-07 (branch `ModelLifecycle-george`)
> Update: 2026-07-07 — Module 07 has expanded the `tts` / `embedding` / `rerank` task families and the InternVL handler (see §4).
> Purpose: catalog every model actually in use across all modules, and assess which ones
> Module 07-ai-ml-services (the AI Lifecycle API / inference gateway) can serve directly, and which need further work.

---

## 1. Module 07's current inference capability (baseline)

- **Runtime (adapter)**: `ollama` (proxies to an external daemon), `hf-transformers` (in-process loading).
- **Task family** (`src/inference/spec.py`): `text-generation` / `vlm` / `asr` / `ocr` /
  `audio-classification` / `video-analysis` / `document-analysis` / `image-captioning` /
  **`tts` / `embedding` / `rerank` (added 2026-07)**.
- **VLM handler**: generic (processor → generate → decode) + **a dedicated Qwen2.5-VL branch** +
  **a dedicated InternVL branch (added 2026-07, dynamic tiling + `.chat()`)**.
- **Extension point**: passing `custom_handler` at registration (a dotted path, or a built-in short name like `"internvl"`) lets any non-standard model plug in without changing Module 07's code.
- **External interface**: native `POST /inference/infer`, `POST /inference/tts`, plus OpenAI-compatible
  `/v1/chat/completions`, `/v1/completions`, `/v1/audio/transcriptions`,
  `/v1/audio/speech`, `/v1/embeddings`, `/v1/rerank`.
- **Still missing**: no task family for `speaker-diarization`;
  no adapter for non-transformers runtimes such as PaddleOCR, faster-whisper (CTranslate2), pyannote, PANNs
  (these can be plugged in via `custom_handler`, but the dependencies would need to be installed into Module 07's image).

---

## 2. Models currently in use, by module

### 2.1 Ollama models (external daemon `192.168.157.135:11434`)

| Model | Consuming module(s) / purpose | Can Module 07 serve it? |
|------|----------------|------------|
| `qwen3.5:9b` | 09 audio summary, 11 video summary/context, 12 document summary + entity extraction, 15 chat (`INFERENCE_MODEL` / `LLM_MODEL`) | ✅ Yes, after `register_ollama` (text-generation) |
| `qwen2.5:7b` | 13 gateway (intent/answer/smolagents), 20 graph-service, 21 agent-protocol | ✅ Same as above |
| `qwen3.5:0.8b` | 18 query-orchestrator (LLM rerank, `QO_INFERENCE_MODEL`) | ✅ Same as above |

> These modules currently all **connect directly to the Ollama daemon** (OpenAI-compatible `/v1`), bypassing Module 07.
> To unify onto Module 07, register the model there and point `base_url` at `raptor-ai-lifecycle-api:8010/v1` — the API surface is compatible.

### 2.2 HuggingFace / locally-loaded models (non-Ollama)

| Model | Consuming module(s) / purpose | Actual runtime | Can Module 07 serve it? |
|------|----------------|--------------|------------|
| `OpenGVLab/InternVL3_5-1B` (default; code falls back to 4B) | 10 image captioning, 11 video-frame-description, 12 document VLM analysis + PDF OCR (`VLM_MODEL_PATH` / `DOCUMENT_ANALYSIS_VLM_MODEL`) | transformers `AutoModel` + a custom `model.chat()` flow (trust_remote_code) | ✅ **Supported (2026-07)**: built-in `InternVLHandler` (dynamic tiling + `.chat()`); register with `task=vlm, model_class=AutoModel, processor_class=AutoTokenizer, torch_dtype=bf16` — auto-detected from path/class name, or set `custom_handler="internvl"` explicitly |
| WhisperX `large-v3` | 09 audio_recognizer (ASR + alignment + word-level timestamps) | whisperx (faster-whisper/CTranslate2 + alignment model + VAD) | ⚠️ Partial. Module 07's asr handler can run the **transformers version** `openai/whisper-large-v3` (plain transcription); WhisperX's word-level alignment/VAD pipeline needs a `custom_handler` or should stay in module 09 |
| `pyannote/speaker-diarization-3.1` | 09 audio_diarization (speaker separation, requires an HF token) | pyannote.audio pipeline | ❌ No corresponding task family, not a generate-style model. Needs a `custom_handler` + a new task, or should stay in module 09 |
| PANNs (AudioTagging + SoundEventDetection) | 09 audio_classifier (audio event tagging) | `panns_inference` (ships its own checkpoint) | ❌ Not an HF model. Module 07's audio-classification handler only supports HF processor+logits-style models (e.g. AST). Needs a `custom_handler` |
| `PP-OCRv5_mobile_det` / `PP-OCRv5_mobile_rec` | 11 video-ocr-frame (`OCR_DET_MODEL` / `OCR_REC_MODEL`) | PaddleOCR (PaddlePaddle) | ❌ Module 07's ocr handler is transformers image-to-text style (TrOCR/Donut). The Paddle runtime needs a new adapter or a `custom_handler` |
| `BAAI/bge-m3` | 17 hybrid-search (embedding); 12 only uses its **tokenizer** for chunk-size calculation (not inference) | FlagEmbedding / sentence-transformers | ✅ **Supported (2026-07)**: `task=embedding, model_class=AutoModel, processor_class=AutoTokenizer`; endpoint `/v1/embeddings` (CLS pooling + L2 normalize, matching BGE's official usage) |
| `BAAI/bge-reranker-v2-m3` | 17 hybrid-search (reranker, `RERANKER_MODEL`) | FlagEmbedding cross-encoder | ✅ **Supported (2026-07)**: `task=rerank, model_class=AutoModelForSequenceClassification`; endpoint `/v1/rerank` (jina/xinference style) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local fallback for 21 reranker_agent | sentence-transformers CrossEncoder | ✅ Same as above (rerank, 2026-07) |
| `microsoft/VibeVoice-ASR` | Configured in 13 gateway (`GATEWAY_VIBEVOICE_ASR_MODEL`) — currently only declared in config, no actual call site found | (not wired up) | ⚠️ If it uses the standard HF transformers ASR interface, Module 07 could serve it — needs verification |
| `vibevoice-tts` | 09 audio_tts_service → calls **Module 07's `POST /inference/tts`** | — | ✅ **Endpoint now available (2026-07)**: `POST /inference/tts` (contract aligned with module 09's tts_client) + `/v1/audio/speech`; a standard HF TTS model (SpeechT5/VITS/Bark) works via `pipeline_task=text-to-speech` — **VibeVoice itself still needs a custom_handler registered, plus a model named `vibevoice-tts` registered in MLflow** (or set `DEFAULT_TTS_MODEL`) |
| `google/gemma-3-1b-it` | 16 training-service (default base model for fine-tuning) | transformers (training use) | (not an inference need; if the trained artifact is registered to MLflow, Module 07 can serve its text-generation inference) |

### 2.3 Leftover / dangling references

| Reference | Location | Status |
|------|------|------|
| `raptor-temporal-model-service:8000/analyze/video` (`GATEWAY_VISION_SERVICE_URL`) | 13 config / .env | No compose file defines this service anywhere — a leftover from the old architecture |

---

## 3. Conclusion: what can Module 07 absorb directly? Do we need a new inference module?

**No need for a separate new inference "module"** — Module 07's spec-driven architecture (`custom_handler` + extensible adapters)
was designed precisely to consolidate these runtimes.

### ✅ Servable by Module 07 right now (after the 2026-07 expansion)
1. All Ollama LLMs (qwen3.5:9b / qwen2.5:7b / qwen3.5:0.8b) — just `register_ollama`.
2. Standard HF text-generation models (including the gemma family trained by module 16).
3. The Qwen2.5-VL family of VLMs (has a dedicated branch).
4. **The InternVL family of VLMs** (InternVL3_5, used by 10/11/12) — via the built-in `InternVLHandler`.
5. The transformers version of Whisper (`openai/whisper-large-v3`, plain transcription, including `/v1/audio/transcriptions`).
6. HF pipeline-style OCR (TrOCR/Donut), audio-classification (AST etc.).
7. **Embedding models** (bge-m3 etc.) — via `/v1/embeddings`.
8. **Rerank models** (bge-reranker-v2-m3, ms-marco cross-encoder, etc.) — via `/v1/rerank`.
9. **Standard HF TTS** (SpeechT5 / VITS / Bark, `pipeline_task=text-to-speech`) —
   via `POST /inference/tts` + `/v1/audio/speech`.

### ⚠️ Still needs a `custom_handler` written first (no change to Module 07's core, one handler class each)
1. **VibeVoice TTS/ASR** — non-standard transformers interface; the endpoint is ready, just missing the handler + model registration.
2. The full **WhisperX** pipeline (word-level timestamps + VAD).
3. **pyannote diarization**, **PANNs** (also worth considering whether it's worth moving off module 09 at all).
4. **PaddleOCR PP-OCRv5** (or keep it embedded in module 11; Paddle's dependencies are heavy — worth evaluating before adding to Module 07's image).

### Per-module migration guide
| Module | Current state | How to switch to Module 07 |
|------|------|----------------|
| 09 audio_tts | Calls `/inference/tts` (previously 404) | **Endpoint now works**; register the `vibevoice-tts` model (or change `DEFAULT_TTS_MODEL`) and it's usable |
| 10/11/12's InternVL | Each loads it in-process separately (3x VRAM) | Register InternVL once to MLflow → have all three modules call `/inference/infer` (task=vlm) or `/v1/chat/completions` instead |
| 17 embedding/rerank | In-process SentenceTransformer/CrossEncoder | Can switch to calling `/v1/embeddings`, `/v1/rerank` (OpenAI/jina-compatible) for centralized GPU scheduling |
| 21 reranker fallback | Local CrossEncoder | Same as above, switch to `/v1/rerank` |
| Each module's Ollama LLM | Connects directly to the daemon's `/v1` | Optional: after `register_ollama`, point `base_url` at Module 07's `/v1` — interface-compatible |

### Follow-up work
1. ~~Token-by-token streaming~~ ✅ **Done (2026-07)**: `/v1/chat/completions` and `/v1/completions`,
   when `stream=true`, now do genuine token-by-token SSE for text-generation models (both the ollama
   and hf-transformers runtimes); other tasks (VLM etc.) automatically fall back to a single-chunk
   pseudo-stream. Live-tested on the dev stack with gemma3-12b (ollama) and an HF model.
2. ~~VibeVoice custom_handler~~ ⚠️ **Code complete, not yet live-tested (2026-07)**:
   built-in `custom_handler="vibevoice"` (`handlers/vibevoice_tts.py`, lazy-loaded),
   with a registration/invocation script at `scripts/09_tts_vibevoice.sh`. **Three external
   prerequisites still missing**: (a) the vibevoice package installed into Module 07's image
   (the official GitHub/HF repo has been taken down — need the community fork
   `vibevoice-community/VibeVoice`, or an internal backup); (b) the model weights (find a mirror);
   (c) a reference voice wav (required for voice cloning, via `options.voice` or `VIBEVOICE_DEFAULT_VOICE`).
3. WhisperX / pyannote / PaddleOCR / PANNs migration — **still undecided**. This is a cross-module
   architectural decision: the benefit of migrating is centralized GPU scheduling and version
   governance; the cost is pulling whisperx/pyannote/paddle's heavy dependencies into Module 07's
   image (image bloat, dependency-conflict risk). Recommendation: wait until 09/11 hit real GPU
   contention problems before migrating; at that point, one custom_handler per runtime is all
   that's needed — Module 07's core doesn't have to change.
