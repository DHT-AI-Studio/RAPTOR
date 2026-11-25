# Audio Processing Module

GPU-isolated audio pipeline: speech-to-text, speaker diarization, sound-event
classification, LLM summarization, and personal-index publishing, driven entirely
by Kafka. Runs on the shared `raptor/media-worker` image (built by
[08-media-worker](../08-media-worker/)) — this module has no `build:` of its own;
`WORKER_TYPE` selects which service `main.py` dispatches to inside the container.

## Services

| Container | `WORKER_TYPE` | GPU | Role |
|---|---|---|---|
| `raptor-audio-kafka-init` | — | — | One-shot: creates every topic below, then exits |
| `raptor-audio-orchestrator` | `audio_orchestrator` | no | Consumes `audio-processing-requests`; fans out to classifier/diarization/recognizer/summary/indexer requests |
| `raptor-audio-analysis` | `audio_analysis` | no | Consumes `audio-analysis-requests`; merges classifier + recognizer + diarization results into one record |
| `raptor-audio-summary` | `audio_summary` | no | Consumes `audio-summary-requests`; LLM summary via Module 07 (Ollama, `INFERENCE_MODEL`, default `gemma4:12b`) |
| `raptor-audio-indexer` | `audio_indexer` | no | Consumes `audio-indexer-requests`; guardrail moderation scan + publishes to the personal index (Module 25) |
| `raptor-audio-tts` | `audio_tts` | no | **Orphaned** (`profiles: [tts]`, not started by default) — superseded by Module 34/35's TTS (`/audio/synthesize`); nothing anywhere in the repo produces to `audio-tts-requests`, so even starting this profile by hand, it never receives work. Also has its own live bug: `tts_client.py` posts to `INFERENCE_URL` with no path appended, but `worker-common`'s shared default for that var has no `/inference/tts` suffix — never hit in practice since nothing reaches this code |
| `raptor-audio-recognizer` | `audio_recognizer` | yes | WhisperX STT (`WHISPER_MODEL`, default `large-v3`); also exposes a synchronous HTTP API (below) |
| `raptor-audio-diarization` | `audio_diarization` | yes | WhisperX + pyannote speaker diarization (needs `HF_TOKEN` with pyannote model access) |
| `raptor-audio-classifier` | `audio_classifier` | yes | PANNs audio tagging / sound-event detection |
| `raptor-audio-cleanup` | — | — | `busybox` sidecar: deletes `/tmp/media_processing/audio_processing/*` older than 24h, hourly |
| `raptor-audio-gpu-watchdog` | — | — | Restarts recognizer/diarization/classifier if their `torch.cuda.is_available()` healthcheck goes unhealthy |

No container is named `aigle-*`, has a static IP, or resolves at `*.aigle.local` —
every service is reachable only by its docker-network alias (`raptor-audio-*`),
same convention as the rest of this repo.

## Kafka topics

Created once by `audio-kafka-init-topics`. Each stage has its own
`*-requests`/`*-results`/`*-dlq` set (DLQs are written to, never consumed — see the
repo-wide known limitation in [`docs/CHANGELOG.md`](../../../docs/CHANGELOG.md)):

```
audio-processing-requests ──▶ audio-orchestrator ──┬─▶ audio-recognizer-requests  ──▶ audio-recognizer  ──▶ audio-recognizer-results  ─┐
                                                     ├─▶ audio-diarization-requests ──▶ audio-diarization ──▶ audio-diarization-results ─┼─▶ audio-analysis-requests ──▶ audio-analysis ──▶ audio-analysis-results
                                                     ├─▶ audio-classifier-requests  ──▶ audio-classifier  ──▶ audio-classifier-results  ─┘
                                                     ├─▶ audio-summary-requests     ──▶ audio-summary     ──▶ audio-summary-results
                                                     └─▶ audio-indexer-requests     ──▶ audio-indexer     ──▶ audio-indexer-results
```

`audio-processing-requests`/`-results` (6 partitions) carry the top-level job;
`audio-analysis-requests`/`-results` (12 partitions — the merge step, highest
throughput need) carry the fan-in. `audio-tts-requests`/`-results` are created but
orphaned — `audio-orchestrator` doesn't produce to `audio-tts-requests`, and no
other service in the repo does either, so `audio-tts` never actually receives
anything even when its profile is started by hand.

## Synchronous HTTP APIs

Three of the GPU services also expose a plain HTTP port, independent of their Kafka
consumer — used for synchronous, blocking calls rather than async job dispatch:

| Service | Port (`PORT_*`) | Purpose |
|---|---|---|
| `audio-recognizer` | `PORT_AUDIO_SYNC_API` (8019) | Upload audio → recognition + diarization + classification → merge → summary → `{text, summary}`, one blocking call |
| `audio-diarization` | `PORT_AUDIO_DIARIZATION_SYNC` (8029) | Called by the recognizer above (`DIARIZATION_SYNC_URL`) |
| `audio-classifier` | `PORT_AUDIO_CLASSIFIER_SYNC` (8039) | Called by the recognizer above (`CLASSIFIER_SYNC_URL`) |

## GPU

`audio-recognizer`, `audio-diarization`, `audio-classifier` each reserve one GPU
(`AUDIO_RECOGNIZER_GPU`/`AUDIO_DIARIZATION_GPU`/`AUDIO_CLASSIFIER_GPU`, default `0`/`0`/`2`
— assign different devices to avoid VRAM contention on a multi-GPU host). Each has a
`torch.cuda.is_available()` healthcheck; `audio-gpu-watchdog` restarts a container that
goes unhealthy (CUDA/NVML going stale on long uptimes), the same pattern module 16's
training-watchdog uses.

## Key environment variables

Full reference: [`docs/AUDIO_PROCESSING_SETTINGS.md`](docs/AUDIO_PROCESSING_SETTINGS.md).

| Variable | Default | Used by |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `audio-recognizer` |
| `HF_TOKEN` | — | `audio-diarization` (pyannote model access, required) |
| `INFERENCE_MODEL` | `gemma4:12b` | `audio-summary` (via Module 07 / Ollama) |
| `GUARDRAIL_URL` | — | `audio-indexer` (moderation scan before publishing) |
| `ASSET_MANAGEMENT_URL` | `http://raptor-asset-management:8000` | `audio-orchestrator` |
| `HYBRID_SEARCH_INGEST_URL` | `http://raptor-hybridsearch-api:8000/api/v1/ingest/file` | shared worker env (legacy path, see module 17's own deprecation note) |

## Deploy

```bash
python build.py -m 09
# with TTS enabled:
COMPOSE_PROFILES=tts python build.py -m 09
```
