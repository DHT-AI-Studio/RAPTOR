# Video Processing Module

GPU-isolated video pipeline: chunking/normalize, per-frame OCR + face detection +
VLM description, scene analysis, LLM summary/contextualization, personal-index
publishing, and graph ingestion — all Kafka-driven. Runs on the shared
`raptor/media-worker` image (built by [08-media-worker](../08-media-worker/)) — this
module has no `build:` of its own; `WORKER_TYPE` selects which service `main.py`
dispatches to inside the container.

## Services

| Container | `WORKER_TYPE` | GPU | Role |
|---|---|---|---|
| `raptor-video-kafka-init` | — | — | One-shot: creates every topic below, then exits |
| `raptor-video-orchestrator` | `video_orchestrator` | no | Consumes `video-processing-requests`; fans out to every stage below |
| `raptor-video-chunking` | `video_chunking` | optional | ffmpeg normalize/chunk (GPU hwaccel tier, falls back to CPU libx264 — see `VIDEO_NORMALIZE_TIMEOUT_SEC` and the NVENC known limitation in [`docs/CHANGELOG.md`](../../../docs/CHANGELOG.md)) |
| `raptor-video-analysis` | `video_analysis` | no | Scene/shot analysis |
| `raptor-video-ocr-frame` | `video_ocr_frame` | yes | Per-frame OCR (PaddleOCR, `OCR_DET_MODEL`/`OCR_REC_MODEL`); also exposes a synchronous HTTP API |
| `raptor-video-face-detection` | `video_face_detection` | yes | Face detection, gated by consent (`FACE_DETECTION_ENABLED`, Module 33); also exposes a synchronous `/embed-face` API sharing the same resident ONNX session as its Kafka consumer |
| `raptor-video-frame-description` | `video_frame_description` | yes | InternVL3.5 VLM frame description; also exposes a synchronous HTTP API |
| `raptor-video-summary` | `video_summary` | no | LLM summary via Module 07 (Ollama, `INFERENCE_MODEL`) |
| `raptor-video-contextualize` | `video_contextualize` | no | LLM contextualization pass, also via Module 07 |
| `raptor-video-indexer` | `video_indexer` | no | Publishes to the personal index (Module 25) |
| `raptor-video-graph` | `video_graph` | no | Consumes `video-indexer-results` (triggered after indexing, not its own `-requests` topic); ingests into the graph layer |
| `raptor-video-cleanup` | — | — | `busybox` sidecar: deletes `/tmp/media_processing/video_processing/*` older than 24h, hourly |
| `raptor-video-gpu-watchdog` | — | — | Restarts chunking/ocr-frame/face-detection/frame-description if their `torch.cuda.is_available()` healthcheck goes unhealthy |

No container is named `aigle-*`, has a static IP, or resolves at `*.aigle.local` —
every service is reachable only by its docker-network alias (`raptor-video-*`), same
convention as the rest of this repo.

## Kafka topics

Created once by `video-kafka-init-topics`. Each stage has its own
`*-requests`/`*-results` set (most also have a `-dlq`, written to but never consumed
— see the repo-wide known limitation in
[`docs/CHANGELOG.md`](../../../docs/CHANGELOG.md)):

```
video-processing-requests ──▶ video-orchestrator ──┬─▶ video-chunking-requests        ──▶ video-chunking        ──▶ video-chunking-results
                                                     ├─▶ video-analysis-requests       ──▶ video-analysis        ──▶ video-analysis-results
                                                     ├─▶ video-ocr-frame-requests      ──▶ video-ocr-frame       ──▶ video-ocr-frame-results
                                                     ├─▶ video-face-detection-requests ──▶ video-face-detection  ──▶ video-face-detection-results
                                                     ├─▶ video-frame-description-requests ──▶ video-frame-description ──▶ video-frame-description-results
                                                     ├─▶ video-summary-requests        ──▶ video-summary         ──▶ video-summary-results
                                                     └─▶ video-contextualize-requests  ──▶ video-contextualize   ──▶ video-contextualize-results
                                                                                                                         │
                                                                               video-indexer-requests ──▶ video-indexer ──▶ video-indexer-results ──▶ video-graph
```

## Synchronous HTTP APIs

| Service | Port (`PORT_*`) | Purpose |
|---|---|---|
| `video-frame-description` | `PORT_VIDEO_FRAME_SYNC_API` (8031) | Direct VLM frame-description call, bypassing Kafka |
| `video-ocr-frame` | `PORT_VIDEO_OCR_SYNC` (8032) | Direct OCR call |
| `video-face-detection` | `PORT_VIDEO_FACE_SYNC` (8033) | `/embed-face`, same resident ONNX session its Kafka consumer uses |

Also calls out to `raptor-audio-recognizer:8019` (`AUDIO_SYNC_URL`, Module 09) for
audio transcription of a video's soundtrack.

## Face detection consent

`video-face-detection` only runs for users who granted `face_detection` consent,
checked against Module 33 (`PRIVACY_COMPLIANCE_URL`). `FACE_DETECTION_ENABLED=false`
stops the orchestrator from sending face-detection requests at all.

## GPU

Four services reserve a GPU: `video-chunking` (`VIDEO_CHUNKING_GPU`, ffmpeg hwaccel
only — see the NVENC multi-GPU known limitation in
[`docs/CHANGELOG.md`](../../../docs/CHANGELOG.md)), `video-ocr-frame`
(`VIDEO_OCR_GPU`), `video-face-detection` (`VIDEO_FACE_GPU`), and
`video-frame-description` (`VIDEO_FRAME_DESC_GPU`, default `2` — kept off the other
three's default `0` to spread VRAM load). Each has a `torch.cuda.is_available()`
healthcheck backing `video-gpu-watchdog`, same pattern as module 16's
training-watchdog.

## Key environment variables

Full reference: [`docs/VISION_PROCESSING_SETTINGS.md`](docs/VISION_PROCESSING_SETTINGS.md).

| Variable | Default | Used by |
|---|---|---|
| `VIDEO_NORMALIZE_TIMEOUT_SEC` | `600` | `video-chunking` |
| `VLM_MODEL_PATH` | `OpenGVLab/InternVL3_5-1B` | `video-frame-description` |
| `OCR_DET_MODEL` / `OCR_REC_MODEL` | `PP-OCRv5_mobile_det` / `_rec` | `video-ocr-frame` |
| `FACE_DETECTION_ENABLED` | `true` | `video-orchestrator` (gates sending face-detection requests) |
| `FACE_DET_THRESHOLD` | `0.5` | `video-face-detection` |
| `INFERENCE_MODEL` | `gemma4:12b` | `video-summary`, `video-contextualize` (Module 07) |

## Deploy

```bash
python build.py -m 11
```
