# Image Processing Module

GPU-isolated image pipeline: VLM description + OCR in one pass, plus optional
moderation, driven by Kafka. Runs on the shared `raptor/media-worker` image (built by
[08-media-worker](../08-media-worker/)) — this module has no `build:` of its own;
`WORKER_TYPE` selects which service `main.py` dispatches to inside the container.

## Services

| Container | `WORKER_TYPE` | GPU | Role |
|---|---|---|---|
| `raptor-image-kafka-init` | — | — | One-shot: creates every topic below, then exits |
| `raptor-image-orchestrator` | `image_orchestrator` | no | Consumes `image-processing-requests`; fans out description + OCR requests, forwards to indexer; optional image moderation (below) |
| `raptor-image-processing` | `image_processing` | yes | InternVL3.5-4B VLM — handles **both** description and OCR requests; also exposes a synchronous HTTP API (below) |
| `raptor-image-indexer` | `image_indexer` | no | Consumes `image-indexer-requests`; publishes to the personal index (Module 25) |
| `raptor-image-cleanup` | — | — | `busybox` sidecar: deletes `/tmp/media_processing/image_processing/*` older than 24h, hourly |
| `raptor-image-gpu-watchdog` | — | — | Restarts `image-processing` if its `torch.cuda.is_available()` healthcheck goes unhealthy |

No container is named `aigle-*`, has a static IP, or resolves at `*.aigle.local` —
every service is reachable only by its docker-network alias (`raptor-image-*`), same
convention as the rest of this repo. (An earlier version of this README described a
"video-frame-description" service as part of this module — that belongs to
[11-video-processing](../11-video-processing/), not here.)

## Kafka topics

Created once by `image-kafka-init-topics`:

```
image-processing-requests ──▶ image-orchestrator ──┬─▶ image-description-requests ─┐
                                                     │         (image-processing)   ├─▶ image-processing-results
                                                     ├─▶ image-ocr-requests ─────────┘
                                                     └─▶ image-indexer-requests ──▶ image-indexer ──▶ image-indexer-results ──▶ personal-index-completed
```

`image-processing` consumes both `image-description-requests` and
`image-ocr-requests` — one GPU worker handling two request types, not two separate
workers. DLQs (`image-processing-dlq`, `image-indexer-dlq`) are written to, never
consumed — see the repo-wide known limitation in
[`docs/CHANGELOG.md`](../../../docs/CHANGELOG.md).

## Synchronous HTTP API

| Service | Port (`PORT_*`) | Purpose |
|---|---|---|
| `image-processing` | `PORT_IMAGE_SYNC_API` (8018) | Direct VLM call, bypassing Kafka |

## Image moderation (CM-2, off by default)

`image-orchestrator` can screen images through Module 23's guard models before they
reach the pipeline: `IMAGE_MODERATION_ENABLED=false` (default), model
`IMAGE_MODERATION_MODEL` (default `shieldgemma-2`), fail-open
(`IMAGE_MODERATION_FAIL_MODE=open`) so a Module 23 outage never blocks processing.
See `worker/services/image_orchestrator_service/image_moderation.py`.

## GPU

`image-processing` reserves one GPU (`IMAGE_PROCESSING_GPU`, default `0`) and caps
per-GPU memory at `MAX_MEMORY_PER_GPU` (default `36GiB`) for the VLM
(`VLM_MODEL_PATH`/`VLM_MODEL_REVISION`, pinned to a specific HF revision). A
`torch.cuda.is_available()` healthcheck backs the `image-gpu-watchdog` sidecar, same
pattern as module 16's training-watchdog.

## Key environment variables

Full reference: [`docs/VISION_PROCESSING_SETTINGS.md`](docs/VISION_PROCESSING_SETTINGS.md).

| Variable | Default | Used by |
|---|---|---|
| `VLM_MODEL_PATH` | `OpenGVLab/InternVL3_5-4B` | `image-processing` |
| `VLM_MODEL_REVISION` | pinned commit hash | `image-processing` |
| `MAX_MEMORY_PER_GPU` | `36GiB` | `image-processing` |
| `IMAGE_MODERATION_ENABLED` | `false` | `image-orchestrator` |
| `GUARDRAIL_URL` | `http://raptor-guardrail-service:8026` | `image-orchestrator` (moderation) |
| `ASSET_MANAGEMENT_URL` | `http://raptor-asset-management:8000` | `image-orchestrator` |

## Deploy

```bash
python build.py -m 10
```
