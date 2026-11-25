# Video Processing Module — Settings Reference

Architecture and service list: [../README.md](../README.md). This file is just the
`.env` reference, grouped the same way as `.env.example`.

## Kafka

| Variable | Default |
|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092` |

## Redis

| Variable | Default |
|---|---|
| `REDIS_HOST` | `raptor-redis-standalone` |
| `REDIS_PORT` | `6379` |
| `REDIS_PASSWORD` | — (required) |

## Downstream services

| Variable | Default | Used by |
|---|---|---|
| `HYBRID_SEARCH_INGEST_URL` | `http://raptor-hybridsearch-api:8000/api/v1/ingest/file` | shared worker env |
| `ASSET_MANAGEMENT_URL` | `http://raptor-asset-management:8000` | `video-orchestrator` |
| `INFERENCE_URL` | `http://raptor-ai-lifecycle-api:8010` | `video-summary`, `video-contextualize` |
| `INFERENCE_MODEL` | `gemma4:12b` | `video-summary`, `video-contextualize` |
| `INFERENCE_THINK` | `false` | `video-summary`, `video-contextualize` |
| `SUMMARY_MAX_TOKENS` | `4000` | `video-summary` |
| `SUMMARY_MAX_LENGTH` | `300` | `video-summary` |
| `AUDIO_SYNC_URL` | `http://raptor-audio-recognizer:8019` | audio transcription of a video's soundtrack (Module 09) |
| `PRIVACY_COMPLIANCE_URL` | `http://raptor-privacy-compliance-service:8033` | `video-face-detection` (consent check) |

## Chunking

| Variable | Default |
|---|---|
| `VIDEO_NORMALIZE_TIMEOUT_SEC` | `600` |

## VLM / OCR / face models

| Variable | Default | Used by |
|---|---|---|
| `HF_TOKEN` | — (required for first-run download, ~8GB) | `video-frame-description` |
| `VLM_MODEL_PATH` | `OpenGVLab/InternVL3_5-1B` | `video-frame-description` |
| `VLM_MODEL_REVISION` | pinned HF commit hash | `video-frame-description` |
| `MAX_MEMORY_PER_GPU` | `36GiB` | `video-frame-description` |
| `OCR_DET_MODEL` | `PP-OCRv5_mobile_det` | `video-ocr-frame` |
| `OCR_REC_MODEL` | `PP-OCRv5_mobile_rec` | `video-ocr-frame` |
| `OCR_PRECISION` | `fp16` | `video-ocr-frame` |
| `FACE_DETECTION_ENABLED` | `true` | `video-orchestrator` |
| `FACE_DET_THRESHOLD` | `0.5` | `video-face-detection` |
| `FACE_MIN_SIZE_PX` | `20` | `video-face-detection` |
| `FACE_SHOT_THRESHOLD` | `27.0` | `video-face-detection` |

## GPU allocation

| Variable | Default |
|---|---|
| `VIDEO_FRAME_DESC_GPU` / `_GPU_COUNT` / `_MEMORY_LIMIT` / `_CPU_LIMIT` | `2` / `1` / `20g` / `4` |
| `VIDEO_OCR_GPU` / `_GPU_COUNT` / `_MEMORY_LIMIT` / `_CPU_LIMIT` | `0` / `1` / `8g` / `2` |
| `VIDEO_FACE_GPU` / `_MEMORY_LIMIT` / `_CPU_LIMIT` | `0` / `6g` / `2` |
| `VIDEO_CHUNKING_GPU` | `0` (see the NVENC multi-GPU known limitation in root `docs/CHANGELOG.md`) |

## CPU-only services

| Variable | Default |
|---|---|
| `VIDEO_ORCHESTRATOR_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `VIDEO_CHUNKING_MEMORY_LIMIT` / `_CPU_LIMIT` | `4g` / `2` |
| `VIDEO_ANALYSIS_MEMORY_LIMIT` / `_CPU_LIMIT` | `4g` / `2` |
| `VIDEO_SCENE_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `VIDEO_SUMMARY_MEMORY_LIMIT` / `_CPU_LIMIT` | `4g` / `2` |
| `VIDEO_CONTEXTUALIZE_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `VIDEO_INDEXER_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |

## Synchronous API ports

| Variable | Default | Service |
|---|---|---|
| `PORT_VIDEO_FRAME_SYNC_API` | `8031` | `video-frame-description` |
| `PORT_VIDEO_OCR_SYNC` | `8032` | `video-ocr-frame` |
| `PORT_VIDEO_FACE_SYNC` | `8033` | `video-face-detection` |

## Shared storage

| Variable | Default |
|---|---|
| `NFS_SERVER` | — (required) |
| `PORT_NFS` | `2049` |
| `NFS_MEDIA_TMP_PATH` | `/media/processing` |

## Other

| Variable | Default |
|---|---|
| `TIMEZONE` | `Asia/Taipei` |
