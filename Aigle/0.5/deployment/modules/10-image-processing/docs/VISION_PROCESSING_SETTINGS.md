# Image Processing Module — Settings Reference

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
| `ASSET_MANAGEMENT_URL` | `http://raptor-asset-management:8000` | `image-orchestrator` |
| `GUARDRAIL_URL` | `http://raptor-guardrail-service:8026` | `image-orchestrator` (CM-2 moderation, only while enabled) |
| `IMAGE_MODERATION_ENABLED` | `false` | `image-orchestrator` |
| `IMAGE_MODERATION_MODEL` | `shieldgemma-2` | `image-orchestrator` |
| `IMAGE_MODERATION_TIMEOUT` | `180` | `image-orchestrator` |
| `IMAGE_MODERATION_FAIL_MODE` | `open` | `image-orchestrator` |

## Model

| Variable | Default | Used by |
|---|---|---|
| `HF_TOKEN` | — (required for first-run download, ~8GB) | `image-processing` |
| `VLM_MODEL_PATH` | `OpenGVLab/InternVL3_5-4B` | `image-processing` |
| `VLM_MODEL_REVISION` | pinned HF commit hash | `image-processing` |
| `MAX_MEMORY_PER_GPU` | `36GiB` | `image-processing` |
| `HF_CACHE_PATH` | `/path/to/huggingface/cache` | shared worker env, mounted at `/hf_cache` |

## GPU allocation

| Variable | Default |
|---|---|
| `IMAGE_PROCESSING_GPU` | `0` |
| `IMAGE_PROCESSING_GPU_COUNT` | `1` |
| `IMAGE_PROCESSING_MEMORY_LIMIT` | `20g` |
| `IMAGE_PROCESSING_CPU_LIMIT` | `4` |

## CPU-only services

| Variable | Default |
|---|---|
| `IMAGE_ORCHESTRATOR_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `IMAGE_INDEXER_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |

## Synchronous API port

| Variable | Default | Service |
|---|---|---|
| `PORT_IMAGE_SYNC_API` | `8018` | `image-processing` |

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
