# Document Processing Module — Settings Reference

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
| `ASSET_MANAGEMENT_URL` | `http://raptor-asset-management:8000` | `document-orchestrator` |
| `INFERENCE_URL` | `http://raptor-ai-lifecycle-api:8010` | `document-summary` (Module 07) |
| `INFERENCE_MODEL` | `gemma4:12b` | `document-summary` |
| `INFERENCE_THINK` | `false` | `document-summary` |

## PII scan at ingestion

| Variable | Default | Used by |
|---|---|---|
| `GUARDRAIL_URL` | `http://raptor-guardrail-service:8026` | `document-indexer` |
| `M12_PII_SCAN_ENABLED` | `true` | `document-indexer` |
| `M12_PII_SCAN_TIMEOUT_MS` | `1000` | `document-indexer` |

## Model

| Variable | Default | Used by |
|---|---|---|
| `HF_TOKEN` | — (required for first-run download) | `document-analysis` |
| `DOCUMENT_ANALYSIS_VLM_MODEL` | `OpenGVLab/InternVL3_5-1B` | `document-analysis` |
| `DOCUMENT_ANALYSIS_VLM_MODEL_REVISION` | pinned HF commit hash | `document-analysis` |
| `DOCUMENT_ANALYSIS_MAX_CHUNK_TOKENS` | `400` | `document-analysis` |
| `DOCUMENT_ANALYSIS_MAX_WORKERS` | `8` | `document-analysis` |
| `DOCUMENT_ANALYSIS_CONTEXTUAL_TIMEOUT` | `120` | `document-analysis` |
| `HF_CACHE_PATH` | `/path/to/huggingface/cache` | shared worker env, mounted at `/hf_cache` |

## GPU allocation

| Variable | Default |
|---|---|
| `DOCUMENT_ANALYSIS_GPU` | `0` |
| `DOCUMENT_ANALYSIS_GPU_COUNT` | `1` |
| `DOCUMENT_ANALYSIS_MEMORY_LIMIT` | `20g` |
| `DOCUMENT_ANALYSIS_CPU_LIMIT` | `4` |

## CPU-only services

| Variable | Default |
|---|---|
| `DOCUMENT_ORCHESTRATOR_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `DOCUMENT_SUMMARY_MEMORY_LIMIT` / `_CPU_LIMIT` | `8g` / `2` |
| `DOCUMENT_INDEXER_MEMORY_LIMIT` / `_CPU_LIMIT` | `4g` / `2` |

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
