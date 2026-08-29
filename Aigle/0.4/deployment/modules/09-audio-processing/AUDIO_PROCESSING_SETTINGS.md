# Audio Processing Module - Settings Reference

## Service IP Addresses and Ports

### Audio Recognizer Service (STT)
- **Service Name**: `audio-recognizer`
- **Container Name**: `aigle-audio-recognizer`
- **Hostname**: `audio-recognizer.aigle.local`
- **IP Address**: `172.30.12.1` (configurable via `AUDIO_RECOGNIZER_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 8GB
- **CPU Limit**: 2 cores

### Audio Diarization Service
- **Service Name**: `audio-diarization`
- **Container Name**: `aigle-audio-diarization`
- **Hostname**: `audio-diarization.aigle.local`
- **IP Address**: `172.30.12.2` (configurable via `AUDIO_DIARIZATION_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 8GB
- **CPU Limit**: 2 cores

### Audio Classifier Service
- **Service Name**: `audio-classifier`
- **Container Name**: `aigle-audio-classifier`
- **Hostname**: `audio-classifier.aigle.local`
- **IP Address**: `172.30.12.3` (configurable via `AUDIO_CLASSIFIER_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 4GB
- **CPU Limit**: 2 cores

### Audio Summary Service
- **Service Name**: `audio-summary`
- **Container Name**: `aigle-audio-summary`
- **Hostname**: `audio-summary.aigle.local`
- **IP Address**: `172.30.12.4` (configurable via `AUDIO_SUMMARY_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 8GB
- **CPU Limit**: 2 cores

## Model Configuration

### WhisperX (STT Model)
- **Model**: `large-v3`
- **Used By**: `audio-recognizer`, `audio-diarization`
- **GPU Memory**: ~4-8GB per GPU
- **Batch Size**: 16 (configurable)

### PANNs (Audio Classification)
- **Model**: Audio Tagging and Sound Event Detection
- **Used By**: `audio-classifier`
- **GPU Memory**: ~2-4GB per GPU
- **Top K**: 5 (configurable)
- **Segment Length**: 30.0 seconds (configurable)

### LLM (Summarization)
- **Model**: `qwenforsummary` (default, via Ollama)
- **Used By**: `audio-summary`
- **GPU Memory**: ~4-8GB per GPU
- **Ollama URL**: Configurable via `OLLAMA_API_BASE`

## Environment Variables

### Common Variables
```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-broker1.aigle.local:9092,kafka-broker2.aigle.local:9092,kafka-broker3.aigle.local:9092

# Redis Configuration
REDIS_HOST=redis-standalone.aigle.local
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}

# Qdrant Configuration
QDRANT_HOST=qdrant.aigle.local
PORT_QDRANT=6333

# Object Storage Configuration
OBJECT_STORAGE_URL=http://seaweedfs-s3.aigle.local:8333
AWS_ACCESS_KEY=${AWS_ACCESS_KEY:-local}
AWS_SECRET_KEY=${AWS_SECRET_KEY:-local}

# Timezone
TZ=${TIMEZONE:-Asia/Taipei}
```

### Audio Recognizer Specific (STT)
```bash
AUDIO_RECOGNIZER_IP=172.30.12.1
AUDIO_RECOGNIZER_MODEL=large-v3
AUDIO_RECOGNIZER_BATCH_SIZE=16
AUDIO_RECOGNIZER_GPU=0
AUDIO_RECOGNIZER_MEMORY_LIMIT=8g
AUDIO_RECOGNIZER_CPU_LIMIT=2
AUDIO_RECOGNIZER_GPU_COUNT=1
```

### Audio Diarization Specific
```bash
AUDIO_DIARIZATION_IP=172.30.12.2
AUDIO_DIARIZATION_GPU=0
AUDIO_DIARIZATION_MIN_SPEAKERS=1
AUDIO_DIARIZATION_MAX_SPEAKERS=10
AUDIO_DIARIZATION_MEMORY_LIMIT=8g
AUDIO_DIARIZATION_CPU_LIMIT=2
AUDIO_DIARIZATION_GPU_COUNT=1
HF_TOKEN=${HF_TOKEN}  # Required for pyannote.audio
```

### Audio Classifier Specific
```bash
AUDIO_CLASSIFIER_IP=172.30.12.3
AUDIO_CLASSIFIER_GPU=0
AUDIO_CLASSIFIER_TOP_K=5
AUDIO_CLASSIFIER_SEGMENT_LENGTH=30.0
AUDIO_CLASSIFIER_MEMORY_LIMIT=4g
AUDIO_CLASSIFIER_CPU_LIMIT=2
AUDIO_CLASSIFIER_GPU_COUNT=1
```

### Audio Summary Specific
```bash
AUDIO_SUMMARY_IP=172.30.12.4
AUDIO_SUMMARY_GPU=0
AUDIO_SUMMARY_MODEL=qwenforsummary
AUDIO_SUMMARY_MEMORY_LIMIT=8g
AUDIO_SUMMARY_CPU_LIMIT=2
AUDIO_SUMMARY_GPU_COUNT=1
OLLAMA_API_BASE=${OLLAMA_API_BASE:-http://host.docker.internal:11434}
```

## Network Configuration

- **Network Name**: `aigle-network` (external)
- **Subnet**: `172.30.12.0/24`
- **IP Allocation**:
  - `172.30.12.1` - audio-recognizer
  - `172.30.12.2` - audio-diarization
  - `172.30.12.3` - audio-classifier
  - `172.30.12.4` - audio-summary
  - `172.30.12.5-254` - Reserved for future services

## GPU Resource Allocation

### Audio Recognizer (STT)
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~4-8GB per GPU
- **Total GPU Memory Required**: ~4-8GB

### Audio Diarization
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~4-8GB per GPU
- **Total GPU Memory Required**: ~4-8GB

### Audio Classifier
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~2-4GB per GPU
- **Total GPU Memory Required**: ~2-4GB

### Audio Summary
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~4-8GB per GPU
- **Total GPU Memory Required**: ~4-8GB

### Total GPU Requirements
- **Minimum GPUs**: 4 (one per service)
- **If sharing GPU**: 1 GPU with ~20-28GB memory (not recommended)

## Kafka Topics

- **audio-recognition**: Consumed by `audio-recognizer` service
- **audio-diarization**: Consumed by `audio-diarization` service
- **audio-classification**: Consumed by `audio-classifier` service
- **audio-summary**: Consumed by `audio-summary` service

## Resource Limits Summary

| Service | Memory | CPU | GPU | IP Address |
|---------|--------|-----|-----|------------|
| audio-recognizer | 8GB | 2 cores | 1 GPU | 172.30.12.1 |
| audio-diarization | 8GB | 2 cores | 1 GPU | 172.30.12.2 |
| audio-classifier | 4GB | 2 cores | 1 GPU | 172.30.12.3 |
| audio-summary | 8GB | 2 cores | 1 GPU | 172.30.12.4 |

## Migration from Media Workers

### Old IP Addresses (07-media-workers)
- `audio-recognizer`: `172.30.7.15` → `172.30.12.1`
- `audio-diarization`: `172.30.7.14` → `172.30.12.2`
- `audio-classifier`: `172.30.7.13` → `172.30.12.3`
- `audio-summary`: `172.30.7.16` → `172.30.12.4`

### Update Required
1. Update `.env` file with new IP addresses
2. Update any hardcoded service references
3. Update API Gateway or service discovery configurations
4. Restart dependent services
