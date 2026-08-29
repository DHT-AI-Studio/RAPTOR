# Vision Processing Module - Settings Reference

## Service IP Addresses and Ports

### Video Frame Description Service
- **Service Name**: `video-frame-description`
- **Container Name**: `aigle-video-frame-description`
- **Hostname**: `video-frame-description.aigle.local`
- **IP Address**: `172.30.11.1` (configurable via `VIDEO_FRAME_DESCRIPTION_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 16GB
- **CPU Limit**: 4 cores

### Image Processing Service
- **Service Name**: `image-processing`
- **Container Name**: `aigle-image-processing`
- **Hostname**: `image-processing.aigle.local`
- **IP Address**: `172.30.11.2` (configurable via `IMAGE_PROCESSING_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 16GB
- **CPU Limit**: 4 cores

## VLM Model Configuration

### Shared Model: InternVL3.5-8B
- **Model Path**: `OpenGVLab/InternVL3_5-8B`
- **Used By**: Both video-frame-description and image-processing
- **GPU Memory**: ~36GB per GPU
- **Quantization**: None (full precision)

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

### Video Frame Description Specific
```bash
VIDEO_FRAME_DESCRIPTION_IP=172.30.11.1
VIDEO_FRAME_DESCRIPTION_MODEL_PATH=OpenGVLab/InternVL3_5-8B
VIDEO_FRAME_DESCRIPTION_GPU=0
VIDEO_FRAME_DESCRIPTION_MEMORY=36GiB
VIDEO_FRAME_DESCRIPTION_MEMORY_LIMIT=16g
VIDEO_FRAME_DESCRIPTION_CPU_LIMIT=4
VIDEO_FRAME_DESCRIPTION_GPU_COUNT=1
```

### Image Processing Specific
```bash
IMAGE_PROCESSING_IP=172.30.11.2
IMAGE_PROCESSING_MODEL_PATH=OpenGVLab/InternVL3_5-8B
IMAGE_PROCESSING_GPU=0
IMAGE_PROCESSING_MEMORY=36GiB
IMAGE_PROCESSING_MEMORY_LIMIT=16g
IMAGE_PROCESSING_CPU_LIMIT=4
IMAGE_PROCESSING_GPU_COUNT=1
```

## Network Configuration

- **Network Name**: `aigle-network` (external)
- **Subnet**: `172.30.11.0/24`
- **IP Allocation**:
  - `172.30.11.1` - video-frame-description
  - `172.30.11.2` - image-processing
  - `172.30.11.3-254` - Reserved for future services

## GPU Resource Allocation

### Video Frame Description
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~36GB per GPU
- **Total GPU Memory Required**: ~36GB

### Image Processing
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~36GB per GPU
- **Total GPU Memory Required**: ~36GB

### Total GPU Requirements
- **Minimum GPUs**: 2 (one per service)
- **If sharing GPU**: 1 GPU with ~72GB memory (not recommended)

## Kafka Topics

- **video-frame-description**: Consumed by `video-frame-description` service
- **image-processing**: Consumed by `image-processing` service

## Resource Limits Summary

| Service | Memory | CPU | GPU | IP Address |
|---------|--------|-----|-----|------------|
| video-frame-description | 16GB | 4 cores | 1 GPU | 172.30.11.1 |
| image-processing | 16GB | 4 cores | 1 GPU | 172.30.11.2 |

## Related Services

### Vision Service (Separate Module)
- **Module**: `05-ai-ml-services`
- **Service**: `vision-service`
- **Model**: Qwen2.5-VL-7B-Instruct
- **IP**: `172.30.5.3`
- **Port**: `8004` (API), `8503` (Streamlit)
- **Note**: This is a separate HTTP API service, not a Kafka worker

## Migration from Media Workers

### Old IP Addresses (07-media-workers)
- `video-frame-description`: `172.30.7.3` → `172.30.11.1`
- `image-processing`: `172.30.7.22` → `172.30.11.2`

### Update Required
1. Update `.env` file with new IP addresses
2. Update any hardcoded service references
3. Update API Gateway or service discovery configurations
4. Restart dependent services
