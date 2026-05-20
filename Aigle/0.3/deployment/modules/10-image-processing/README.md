# Vision Processing Module

## Overview

The Vision Processing Module consolidates all VLM (Vision Language Model) based services for video and image processing. This module extracts GPU-intensive vision processing services from the media-workers module to provide better resource isolation and management.

## Services

### 1. Video Frame Description Service
- **Container**: `aigle-video-frame-description`
- **Hostname**: `video-frame-description.aigle.local`
- **IP Address**: `172.30.11.1` (configurable via `VIDEO_FRAME_DESCRIPTION_IP`)
- **Model**: InternVL3.5-8B (`OpenGVLab/InternVL3_5-8B`)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 16GB (configurable)
- **CPU**: 4 cores (configurable)
- **Kafka Topic**: `video-frame-description`
- **Purpose**: Generates descriptions for video frames using VLM

### 2. Image Processing Service
- **Container**: `aigle-image-processing`
- **Hostname**: `image-processing.aigle.local`
- **IP Address**: `172.30.11.2` (configurable via `IMAGE_PROCESSING_IP`)
- **Model**: InternVL3.5-8B (`OpenGVLab/InternVL3_5-8B`)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 16GB (configurable)
- **CPU**: 4 cores (configurable)
- **Kafka Topic**: `image-processing`
- **Purpose**: Processes images and generates descriptions using VLM

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Kafka Cluster                              │
│  Topics: video-frame-description, image-processing      │
└──────────────┬──────────────────────────────────────────┘
               │
               │ Messages
               │
┌──────────────▼──────────────────────────────────────────┐
│         Vision Processing Module                        │
│  ┌────────────────────┐  ┌────────────────────┐      │
│  │ video-frame-       │  │ image-processing   │      │
│  │ description        │  │                     │      │
│  │ (InternVL3.5-8B)   │  │ (InternVL3.5-8B)    │      │
│  │ GPU: 1             │  │ GPU: 1              │      │
│  │ IP: 172.30.11.1    │  │ IP: 172.30.11.2     │      │
│  └────────────────────┘  └────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## VLM Models Used

### InternVL3.5-8B
- **Model**: `OpenGVLab/InternVL3_5-8B`
- **Purpose**: Vision-language understanding for both video frames and images
- **Shared by**: Both video-frame-description and image-processing services
- **GPU Memory**: ~36GB per GPU (configurable)

## Configuration

### Environment Variables

#### Video Frame Description Service
```bash
VIDEO_FRAME_DESCRIPTION_IP=172.30.11.1
VIDEO_FRAME_DESCRIPTION_MODEL_PATH=OpenGVLab/InternVL3_5-8B
VIDEO_FRAME_DESCRIPTION_GPU=0
VIDEO_FRAME_DESCRIPTION_MEMORY=36GiB
VIDEO_FRAME_DESCRIPTION_MEMORY_LIMIT=16g
VIDEO_FRAME_DESCRIPTION_CPU_LIMIT=4
VIDEO_FRAME_DESCRIPTION_GPU_COUNT=1
```

#### Image Processing Service
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

- **Network**: `aigle-network` (external)
- **Subnet**: `172.30.11.0/24` (Vision Processing Module)
- **IP Range**: `172.30.11.1 - 172.30.11.254`

## Dependencies

- **Kafka Cluster**: Required for message queue
- **Redis**: Required for caching
- **Qdrant**: Required for vector storage
- **SeaweedFS**: Required for object storage
- **GPU**: NVIDIA GPU with CUDA support required

## Deployment

```bash
cd /opt/dht/apps/raptor/deployment/modules/11-vision-processing
docker compose up -d
```

## Monitoring

Check service status:
```bash
docker ps | grep vision-processing
docker logs aigle-video-frame-description
docker logs aigle-image-processing
```

## Migration Notes

These services were previously part of the `07-media-workers` module:
- `video-frame-description`: Moved from `172.30.7.3` to `172.30.11.1`
- `image-processing`: Moved from `172.30.7.22` to `172.30.11.2`

Update any hardcoded IP addresses or service references when migrating.
