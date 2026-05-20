# Document Processing Module - Settings Reference

## Service IP Addresses and Ports

### Document Analysis Service
- **Service Name**: `document-analysis`
- **Container Name**: `aigle-document-analysis`
- **Hostname**: `document-analysis.aigle.local`
- **IP Address**: `172.30.13.1` (configurable via `DOCUMENT_ANALYSIS_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU) - **Added in this module** (was missing in media-workers)
- **Memory Limit**: 16GB
- **CPU Limit**: 4 cores

### Document Summary Service
- **Service Name**: `document-summary`
- **Container Name**: `aigle-document-summary`
- **Hostname**: `document-summary.aigle.local`
- **IP Address**: `172.30.13.2` (configurable via `DOCUMENT_SUMMARY_IP`)
- **Port**: N/A (Kafka worker, no HTTP port)
- **GPU**: Required (1 GPU)
- **Memory Limit**: 8GB
- **CPU Limit**: 2 cores

## Model Configuration

### InternVL3.5-4B (VLM/OCR Model)
- **Model Path**: `OpenGVLab/InternVL3_5-4B`
- **Used By**: `document-analysis`
- **Purpose**: 
  - OCR for PDF documents
  - Image description for Word/PowerPoint documents
  - Handwriting recognition from images
- **GPU Memory**: ~20-36GB per GPU
- **Supported Document Types**: PDF, Word (doc/docx), Excel (xlsx/xls), PowerPoint (ppt/pptx), HTML, TXT, Handwriting images

### LLM (Summarization)
- **Model**: `qwenforsummary` (default, via Ollama)
- **Used By**: `document-summary`
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
QDRANT_PORT=6333

# Object Storage Configuration
OBJECT_STORAGE_URL=http://seaweedfs-s3.aigle.local:8333
AWS_ACCESS_KEY=${AWS_ACCESS_KEY:-local}
AWS_SECRET_KEY=${AWS_SECRET_KEY:-local}

# Timezone
TZ=${TIMEZONE:-Asia/Taipei}
```

### Document Analysis Specific
```bash
DOCUMENT_ANALYSIS_IP=172.30.13.1
DOCUMENT_ANALYSIS_VLM_MODEL=OpenGVLab/InternVL3_5-4B
DOCUMENT_ANALYSIS_GPU=0
DOCUMENT_ANALYSIS_MEMORY=36GiB
DOCUMENT_ANALYSIS_MAX_CHUNK_TOKENS=400
DOCUMENT_ANALYSIS_MEMORY_LIMIT=16g
DOCUMENT_ANALYSIS_CPU_LIMIT=4
DOCUMENT_ANALYSIS_GPU_COUNT=1
```

### Document Summary Specific
```bash
DOCUMENT_SUMMARY_IP=172.30.13.2
DOCUMENT_SUMMARY_MODEL=qwenforsummary
DOCUMENT_SUMMARY_GPU=0
DOCUMENT_SUMMARY_MEMORY_LIMIT=8g
DOCUMENT_SUMMARY_CPU_LIMIT=2
DOCUMENT_SUMMARY_GPU_COUNT=1
OLLAMA_API_BASE=${OLLAMA_API_BASE:-http://host.docker.internal:11434}
```

## Network Configuration

- **Network Name**: `aigle-network` (external)
- **Subnet**: `172.30.13.0/24`
- **IP Allocation**:
  - `172.30.13.1` - document-analysis
  - `172.30.13.2` - document-summary
  - `172.30.13.3-254` - Reserved for future services

## GPU Resource Allocation

### Document Analysis (VLM/OCR)
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~20-36GB per GPU
- **Total GPU Memory Required**: ~20-36GB
- **Note**: GPU was added in this module (previously missing in media-workers)

### Document Summary (LLM)
- **GPU Count**: 1
- **GPU Device**: CUDA device 0 (configurable)
- **GPU Memory**: ~4-8GB per GPU
- **Total GPU Memory Required**: ~4-8GB

### Total GPU Requirements
- **Minimum GPUs**: 2 (one per service)
- **If sharing GPU**: 1 GPU with ~24-44GB memory (not recommended)

## Kafka Topics

- **document-analysis**: Consumed by `document-analysis` service
- **document-summary**: Consumed by `document-summary` service

## Resource Limits Summary

| Service | Memory | CPU | GPU | IP Address |
|---------|--------|-----|-----|------------|
| document-analysis | 16GB | 4 cores | 1 GPU | 172.30.13.1 |
| document-summary | 8GB | 2 cores | 1 GPU | 172.30.13.2 |

## Document Type Support

### Supported File Types

| Type | Extensions | Processor | Model Used |
|------|------------|-----------|------------|
| PDF | pdf | PDFOCRProcessor | InternVL3.5-4B (VLM OCR) |
| Word | doc, docx | OfficeDocumentProcessor | InternVL3.5-4B (VLM) |
| Excel | xlsx, xls | CSVXLSXProcessor | None (direct parsing) |
| PowerPoint | ppt, pptx | OfficeDocumentProcessor | InternVL3.5-4B (VLM) |
| HTML | html, htm | HTMLProcessor | None (HTML parsing) |
| Text | txt | TxtProcessor | None (text reading) |
| Handwriting Images | jpg, png, etc. | PDFOCRProcessor/VLM | InternVL3.5-4B (VLM OCR) |

## Migration from Media Workers

### Old IP Addresses (07-media-workers)
- `document-analysis`: `172.30.7.32` → `172.30.13.1` (**GPU added**)
- `document-summary`: `172.30.7.33` → `172.30.13.2`

### Update Required
1. Update `.env` file with new IP addresses
2. Update any hardcoded service references
3. Update API Gateway or service discovery configurations
4. Restart dependent services
5. **Important**: `document-analysis` now has GPU resources (previously missing)

## Related Services

### Document Orchestrator (Non-GPU)
- **Module**: `07-media-workers`
- **Service**: `document-orchestrator`
- **IP**: `172.30.7.31`
- **Purpose**: Orchestrates document processing workflow (no GPU needed)

### Document Save to Qdrant (Non-GPU)
- **Module**: `07-media-workers`
- **Service**: `document-save2qdrant`
- **IP**: `172.30.7.34`
- **Purpose**: Saves processed documents to Qdrant (no GPU needed)
