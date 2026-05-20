# Document Processing Module

## Overview

The Document Processing Module consolidates all LLM and OCR/VLM-based document processing services that require GPU resources. This module extracts GPU-intensive document processing services from the media-workers module to provide better resource isolation and management.

## Services

### 1. Document Analysis Service
- **Container**: `aigle-document-analysis`
- **Hostname**: `document-analysis.aigle.local`
- **IP Address**: `172.30.13.1` (configurable via `DOCUMENT_ANALYSIS_IP`)
- **Model**: InternVL3.5-4B (`OpenGVLab/InternVL3_5-4B`) - VLM for OCR and image processing
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 16GB (configurable)
- **CPU**: 4 cores (configurable)
- **Kafka Topic**: `document-analysis`
- **Purpose**: Extracts content from various document types using VLM/OCR
- **Supported Formats**:
  - **PDF**: Via PDFOCRProcessor with VLM OCR
  - **Word Documents**: doc, docx (via OfficeDocumentProcessor with VLM)
  - **Excel**: xlsx, xls (via CSVXLSXProcessor)
  - **PowerPoint**: ppt, pptx (via OfficeDocumentProcessor with VLM)
  - **HTML**: htm, html (via HTMLProcessor)
  - **Text**: txt (via TxtProcessor)
  - **Handwriting Images**: Via VLM OCR for image-based documents

### 2. Document Summary Service
- **Container**: `aigle-document-summary`
- **Hostname**: `document-summary.aigle.local`
- **IP Address**: `172.30.13.2` (configurable via `DOCUMENT_SUMMARY_IP`)
- **Model**: LLM (via Ollama, default: `qwenforsummary`)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 8GB (configurable)
- **CPU**: 2 cores (configurable)
- **Kafka Topic**: `document-summary`
- **Purpose**: Generates summaries of document content using LLM

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Kafka Cluster                              │
│  Topics: document-analysis, document-summary            │
└──────────────┬──────────────────────────────────────────┘
               │
               │ Messages
               │
┌──────────────▼──────────────────────────────────────────┐
│         Document Processing Module                      │
│  ┌────────────────────┐  ┌────────────────────┐        │
│  │ document-analysis   │  │ document-summary  │        │
│  │ (InternVL3.5-4B)    │  │ (LLM/Ollama)      │        │
│  │ GPU: 1              │  │ GPU: 1            │        │
│  │ IP: 172.30.13.1     │  │ IP: 172.30.13.2   │        │
│  │ PDF/Word/Excel/OCR  │  │ Summarization     │        │
│  └────────────────────┘  └────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Models Used

### InternVL3.5-4B (VLM/OCR)
- **Model**: `OpenGVLab/InternVL3_5-4B`
- **Used By**: `document-analysis`
- **Purpose**: 
  - OCR for PDF documents
  - Image description for Word/PowerPoint documents
  - Handwriting recognition from images
- **GPU Memory**: ~20-36GB per GPU
- **Document Types**: PDF, Word, PowerPoint, Handwriting images

### LLM (via Ollama)
- **Model**: `qwenforsummary` (default)
- **Used By**: `document-summary`
- **Purpose**: Document content summarization
- **GPU Memory**: ~4-8GB per GPU

## Document Type Support

### PDF Files
- **Processor**: `PDFOCRProcessor`
- **Method**: PDF → Images → VLM OCR
- **Features**: Text extraction, layout analysis, OCR for scanned PDFs

### Word Documents (doc/docx)
- **Processor**: `OfficeDocumentProcessor`
- **Method**: Document → Images → VLM image description
- **Features**: Text extraction, image/chart description

### Excel Files (xlsx/xls)
- **Processor**: `CSVXLSXProcessor`
- **Method**: Direct spreadsheet parsing
- **Features**: Sheet-by-sheet data extraction

### PowerPoint (ppt/pptx)
- **Processor**: `OfficeDocumentProcessor`
- **Method**: Slides → Images → VLM image description
- **Features**: Slide content extraction, image description

### HTML Files
- **Processor**: `HTMLProcessor`
- **Method**: HTML parsing and text extraction
- **Features**: Web content extraction

### Text Files (txt)
- **Processor**: `TxtProcessor`
- **Method**: Direct text reading
- **Features**: Plain text extraction

### Handwriting Images
- **Processor**: `PDFOCRProcessor` or `OfficeDocumentProcessor`
- **Method**: Image → VLM OCR
- **Features**: Handwriting recognition via VLM OCR

## Configuration

### Environment Variables

#### Document Analysis Service
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

#### Document Summary Service
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

- **Network**: `aigle-network` (external)
- **Subnet**: `172.30.13.0/24`
- **IP Range**: `172.30.13.1 - 172.30.13.254`

## Dependencies

- **Kafka Cluster**: Required for message queue
- **Redis**: Required for caching
- **Qdrant**: Required for vector storage
- **SeaweedFS**: Required for object storage
- **GPU**: NVIDIA GPU with CUDA support required
- **Ollama**: Required for document-summary service (LLM)

## Deployment

```bash
cd /opt/dht/apps/raptor/deployment/modules/13-document-processing
docker compose up -d
```

## Monitoring

Check service status:
```bash
docker ps | grep document-processing
docker logs aigle-document-analysis
docker logs aigle-document-summary
```

## Migration Notes

These services were previously part of the `07-media-workers` module:
- `document-analysis`: Moved from `172.30.7.32` to `172.30.13.1` (GPU added)
- `document-summary`: Moved from `172.30.7.33` to `172.30.13.2`

**Important**: `document-analysis` now has GPU resources allocated (previously did not have GPU in docker-compose.yml, but code uses CUDA).

Update any hardcoded IP addresses or service references when migrating.
