# Document Processing Module - Summary

## Module Created Successfully ✅

The Document Processing Module has been created at:
**`/opt/dht/apps/raptor/deployment/modules/13-document-processing/`**

## Services Extracted

### 1. Document Analysis Service
- **Old Location**: `07-media-workers` module
- **Old IP**: `172.30.7.32`
- **New Location**: `13-document-processing` module
- **New IP**: `172.30.13.1`
- **Model**: InternVL3.5-4B (`OpenGVLab/InternVL3_5-4B`) - VLM for OCR and image processing
- **GPU**: Required (1 GPU) - **Added in this module** (was missing in media-workers)
- **Memory**: 16GB
- **CPU**: 4 cores
- **Port**: N/A (Kafka worker)
- **Supported Document Types**:
  - PDF (via VLM OCR)
  - Word Documents (doc/docx)
  - Excel (xlsx/xls)
  - PowerPoint (ppt/pptx)
  - HTML (html/htm)
  - Text (txt)
  - Handwriting Images (via VLM OCR)

### 2. Document Summary Service
- **Old Location**: `07-media-workers` module
- **Old IP**: `172.30.7.33`
- **New Location**: `13-document-processing` module
- **New IP**: `172.30.13.2`
- **Model**: LLM (via Ollama, default: `qwenforsummary`)
- **GPU**: Required (1 GPU)
- **Memory**: 8GB
- **CPU**: 2 cores
- **Port**: N/A (Kafka worker)

## LLM and OCR Model Usage

### OCR/VLM Model (InternVL3.5-4B)
- **Model**: `OpenGVLab/InternVL3_5-4B`
- **Used By**: `document-analysis`
- **Purpose**: 
  - OCR for PDF documents
  - Image description for Word/PowerPoint documents
  - Handwriting recognition from images
- **GPU Memory**: ~20-36GB per GPU

### LLM Model (Summarization)
- **Model**: `qwenforsummary` (default, via Ollama)
- **Used By**: `document-summary`
- **Purpose**: Document content summarization
- **GPU Memory**: ~4-8GB per GPU

## Network Configuration

- **Network**: `aigle-network` (external)
- **Subnet**: `172.30.13.0/24`
- **IP Range**: `172.30.13.1 - 172.30.13.254`

## Files Created

1. **`docker-compose.yml`** - Service definitions with GPU resources
2. **`README.md`** - Module overview and architecture
3. **`DOCUMENT_PROCESSING_SETTINGS.md`** - Detailed settings reference
4. **`DOCUMENT_PROCESSING_SUMMARY.md`** - Quick summary
5. **`worker/`** - Worker Dockerfile and source code

## Environment Variables

Added to `deployment/.env.example`:
- `DOCUMENT_ANALYSIS_IP=172.30.13.1`
- `DOCUMENT_SUMMARY_IP=172.30.13.2`
- Model, GPU, memory, and CPU configuration variables for each service

## Next Steps

1. ✅ Module created
2. ✅ Docker Compose configuration created
3. ✅ Environment variables added to `.env.example`
4. ⏭️ Update `.env` file with new IP addresses
5. ⏭️ Stop old services in `07-media-workers` module
6. ⏭️ Remove services from `07-media-workers/docker-compose.yml`
7. ⏭️ Deploy new module: `cd 13-document-processing && docker compose up -d`
8. ⏭️ Verify services are running correctly

## Benefits

1. ✅ **Resource Isolation**: GPU resources isolated for document processing
2. ✅ **Model Consolidation**: Both services use shared VLM/LLM models
3. ✅ **Better Scalability**: Independent scaling of document workloads
4. ✅ **Easier Maintenance**: Document-specific configurations in one place
5. ✅ **Clear Separation**: Document processing separated from other media workers
6. ✅ **GPU Added**: `document-analysis` now has proper GPU allocation

## Documentation

- **README.md** - Module overview and architecture
- **DOCUMENT_PROCESSING_SETTINGS.md** - Detailed settings reference
- **DOCUMENT_PROCESSING_SUMMARY.md** - Quick summary
- **docker-compose.yml** - Service configuration
