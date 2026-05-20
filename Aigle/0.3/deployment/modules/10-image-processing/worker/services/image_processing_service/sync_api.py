# services/image_processing_service/sync_api.py
# Sync analysis API: upload image -> description + OCR -> return result (no indexing)

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Image Sync API", version="0.1")

_analysis_client = None
_thread_pool = None


def init(analysis_client, thread_pool):
    global _analysis_client, _thread_pool
    _analysis_client = analysis_client
    _thread_pool = thread_pool


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _analysis_client is not None}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    processing_mode: str = Form("default"),
):
    """
    Synchronous image analysis — returns description + OCR result.
    
    Args:
        file: Image file to analyze
        processing_mode: Processing mode (default)
    
    Returns:
        JSON response with description and OCR result
    """
    if _analysis_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    request_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ".bin"
    tmp_path = Path(f"/tmp/media_processing/image_processing/sync_{request_id}{suffix}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    description_result_path = None
    ocr_result_path = None

    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        request_payload = {
            "request_id": request_id,
            "parameters": {
                "temp_file_path": str(tmp_path),
                "primary_filename": file.filename or f"upload{suffix}",
            },
            "metadata": {
                "original_metadata": {
                    "original_metadata": {
                        "processing_mode": processing_mode
                    }
                }
            },
        }

        loop = asyncio.get_event_loop()

        # 同時執行 description 和 OCR 分析
        description_result_path, ocr_result_path = await loop.run_in_executor(
            _thread_pool,
            _analysis_client.process_image_sync,
            request_payload,
        )

        if not description_result_path or not Path(description_result_path).exists():
            raise HTTPException(status_code=500, detail="Description analysis failed")
        if not ocr_result_path or not Path(ocr_result_path).exists():
            raise HTTPException(status_code=500, detail="OCR analysis failed")

        # 讀取 description 結果
        with open(description_result_path, "r", encoding="utf-8") as f:
            description_data = json.load(f)

        # 讀取 OCR 結果
        with open(ocr_result_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        # 返回完整結果（description 作為 summary，OCR 作為 text）
        return JSONResponse({
            "filename": file.filename,
            "summary": description_data.get("description", ""),
            "text": ocr_data.get("extracted_text", ""),
        })

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if description_result_path and Path(description_result_path).exists():
            Path(description_result_path).unlink(missing_ok=True)
        if ocr_result_path and Path(ocr_result_path).exists():
            Path(ocr_result_path).unlink(missing_ok=True)
