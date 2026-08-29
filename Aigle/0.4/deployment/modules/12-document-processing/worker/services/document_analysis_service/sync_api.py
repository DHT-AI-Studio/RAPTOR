# services/document_analysis_service/sync_api.py
# Sync analysis API: upload file -> analysis + summary -> return result (no indexing)

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Document Sync API", version="0.1")

_analysis_client = None
_summarizer = None
_thread_pool = None


def init(analysis_client, summarizer, thread_pool):
    global _analysis_client, _summarizer, _thread_pool
    _analysis_client = analysis_client
    _summarizer = summarizer
    _thread_pool = thread_pool


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _analysis_client is not None}


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    processing_mode: str = Form("default"),
):
    if _analysis_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    request_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ".bin"
    tmp_path = Path(f"/tmp/media_processing/document_processing/sync_{request_id}{suffix}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_result_path = None

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

        analysis_result = await loop.run_in_executor(
            _thread_pool,
            _analysis_client.analyze_document_sync,
            request_payload,
        )

        if analysis_result.get("status") != "success":
            err = analysis_result.get("error", {})
            raise HTTPException(status_code=422, detail=err)

        analysis_result_path = analysis_result["parameters"]["analysis_result_path"]

        with open(analysis_result_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # summarize_from_json is synchronous (uses requests), run in thread pool
        summary_result = await loop.run_in_executor(
            _thread_pool,
            _summarizer.summarize_from_json,
            analysis_result_path,
        )

        summary_text = str(summary_result)

        return JSONResponse({
            "filename": file.filename,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "summary": summary_text,
        })

    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if analysis_result_path:
            Path(analysis_result_path).unlink(missing_ok=True)
