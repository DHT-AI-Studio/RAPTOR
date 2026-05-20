# Sync OCR API: accepts keyframe directory path → runs PaddleOCR → returns results
# Called by video-frame-description service during sync video analysis.

import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Video OCR Sync API", version="0.1")

_ocr_processor = None
_semaphore = None   # serializes concurrent calls to protect GPU


class OCRRequest(BaseModel):
    keyframe_dir: str
    request_id: str


def init(ocr_processor):
    global _ocr_processor, _semaphore
    _ocr_processor = ocr_processor
    _semaphore = asyncio.Semaphore(1)


@app.get("/health")
async def health():
    return {"status": "ok", "ocr_ready": _ocr_processor is not None}


@app.post("/ocr")
async def run_ocr(req: OCRRequest):
    """
    Run OCR on all scene_frame_*.jpg files inside keyframe_dir.
    Returns {"ocr_results": {frame_key: {"text": "...", "regions": {...}}}}
    """
    if _ocr_processor is None:
        raise HTTPException(status_code=503, detail="OCR service not ready")

    async with _semaphore:
        try:
            result = await _ocr_processor.process_scene_frames(
                request_id=req.request_id,
                primary_filename="sync",
                scene_output_directory=req.keyframe_dir,
                cleanup_after=False,  # keep engine alive across calls
            )
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"OCR processing error for {req.keyframe_dir}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
