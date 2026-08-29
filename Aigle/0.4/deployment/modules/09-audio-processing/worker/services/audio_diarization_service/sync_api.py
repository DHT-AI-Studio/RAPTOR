# Sync diarization API: accepts file_path on shared volume → returns speaker segments

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Diarization Sync API", version="0.1")

_diarization_client = None
_thread_pool = None


def init(diarization_client, thread_pool):
    global _diarization_client, _thread_pool
    _diarization_client = diarization_client
    _thread_pool = thread_pool


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _diarization_client is not None}


class DiarizeRequest(BaseModel):
    file_path: str
    filename: str = "audio.wav"
    request_id: str = ""


@app.post("/diarize")
async def diarize_audio(req: DiarizeRequest):
    if _diarization_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    request_id = req.request_id or str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "parameters": {
            "file_path": req.file_path,
            "primary_filename": req.filename,
            "diarization_type": "basic",
            "min_speakers": 1,
            "max_speakers": 10,
        },
    }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _thread_pool,
        _diarization_client.diarize_audio_sync,
        payload,
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail=result.get("error", {}).get("message", "Diarization failed"),
        )

    result_path = result.get("parameters", {}).get("diarization_result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=500, detail="Diarization result file not found")

    with open(result_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    Path(result_path).unlink(missing_ok=True)

    return JSONResponse({"segments": segments})
