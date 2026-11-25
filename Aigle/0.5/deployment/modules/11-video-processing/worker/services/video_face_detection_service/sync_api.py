# Sync face-embedding API: accepts a single image → face detection + ArcFace → returns a 512-dim embedding.
# Seed-photo pipeline step 1 (v1.4 §4.5): the Gateway forwards the user's photo here to get its embedding.
# Storing the photo in Module 04 (step 2) belongs to the Identity API sprint and is not done here.

import asyncio
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Face Detection Sync API", version="0.1")

_detector = None
_semaphore = None   # serializes concurrent calls to protect GPU


def init(detector):
    global _detector, _semaphore
    _detector = detector
    _semaphore = asyncio.Semaphore(1)


@app.get("/health")
async def health():
    return {"status": "ok", "detector_ready": _detector is not None}


@app.post("/embed-face")
async def embed_face(file: UploadFile = File(...)):
    """
    Detect faces in one image and return the embedding of the main face.
    400 if the image cannot be decoded or contains no face.
    Returns {"embedding": [512 floats], "bbox": [x1, y1, x2, y2], "confidence": float, "face_count": int}
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Face detector not ready")

    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    async with _semaphore:
        try:
            result = await asyncio.to_thread(_detector.embed_best_face, image)
        except Exception as e:  # noqa: BLE001 -- ONNX/insightface inference, failure modes vary, logged and surfaced as 500
            logger.error(f"Face embedding error for {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(status_code=400, detail="No face detected")

    face, embedding, face_count = result
    return {
        "embedding": embedding.tolist(),
        "bbox": [round(float(v), 1) for v in face.bbox],
        "confidence": face.confidence,
        "face_count": face_count,
    }
