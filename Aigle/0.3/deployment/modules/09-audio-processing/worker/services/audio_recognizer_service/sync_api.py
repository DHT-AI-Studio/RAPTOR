# Sync transcription API: upload audio -> recognition + diarization + classification
# -> merge -> summary -> return {text, summary} matching Redis state format

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Sync API", version="0.2")

_recognition_client = None
_summarizer = None
_thread_pool = None

DIARIZATION_SYNC_URL = os.getenv("DIARIZATION_SYNC_URL", "http://raptor-audio-diarization:8029")
CLASSIFIER_SYNC_URL  = os.getenv("CLASSIFIER_SYNC_URL",  "http://raptor-audio-classifier:8039")
CROSS_SERVICE_TIMEOUT = int(os.getenv("CROSS_SERVICE_TIMEOUT", "300"))


def init(recognition_client, summarizer, thread_pool):
    global _recognition_client, _summarizer, _thread_pool
    _recognition_client = recognition_client
    _summarizer = summarizer
    _thread_pool = thread_pool


# ── helpers ───────────────────────────────────────────────────────────────────

def _time_overlap(s1, e1, s2, e2):
    return max(0, min(e1, e2) - max(s1, s2))


def _call_diarize(file_path: str, filename: str, request_id: str) -> List[Dict]:
    """Blocking HTTP call to diarization service — run in executor."""
    try:
        resp = requests.post(
            f"{DIARIZATION_SYNC_URL}/diarize",
            json={"file_path": file_path, "filename": filename, "request_id": request_id},
            timeout=CROSS_SERVICE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("segments", [])
    except Exception as e:
        logger.warning(f"Diarization service call failed: {e}; speaker labels will be UNKNOWN")
        return []


def _call_classify(file_path: str, filename: str, request_id: str) -> List[Dict]:
    """Blocking HTTP call to classifier service — run in executor."""
    try:
        resp = requests.post(
            f"{CLASSIFIER_SYNC_URL}/classify",
            json={"file_path": file_path, "filename": filename, "request_id": request_id},
            timeout=CROSS_SERVICE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("segments", [])
    except Exception as e:
        logger.warning(f"Classifier service call failed: {e}; audio_labels will be empty")
        return []


def _merge_segments(
    recognizer_segments: List[Dict],
    diarization_segments: List[Dict],
    classification_segments: List[Dict],
) -> List[Dict]:
    """Build merged items with payload wrapper compatible with AudioSummary.extract_audio_info."""
    merged = []
    for idx, seg in enumerate(recognizer_segments):
        start = seg["start"]
        end   = seg["end"]
        text  = seg.get("text", "").strip()

        # Best-overlap speaker
        speaker = None
        best_overlap = 0
        for d in diarization_segments:
            ov = _time_overlap(start, end, d.get("start", 0), d.get("end", 0))
            if ov > best_overlap:
                best_overlap = ov
                speaker = d.get("speaker")

        # Audio labels above threshold
        label_set: set = set()
        for c in classification_segments:
            ov = _time_overlap(start, end, c.get("segment_start", 0), c.get("segment_end", 0))
            if ov > 0:
                for cls in c.get("top_classes", []):
                    if cls.get("probability", 0) > 0.4:
                        label_set.add(cls["label"])

        merged.append({
            "id": str(uuid.uuid4()),
            "payload": {
                "start_time":   str(start),
                "end_time":     str(end),
                "speaker":      speaker or "UNKNOWN",
                "text":         text,
                "audio_labels": list(label_set),
                "embedding_type": "text",
            },
        })
    return merged


def _format_text(merged_items: List[Dict]) -> str:
    """Format text identical to orchestrator Redis state."""
    parts = []
    for item in merged_items:
        p = item.get("payload", {})
        if p.get("embedding_type") == "text" and p.get("text"):
            parts.append(
                f'"{p["start_time"]}-{p["end_time"]}sec" "{p["speaker"]}": "{p["text"]}"'
            )
    return " ".join(parts)


def _run_summary(summarizer, merged_items: List[Dict], request_id: str) -> str:
    """Write merged data to temp file, call process_summary, clean up."""
    tmp = Path(f"/tmp/media_processing/audio_processing/summary_input_{request_id}.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(json.dumps(merged_items, ensure_ascii=False), encoding="utf-8")
        return summarizer.process_summary(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _recognition_client is not None}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Synchronous audio analysis — returns text (with speaker labels) and summary.
    Output matches the Redis state format produced by the full Kafka pipeline.
    """
    if _recognition_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    request_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp_path = Path(f"/tmp/media_processing/audio_processing/sync_{request_id}{suffix}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        file_path = str(tmp_path)
        filename  = file.filename or f"upload{suffix}"

        recognition_payload = {
            "request_id": request_id,
            "parameters": {
                "file_path": file_path,
                "primary_filename": filename,
            },
        }

        loop = asyncio.get_event_loop()

        # Run recognition (thread pool), diarization and classification (default executor) in parallel
        recognition_task = loop.run_in_executor(
            _thread_pool,
            _recognition_client.recognize_audio_sync,
            recognition_payload,
        )
        diarization_task = loop.run_in_executor(
            None, _call_diarize, file_path, filename, request_id
        )
        classification_task = loop.run_in_executor(
            None, _call_classify, file_path, filename, request_id
        )

        recognition_result, diarization_segments, classification_segments = await asyncio.gather(
            recognition_task, diarization_task, classification_task
        )

        if recognition_result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=recognition_result.get("error", {}).get("message", "Recognition failed"),
            )

        params = recognition_result.get("parameters", {})

        # Load recognition segments
        recognizer_segments = []
        recognition_result_path = params.get("recognition_result_path")
        if recognition_result_path and Path(recognition_result_path).exists():
            with open(recognition_result_path, "r", encoding="utf-8") as f:
                recognizer_segments = json.load(f)
            Path(recognition_result_path).unlink(missing_ok=True)

        # Merge
        merged_items = _merge_segments(
            recognizer_segments, diarization_segments, classification_segments
        )

        # Summary (blocking — run in thread pool)
        summary = await loop.run_in_executor(
            _thread_pool, _run_summary, _summarizer, merged_items, request_id
        )

        text = _format_text(merged_items)

        segments = [
            {
                "start_time": float(item["payload"]["start_time"]),
                "end_time":   float(item["payload"]["end_time"]),
                "speaker":    item["payload"].get("speaker", "UNKNOWN"),
                "text":       item["payload"].get("text", ""),
            }
            for item in merged_items
            if item["payload"].get("text", "").strip()
        ]

        return JSONResponse({
            "filename": file.filename,
            "language": params.get("language"),
            "duration": params.get("duration"),
            "summary":  summary,
            "text":     text,
            "segments": segments,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
