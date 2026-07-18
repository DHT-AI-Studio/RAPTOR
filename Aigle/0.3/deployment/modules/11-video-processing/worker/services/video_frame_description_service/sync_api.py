# Sync video analysis API: upload video → moments → VLM + OCR → summary → contextual prefixes
# Returns per-moment results matching the Kafka pipeline output (before hybrid search indexing).

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

import opencc
import requests
from decord import VideoReader, cpu
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Frame Sync API", version="0.3")

_processor = None       # FrameDescriptionProcessor
_summarizer = None      # VideoSummary
_thread_pool = None

OCR_SYNC_URL = os.getenv("VIDEO_OCR_SYNC_URL", "http://raptor-video-ocr-frame:8031")
AUDIO_SYNC_URL = os.getenv("AUDIO_SYNC_URL", "http://raptor-audio-recognizer:8019")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen3.5:9b")
CROSS_SERVICE_TIMEOUT = int(os.getenv("CROSS_SERVICE_TIMEOUT", "600"))
MOMENT_DURATION_SEC = 10.0
KEYFRAMES_PER_MOMENT = 2
_cc = opencc.OpenCC("s2twp")


def init(processor, summarizer, thread_pool):
    global _processor, _summarizer, _thread_pool
    _processor = processor
    _summarizer = summarizer
    _thread_pool = thread_pool


# ── helpers ───────────────────────────────────────────────────────────────────

def _probe_duration(video_path: str) -> float:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    return len(vr) / float(vr.get_avg_fps())


def _extract_keyframes(video_path: str, start: float, end: float, out_dir: str) -> List[str]:
    """Extract KEYFRAMES_PER_MOMENT frames from [start, end) and save as scene_frame_NNN.jpg."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    max_frame = len(vr) - 1
    start_idx = max(0, round(start * fps))
    end_idx = min(round(end * fps), max_frame)

    if start_idx >= end_idx:
        return []

    seg = max(1, (end_idx - start_idx) // KEYFRAMES_PER_MOMENT)
    indices = [min(start_idx + i * seg, end_idx) for i in range(KEYFRAMES_PER_MOMENT)]

    paths = []
    for i, fi in enumerate(indices):
        img = Image.fromarray(vr[fi].asnumpy()).convert("RGB")
        path = os.path.join(out_dir, f"scene_frame_{i:03d}.jpg")
        img.save(path, "JPEG", quality=85)
        paths.append(path)
    return paths


def _call_ocr(keyframe_dir: str, request_id: str) -> Dict[str, Any]:
    """Blocking HTTP call to OCR service. Returns {} on failure (graceful degradation)."""
    try:
        resp = requests.post(
            f"{OCR_SYNC_URL}/ocr",
            json={"keyframe_dir": keyframe_dir, "request_id": request_id},
            timeout=CROSS_SERVICE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("ocr_results", {})
    except Exception as e:
        logger.warning(f"OCR service call failed for {keyframe_dir}: {e}; using empty OCR")
        return {}


def _ocr_text_for_moment(ocr_results: Dict[str, Any]) -> str:
    texts = [v["text"] for v in ocr_results.values() if v.get("text")]
    return " | ".join(texts)


def _asr_text_for_moment(audio_segments: List[Dict], start: float, end: float) -> str:
    texts = [
        s["text"] for s in audio_segments
        if s.get("text", "").strip()
        and float(s.get("end_time", 0)) > start
        and float(s.get("start_time", 0)) < end
    ]
    return " ".join(texts)


def _call_audio_transcribe(video_path: str, request_id: str) -> List[Dict]:
    """Blocking HTTP call to audio sync API. Returns [] on failure (graceful degradation)."""
    try:
        with open(video_path, "rb") as f:
            filename = os.path.basename(video_path)
            resp = requests.post(
                f"{AUDIO_SYNC_URL}/transcribe",
                files={"file": (filename, f, "video/mp4")},
                timeout=CROSS_SERVICE_TIMEOUT,
            )
        resp.raise_for_status()
        raw = resp.json().get("segments", [])
        return [
            {
                "start_time": float(s["start_time"]),
                "end_time":   float(s["end_time"]),
                "speaker":    s.get("speaker", "UNKNOWN"),
                "text":       s.get("text", ""),
            }
            for s in raw
            if s.get("text", "").strip()
        ]
    except Exception as e:
        logger.warning(f"Audio transcribe call failed for {request_id}: {e}; using empty audio")
        return []


# ── contextual prefix (inline, avoids config import conflict with contextualize service) ──

def _build_contextual_prompt(global_summary: str, batch: List[Dict]) -> str:
    chunks = "\n".join(
        f'<chunk index="{m["moment_index"]}">'
        f'ocr: {m.get("ocr_text", "")} / '
        f'asr: {m.get("asr_text", "")} / '
        f'lvlm: {m.get("lvlm_desc", "")}'
        f'</chunk>'
        for m in batch
    )
    return (
        f"<document>\n{global_summary}\n</document>\n\n"
        "你是一個影片內容分析專家。以下是影片中數個連續的 10 秒片段，"
        "請根據上方的影片全局總結，為每個 <chunk> 產生一句精簡的繁體中文上下文"
        "（不超過 50 字），說明此片段在整支影片中的角色或主題定位。\n\n"
        "請嚴格只輸出 JSON 陣列格式，不要有任何其他文字：\n"
        '[{"index": <int>, "context": "<str>"}, ...]\n\n'
        f"{chunks}"
    )


def _parse_contextual_response(raw: str, batch: List[Dict]) -> List[str]:
    def _extract(data):
        index_map = {item["index"]: item.get("context", "") for item in data}
        return [index_map.get(m["moment_index"], "") for m in batch]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return _extract(parsed)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return _extract(json.loads(match.group()))
        except Exception:
            pass
    return [""] * len(batch)


def _call_contextual_batch(global_summary: str, batch: List[Dict]) -> List[str]:
    """Blocking HTTP call to inference for one batch of moments."""
    prompt = _build_contextual_prompt(global_summary, batch)
    model = os.getenv("CONTEXTUALIZE_MODEL_NAME") or INFERENCE_MODEL
    payload = {
        "task": "text-generation-ollama",
        "engine": "ollama",
        "model_name": model,
        "data": {"inputs": prompt},
        "options": {"max_length": 600, "temperature": 0.2},
    }
    try:
        resp = requests.post(
            f"{INFERENCE_URL}/inference/infer",
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("result", {}).get("generated_text", "").strip()
        return _parse_contextual_response(raw, batch)
    except Exception as e:
        logger.warning(f"Contextual prefix call failed for batch starting at {batch[0]['moment_index']}: {e}")
        return [""] * len(batch)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _processor is not None}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Full synchronous video analysis.
    Pipeline: upload → moments → VLM batch + OCR (parallel) → global summary → contextual prefixes.
    Returns per-moment results with contextual_prefix, plus global_summary.
    """
    if _processor is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    request_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    base_dir = Path(f"/tmp/media_processing/video_processing/sync_{request_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(base_dir / f"video{suffix}")

    try:
        # Save uploaded file
        content = await file.read()
        Path(video_path).write_bytes(content)

        loop = asyncio.get_event_loop()

        # 1. Probe duration, build moment list
        duration = await loop.run_in_executor(None, _probe_duration, video_path)
        moments: List[Dict] = []
        start = 0.0
        idx = 0
        while start < duration:
            end = min(start + MOMENT_DURATION_SEC, duration)
            moments.append({
                "moment_index": idx,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
            })
            idx += 1
            start = end

        # 2. Extract keyframes per moment (concurrent, CPU-bound via default executor)
        async def _extract_async(m):
            kf_dir = str(base_dir / f"moment_{m['moment_index']}")
            os.makedirs(kf_dir, exist_ok=True)
            paths = await loop.run_in_executor(
                None, _extract_keyframes, video_path, m["start_time"], m["end_time"], kf_dir
            )
            return kf_dir, paths

        kf_results = await asyncio.gather(*[_extract_async(m) for m in moments])
        keyframe_dirs = [r[0] for r in kf_results]
        keyframe_paths_per_moment = [r[1] for r in kf_results]

        # 3. VLM batch + OCR per moment + audio transcription — run in parallel
        vlm_batch = [
            {
                "keyframe_paths": keyframe_paths_per_moment[i],
                "start_time": m["start_time"],
                "end_time": m["end_time"],
            }
            for i, m in enumerate(moments)
        ]

        vlm_task = loop.run_in_executor(
            _thread_pool,
            _processor.generate_moment_descriptions_batch,
            vlm_batch,
        )
        ocr_tasks = [
            loop.run_in_executor(
                None, _call_ocr, keyframe_dirs[i], f"{request_id}_m{m['moment_index']}"
            )
            for i, m in enumerate(moments)
        ]
        audio_task = loop.run_in_executor(None, _call_audio_transcribe, video_path, request_id)

        all_results = await asyncio.gather(vlm_task, *ocr_tasks, audio_task)
        vlm_descriptions = all_results[0]
        all_ocr_results = all_results[1:-1]
        audio_segments = all_results[-1]

        # 4. Build per-moment structures
        moment_descriptions = [
            {
                "moment_index": m["moment_index"],
                "start_time": m["start_time"],
                "end_time": m["end_time"],
                "lvlm_description": _cc.convert(vlm_descriptions[i]) if vlm_descriptions[i] else "",
            }
            for i, m in enumerate(moments)
        ]

        # 5. Global summary
        frames_for_summary = [
            {"timestamp": m["start_time"], "text": _ocr_text_for_moment(all_ocr_results[i])}
            for i, m in enumerate(moments)
            if _ocr_text_for_moment(all_ocr_results[i])
        ]
        video_analysis_data = {
            "event_summary": "",
            "frames": frames_for_summary,
            "moment_descriptions": moment_descriptions,
        }
        va_path = base_dir / "video_analysis.json"
        aa_path = base_dir / "audio_analysis.json"
        va_path.write_text(json.dumps(video_analysis_data, ensure_ascii=False), encoding="utf-8")
        aa_path.write_text(json.dumps(audio_segments, ensure_ascii=False), encoding="utf-8")

        global_summary = await loop.run_in_executor(
            _thread_pool, _summarizer.process_summary, str(va_path), str(aa_path)
        )

        # 6. Contextual prefixes (parallel batches, each batch is a blocking HTTP call in executor)
        CONTEXTUALIZE_BATCH_SIZE = 8
        contextualize_moments = [
            {
                "moment_index": m["moment_index"],
                "ocr_text": _ocr_text_for_moment(all_ocr_results[i]),
                "asr_text": _asr_text_for_moment(audio_segments, m["start_time"], m["end_time"]),
                "lvlm_desc": moment_descriptions[i]["lvlm_description"],
            }
            for i, m in enumerate(moments)
        ]
        batches = [
            contextualize_moments[i:i + CONTEXTUALIZE_BATCH_SIZE]
            for i in range(0, len(contextualize_moments), CONTEXTUALIZE_BATCH_SIZE)
        ]
        batch_results = await asyncio.gather(*[
            loop.run_in_executor(None, _call_contextual_batch, global_summary, b)
            for b in batches
        ])
        flat_prefixes = [p for batch_prefixes in batch_results for p in batch_prefixes]

        # 7. Build final result
        result_moments = [
            {
                "moment_index": m["moment_index"],
                "start_time": m["start_time"],
                "end_time": m["end_time"],
                "lvlm_description": moment_descriptions[i]["lvlm_description"],
                "ocr_text": _ocr_text_for_moment(all_ocr_results[i]),
                "contextual_prefix": flat_prefixes[i] if i < len(flat_prefixes) else "",
            }
            for i, m in enumerate(moments)
        ]

        # 8. Build embedding_type:text entries (same format as Kafka pipeline / Redis)
        import uuid as _uuid
        text_entries = []
        for i, m in enumerate(moments):
            ocr_text = _ocr_text_for_moment(all_ocr_results[i])
            asr_text = _asr_text_for_moment(audio_segments, m["start_time"], m["end_time"])
            lvlm_desc = moment_descriptions[i]["lvlm_description"]

            parts = []
            if ocr_text:
                parts.append(f"ocr:{{{ocr_text}}}")
            if asr_text:
                parts.append(f"asr:{{{asr_text}}}")
            if lvlm_desc:
                parts.append(f"lvlm:{{{lvlm_desc}}}")
            combined = " / ".join(parts)
            if not combined:
                continue

            prefix = flat_prefixes[i] if i < len(flat_prefixes) else ""
            if prefix:
                combined = f"[上下文補充] {prefix} {combined}"

            text_entries.append({
                "id": str(_uuid.uuid4()),
                "payload": {
                    "video_id": f"{file.filename}_moment_{m['moment_index']}",
                    "text": combined,
                    "filename": file.filename,
                    "chunk_index": m["moment_index"],
                    "start_time": str(m["start_time"]),
                    "end_time": str(m["end_time"]),
                    "embedding_type": "text",
                },
            })

        return JSONResponse({
            "filename": file.filename,
            "duration": round(duration, 2),
            "moment_count": len(moments),
            "summary": global_summary,
            "text": text_entries,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(str(base_dir), ignore_errors=True)
