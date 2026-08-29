# src/inference/handlers/tts.py
"""
TTS handler — 文字轉語音。

支援兩條路徑：
  1) pipeline 路線：註冊時 pipeline_task="text-to-speech"（SpeechT5 / Bark / VITS 等
     transformers 原生支援的模型），adapter 直接跑 pipeline，本 handler 的
     postprocess 把 raw audio 統一編碼成 base64 WAV。
  2) model+processor 路線：processor(text) → model.generate → waveform。
     介面差異大的模型（如 VibeVoice）請用 custom_handler。

data:
    text: str                     （別名 inputs 亦可）
    voice: str（可選 — 有 speaker embedding / voice preset 概念的模型才用得到）
options:
    speed: float（可選；由具體模型決定是否支援）
    output_format: 目前僅 "wav"

回傳：
    {"audio_base64": ..., "format": "wav", "sample_rate": ..., "metadata": {...}}
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict

from .base import BaseHandler


class TTSHandler(BaseHandler):
    # ---- pipeline 路線 hooks ----

    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        text = data.get("text", data.get("inputs"))
        if not isinstance(text, str) or not text:
            raise ValueError("tts handler requires data['text'] (str)")
        return {"text": text}

    def postprocess(self, raw: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        # transformers text-to-speech pipeline 回傳 {"audio": np.ndarray, "sampling_rate": int}
        if isinstance(raw, dict) and "audio" in raw:
            return _wav_result(raw["audio"], int(raw.get("sampling_rate", 16000)))
        return raw

    # ---- model+processor 路線 ----

    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        text = data.get("text", data.get("inputs"))
        if not isinstance(text, str) or not text:
            raise ValueError("tts handler requires data['text'] (str)")

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        inputs = processor(text=text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs)

        waveform = out
        if hasattr(out, "cpu"):
            waveform = out.squeeze().float().cpu().numpy()
        sample_rate = int(
            options.get("sample_rate")
            or getattr(getattr(model, "config", None), "sampling_rate", None)
            or getattr(getattr(model, "generation_config", None), "sample_rate", None)
            or 16000
        )
        return _wav_result(waveform, sample_rate)


def _wav_result(audio, sample_rate: int) -> Dict[str, Any]:
    """numpy waveform → base64 WAV（統一回應格式）。"""
    import numpy as np

    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim > 1:  # (channels, samples) → mono
        arr = arr.mean(axis=0)
    buf = io.BytesIO()
    _write_wav(buf, arr, sample_rate)
    return {
        "audio_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "format": "wav",
        "sample_rate": sample_rate,
        "metadata": {"duration_seconds": round(len(arr) / max(sample_rate, 1), 3)},
    }


def _write_wav(buf: io.BytesIO, arr, sample_rate: int) -> None:
    """float32 [-1,1] → 16-bit PCM WAV（僅標準庫，不依賴 soundfile）。"""
    import struct
    import numpy as np

    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
