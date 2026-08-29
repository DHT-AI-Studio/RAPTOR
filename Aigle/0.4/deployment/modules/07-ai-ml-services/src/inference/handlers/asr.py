# src/inference/handlers/asr.py
"""ASR handler — raw model+processor 路線。pipeline 路線由 adapter 直接處理。"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseHandler


class ASRHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if "audio" not in data:
            raise ValueError("asr handler requires data['audio']")

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        audio = _load_audio(data["audio"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {}
        if "language" in options:
            gen_kwargs["language"] = options["language"]
        if "task" in options:
            gen_kwargs["task"] = options["task"]

        with torch.no_grad():
            ids = model.generate(**inputs, **gen_kwargs)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return {"text": text, "metadata": {}}


def _load_audio(src: Any):
    """src 可以是 path 或 numpy array；用 librosa/soundfile 解碼為 16kHz mono。"""
    import numpy as np

    if isinstance(src, np.ndarray):
        return src
    try:
        import librosa
        wav, _ = librosa.load(src, sr=16000, mono=True)
        return wav
    except ImportError:
        import soundfile as sf
        wav, sr = sf.read(src)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav
