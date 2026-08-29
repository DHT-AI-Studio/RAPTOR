# src/inference/handlers/audio_classification.py
"""音頻分類 handler — 多數情況建議直接走 pipeline_task='audio-classification'。"""

from __future__ import annotations

from typing import Any, Dict

from .asr import _load_audio
from .base import BaseHandler


class AudioClassificationHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if "audio" not in data:
            raise ValueError("audio-classification handler requires data['audio']")

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")
        top_k = int(options.get("top_k", 5))

        audio = _load_audio(data["audio"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits[0], dim=-1)
        scores, indices = probs.topk(min(top_k, probs.shape[0]))
        id2label = getattr(model.config, "id2label", {})
        results = [
            {"label": id2label.get(int(i), str(int(i))), "score": float(s)}
            for s, i in zip(scores.tolist(), indices.tolist())
        ]
        return {"classifications": results, "metadata": {}}
