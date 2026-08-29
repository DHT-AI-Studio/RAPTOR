# src/inference/handlers/video_analysis.py
"""視頻分析 handler — 預設 fallback 行為；不同 video 模型差異大，多數情況需自訂 handler。"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseHandler


class VideoAnalysisHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if "video" not in data:
            raise ValueError("video-analysis handler requires data['video']")

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        # 由 processor 自行解析（部分 processor 接受路徑字串，部分要 frames array）
        kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        if "prompt" in data:
            kwargs["text"] = data["prompt"]
        inputs = processor(videos=data["video"], **kwargs)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": int(options.get("max_new_tokens", options.get("max_length", 256)))}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"response": text, "metadata": {}}
