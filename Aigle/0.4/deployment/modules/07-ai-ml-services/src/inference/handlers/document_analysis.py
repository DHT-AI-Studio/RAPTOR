# src/inference/handlers/document_analysis.py
"""文檔分析 handler — 同 video 一樣是 fallback；具體模型如 Donut/LayoutLM 通常需專屬 handler。"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseHandler


class DocumentAnalysisHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if "document" not in data:
            raise ValueError("document-analysis handler requires data['document']")

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        if "query" in data:
            kwargs["text"] = data["query"]
        inputs = processor(images=data["document"], **kwargs)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": int(options.get("max_new_tokens", options.get("max_length", 256)))}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"response": text, "metadata": {}}
