# src/inference/handlers/ocr.py
"""OCR handler — raw model+processor 路線。簡單模型可直接走 pipeline_task=image-to-text。"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseHandler
from .vlm import _load_image


class OCRHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if "image" not in data:
            raise ValueError("ocr handler requires data['image']")

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        image = _load_image(data["image"])
        inputs = processor(images=image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            ids = model.generate(**inputs)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return {"text": text, "metadata": {}}
