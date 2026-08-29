# src/inference/handlers/vlm.py
"""
視覺語言模型 handler

支援三條路徑：
  1) 通用 VLM（標準 AutoProcessor 接口）
  2) Qwen2.5-VL（需要 chat-template + qwen_vl_utils.process_vision_info）
  3) InternVL（.chat() 介面 + 動態切圖 — 委派給 internvl.InternVLHandler）

選擇哪條路徑由 ModelSpec 決定：
  - spec.processor_class 含 "Qwen2_5_VL" / spec.model_class 含 "Qwen2_5_VL" → Qwen 路線
  - spec.model_class / physical_path 含 "internvl"（不分大小寫）→ InternVL 路線
  - 其他 → 通用路線

註冊新模型時只要設對 model_class / processor_class，handler 就會走對的分支，
不需要在程式碼裡加 if 'qwen' in path.lower()。
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict

from PIL import Image

from .base import BaseHandler

logger = logging.getLogger(__name__)


class VLMHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in data or "prompt" not in data:
            raise ValueError("vlm handler requires data['image'] and data['prompt']")

        image = _load_image(data["image"])
        prompt = data["prompt"]

        if _is_qwen_vl(spec):
            return self._infer_qwen_vl(loaded, image, prompt, options)
        if _is_internvl(spec):
            from .internvl import InternVLHandler
            return InternVLHandler().run(loaded, spec, data, options)
        return self._infer_generic(loaded, image, prompt, options)

    # ---- 通用 VLM ----
    @staticmethod
    def _infer_generic(loaded: Dict[str, Any], image: Image.Image, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = _gen_kwargs(options)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"response": response, "metadata": {"output_length": int(outputs.shape[-1])}}

    # ---- Qwen2.5-VL ----
    @staticmethod
    def _infer_qwen_vl(loaded: Dict[str, Any], image: Image.Image, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        model = loaded["model"]
        processor = loaded["processor"]
        device = loaded.get("device", "cpu")

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            logger.warning("qwen_vl_utils unavailable; falling back to direct image input")
            image_inputs, video_inputs = [image], None

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = _gen_kwargs(options)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # 只 decode 新生成的部分
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {"response": response, "metadata": {"output_length": int(outputs.shape[-1])}}


# ===== helpers =====


def _is_qwen_vl(spec) -> bool:
    return any(
        "Qwen2_5_VL" in (getattr(spec, attr, "") or "")
        for attr in ("model_class", "processor_class")
    )


def _is_internvl(spec) -> bool:
    # InternVL 註冊時 model_class 通常只是 "AutoModel"，故同時看 physical_path
    return any(
        "internvl" in (getattr(spec, attr, "") or "").lower()
        for attr in ("model_class", "physical_path")
    )


def _load_image(src: Any) -> Image.Image:
    """支援 PIL.Image / 檔案路徑 / data-URL / base64 字串。"""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, str):
        if src.startswith("data:image"):
            src = src.split(",", 1)[1]
        try:
            raw = base64.b64decode(src, validate=False)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return Image.open(src).convert("RGB")
    raise ValueError(f"unsupported image type: {type(src)}")


def _gen_kwargs(options: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("max_length", "max_new_tokens", "temperature", "top_p", "top_k", "do_sample", "num_beams")
    out = {k: options[k] for k in keys if k in options}
    out.setdefault("do_sample", True)
    if "max_new_tokens" not in out and "max_length" not in out:
        out["max_new_tokens"] = 256
    return out
