# src/inference/handlers/internvl.py
"""
InternVL handler — OpenGVLab InternVL 系列（InternVL2 / InternVL3 / InternVL3_5）。

InternVL 不走標準 processor→generate 介面，而是：
    AutoModel(trust_remote_code) + AutoTokenizer
    影像 → 動態切圖（tiling）→ pixel_values → model.chat(tokenizer, pixel_values, question)

與 10-image / 11-video / 12-document 各模組內嵌的推理程式碼等價，
集中到本 handler 後，這三個模組可改呼叫 07 的 /inference/infer（task=vlm）。

註冊方式：
    task            = "vlm"
    model_class     = "AutoModel"
    processor_class = "AutoTokenizer"
    torch_dtype     = "bf16"          （官方建議）
    custom_handler  = "internvl"      （內建名；或由 VLMHandler 依 InternVL 字樣自動分派）

data:
    image:  路徑 / PIL.Image / base64 / data URI
    prompt: str
options:
    max_num（切圖上限，預設 12）、max_new_tokens、temperature、do_sample ...
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base import BaseHandler

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        from .vlm import _load_image

        if "image" not in data or "prompt" not in data:
            raise ValueError("internvl handler requires data['image'] and data['prompt']")

        model = loaded["model"]
        tokenizer = loaded["processor"]
        if tokenizer is None:
            raise ValueError("internvl handler requires processor_class='AutoTokenizer' at registration")

        image = _load_image(data["image"])
        prompt = data["prompt"]
        max_num = int(options.get("max_num", 12))
        input_size = int(options.get("input_size", 448))

        pixel_values = _to_pixel_values(image, input_size=input_size, max_num=max_num)
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        pixel_values = pixel_values.to(dtype=model_dtype, device=model_device)

        generation_config = {
            "max_new_tokens": int(options.get("max_new_tokens", options.get("max_length", 512))),
            "do_sample": bool(options.get("do_sample", False)),
        }
        if "temperature" in options:
            generation_config["temperature"] = float(options["temperature"])
            generation_config["do_sample"] = True
        if "top_p" in options:
            generation_config["top_p"] = float(options["top_p"])

        # InternVL 的 <image> placeholder 由 model.chat 內部處理，prompt 需含 "<image>\n" 前綴
        question = prompt if "<image>" in prompt else f"<image>\n{prompt}"

        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, question, generation_config)

        return {
            "response": response,
            "metadata": {"num_patches": int(pixel_values.shape[0]), "input_size": input_size},
        }


# ===== InternVL 官方前處理（與 10/11/12 模組內嵌版一致）=====


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target_ar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _to_pixel_values(image, input_size=448, max_num=12):
    import torch

    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(t) for t in tiles])
