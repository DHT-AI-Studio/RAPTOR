"""Llama Guard 3 11B Vision inference.

Loaded models are owned by the shared resource manager
(app/services/model_resource_manager.py), separate from the Ollama-based
adapters used by the text guard pipeline.

Memory note: 11B parameters in bfloat16 ≈ 22 GB — exceeds any single
consumer GPU. Default uses NF4 4-bit quantization with double quant
(LLAMA_GUARD_LOAD_IN_4BIT=true) for ≈ 5.5-6 GB.
"""
from __future__ import annotations

import io
import logging
import os

from app.services.model_resource_manager import ModelSpec, env_mb, get_manager

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Check this image for unsafe content."

# Pinned to a specific upstream commit so a service restart never silently
# pulls a different set of weights.  Update intentionally when upgrading.
_MODEL_REVISION = "62d4275543ec7503de66c486de1c0c2103e365ac"

_MANAGER_KEY_PREFIX = "llama-guard-3-11b-vision"


def _load_in_4bit() -> bool:
    return os.environ.get("LLAMA_GUARD_LOAD_IN_4BIT", "true").lower() == "true"


def _load_model(model_path: str, device_index: int) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_in_4bit = _load_in_4bit()

    processor = AutoProcessor.from_pretrained(model_path, revision=_MODEL_REVISION)

    # device_map pins the whole model to the managed GPU.  "auto" would let
    # accelerate spill layers to CPU when VRAM is short, which bitsandbytes
    # then rejects ("Some modules are dispatched on the CPU or the disk") —
    # the resource manager checks free VRAM before we get here instead.
    device_map = {"": device_index}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # saves ~0.4 bits/param extra
            bnb_4bit_quant_type="nf4",       # better accuracy than fp4
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            revision=_MODEL_REVISION,
            quantization_config=bnb_config,
            device_map=device_map,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            revision=_MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
    model.eval()

    logger.info(
        "llama-guard-vision: loaded model '%s' on %s (4bit=%s)", model_path, model.device, load_in_4bit
    )
    return {"model": model, "processor": processor}


def _register(model_path: str) -> str:
    load_in_4bit = _load_in_4bit()
    key = f"{_MANAGER_KEY_PREFIX}:{model_path}:{'4bit' if load_in_4bit else 'bf16'}"
    # Defaults measured on an RTX 4060 Ti (4-bit: 6.85 GB allocated, 7.75 GB
    # peak while loading).  Override per deployment if needed.
    get_manager().register(ModelSpec(
        key=key,
        loader=lambda device_index: _load_model(model_path, device_index),
        vram_bytes=env_mb("LLAMA_GUARD_VRAM_ESTIMATE_MB", 7000 if load_in_4bit else 21500),
        inference_reserve_bytes=env_mb("LLAMA_GUARD_INFERENCE_RESERVE_MB", 1024),
        ram_bytes=env_mb("LLAMA_GUARD_RAM_ESTIMATE_MB", 4096),
    ))
    return key


def run_llama_guard_vision_inference(
    model_path: str, image_bytes: bytes, prompt: str | None
) -> str:
    from PIL import Image as PIL_Image

    image = PIL_Image.open(io.BytesIO(image_bytes)).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or _DEFAULT_PROMPT},
                {"type": "image"},
            ],
        }
    ]

    with get_manager().lease(_register(model_path)) as loaded:
        model, processor = loaded["model"], loaded["processor"]

        input_prompt = processor.apply_chat_template(conversation, return_tensors="pt")

        inputs = processor(
            text=input_prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device)

        prompt_len = len(inputs["input_ids"][0])

        output = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=0,
            do_sample=False,
        )

        generated_tokens = output[:, prompt_len:]
        text = processor.decode(generated_tokens[0], skip_special_tokens=True)
        # Drop GPU tensors before the lease ends so an unload can free them.
        del inputs, output, generated_tokens, model, processor, loaded
    return text
