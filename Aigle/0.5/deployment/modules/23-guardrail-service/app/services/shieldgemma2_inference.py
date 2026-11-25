"""Google ShieldGemma 2 (google/shieldgemma-2-4b-it) inference.

Classifies an uploaded image against the model's 3 built-in safety
policies (dangerous content, sexually explicit content, violence/gore).

Loaded models are owned by the shared resource manager
(app/services/model_resource_manager.py), separate from the Ollama-based
adapters used by the text guard pipeline.

Memory note: 4B parameters in float32 (the transformers default) ≈ 16 GB.
This module loads in 8-bit (≈ 4.8 GB, default) or bfloat16 (≈ 8.2 GB).
Set SHIELDGEMMA2_LOAD_IN_8BIT=false to use bfloat16 instead.
"""
from __future__ import annotations

import io
import logging
import os

from app.services.model_resource_manager import ModelSpec, env_mb, get_manager

logger = logging.getLogger(__name__)

# Pinned to a specific upstream commit so a service restart never silently
# pulls a different set of weights.  Update intentionally when upgrading.
_MODEL_REVISION = "eaf60452b5fc41a911338a022e628b0c15283897"

_MANAGER_KEY_PREFIX = "shieldgemma-2-4b-it"


def _load_in_8bit() -> bool:
    return os.environ.get("SHIELDGEMMA2_LOAD_IN_8BIT", "true").lower() == "true"


def _fix_shieldgemma2_lm_head(model) -> None:
    """Tie the lm_head back to the input embedding.

    ShieldGemma2ForImageClassification.get_output_embeddings() delegates to
    self.model.language_model (the bare text trunk, which has no lm_head of
    its own). This causes transformers' weight tying during from_pretrained()
    to find nothing to tie, leaving the real lm_head at random initialization
    ("newly initialized: ['model.lm_head.weight']" in the load log). Gemma
    models use tied embeddings, so the checkpoint never stores a separate
    lm_head weight — tie it back here.

    Required in every mode: ShieldGemma2's forward reads its Yes/No
    probabilities from the lm_head logits, so an untied head yields random
    verdicts.  bitsandbytes 8-bit leaves lm_head unquantized by default, so the
    plain assignment works there too.
    """
    input_embeddings = model.get_input_embeddings().weight
    for name, module in model.named_modules():
        if (
            name.endswith("lm_head")
            and getattr(module, "weight", None) is not None
            and module.weight.shape == input_embeddings.shape
        ):
            if type(module.weight).__name__ != "Parameter" or module.weight.dtype != input_embeddings.dtype:
                raise RuntimeError(
                    f"ShieldGemma2 {name} is {type(module.weight).__name__}/{module.weight.dtype}; "
                    "cannot tie it to the input embeddings"
                )
            module.weight = input_embeddings
            logger.info("shieldgemma2: tied %s.weight to input embeddings", name)
            return
    raise RuntimeError("could not find a matching lm_head to tie for ShieldGemma2")


def _load_model(model_path: str, device_index: int) -> dict:
    import torch
    from transformers import AutoProcessor, ShieldGemma2ForImageClassification

    load_in_8bit = _load_in_8bit()

    # The whole model is pinned to the managed GPU.  device_map="auto" (and the
    # old hard-coded inputs.to("cuda")) let weights end up on CPU while inputs
    # were on cuda:0, which fails in the Gemma3 token embedding with
    # "Expected all tensors to be on the same device ... wrapper_CUDA__index_select".
    # Non-quantized modules run in bfloat16: in float16 (the bitsandbytes
    # default) Gemma3 produced nonsense scores (P(sexual)=0.91 for a blank image).
    kwargs = {"revision": _MODEL_REVISION, "device_map": {"": device_index}, "dtype": torch.bfloat16}
    if load_in_8bit:
        # 8-bit via bitsandbytes: ≈ 4.8 GB vs ≈ 8.2 GB for bfloat16.
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = ShieldGemma2ForImageClassification.from_pretrained(model_path, **kwargs).eval()  # nosec B615 -- revision is in kwargs above; bandit can't see through **kwargs
    _fix_shieldgemma2_lm_head(model)

    processor = AutoProcessor.from_pretrained(model_path, revision=_MODEL_REVISION)
    logger.info(
        "shieldgemma2: loaded model '%s' on %s (8bit=%s)", model_path, model.device, load_in_8bit
    )
    return {"model": model, "processor": processor}


def _register(model_path: str) -> str:
    load_in_8bit = _load_in_8bit()
    key = f"{_MANAGER_KEY_PREFIX}:{model_path}:{'8bit' if load_in_8bit else 'bf16'}"
    # Defaults measured on an RTX 4060 Ti (8-bit: 4.8 GB allocated, ~6.2 GB
    # reserved while loading; bf16: 8.2 GB).  Override per deployment if needed.
    get_manager().register(ModelSpec(
        key=key,
        loader=lambda device_index: _load_model(model_path, device_index),
        vram_bytes=env_mb("SHIELDGEMMA2_VRAM_ESTIMATE_MB", 4900 if load_in_8bit else 8300),
        inference_reserve_bytes=env_mb("SHIELDGEMMA2_INFERENCE_RESERVE_MB", 1536),
        ram_bytes=env_mb("SHIELDGEMMA2_RAM_ESTIMATE_MB", 4096),
    ))
    return key


def run_shieldgemma2_inference(model_path: str, image_bytes: bytes) -> dict:
    import torch
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with get_manager().lease(_register(model_path)) as loaded:
        model, processor = loaded["model"], loaded["processor"]
        model_inputs = processor(images=[image], return_tensors="pt").to(model.device)
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(model.dtype)

        with torch.inference_mode():
            scores = model(**model_inputs)
        probabilities = scores.probabilities.float().cpu()
        policy_definitions = dict(processor.policy_definitions)
        # Drop GPU tensors before the lease ends so an unload can free them.
        del model_inputs, scores, model, processor, loaded

    # scores.probabilities shape: (num_policies, 2)
    #   col 0 = P(Yes) = unsafe probability
    #   col 1 = P(No)  = safe probability
    # Row order = processor.policy_definitions key order (3 built-in policies)
    policies = list(policy_definitions.keys())

    policy_results: dict = {}
    overall_unsafe = False
    for policy_key, row in zip(policies, probabilities):
        unsafe_prob, safe_prob = row.tolist()
        verdict = "unsafe" if unsafe_prob > safe_prob else "safe"
        overall_unsafe = overall_unsafe or verdict == "unsafe"
        policy_results[policy_key] = {
            "unsafe_probability": unsafe_prob,
            "safe_probability": safe_prob,
            "verdict": verdict,
            "policy_definition": policy_definitions[policy_key],
        }

    return {
        "overall_verdict": "unsafe" if overall_unsafe else "safe",
        "policies": policy_results,
    }
