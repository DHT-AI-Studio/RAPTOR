"""Local inference API for freshly fine-tuned models.

Serves text-generation directly from a model directory produced by the
training service (HF `save_pretrained` format at `final_model_path`), so a
benchmark run can score a just-trained model without exporting it to another
service.

Design notes:
- torch/transformers are imported lazily inside the handler (not at module
  load) to keep service startup cheap and avoid any import-order coupling.
- A tiny 1-slot model cache avoids reloading weights on every request; loading
  a new model evicts the previous one to bound GPU memory (single small GPU).
- The endpoint is a sync `def` so FastAPI runs it in a worker thread and the
  blocking load/generate does not stall the event loop.
"""

import logging
import os
import threading
import time
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Local Inference"])

# ── 1-slot model cache (path -> (model, tokenizer)) ──────────────────
_cache_lock = threading.Lock()
_loaded_path: Optional[str] = None
_loaded_model = None
_loaded_tokenizer = None


class InferRequest(BaseModel):
    model_path: str = Field(..., description="Path to a HF-format model dir (e.g. training final_model_path)")
    inputs: str = Field(..., description="Prompt text")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    model_config = {
        "protected_namespaces": (),  # allow the `model_path` field name
        "json_schema_extra": {
            "example": {
                "model_path": "/app/tmp/models/gemma_squad_finetune_ab12",
                "inputs": "What is the capital of France?",
                "max_new_tokens": 128,
                "temperature": 0.2,
            }
        }
    }


class InferResponse(BaseModel):
    output: str
    model_path: str
    latency_ms: float


def _evict() -> Optional[str]:
    """Drop the cached model and release its VRAM. Returns the path evicted, if any.

    Caller must hold ``_cache_lock``.
    """
    global _loaded_path, _loaded_model, _loaded_tokenizer
    if _loaded_model is None:
        return None
    path = _loaded_path
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_path = None
    try:
        import gc

        import torch

        # HF models hold reference cycles (config/hooks) — a gc pass is needed
        # before empty_cache() actually returns the weight tensors to the driver.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass
    logger.info("Unloaded model %s (VRAM released)", path)
    return path


def _load(model_path: str):
    """Return (model, tokenizer) for model_path, using the 1-slot cache."""
    global _loaded_path, _loaded_model, _loaded_tokenizer

    if _loaded_path == model_path and _loaded_model is not None:
        return _loaded_model, _loaded_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Evict the previous model to bound GPU memory before loading a new one.
    _evict()

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    logger.info("Loading model from %s (cuda=%s)", model_path, use_cuda)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda" if use_cuda else None,
    )
    model.eval()

    _loaded_path, _loaded_model, _loaded_tokenizer = model_path, model, tokenizer
    return model, tokenizer


@router.post("/infer", response_model=InferResponse, summary="Generate text from a local fine-tuned model")
def infer(req: InferRequest) -> InferResponse:
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"model_path not found: {req.model_path}")

    start = time.perf_counter()
    try:
        import torch

        with _cache_lock:
            model, tokenizer = _load(req.model_path)

            # Instruct models (e.g. *-it) carry a chat template — use it so the
            # prompt is framed as a user turn (mirrors test_trainer/compare_models).
            if getattr(tokenizer, "chat_template", None):
                enc = dict(tokenizer.apply_chat_template(
                    [{"role": "user", "content": req.inputs}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ))
            else:
                enc = dict(tokenizer(req.inputs, return_tensors="pt"))
            enc.setdefault("attention_mask", torch.ones_like(enc["input_ids"]))
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}

            gen_kwargs = {
                "max_new_tokens": req.max_new_tokens,
                "do_sample": req.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if req.temperature > 0:
                gen_kwargs["temperature"] = req.temperature

            with torch.no_grad():
                out_ids = model.generate(**enc, **gen_kwargs)

            # Decode only the newly generated tokens.
            new_tokens = out_ids[0][enc["input_ids"].shape[1]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed for %s", req.model_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"inference failed: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000.0
    return InferResponse(output=output, model_path=req.model_path, latency_ms=latency_ms)


# ── Batched generation (new endpoint; does NOT touch /infer) ─────────
class InferBatchRequest(BaseModel):
    model_path: str = Field(..., description="Path to a HF-format model dir")
    inputs: List[str] = Field(..., min_length=1, description="Prompts to generate for, in order")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    batch_size: int = Field(16, ge=1, le=128, description="Internal sub-batch size (bounds VRAM)")

    model_config = {"protected_namespaces": ()}


class InferBatchResponse(BaseModel):
    outputs: List[str]
    model_path: str
    latency_ms: float
    count: int


def _encode_prompt(tokenizer, text: str):
    """Tokenize one prompt exactly like /infer (chat template if present) → 1D input_ids."""
    if getattr(tokenizer, "chat_template", None):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        return enc["input_ids"][0]
    return tokenizer(text, return_tensors="pt")["input_ids"][0]


@router.post("/infer_batch", response_model=InferBatchResponse,
             summary="Batched text-generation for many prompts in one call")
def infer_batch(req: InferBatchRequest) -> InferBatchResponse:
    """Generate for a list of prompts with one model load + batched forward passes.

    Semantically identical to calling /infer per prompt (same chat template, same
    per-row prompt stripping) but with far fewer round-trips and GPU calls. Uses
    LEFT padding — required for correct decoder-only batched generation — and runs
    in sub-batches of ``batch_size`` to bound VRAM. Independent of /infer.
    """
    if not os.path.isdir(req.model_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"model_path not found: {req.model_path}")

    start = time.perf_counter()
    try:
        import torch

        with _cache_lock:
            model, tokenizer = _load(req.model_path)
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id

            outputs: List[str] = []
            for lo in range(0, len(req.inputs), req.batch_size):
                seqs = [_encode_prompt(tokenizer, t) for t in req.inputs[lo:lo + req.batch_size]]
                maxlen = max(s.size(0) for s in seqs)
                n = len(seqs)
                input_ids = torch.full((n, maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((n, maxlen), dtype=torch.long)
                for i, s in enumerate(seqs):
                    input_ids[i, maxlen - s.size(0):] = s   # LEFT pad
                    attn[i, maxlen - s.size(0):] = 1
                if torch.cuda.is_available():
                    input_ids, attn = input_ids.to("cuda"), attn.to("cuda")

                gen_kwargs = {
                    "max_new_tokens": req.max_new_tokens,
                    "do_sample": req.temperature > 0,
                    "pad_token_id": pad_id,
                }
                if req.temperature > 0:
                    gen_kwargs["temperature"] = req.temperature

                with torch.no_grad():
                    out_ids = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
                # Left padding → every row's prompt occupies the first `maxlen` columns,
                # so the newly generated tokens are the same slice for all rows.
                new = out_ids[:, maxlen:]
                outputs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch inference failed for %s", req.model_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"batch inference failed: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000.0
    return InferBatchResponse(outputs=outputs, model_path=req.model_path,
                              latency_ms=latency_ms, count=len(outputs))


class UnloadRequest(BaseModel):
    model_path: Optional[str] = Field(
        None, description="Only unload if this exact model is loaded; omit to unload whatever is resident"
    )
    model_config = {"protected_namespaces": ()}


@router.post("/unload", summary="Unload the resident model and free GPU memory")
def unload(req: Optional[UnloadRequest] = Body(None)) -> dict:
    """Evict the 1-slot model cache to release VRAM.

    Benchmark runs call this after scoring so the model does not sit resident on
    the GPU between iterations (the loop's next training job needs that VRAM).
    Safe to call when nothing is loaded (no-op).
    """
    want = req.model_path if req else None
    with _cache_lock:
        if want and _loaded_path != want:
            # A different model (or none) is loaded — leave it alone.
            return {"unloaded": False, "model_path": _loaded_path, **_vram_stats()}
        path = _evict()
        return {"unloaded": path is not None, "model_path": path, **_vram_stats()}


def _vram_stats() -> dict:
    """torch's own VRAM accounting — the authoritative check that weights were
    freed. (nvidia-smi also counts the ~500MB CUDA context, which never releases
    while the process is alive, so `allocated` near 0 == weights are gone.)"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            }
    except Exception:  # noqa: BLE001
        pass
    return {}
