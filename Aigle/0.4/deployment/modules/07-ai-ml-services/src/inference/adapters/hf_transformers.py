# src/inference/adapters/hf_transformers.py
"""
HuggingFace Transformers Runtime Adapter — spec-driven dispatch

關鍵設計：不再用字串匹配模型名稱來選 model class。
spec.model_class / spec.processor_class 直接告訴我們要用哪個 transformers class，
讓註冊端負責「該模型怎麼跑」的決策，推論端只負責執行。

對於需要客製 preprocess 或 generate 邏輯的模型（如 Qwen-VL），
透過 spec.custom_handler 載入該模型專屬的 handler，否則走預設 task-family handler。
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

from ..exceptions import EngineError, ModelLoadError, UnsupportedTaskError, ValidationError
from ..handlers import resolve_handler
from ..spec import ModelSpec, RUNTIME_HF
from .base import BaseAdapter

logger = logging.getLogger(__name__)


# 延遲 import，避免測試環境沒裝 torch/transformers 時整個模組炸掉
def _lazy_imports():
    import torch
    import transformers
    return torch, transformers


class HFTransformersAdapter(BaseAdapter):
    runtime = RUNTIME_HF

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, max_cached_models=int((config or {}).get("max_cached_models", 2)))
        self.default_device = (self.config or {}).get("device", "auto")
        self.default_dtype = (self.config or {}).get("torch_dtype", "auto")
        self.trust_remote_code_default = bool((self.config or {}).get("trust_remote_code", True))
        logger.info(
            f"HFTransformersAdapter ready — device={self.default_device} "
            f"dtype={self.default_dtype} cache={self._max_cached}"
        )

    # ===== load_model: spec-driven =====

    def load_model(self, spec: ModelSpec) -> Dict[str, Any]:
        torch, tx = _lazy_imports()

        if spec.runtime != RUNTIME_HF:
            raise ValidationError(f"HFTransformersAdapter received non-HF spec: {spec.runtime}")

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, spec.torch_dtype or self.default_dtype, device)
        trust_remote = spec.trust_remote_code if spec.trust_remote_code is not None else self.trust_remote_code_default

        # ---- 路線 A：spec 指定了 pipeline_task → 直接走 HF pipeline shortcut ----
        if spec.pipeline_task:
            logger.info(f"loading via pipeline: task={spec.pipeline_task} path={spec.physical_path}")
            try:
                pipe = tx.pipeline(
                    spec.pipeline_task,
                    model=spec.physical_path,
                    device=0 if device == "cuda" else -1,
                    dtype=dtype,
                    trust_remote_code=trust_remote,
                )
            except Exception as e:
                raise ModelLoadError(f"pipeline('{spec.pipeline_task}') failed for {spec.physical_path}: {e}") from e
            return {"kind": "pipeline", "pipeline": pipe, "spec": spec}

        # ---- 路線 B：spec 指定了 model_class / processor_class → 反射加載 ----
        if not spec.model_class:
            raise ValidationError(
                f"spec for '{spec.model_name}' is missing both 'pipeline_task' and 'model_class'; "
                f"cannot load HF model"
            )

        ModelCls = self._resolve_class(tx, spec.model_class)
        ProcCls = self._resolve_class(tx, spec.processor_class) if spec.processor_class else None

        model_kwargs = self._build_model_kwargs(torch, tx, spec, dtype, trust_remote)

        logger.info(f"loading {spec.model_class}.from_pretrained({spec.physical_path}) kwargs_keys={list(model_kwargs)}")
        try:
            model = ModelCls.from_pretrained(spec.physical_path, **model_kwargs)
        except Exception as e:
            raise ModelLoadError(f"{spec.model_class}.from_pretrained failed for {spec.physical_path}: {e}") from e

        # 若沒有 device_map，手動移至 GPU
        if "device_map" not in model_kwargs and device == "cuda":
            try:
                model = model.to(device)
            except Exception as e:
                logger.warning(f"model.to('cuda') failed (likely already on device): {e}")

        processor = None
        if ProcCls is not None:
            try:
                processor = ProcCls.from_pretrained(spec.physical_path, trust_remote_code=trust_remote)
            except Exception as e:
                raise ModelLoadError(f"{spec.processor_class}.from_pretrained failed: {e}") from e

        return {
            "kind": "model+processor",
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype,
            "spec": spec,
        }

    # ===== infer: dispatch to handler =====

    def infer(self, model: Any, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        handler = resolve_handler(spec)

        # pipeline 路線
        if isinstance(model, dict) and model.get("kind") == "pipeline":
            inputs = handler.preprocess(data, options) if handler else data
            try:
                raw = self._run_pipeline(model["pipeline"], spec, inputs, options)
            except Exception as e:
                raise EngineError(f"pipeline inference failed: {e}") from e
            return handler.postprocess(raw, options) if handler else raw

        # model+processor 路線：完全交給 handler 處理
        if isinstance(model, dict) and model.get("kind") == "model+processor":
            if handler is None:
                raise ValidationError(
                    f"task '{spec.task_family}' on raw model+processor needs a handler, "
                    f"but none registered. Either set spec.pipeline_task or register a handler."
                )
            return handler.run(model, spec, data, options)

        raise EngineError(f"unknown loaded-model shape: {type(model)}")

    # ===== 串流（text-generation 專用）=====

    def infer_stream(self, model: Any, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]):
        """逐 token 串流：TextIteratorStreamer + 背景執行緒跑 generate。

        僅支援 text-generation（其他 task 拋 UnsupportedTaskError，呼叫端 fallback
        到非串流路徑）。pipeline 與 model+processor 兩種載入形態都支援。
        """
        if spec.task_family != "text-generation":
            raise UnsupportedTaskError(
                f"hf-transformers streaming only supports text-generation, got '{spec.task_family}'"
            )
        torch, tx = _lazy_imports()
        import threading

        if isinstance(model, dict) and model.get("kind") == "pipeline":
            mdl = model["pipeline"].model
            tok = model["pipeline"].tokenizer
        elif isinstance(model, dict) and model.get("kind") == "model+processor":
            mdl = model["model"]
            tok = model["processor"]
            if tok is None:
                raise ValidationError("streaming needs a tokenizer; set processor_class at registration")
        else:
            raise EngineError(f"unknown loaded-model shape: {type(model)}")

        device = next(mdl.parameters()).device
        if "messages" in data:
            input_ids = tok.apply_chat_template(
                data["messages"], add_generation_prompt=True, return_tensors="pt"
            )
            inputs = {"input_ids": input_ids.to(device)}
        elif "inputs" in data:
            enc = tok(data["inputs"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in enc.items()}
        else:
            raise ValidationError("streaming requires data['inputs'] or data['messages']")

        streamer = tx.TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = _gen_kwargs(options)
        errors: list = []

        def _worker():
            try:
                with torch.no_grad():
                    mdl.generate(**inputs, streamer=streamer, **gen_kwargs)
            except Exception as e:  # noqa: BLE001 — 錯誤帶回主執行緒統一拋出
                errors.append(e)
                streamer.end()  # 讓 iterator 停止，避免 gen() 卡死

        thread = threading.Thread(target=_worker, daemon=True)

        def gen():
            thread.start()
            completion_tokens = 0
            try:
                for piece in streamer:
                    if piece:
                        completion_tokens += 1  # 粗估：一段 ≈ 一個 decode step
                        yield piece
            finally:
                thread.join(timeout=5)
            if errors:
                raise EngineError(f"streaming generation failed: {errors[0]}") from errors[0]
            yield {"metadata": {
                "prompt_eval_count": int(inputs["input_ids"].shape[-1]),
                "eval_count": completion_tokens,
            }}

        return gen()

    # ===== unload =====

    def _on_unload(self, entry: Any) -> None:
        """卸載一個被 LRU 趕出/手動 unload 的模型 entry。

        不主動 pop entry 的欄位（model / processor / pipeline）— 另一個 thread
        可能仍持有 entry 引用並正在跑 inference。讓 entry 自然被 GC 後，
        torch.cuda.empty_cache 才會真正釋放 VRAM；若還有人在用，記憶體會延後到
        所有引用消失後才被釋放，這是預期行為。
        """
        del entry  # explicit: drop our own ref before GC
        try:
            torch, _ = _lazy_imports()
        except Exception:
            return
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ===== helpers =====

    @staticmethod
    def _resolve_class(tx_module, class_name: str):
        """先查 transformers 主命名空間；查不到才嘗試 dotted path。"""
        if hasattr(tx_module, class_name):
            return getattr(tx_module, class_name)
        if "." in class_name:
            module_path, attr = class_name.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_path)
                return getattr(mod, attr)
            except Exception as e:
                raise ValidationError(f"cannot import class '{class_name}': {e}") from e
        raise ValidationError(
            f"class '{class_name}' not found in transformers; "
            f"use a fully-qualified dotted path if it lives elsewhere"
        )

    def _resolve_device(self, torch) -> str:
        if self.default_device != "auto":
            return self.default_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(torch, dtype_str: Optional[str], device: str):
        if not dtype_str or dtype_str == "auto":
            return torch.float16 if device == "cuda" else torch.float32
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(dtype_str, torch.float16 if device == "cuda" else torch.float32)

    def _build_model_kwargs(self, torch, tx, spec: ModelSpec, dtype, trust_remote: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote,
            "low_cpu_mem_usage": True,
        }

        if spec.quantization not in {"4bit", "8bit"}:
            # 非量化路徑：dtype 走 spec / config
            kwargs["dtype"] = dtype
            return kwargs

        # 量化路徑：dtype 由 bnb_*_compute_dtype 承擔；不應再寫頂層 dtype
        BnB = getattr(tx, "BitsAndBytesConfig", None)
        if BnB is None:
            raise ModelLoadError(
                f"spec for '{spec.model_name}' requests quantization='{spec.quantization}' "
                f"but transformers.BitsAndBytesConfig is unavailable (bitsandbytes not installed). "
                f"Install bitsandbytes or remove the quantization tag."
            )

        # compute_dtype：spec 顯式指定 fp16/bf16 時優先採用；否則用 bf16
        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        if spec.quantization == "4bit":
            kwargs["quantization_config"] = BnB(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:  # 8bit
            kwargs["quantization_config"] = BnB(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
        kwargs["device_map"] = "auto"
        return kwargs

    @staticmethod
    def _run_pipeline(pipe, spec: ModelSpec, inputs: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        gen_kwargs = _gen_kwargs(options)
        # 不同 pipeline task 的輸入欄位
        if spec.pipeline_task == "text-generation":
            if isinstance(inputs, dict) and "messages" in inputs:
                # OpenAI 式 messages — 新版 transformers pipeline 原生支援 chat 格式
                out = pipe(inputs["messages"], **gen_kwargs)
                generated = out[0]["generated_text"]
                if isinstance(generated, list):  # chat 格式回傳完整對話，取最後一則 assistant 訊息
                    generated = generated[-1].get("content", "")
                return {"response": generated, "metadata": {}}
            text = inputs.get("inputs") if isinstance(inputs, dict) else inputs
            out = pipe(text, **gen_kwargs)
            return {"response": out[0]["generated_text"], "metadata": {}}
        if spec.pipeline_task == "automatic-speech-recognition":
            audio = inputs.get("audio") if isinstance(inputs, dict) else inputs
            out = pipe(audio)
            return {"text": out.get("text", ""), "metadata": {}}
        if spec.pipeline_task == "audio-classification":
            audio = inputs.get("audio") if isinstance(inputs, dict) else inputs
            out = pipe(audio)
            return {"classifications": out, "metadata": {}}
        if spec.pipeline_task == "image-to-text":
            image = inputs.get("image") if isinstance(inputs, dict) else inputs
            out = pipe(image)
            return {"text": out[0]["generated_text"], "metadata": {}}
        if spec.pipeline_task == "text-to-speech":
            text = inputs.get("text") if isinstance(inputs, dict) else inputs
            out = pipe(text)
            return out  # {"audio": np.ndarray, "sampling_rate": int} — TTSHandler.postprocess 編碼成 WAV
        # fallback
        out = pipe(inputs)
        return {"response": str(out), "metadata": {}}


def _gen_kwargs(options: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "max_length", "max_new_tokens", "temperature", "top_p", "top_k",
        "do_sample", "num_beams", "repetition_penalty", "length_penalty",
        "early_stopping",
    )
    out = {k: options[k] for k in keys if k in options}
    if "do_sample" not in out:
        out["do_sample"] = True
    if "max_new_tokens" not in out and "max_length" not in out:
        out["max_new_tokens"] = 512
    return out
