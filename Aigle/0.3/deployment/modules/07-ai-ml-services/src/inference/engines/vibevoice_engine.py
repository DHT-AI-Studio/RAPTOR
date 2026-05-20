# src/inference/engines/vibevoice_engine.py
"""
VibeVoice engine — wraps microsoft/VibeVoice-ASR (ASR task)
and microsoft/VibeVoice-1.5B (TTS task) as a BaseEngine subclass.
"""

import base64
import io
import logging
import os
from typing import Any, Dict, Optional
from .base import BaseEngine

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline as hf_pipeline,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers not installed — VibeVoice engine unavailable")

ASR_MODEL_ID = os.getenv("VIBEVOICE_ASR_MODEL", "microsoft/VibeVoice-ASR")
TTS_MODEL_ID = os.getenv("VIBEVOICE_TTS_MODEL", "microsoft/VibeVoice-1.5B")


class VibeVoiceEngine(BaseEngine):
    """
    Engine for VibeVoice ASR and TTS models.

    Supported tasks:
    - "asr"  — automatic speech recognition via microsoft/VibeVoice-ASR
    - "tts"  — text-to-speech synthesis via microsoft/VibeVoice-1.5B

    Input data for "asr":  {"audio_path": "/path/to/file.wav"}
                        or {"audio_bytes": "<base64-encoded WAV>"}
    Input data for "tts":  {"text": "...", "voice": "default", "speed": 1.0}
    """

    SUPPORTED_TASKS = ["asr", "tts"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required for VibeVoiceEngine")

        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._loaded_models: Dict[str, Any] = {}
        self.is_initialized = True
        self.version = "0.3.0"
        logger.info(f"VibeVoiceEngine initialised on {self.device}")

    # ── BaseEngine interface ────────────────────────────────────────────────

    def load_model(self, model_name: str, **kwargs) -> Any:
        task = kwargs.get("task", "asr")
        cache_key = f"{model_name}:{task}"

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        if task == "asr":
            model = self._load_asr(model_name)
        elif task == "tts":
            model = self._load_tts(model_name)
        else:
            raise ValueError(f"VibeVoiceEngine does not support task={task}")

        self._loaded_models[cache_key] = model
        return model

    def infer(self, model: Any, inputs: Dict[str, Any], options: Dict[str, Any]) -> Any:
        task = options.get("task", "asr")
        if task == "asr":
            return self._infer_asr(model, inputs)
        elif task == "tts":
            return self._infer_tts(model, inputs, options)
        else:
            raise ValueError(f"VibeVoiceEngine does not support task={task}")

    def unload_model(self, model: Any) -> bool:
        for key, m in list(self._loaded_models.items()):
            if m is model:
                del self._loaded_models[key]
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
        return False

    def get_info(self) -> Dict[str, Any]:
        base = super().get_info()
        base.update({
            "device": self.device,
            "supported_tasks": self.SUPPORTED_TASKS,
            "loaded_models": list(self._loaded_models.keys()),
        })
        return base

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_asr(self, model_id: str) -> Any:
        logger.info(f"Loading VibeVoice ASR model: {model_id}")
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=self.torch_dtype,
            device=0 if self.device == "cuda" else -1,
        )
        return {"pipeline": pipe, "task": "asr"}

    def _load_tts(self, model_id: str) -> Any:
        logger.info(f"Loading VibeVoice TTS model: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype
        )
        if self.device == "cuda":
            model = model.to(self.device)
        return {"model": model, "processor": processor, "task": "tts"}

    def _infer_asr(self, model_dict: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        pipe = model_dict["pipeline"]
        if "audio_bytes" in inputs:
            audio_data = base64.b64decode(inputs["audio_bytes"])
            import soundfile as sf
            audio_array, sr = sf.read(io.BytesIO(audio_data))
            result = pipe({"array": audio_array, "sampling_rate": sr})
        else:
            audio_path = inputs.get("audio_path", "")
            result = pipe(audio_path)
        return {"text": result.get("text", ""), "metadata": {}}

    def _infer_tts(
        self, model_dict: Dict[str, Any], inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        import numpy as np
        import soundfile as sf

        model     = model_dict["model"]
        processor = model_dict["processor"]
        text      = inputs.get("text", "")
        speed     = float(inputs.get("speed", options.get("speed", 1.0)))

        processed = processor(text=text, return_tensors="pt")
        if self.device == "cuda":
            processed = {k: v.to(self.device) for k, v in processed.items()}

        with torch.no_grad():
            speech = model.generate_speech(
                processed["input_ids"],
                speaker_embeddings=None,
                vocoder=None,
            )

        audio_np = speech.cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, samplerate=16000, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        duration  = len(audio_np) / 16000.0

        return {
            "audio_base64": audio_b64,
            "duration_seconds": duration,
            "format": "wav",
        }
