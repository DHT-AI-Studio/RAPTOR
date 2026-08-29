# src/inference/handlers/vibevoice_tts.py
"""
VibeVoice TTS handler（實驗性 — 尚未在本環境實測）

VibeVoice（microsoft/VibeVoice）不是標準 transformers TTS 介面：
    - 模型類：vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference
    - 處理器：vibevoice.processor.vibevoice_processor.VibeVoiceProcessor
    - 輸入：對話腳本（"Speaker 0: ..."）+ 每個 speaker 的參考語音 wav（voice cloning）
    - 輸出：outputs.speech_outputs[0]（24kHz waveform tensor）

前置：
    1. 07 映像檔安裝 vibevoice 套件（官方 GitHub repo 已下架，用社群 fork）：
       pip install git+https://github.com/vibevoice-community/VibeVoice.git
    2. 模型權重上傳 lakeFS 後註冊（見 scripts/09_tts_vibevoice.sh）：
       task            = "tts"
       model_class     = "vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference"
       processor_class = "vibevoice.processor.vibevoice_processor.VibeVoiceProcessor"
       torch_dtype     = "bf16"
       custom_handler  = "vibevoice"
    3. 參考語音：options.voice（wav 路徑）或環境變數 VIBEVOICE_DEFAULT_VOICE

data:
    text: str（純文字；或已含 "Speaker N:" 前綴的多人腳本）
options:
    voice: 參考語音 wav 路徑（單人）；cfg_scale（預設 1.3）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from .base import BaseHandler
from .tts import _wav_result

logger = logging.getLogger(__name__)

VIBEVOICE_SAMPLE_RATE = 24000


class VibeVoiceTTSHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        text = data.get("text", data.get("inputs"))
        if not isinstance(text, str) or not text:
            raise ValueError("vibevoice handler requires data['text'] (str)")

        model = loaded["model"]
        processor = loaded["processor"]
        if processor is None:
            raise ValueError(
                "vibevoice handler needs processor_class="
                "'vibevoice.processor.vibevoice_processor.VibeVoiceProcessor' at registration"
            )

        # 純文字 → 單人腳本；已含 "Speaker N:" 的多人腳本原樣使用
        script = text if text.lstrip().lower().startswith("speaker") else f"Speaker 0: {text}"

        voice = options.get("voice") or data.get("voice") or os.getenv("VIBEVOICE_DEFAULT_VOICE")
        if not voice or not os.path.isfile(str(voice)):
            raise ValueError(
                "VibeVoice needs a reference voice wav for voice cloning. "
                "Pass options.voice=<wav path> or set env VIBEVOICE_DEFAULT_VOICE."
            )
        n_speakers = len({ln.split(":", 1)[0].strip() for ln in script.splitlines()
                          if ":" in ln and ln.lstrip().lower().startswith("speaker")}) or 1
        voice_samples = [[str(voice)] * n_speakers]

        inputs = processor(text=[script], voice_samples=voice_samples, padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=float(options.get("cfg_scale", 1.3)),
                max_new_tokens=None,
            )

        speech = outputs.speech_outputs[0]
        if hasattr(speech, "float"):
            speech = speech.float().cpu().numpy()
        return _wav_result(speech, VIBEVOICE_SAMPLE_RATE)
