# src/inference/handlers/__init__.py
"""
Handler registry — task_family → handler 實例（單例）。

resolve_handler(spec) 順序：
    1) 若 spec.custom_handler 有值 → 先查內建名（_NAMED），再動態 import（dotted path）
    2) 否則查 _DEFAULT 表
    3) 找不到回 None（adapter 視情況決定要不要報錯）
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

from .base import BaseHandler
from .text_generation import TextGenerationHandler
from .vlm import VLMHandler
from .asr import ASRHandler
from .ocr import OCRHandler
from .audio_classification import AudioClassificationHandler
from .video_analysis import VideoAnalysisHandler
from .document_analysis import DocumentAnalysisHandler
from .internvl import InternVLHandler
from .tts import TTSHandler
from .embedding import EmbeddingHandler
from .rerank import RerankHandler

logger = logging.getLogger(__name__)


_DEFAULT: dict[str, BaseHandler] = {
    "text-generation": TextGenerationHandler(),
    "vlm": VLMHandler(),
    "image-captioning": VLMHandler(),
    "asr": ASRHandler(),
    "ocr": OCRHandler(),
    "audio-classification": AudioClassificationHandler(),
    "video-analysis": VideoAnalysisHandler(),
    "document-analysis": DocumentAnalysisHandler(),
    "tts": TTSHandler(),
    "embedding": EmbeddingHandler(),
    "rerank": RerankHandler(),
}

# 內建具名 handler — custom_handler tag 可直接填這些短名，不需 dotted path
# （lazy import，避免可選依賴（如 vibevoice 套件）拖垮整個 registry）
_NAMED: dict[str, BaseHandler] = {
    "internvl": InternVLHandler(),
}

_NAMED_LAZY: dict[str, str] = {
    "vibevoice": "src.inference.handlers.vibevoice_tts:VibeVoiceTTSHandler",
}


def resolve_handler(spec) -> Optional[BaseHandler]:
    if getattr(spec, "custom_handler", None):
        named = _NAMED.get(spec.custom_handler)
        if named is not None:
            return named
        lazy = _NAMED_LAZY.get(spec.custom_handler)
        if lazy is not None:
            inst = _load_custom(lazy)
            if inst is not None:
                _NAMED[spec.custom_handler] = inst  # 快取，之後直接命中
            return inst
        return _load_custom(spec.custom_handler)
    return _DEFAULT.get(spec.task_family)


def _load_custom(dotted: str) -> Optional[BaseHandler]:
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    elif "." in dotted:
        module_path, attr = dotted.rsplit(".", 1)
    else:
        logger.error(f"custom_handler must be 'pkg.mod:Class' or 'pkg.mod.Class', got: {dotted}")
        return None
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, attr)
    except Exception as e:
        logger.error(f"failed to load custom handler '{dotted}': {e}")
        return None
    inst = cls() if isinstance(cls, type) else cls
    if not isinstance(inst, BaseHandler):
        logger.error(f"custom handler '{dotted}' is not BaseHandler subclass")
        return None
    return inst


__all__ = [
    "BaseHandler",
    "TextGenerationHandler", "VLMHandler", "ASRHandler", "OCRHandler",
    "AudioClassificationHandler", "VideoAnalysisHandler", "DocumentAnalysisHandler",
    "InternVLHandler", "TTSHandler", "EmbeddingHandler", "RerankHandler",
    "resolve_handler",
]
