# src/inference/adapters/__init__.py
"""Adapter registry — runtime → adapter instance（單例）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..exceptions import ValidationError
from ..spec import RUNTIME_HF, RUNTIME_OLLAMA
from .base import BaseAdapter

logger = logging.getLogger(__name__)


_ADAPTERS: Dict[str, BaseAdapter] = {}


def get_adapter(runtime: str, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> BaseAdapter:
    """取得 runtime 對應的 adapter 單例（首次呼叫時 lazy-init）。

    configs: 整份引擎配置 dict（key=runtime name），通常從 inference.yaml.engines 來。
    """
    if runtime in _ADAPTERS:
        return _ADAPTERS[runtime]

    cfg = (configs or {}).get(runtime) or (configs or {}).get(_legacy_engine_alias(runtime)) or {}

    if runtime == RUNTIME_OLLAMA:
        from .ollama import OllamaAdapter
        adapter: BaseAdapter = OllamaAdapter(config=cfg)
    elif runtime == RUNTIME_HF:
        from .hf_transformers import HFTransformersAdapter
        adapter = HFTransformersAdapter(config=cfg)
    else:
        raise ValidationError(f"no adapter registered for runtime '{runtime}'")

    _ADAPTERS[runtime] = adapter
    return adapter


def all_adapters() -> Dict[str, BaseAdapter]:
    return dict(_ADAPTERS)


def _legacy_engine_alias(runtime: str) -> str:
    """允許 inference.yaml 仍用舊名 (transformers / ollama) 設定。"""
    if runtime == RUNTIME_HF:
        return "transformers"
    return runtime


__all__ = ["BaseAdapter", "get_adapter", "all_adapters"]
