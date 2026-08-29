# src/inference/handlers/base.py
"""
Handler 基類

Handler 的職責：給定 (loaded model+processor, spec, data, options)，回傳推理結果。
和舊設計不同的是，handler 現在「擁有」整個推理流程（encode→generate→decode），
不再只是 preprocess/postprocess 兩個 hook，因為 raw model 推理時 encoder 與 decoder
配對非常緊耦合，硬拆只會讓 adapter 又長出特例分支。

Pipeline route 仍保留 preprocess/postprocess hooks（成本低、共用容易）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseHandler(ABC):
    """子類至少要實作 run()。preprocess/postprocess 對 pipeline route 才用得到。"""

    @abstractmethod
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """單一推理：encode → generate → decode → 統一格式 dict。"""

    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """pipeline route 的可選 hook；預設不動。"""
        return data

    def postprocess(self, raw: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """pipeline route 的可選 hook；預設不動。"""
        return raw
