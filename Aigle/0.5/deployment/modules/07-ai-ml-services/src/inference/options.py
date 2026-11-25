# src/inference/options.py
"""
共用文字生成參數正規化（text-generation 專用）

InferenceService 在 dispatch 到 adapter 前呼叫 normalize_generation_options()，
讓 Ollama / HF raw / HF pipeline / 串流路徑看到同一份語意一致的 options：

    輸入                         正規化後
    max_tokens | max_new_tokens  → options["max_tokens"]（int > 0；兩者不同 → 400 conflicting_parameters）
    stop | stop_sequences        → options["stop"]（list[str]；空 list = 無額外停止條件）
    temperature                  → 0～2；0 = greedy（與明確 do_sample=true 衝突 → 400）
    max_length                   → 保留各 runtime 既有語意（HF = 總序列長度、Ollama = 生成上限），
                                   與 max_tokens / max_new_tokens 混用 → 400 conflicting_parameters

其他 options（runtime 專屬參數、keep_alive、think ...）原樣保留。值為 None 的共用欄位視同未提供。
非 text-generation task 不經過本模組，維持各自既有路徑。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .exceptions import ConflictingParametersError, InvalidParameterError

# 本次保證「實際套用或明確拒絕」的共用欄位（model-info 的 supported_options）
SHARED_TEXT_OPTIONS: tuple[str, ...] = (
    "max_tokens", "max_new_tokens", "stop", "stop_sequences", "temperature",
    "top_p", "top_k", "do_sample", "max_length", "think", "keep_alive",
)

TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0


def normalize_generation_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """回傳正規化後的新 dict（不修改輸入）。違規時拋 InvalidParameterError / ConflictingParametersError。"""
    opts = {k: v for k, v in (options or {}).items()}
    for k in ("max_tokens", "max_new_tokens", "max_length", "stop", "stop_sequences",
              "temperature", "do_sample"):
        if k in opts and opts[k] is None:
            del opts[k]

    # --- 生成長度 ---
    max_tokens = _positive_int(opts.pop("max_tokens", None), "max_tokens")
    max_new_tokens = _positive_int(opts.pop("max_new_tokens", None), "max_new_tokens")
    if max_tokens is not None and max_new_tokens is not None and max_tokens != max_new_tokens:
        raise ConflictingParametersError(
            f"max_tokens={max_tokens} and max_new_tokens={max_new_tokens} are synonyms but differ; send only max_tokens",
            parameters=["max_tokens", "max_new_tokens"],
        )
    if max_tokens is None:
        max_tokens = max_new_tokens
    if "max_length" in opts:
        opts["max_length"] = _positive_int(opts["max_length"], "max_length")
        if max_tokens is not None:
            raise ConflictingParametersError(
                "max_length keeps its legacy per-runtime meaning (HF: total sequence length) and cannot be "
                "combined with max_tokens / max_new_tokens; send only max_tokens",
                parameters=["max_tokens", "max_length"],
            )
    if max_tokens is not None:
        opts["max_tokens"] = max_tokens

    # --- 停止字串 ---
    stop = _stop_list(opts.pop("stop", None), "stop")
    stop_sequences = _stop_list(opts.pop("stop_sequences", None), "stop_sequences")
    if stop is not None and stop_sequences is not None and stop != stop_sequences:
        raise ConflictingParametersError(
            "stop and stop_sequences are synonyms but differ; send only one of them",
            parameters=["stop", "stop_sequences"],
        )
    merged = stop if stop is not None else stop_sequences
    if merged is not None:
        opts["stop"] = merged

    # --- temperature / greedy ---
    if "temperature" in opts:
        t = opts["temperature"]
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise InvalidParameterError(f"temperature must be a number, got {t!r}", parameter="temperature")
        if not (TEMPERATURE_MIN <= float(t) <= TEMPERATURE_MAX):
            raise InvalidParameterError(
                f"temperature must be within [{TEMPERATURE_MIN:g}, {TEMPERATURE_MAX:g}], got {t}",
                parameter="temperature",
            )
        opts["temperature"] = float(t)
    if "do_sample" in opts and not isinstance(opts["do_sample"], bool):
        raise InvalidParameterError(f"do_sample must be a boolean, got {opts['do_sample']!r}", parameter="do_sample")
    if opts.get("temperature") == 0.0 and opts.get("do_sample") is True:
        raise ConflictingParametersError(
            "temperature=0 means greedy decoding and contradicts do_sample=true",
            parameters=["temperature", "do_sample"],
        )
    return opts


def is_greedy(options: dict[str, Any]) -> bool:
    """正規化後的 options 是否要求 greedy decoding。"""
    return options.get("temperature") == 0.0 or options.get("do_sample") is False


def truncate_at_stop(text: str, stops: Iterable[str]) -> tuple[str, bool]:
    """在第一個出現的停止字串處截斷；回傳 (截斷後文字, 是否命中)。"""
    idx = _earliest(text, stops)
    if idx < 0:
        return text, False
    return text[:idx], True


class StopStringFilter:
    """串流用停止字串過濾器。

    feed() 回傳可安全送給 client 的文字：可能是停止字串前綴的尾端會先扣住，
    直到確定不是停止字串（或串流結束 flush）為止；命中後 stopped=True，
    之後的輸入全部丟棄。停止字串本身與其後文字都不會送出。
    """

    def __init__(self, stops: Iterable[str] | None):
        self.stops: list[str] = [s for s in (stops or []) if s]
        self.stopped = False
        self._buf = ""
        self._hold = max((len(s) for s in self.stops), default=1) - 1

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        if not self.stops:
            return text
        self._buf += text
        idx = _earliest(self._buf, self.stops)
        if idx >= 0:
            out, self._buf = self._buf[:idx], ""
            self.stopped = True
            return out
        keep = self._partial_suffix_len(self._buf)
        cut = len(self._buf) - keep
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return out

    def flush(self) -> str:
        out, self._buf = ("" if self.stopped else self._buf), ""
        return out

    def _partial_suffix_len(self, buf: str) -> int:
        for k in range(min(len(buf), self._hold), 0, -1):
            tail = buf[-k:]
            if any(s.startswith(tail) for s in self.stops):
                return k
        return 0


def usage_from_metadata(meta: dict[str, Any] | None) -> dict[str, int] | None:
    """由 adapter metadata 的 prompt_tokens / completion_tokens 組出 usage。

    兩者皆為真實計數（int）時才回傳；缺任何一個回 None，不以 0 冒充。
    """
    if not isinstance(meta, dict):
        return None
    p, c = meta.get("prompt_tokens"), meta.get("completion_tokens")
    if not _is_count(p) or not _is_count(c):
        return None
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


# ===== 內部 helpers =====


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _positive_int(v: Any, name: str) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise InvalidParameterError(f"{name} must be a positive integer, got {v!r}", parameter=name)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if not isinstance(v, int) or v <= 0:
        raise InvalidParameterError(f"{name} must be a positive integer, got {v!r}", parameter=name)
    return v


def _stop_list(v: Any, name: str) -> list[str] | None:
    if v is None:
        return None
    items = [v] if isinstance(v, str) else v
    if not isinstance(items, (list, tuple)) or not all(isinstance(s, str) for s in items):
        raise InvalidParameterError(f"{name} must be a string or a list of strings", parameter=name)
    if any(s == "" for s in items):
        raise InvalidParameterError(f"{name} must not contain empty strings", parameter=name)
    out: list[str] = []
    for s in items:
        if s not in out:
            out.append(s)
    return out


def _earliest(text: str, stops: Iterable[str]) -> int:
    best = -1
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best
