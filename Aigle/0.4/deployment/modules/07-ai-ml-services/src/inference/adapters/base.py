# src/inference/adapters/base.py
"""
Adapter 抽象基類

Adapter 取代了舊的 BaseEngine + ModelExecutor 兩層。
單一職責：給定 ModelSpec，把 raw inputs 變成推理結果。

生命週期：
    1. service 建立 adapter 單例（每個 runtime 一個）
    2. service.infer() → adapter.run(spec, data, options)
    3. adapter 內部：load_model(spec) → preprocess → infer → postprocess
    4. 模型實例由 adapter 用 OrderedDict 做 LRU 快取，並帶閒置逾時（ollama 式 keep_alive）

並發策略（_get_or_load）：
    - cache hit：在 _cache_lock 下 move_to_end 後直接回傳
    - cache miss：使用 per-key load lock 確保「同一模型同時間只有一個 loader」，
      其他 caller 等到 loader 完成後直接拿到快取結果，避免重複 from_pretrained 兩次

閒置卸載（idle eviction）：
    - 每個 cache entry 記錄 last_used / expires_at / in_flight
    - options["keep_alive"] 可逐請求覆寫存活時間（秒數或 "30s"/"5m"/"1h"；
      0 = 推理完立即卸載；負值 = 常駐不卸載；未填 = 用 adapter 的 idle_timeout）
    - InferenceService 的 reaper 執行緒定期呼叫 evict_idle() 回收過期模型並清 GPU cache
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..exceptions import UnsupportedTaskError, ValidationError
from ..spec import ModelSpec

logger = logging.getLogger(__name__)


def parse_keep_alive(value: Any) -> Optional[float]:
    """把 keep_alive 參數正規化為秒數。

    None → None（表示「呼叫端沒指定」，由 adapter 預設值決定）
    數字 → 秒；負值表示常駐
    字串 → "500ms" / "30s" / "5m" / "1h" / "300" / "-1"
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError(f"invalid keep_alive: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        if s.endswith("s"):
            return float(s[:-1])
        return float(s)
    except ValueError:
        raise ValidationError(
            f"invalid keep_alive '{value}'; use seconds or a duration like '30s'/'5m'/'1h' "
            f"(0 = unload immediately, negative = keep loaded forever)"
        ) from None


@dataclass
class CacheEntry:
    """一個已載入模型在 adapter 快取中的完整狀態。"""

    model: Any
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # None = 不會被 idle reaper 回收
    use_count: int = 0
    in_flight: int = 0  # 正在使用此模型的推理請求數；>0 時 reaper 不回收


class BaseAdapter(ABC):
    """所有 runtime 的共同介面。"""

    runtime: str = ""  # 子類必須覆寫，供 service registry lookup

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_cached_models: int = 4,
        default_idle_timeout: float = 300.0,
    ):
        self.config = config or {}
        self._cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._cache_lock = threading.RLock()
        self._max_cached = max_cached_models
        # per-model load locks — 持久存在；模型卸載時不清除（lock 物件成本極低）
        self._load_locks: Dict[str, threading.Lock] = {}
        # 閒置逾時（秒）；<=0 表示不自動卸載
        raw_idle = self.config.get("idle_timeout", default_idle_timeout)
        idle = parse_keep_alive(raw_idle)
        self.idle_timeout: Optional[float] = idle if (idle is not None and idle > 0) else None

    # ----- 子類必須實作 -----

    @abstractmethod
    def load_model(self, spec: ModelSpec) -> Any:
        """根據 spec 把模型載入到記憶體（或建立可呼叫物件）。

        實作者只能讀 spec，不應該再去碰 MLflow / 檔案系統猜資訊。
        若 spec 缺欄位請拋 ValidationError。
        """

    @abstractmethod
    def infer(self, model: Any, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """對已載入的模型執行一次推理。回傳統一格式 dict。"""

    # ----- 預設實作（多數情況夠用，子類可覆寫）-----

    def run(self, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """單一進入點：取得（或載入）模型 → 推理 → 更新存活期限。"""
        keep_alive = parse_keep_alive(options.get("keep_alive"))
        entry = self._get_or_load(spec, keep_alive=keep_alive)
        key = spec.model_name
        with self._cache_lock:
            entry.in_flight += 1
        try:
            return self.infer(entry.model, spec, data, options)
        finally:
            self._settle(entry, key, keep_alive)

    def run_stream(self, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]):
        """串流版 run()：回傳逐段結果的 generator。

        infer_stream 在「呼叫時」就完成驗證與請求建立（可拋錯 — 此時 HTTP 層
        還沒送出 headers，能正常回錯誤碼）；回傳的 generator 才開始逐段產出。
        in_flight 在整個 generator 生命週期內保持 +1，關閉時結算 keep_alive。
        """
        keep_alive = parse_keep_alive(options.get("keep_alive"))
        entry = self._get_or_load(spec, keep_alive=keep_alive)
        key = spec.model_name
        with self._cache_lock:
            entry.in_flight += 1
        try:
            inner = self.infer_stream(entry.model, spec, data, options)
        except BaseException:
            self._settle(entry, key, keep_alive)
            raise

        def wrapped():
            try:
                yield from inner
            finally:
                self._settle(entry, key, keep_alive)

        return wrapped()

    def infer_stream(self, model: Any, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]):
        """子類可覆寫以支援串流。預設不支援（呼叫端 fallback 到非串流路徑）。

        合約：本方法為「一般函式」— 驗證與建立請求在呼叫時完成（可拋錯），
        回傳一個 generator：yield str（文字增量）；最後可選擇 yield 一個
        {"metadata": {...}} dict（usage 等統計）。
        """
        raise UnsupportedTaskError(
            f"runtime '{self.runtime}' does not support streaming for task '{spec.task_family}'"
        )

    def _settle(self, entry: "CacheEntry", key: str, keep_alive: Optional[float]) -> None:
        """一次推理（或串流）結束後的快取記帳與 keep_alive=0 即時卸載。"""
        now = time.time()
        with self._cache_lock:
            entry.in_flight -= 1
            entry.last_used = now
            entry.use_count += 1
            entry.expires_at = self._expiry(now, keep_alive)
            immediate = keep_alive is not None and keep_alive == 0 and entry.in_flight == 0
        if immediate:
            logger.info(f"[{self.runtime}] keep_alive=0 — unloading '{key}' right after inference")
            self.unload(key)

    def unload(self, model_name: str) -> bool:
        with self._cache_lock:
            entry = self._cache.pop(model_name, None)
        if entry is None:
            return False
        self._on_unload(entry.model)
        return True

    def unload_all(self) -> int:
        with self._cache_lock:
            entries = list(self._cache.values())
            self._cache.clear()
        for entry in entries:
            self._on_unload(entry.model)
        return len(entries)

    def evict_idle(self, now: Optional[float] = None) -> List[str]:
        """卸載所有已過 expires_at 且目前沒有請求在用的模型。回傳被卸載的名字。"""
        now = now if now is not None else time.time()
        expired: List[Tuple[str, CacheEntry]] = []
        with self._cache_lock:
            for key, entry in list(self._cache.items()):
                if entry.in_flight > 0:
                    continue
                if entry.expires_at is not None and entry.expires_at <= now:
                    expired.append((key, self._cache.pop(key)))
        for key, entry in expired:
            idle_for = now - entry.last_used
            logger.info(f"[{self.runtime}] idle-evicting '{key}' (idle {idle_for:.0f}s)")
            self._on_unload(entry.model)
        return [k for k, _ in expired]

    def loaded_models(self) -> list[str]:
        with self._cache_lock:
            return list(self._cache.keys())

    def loaded_models_info(self) -> List[Dict[str, Any]]:
        """快取內容快照（給監控/API 用）。"""
        now = time.time()
        with self._cache_lock:
            return [
                {
                    "model_name": key,
                    "loaded_at": entry.loaded_at,
                    "last_used": entry.last_used,
                    "idle_seconds": round(now - entry.last_used, 1),
                    "expires_at": entry.expires_at,
                    "expires_in_seconds": (
                        round(entry.expires_at - now, 1) if entry.expires_at is not None else None
                    ),
                    "use_count": entry.use_count,
                    "in_flight": entry.in_flight,
                }
                for key, entry in self._cache.items()
            ]

    # ----- 內部：LRU 快取 -----

    def _expiry(self, now: float, keep_alive: Optional[float]) -> Optional[float]:
        """依 keep_alive（請求層）與 idle_timeout（adapter 層）算出過期時間。"""
        ttl = keep_alive if keep_alive is not None else self.idle_timeout
        if ttl is None or ttl < 0:
            return None  # 常駐
        return now + ttl

    def _get_or_load(self, spec: ModelSpec, keep_alive: Optional[float] = None) -> CacheEntry:
        key = spec.model_name

        # 1. fast path — cache hit
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            load_lock = self._load_locks.setdefault(key, threading.Lock())

        # 2. slow path — acquire per-key load lock so only one thread loads
        evicted: List[Tuple[str, CacheEntry]] = []
        with load_lock:
            # double-checked locking: another thread may have loaded while we waited
            with self._cache_lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]

            logger.info(f"[{self.runtime}] loading model: {key} (spec.physical_path={spec.physical_path})")
            model = self.load_model(spec)

            now = time.time()
            entry = CacheEntry(model=model, loaded_at=now, last_used=now, expires_at=self._expiry(now, keep_alive))
            with self._cache_lock:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                while len(self._cache) > self._max_cached:
                    evict_key, evict_entry = self._cache.popitem(last=False)
                    if evict_entry.in_flight > 0:
                        # 有請求還在用 — 放回快取尾端，改挑下一個最舊的
                        self._cache[evict_key] = evict_entry
                        self._cache.move_to_end(evict_key)
                        if all(e.in_flight > 0 for e in self._cache.values()):
                            break  # 全部都在用，暫時超額，等 reaper 之後回收
                        continue
                    logger.info(f"[{self.runtime}] evicting LRU model: {evict_key}")
                    evicted.append((evict_key, evict_entry))

        # 3. _on_unload outside both locks — it may touch CUDA / be slow
        for _, evict_entry in evicted:
            self._on_unload(evict_entry.model)
        return entry

    def _on_unload(self, model: Any) -> None:
        """子類可覆寫以執行 GPU 清理等。預設 no-op。

        ⚠️ 注意：實作者「不應」主動 pop / clear 載入結果 dict 的欄位（model / processor
        / pipeline），因為其他 thread 可能仍持有引用並在跑 inference。
        正確做法是「只放掉本方法持有的引用 + 觸發 GC + empty_cache」，
        實際記憶體會在所有引用消失後才釋放。
        """
        return None
