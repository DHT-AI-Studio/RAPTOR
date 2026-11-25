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
    - 快取識別為 spec.cache_key（模型名 + 解析後版本 + runtime / 實體位置 / 載入參數），
      同名不同版本不共用實例；load lock、in-flight 記帳、idle eviction、串流都用同一個 key
    - in_flight 在取得 entry 的同一個 lock 區段內 +1，避免「剛拿到就被 reaper 回收」

卸載（unload）：
    - unload(model_name) 卸載該名稱「所有版本」的快取；仍有請求在用的 entry 先移出快取
      （新請求改載新實例），標記 retired，等最後一個請求結束才真正釋放

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
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import UnsupportedTaskError, ValidationError
from ..spec import ModelSpec

logger = logging.getLogger(__name__)


def parse_keep_alive(value: Any) -> float | None:
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
    model_name: str = ""
    version: str = ""
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    expires_at: float | None = None  # None = 不會被 idle reaper 回收
    use_count: int = 0
    in_flight: int = 0  # 正在使用此模型的推理請求數；>0 時 reaper 不回收
    retired: bool = False  # 已移出快取但仍有請求在用；最後一個請求結束時釋放
    retire_explicit: bool = False  # retired 的原因是否為明確 unload（影響 _on_explicit_unload）


class BaseAdapter(ABC):
    """所有 runtime 的共同介面。"""

    runtime: str = ""  # 子類必須覆寫，供 service registry lookup

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        max_cached_models: int = 4,
        default_idle_timeout: float = 300.0,
    ):
        self.config = config or {}
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._max_cached = max_cached_models
        # per-key load locks — 持久存在；模型卸載時不清除（lock 物件成本極低）
        self._load_locks: dict[str, threading.Lock] = {}
        # 已移出快取、等在用請求結束才釋放的 entries
        self._retired: list[CacheEntry] = []
        # 閒置逾時（秒）；<=0 表示不自動卸載
        raw_idle = self.config.get("idle_timeout", default_idle_timeout)
        idle = parse_keep_alive(raw_idle)
        self.idle_timeout: float | None = idle if (idle is not None and idle > 0) else None

    # ----- 子類必須實作 -----

    @abstractmethod
    def load_model(self, spec: ModelSpec) -> Any:
        """根據 spec 把模型載入到記憶體（或建立可呼叫物件）。

        實作者只能讀 spec，不應該再去碰 MLflow / 檔案系統猜資訊。
        若 spec 缺欄位請拋 ValidationError。
        """

    @abstractmethod
    def infer(self, model: Any, spec: ModelSpec, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """對已載入的模型執行一次推理。回傳統一格式 dict。"""

    # ----- 預設實作（多數情況夠用，子類可覆寫）-----

    def run(self, spec: ModelSpec, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """單一進入點：取得（或載入）模型 → 推理 → 更新存活期限。"""
        keep_alive = parse_keep_alive(options.get("keep_alive"))
        key, entry = self._get_or_load(spec, keep_alive=keep_alive)
        try:
            return self.infer(entry.model, spec, data, options)
        finally:
            self._settle(entry, key, keep_alive)

    def run_stream(self, spec: ModelSpec, data: dict[str, Any], options: dict[str, Any]):
        """串流版 run()：回傳逐段結果的 generator。

        infer_stream 在「呼叫時」就完成驗證與請求建立（可拋錯 — 此時 HTTP 層
        還沒送出 headers，能正常回錯誤碼）；回傳的 generator 才開始逐段產出。
        in_flight 在整個 generator 生命週期內保持 +1，關閉時結算 keep_alive；
        generator 從未被迭代就被丟棄時，由 weakref.finalize 補做結算。
        """
        keep_alive = parse_keep_alive(options.get("keep_alive"))
        key, entry = self._get_or_load(spec, keep_alive=keep_alive)
        try:
            inner = self.infer_stream(entry.model, spec, data, options)
        except BaseException:
            self._settle(entry, key, keep_alive)
            raise

        settle = _Once(lambda: self._settle(entry, key, keep_alive))

        def wrapped():
            try:
                yield from inner
            finally:
                settle()

        gen = wrapped()
        weakref.finalize(gen, settle)
        return gen

    def infer_stream(self, model: Any, spec: ModelSpec, data: dict[str, Any], options: dict[str, Any]):
        """子類可覆寫以支援串流。預設不支援（呼叫端 fallback 到非串流路徑）。

        合約：本方法為「一般函式」— 驗證與建立請求在呼叫時完成（可拋錯），
        回傳一個 generator：yield str（文字增量）；最後可選擇 yield 一個
        {"metadata": {...}} dict（usage 等統計）。
        """
        raise UnsupportedTaskError(
            f"runtime '{self.runtime}' does not support streaming for task '{spec.task_family}'"
        )

    def _settle(self, entry: CacheEntry, key: str, keep_alive: float | None) -> None:
        """一次推理（或串流）結束後的快取記帳、keep_alive=0 即時卸載、retired entry 釋放。"""
        now = time.time()
        release: bool | None = None  # None = 不釋放；True/False = 釋放且是否為明確卸載
        with self._cache_lock:
            entry.in_flight -= 1
            entry.last_used = now
            entry.use_count += 1
            entry.expires_at = self._expiry(now, keep_alive)
            if entry.in_flight == 0:
                if entry.retired:
                    self._drop_retired(entry)
                    release = entry.retire_explicit
                elif keep_alive is not None and keep_alive == 0 and self._cache.get(key) is entry:
                    logger.info(f"[{self.runtime}] keep_alive=0 — unloading '{key}' right after inference")
                    self._cache.pop(key)
                    release = True
        if release is not None:
            self._release(entry, explicit=release)

    def unload(self, model_name: str) -> bool:
        """卸載該 model_name 的所有快取版本；在用中的延後到請求結束才釋放。"""
        with self._cache_lock:
            victims = [k for k, e in self._cache.items() if e.model_name == model_name or k == model_name]
            entries = [self._cache.pop(k) for k in victims]
            now_release = self._retire_busy(entries, explicit=True)
        for entry in now_release:
            self._release(entry, explicit=True)
        return bool(entries)

    def unload_all(self) -> int:
        with self._cache_lock:
            entries = list(self._cache.values())
            self._cache.clear()
            now_release = self._retire_busy(entries, explicit=False)
        for entry in now_release:
            self._release(entry, explicit=False)
        return len(entries)

    def evict_idle(self, now: float | None = None) -> list[str]:
        """卸載所有已過 expires_at 且目前沒有請求在用的模型。回傳被卸載的 cache key。"""
        now = now if now is not None else time.time()
        expired: list[tuple[str, CacheEntry]] = []
        with self._cache_lock:
            for key, entry in list(self._cache.items()):
                if entry.in_flight > 0:
                    continue
                if entry.expires_at is not None and entry.expires_at <= now:
                    expired.append((key, self._cache.pop(key)))
        for key, entry in expired:
            idle_for = now - entry.last_used
            logger.info(f"[{self.runtime}] idle-evicting '{key}' (idle {idle_for:.0f}s)")
            self._release(entry, explicit=False)
        return [k for k, _ in expired]

    def loaded_models(self) -> list[str]:
        """已載入的模型名稱（同名多版本只列一次；版本細節見 loaded_models_info）。"""
        with self._cache_lock:
            return list(dict.fromkeys(e.model_name or k for k, e in self._cache.items()))

    def loaded_models_info(self) -> list[dict[str, Any]]:
        """快取內容快照（給監控/API 用）；含已移出快取、等待在用請求結束的 entries。"""
        now = time.time()
        with self._cache_lock:
            rows = [(key, entry) for key, entry in self._cache.items()]
            rows += [(None, entry) for entry in self._retired]
            return [
                {
                    "model_name": entry.model_name or key,
                    "model_version": entry.version or None,
                    "cache_key": key,
                    "loaded_at": entry.loaded_at,
                    "last_used": entry.last_used,
                    "idle_seconds": round(now - entry.last_used, 1),
                    "expires_at": entry.expires_at,
                    "expires_in_seconds": (
                        round(entry.expires_at - now, 1) if entry.expires_at is not None else None
                    ),
                    "use_count": entry.use_count,
                    "in_flight": entry.in_flight,
                    "pending_unload": entry.retired,
                }
                for key, entry in rows
            ]

    # ----- 內部：LRU 快取 -----

    def _expiry(self, now: float, keep_alive: float | None) -> float | None:
        """依 keep_alive（請求層）與 idle_timeout（adapter 層）算出過期時間。"""
        ttl = keep_alive if keep_alive is not None else self.idle_timeout
        if ttl is None or ttl < 0:
            return None  # 常駐
        return now + ttl

    def _get_or_load(self, spec: ModelSpec, keep_alive: float | None = None) -> tuple[str, CacheEntry]:
        """取得（或載入）spec 對應的 entry，並在同一 lock 區段內把 in_flight +1。"""
        key = spec.cache_key

        # 1. fast path — cache hit
        with self._cache_lock:
            if key in self._cache:
                return key, self._acquire(key)
            load_lock = self._load_locks.setdefault(key, threading.Lock())

        # 2. slow path — acquire per-key load lock so only one thread loads
        evicted: list[tuple[str, CacheEntry]] = []
        with load_lock:
            # double-checked locking: another thread may have loaded while we waited
            with self._cache_lock:
                if key in self._cache:
                    return key, self._acquire(key)

            logger.info(
                f"[{self.runtime}] loading model: {spec.model_name} v{spec.version} "
                f"(spec.physical_path={spec.physical_path})"
            )
            model = self.load_model(spec)

            now = time.time()
            entry = CacheEntry(
                model=model, model_name=spec.model_name, version=str(spec.version),
                loaded_at=now, last_used=now, expires_at=self._expiry(now, keep_alive),
            )
            with self._cache_lock:
                entry.in_flight += 1
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

        # 3. release outside both locks — it may touch CUDA / be slow
        for _, evict_entry in evicted:
            self._release(evict_entry, explicit=False)
        return key, entry

    def _acquire(self, key: str) -> CacheEntry:
        """（呼叫端須持有 _cache_lock）標記使用並 +1 in_flight。"""
        self._cache.move_to_end(key)
        entry = self._cache[key]
        entry.in_flight += 1
        return entry

    def _retire_busy(self, entries: list[CacheEntry], explicit: bool) -> list[CacheEntry]:
        """（呼叫端須持有 _cache_lock）在用的標記 retired 延後釋放；回傳可立即釋放的。"""
        idle = []
        for entry in entries:
            if entry.in_flight > 0:
                entry.retired = True
                entry.retire_explicit = entry.retire_explicit or explicit
                self._retired.append(entry)
            else:
                idle.append(entry)
        return idle

    def _drop_retired(self, entry: CacheEntry) -> None:
        self._retired = [e for e in self._retired if e is not entry]

    def _entries_using(self, model: Any) -> list[CacheEntry]:
        """快取中（含 retired）持有同一個 model 物件／識別的 entries。"""
        with self._cache_lock:
            return [e for e in list(self._cache.values()) + self._retired if e.model == model]

    def _release(self, entry: CacheEntry, explicit: bool) -> None:
        self._on_unload(entry.model)
        if explicit:
            self._on_explicit_unload(entry)

    def _on_unload(self, model: Any) -> None:
        """子類可覆寫以執行 GPU 清理等。預設 no-op。

        ⚠️ 注意：實作者「不應」主動 pop / clear 載入結果 dict 的欄位（model / processor
        / pipeline），因為其他 thread 可能仍持有引用並在跑 inference。
        正確做法是「只放掉本方法持有的引用 + 觸發 GC + empty_cache」，
        實際記憶體會在所有引用消失後才釋放。
        """
        return

    def _on_explicit_unload(self, entry: CacheEntry) -> None:
        """明確卸載（unload(model_name) / keep_alive=0）後的額外動作；預設 no-op。"""
        return


class _Once:
    """只執行一次的 callable（串流結算同時掛在 finally 與 weakref.finalize）。"""

    def __init__(self, fn):
        self._fn = fn
        self._done = False
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._fn()
