# src/inference/service.py
"""
InferenceService — 單一推理進入點

取代舊架構的 InferenceManager + TaskRouter + ModelExecutor + ModelCache 四層。
新流程：
    request → 驗證 → resolve ModelSpec from MLflow → 取得 adapter → adapter.run(spec, data, options)

只有兩層：service (本檔) + adapter（runtime-specific）。
模型快取由 adapter 自己用 OrderedDict 管理（在 BaseAdapter.LRU 裡）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

from .adapters import all_adapters, get_adapter
from .exceptions import (
    InferenceError,
    InferenceExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    ResourceExhaustedError,
    ResourceNotFoundError,
    UnsupportedTaskError,
    ValidationError,
)
from .spec import RUNTIME_OLLAMA, ModelSpec, canonicalize_task

logger = logging.getLogger(__name__)


# 任務類型 → 必須的 data 欄位；tuple 表示「至少要有其中一個」
_REQUIRED_DATA_FIELDS = {
    "text-generation": [("inputs", "messages")],
    "vlm": ["image", "prompt"],
    "asr": ["audio"],
    "ocr": ["image"],
    "audio-classification": ["audio"],
    "video-analysis": ["video"],
    "document-analysis": ["document"],
    "image-captioning": ["image"],
    "tts": [("text", "inputs")],
    "embedding": [("inputs", "input")],
    "rerank": ["query", "documents"],
}


class InferenceService:
    """單例式 service。InferenceService() 多次呼叫得到同一個物件。"""

    _instance: Optional["InferenceService"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "InferenceService":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._stats = {"total": 0, "success": 0, "failed": 0}
        self._stats_lock = threading.Lock()
        self._engine_configs: Dict[str, Dict[str, Any]] = {}
        reaper_cfg: Dict[str, Any] = {}
        try:
            from ..core.config import config
            self._engine_configs = config.get_config("inference", "engines") or {}
            reaper_cfg = config.get_config("inference", "idle_reaper") or {}
        except Exception as e:
            logger.warning(f"could not load engine configs from config.inference.engines: {e}")
        self._reaper_interval = float(reaper_cfg.get("interval", 30))
        self._reaper_enabled = bool(reaper_cfg.get("enabled", True))
        if self._reaper_enabled:
            self._start_idle_reaper()
        self._initialized = True
        logger.info("InferenceService initialized")

    # ===== 主進入點 =====

    def infer(
        self,
        task: Optional[str],
        model_name: str,
        data: Dict[str, Any],
        engine: Optional[str] = None,  # 只在模型未註冊時作為 fallback 依據，見 _spec_from_registry
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """task 可為 None — 此時由模型註冊時的 task_family 決定（xinference 式：只給模型名即可調用）。"""
        t0 = time.time()
        with self._stats_lock:
            self._stats["total"] += 1

        try:
            spec = self._spec_from_registry(model_name, engine=engine, task=task)
            if task:
                task_family = canonicalize_task(task)
                if spec.task_family != task_family:
                    raise ValidationError(
                        f"task mismatch: request task='{task_family}' but model "
                        f"'{model_name}' is registered as task='{spec.task_family}'. "
                        f"Omit 'task' to use the model's registered task family, or pick a different model."
                    )
            self._validate_data(spec.task_family, data)
            self._materialize(spec)

            adapter = get_adapter(spec.runtime, configs=self._engine_configs)
            logger.info(
                f"infer model={spec.model_name} v{spec.version} runtime={spec.runtime} task={spec.task_family}"
            )
            result = adapter.run(spec, data, options or {})

            with self._stats_lock:
                self._stats["success"] += 1
            return {
                "result": result,
                "task": spec.task_family,
                "engine": spec.runtime,
                "model_name": spec.model_name,
                "model_version": spec.version,
                "processing_time": time.time() - t0,
                "timestamp": time.time(),
            }

        except (ValidationError, UnsupportedTaskError, ModelNotFoundError,
                ResourceNotFoundError, ModelLoadError, InferenceExecutionError,
                ResourceExhaustedError, InferenceError):
            with self._stats_lock:
                self._stats["failed"] += 1
            raise
        except Exception as e:
            with self._stats_lock:
                self._stats["failed"] += 1
            logger.error("unexpected inference error", exc_info=True)
            raise InferenceExecutionError(f"unexpected error: {e}") from e

    def infer_stream(
        self,
        task: Optional[str],
        model_name: str,
        data: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
    ):
        """串流版 infer()：回傳逐段產出的 generator（str 增量 + 最後的 metadata dict）。

        驗證/載入/建立串流連線在「呼叫時」完成 — 失敗會直接拋錯（此時 API 層
        還能回正常錯誤碼）。runtime 不支援該 task 串流時拋 UnsupportedTaskError，
        呼叫端應 fallback 到非串流路徑。

        engine: 同 infer()/describe_model() 的 fallback 依據 -- 沒有這個，
        engine="ollama" 的呼叫端對未註冊模型走串流路徑會在這裡就 404，
        跟非串流路徑（infer()）不一致。
        """
        with self._stats_lock:
            self._stats["total"] += 1
        try:
            spec = self._spec_from_registry(model_name, engine=engine)
            if task:
                task_family = canonicalize_task(task)
                if spec.task_family != task_family:
                    raise ValidationError(
                        f"task mismatch: request task='{task_family}' but model "
                        f"'{model_name}' is registered as task='{spec.task_family}'."
                    )
            self._validate_data(spec.task_family, data)
            self._materialize(spec)
            adapter = get_adapter(spec.runtime, configs=self._engine_configs)
            logger.info(
                f"infer(stream) model={spec.model_name} v{spec.version} runtime={spec.runtime} task={spec.task_family}"
            )
            inner = adapter.run_stream(spec, data, options or {})
        except BaseException:
            with self._stats_lock:
                self._stats["failed"] += 1
            raise

        def gen():
            try:
                yield from inner
            except BaseException:
                with self._stats_lock:
                    self._stats["failed"] += 1
                raise
            else:
                with self._stats_lock:
                    self._stats["success"] += 1

        return gen()

    # ===== 模型管理 =====

    def unload_model(self, model_name: str) -> Dict[str, Any]:
        unloaded = []
        for runtime, adapter in all_adapters().items():
            if adapter.unload(model_name):
                unloaded.append(runtime)
        return {
            "success": True,
            "model_name": model_name,
            "unloaded_runtimes": unloaded,
            "gpu_memory_freed": _clear_gpu_cache(),
        }

    def unload_all(self) -> Dict[str, Any]:
        total = 0
        for adapter in all_adapters().values():
            total += adapter.unload_all()
        return {
            "success": True,
            "total_unloaded": total,
            "gpu_memory_freed": _clear_gpu_cache(),
        }

    def loaded_models(self) -> Dict[str, Any]:
        out: Dict[str, list[str]] = {}
        details: Dict[str, list] = {}
        for runtime, adapter in all_adapters().items():
            out[runtime] = adapter.loaded_models()
            details[runtime] = adapter.loaded_models_info()
        total = sum(len(v) for v in out.values())
        return {"total": total, "by_runtime": out, "details": details}

    def clear_cache(self) -> None:
        self.unload_all()

    # ===== 內省 =====

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            snap = dict(self._stats)
        snap["success_rate"] = snap["success"] / max(snap["total"], 1)
        snap["loaded"] = self.loaded_models()
        return snap

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "adapters_initialized": list(all_adapters().keys()),
            "stats": self.get_stats(),
            "timestamp": time.time(),
        }

    def get_supported_tasks(self) -> Dict[str, Any]:
        from .spec import TASK_FAMILIES

        def _fmt(item):  # tuple = 擇一欄位，呈現為 "a | b"
            return " | ".join(item) if isinstance(item, tuple) else item

        return {
            t: {"required_fields": [_fmt(i) for i in _REQUIRED_DATA_FIELDS.get(t, [])]}
            for t in sorted(TASK_FAMILIES)
        }

    # ===== 內部 =====

    def _validate_data(self, task_family: str, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict) or not data:
            raise ValidationError("data must be a non-empty dict")
        required = _REQUIRED_DATA_FIELDS.get(task_family, [])
        missing = []
        for item in required:
            if isinstance(item, tuple):  # 任一欄位滿足即可
                if not any(k in data for k in item):
                    missing.append(" or ".join(item))
            elif item not in data:
                missing.append(item)
        if missing:
            raise ValidationError(f"task '{task_family}' missing required data fields: {missing}")

    def describe_model(self, model_name: str, engine: Optional[str] = None) -> ModelSpec:
        """讀取模型的推理規格（只讀 MLflow tag，不觸發 lakeFS 下載）。

        供 OpenAI 相容層等呼叫端在建構請求前查詢 task_family / runtime。

        engine: 與 infer() 同一個 fallback 依據 -- 沒有這個之前，/v1/chat/completions
        在呼叫這裡查 task_family 時就已經因為模型未註冊而 404，根本到不了
        infer() 那邊真正有 fallback 的路徑。呼叫端明確傳 engine="ollama" 才會
        觸發，不影響一般（未帶 engine）呼叫仍要求先在 MLflow 註冊過的行為。
        """
        return self._spec_from_registry(model_name, engine=engine)

    def _spec_from_registry(
        self, model_name: str, engine: Optional[str] = None, task: Optional[str] = None
    ) -> ModelSpec:
        try:
            from ..core.model_manager import model_manager
        except ImportError as e:
            raise InferenceError(f"model_manager unavailable: {e}") from e

        info = model_manager.get_model_details_from_mlflow(model_name)
        if not info or info.get("error"):
            if engine == RUNTIME_OLLAMA:
                # 舊架構（0.3）行為：MLflow 查不到時，若呼叫端明確聲明
                # engine="ollama"，退回把 model_name 當成 Ollama daemon 上的
                # 原生 tag 直接嘗試（OllamaAdapter 會視 auto_pull 設定決定
                # 是否自動 pull）。只在呼叫端明確要求 ollama 時觸發，不影響
                # 一般（未帶 engine）呼叫仍然要求先在 MLflow 註冊過。
                logger.warning(
                    f"model '{model_name}' not in MLflow registry; engine='ollama' was "
                    f"explicitly requested, falling back to treating model_name as a raw "
                    f"Ollama tag on the daemon"
                )
                task_family = canonicalize_task(task) if task else "text-generation"
                return ModelSpec(
                    model_name=model_name,
                    version="unregistered",
                    task_family=task_family,
                    runtime=RUNTIME_OLLAMA,
                    physical_path=model_name,
                    ollama_model_name=model_name,
                )
            raise ModelNotFoundError(f"model '{model_name}' not in MLflow registry")

        version = str(info.get("version", "0"))
        tags = info.get("tags", {}) or {}
        return ModelSpec.from_mlflow_tags(model_name=model_name, version=version, tags=tags)

    def _materialize(self, spec: ModelSpec) -> None:
        """lakefs:// 位址無法被 transformers/from_pretrained 直接讀取，
        需先把 commit 內的模型檔下載到本地，再把 physical_path 換成本地路徑。
        download_from_lakefs_by_uri 內含快取：已下載過則直接回傳本地路徑。
        """
        if not spec.physical_path.startswith("lakefs://"):
            return
        from ..core.model_manager import model_manager
        local_path = model_manager.download_from_lakefs_by_uri(spec.physical_path)
        if not local_path or str(local_path).startswith("Error"):
            raise ModelLoadError(
                f"failed to materialize lakefs model '{spec.model_name}' from "
                f"{spec.physical_path}: {local_path}"
            )
        model_root = _resolve_local_model_root(local_path)
        logger.info(
            f"materialized lakefs model '{spec.model_name}' {spec.physical_path} -> {model_root}"
        )
        spec.physical_path = model_root

    # ===== 閒置回收（ollama 式 keep_alive）=====

    def _start_idle_reaper(self) -> None:
        t = threading.Thread(target=self._reaper_loop, name="model-idle-reaper", daemon=True)
        t.start()
        logger.info(f"idle reaper started (interval={self._reaper_interval}s)")

    def _reaper_loop(self) -> None:
        while True:
            time.sleep(self._reaper_interval)
            try:
                evicted: Dict[str, list] = {}
                for runtime, adapter in all_adapters().items():
                    names = adapter.evict_idle()
                    if names:
                        evicted[runtime] = names
                if evicted:
                    freed = _clear_gpu_cache()
                    logger.info(f"idle reaper unloaded {evicted}; gpu_memory_freed={freed}")
            except Exception:
                logger.exception("idle reaper iteration failed")


# ===== module-level helpers =====


# transformers 認得的權重檔名（含 sharded index）
_WEIGHT_FILES = (
    "model.safetensors", "model.safetensors.index.json",
    "pytorch_model.bin", "pytorch_model.bin.index.json",
    "tf_model.h5", "flax_model.msgpack", "model.ckpt.index",
)


def _resolve_local_model_root(path: str) -> str:
    """從下載目錄找出真正含模型權重的資料夾。

    上傳到 lakeFS 時模型常被包在一層以模型名命名的子資料夾下，
    或夾帶 HF 快取目錄（models--*），導致 from_pretrained 在頂層找不到權重。
    以 BFS 找出「最淺、同時含 config.json 與權重檔」的目錄；找不到則回原路徑，
    讓 from_pretrained 自行報出清楚的錯誤。
    """
    queue = deque([path])
    while queue:
        d = queue.popleft()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        names = {e.name for e in entries}
        if "config.json" in names and any(w in names for w in _WEIGHT_FILES):
            return d
        for e in entries:
            # 跳過 HF 快取目錄，優先取扁平存放的模型目錄
            if e.is_dir() and e.name != ".cache" and not e.name.startswith("models--"):
                queue.append(e.path)
    return path


def _clear_gpu_cache() -> Dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        before = torch.cuda.memory_allocated() / (1024 ** 3)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / (1024 ** 3)
        return {
            "cuda_available": True,
            "memory_before_gb": round(before, 2),
            "memory_after_gb": round(after, 2),
            "memory_freed_gb": round(before - after, 2),
        }
    except ImportError:
        return {"cuda_available": False}


# 全局單例
inference_service = InferenceService()
