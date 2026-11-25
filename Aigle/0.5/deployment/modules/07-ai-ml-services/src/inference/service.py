# src/inference/service.py
"""
InferenceService — 單一推理進入點

取代舊架構的 InferenceManager + TaskRouter + ModelExecutor + ModelCache 四層。
新流程：
    request → 驗證 → resolve ModelSpec from MLflow → 取得 adapter → adapter.run(spec, data, options)

只有兩層：service (本檔) + adapter（runtime-specific）。
模型快取由 adapter 自己用 OrderedDict 管理（在 BaseAdapter.LRU 裡）。

版本選擇（model_version / model_alias，互斥）在請求開始時解析為一份固定的 ModelSpec，
之後的驗證、載入、執行與 model-info 都用同一個 resolver（src/inference/registry.py）；
未帶 selector 時沿用「最大版本號」政策。

text-generation 的 options 先經 options.normalize_generation_options() 正規化，
再依已解析版本的 max_output_tokens 在生成前檢查 max_tokens；結果附上統一的 usage。

結構化輸出（options["format"]，見 structured.py）：只有 Ollama 的 text-generation 能照辦，
其他回 400 unsupported_capability；輸出不合格回 502 structured_output_error。
合格時 result.response 是正規化後的 JSON 文字，result.parsed 是解析後的值。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from typing_extensions import Self

from .adapters import all_adapters, get_adapter
from .exceptions import (
    InferenceError,
    InferenceExecutionError,
    MaxTokensExceededError,
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedCapabilityError,
    UnsupportedTaskError,
    ValidationError,
)
from .options import (
    SHARED_TEXT_OPTIONS,
    normalize_generation_options,
    usage_from_metadata,
)
from .registry import validate_selector
from .spec import LIMIT_TAGS, RUNTIME_OLLAMA, ModelSpec, canonicalize_task, parse_limit
from .structured import OutputContract, output_contract

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

    _instance: InferenceService | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> Self:
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
        # (model_name, model_version, model_alias) → {"version", "tags"}；測試可替換
        self._resolver: Callable[..., dict[str, Any]] | None = None
        self._engine_configs: dict[str, dict[str, Any]] = {}
        reaper_cfg: dict[str, Any] = {}
        try:
            from ..core.config import config
            self._engine_configs = config.get_config("inference", "engines") or {}
            reaper_cfg = config.get_config("inference", "idle_reaper") or {}
        except Exception as e:  # noqa: BLE001 -- config load across file IO/YAML parsing; already logged, fail-open
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
        task: str | None,
        model_name: str,
        data: dict[str, Any],
        engine: str | None = None,  # 只在模型未註冊時作為 fallback 依據，見 _spec_from_registry
        options: dict[str, Any] | None = None,
        model_version: str | None = None,
        model_alias: str | None = None,
    ) -> dict[str, Any]:
        """task 可為 None — 此時由模型註冊時的 task_family 決定（xinference 式：只給模型名即可調用）。

        model_version / model_alias：可選且互斥；未提供時使用最大版本號（既有行為）。
        回傳新增 usage（text-generation 的真實 token 計數；無法提供時為 None）。
        """
        t0 = time.time()
        with self._stats_lock:
            self._stats["total"] += 1

        try:
            spec, adapter, opts, contract = self._prepare(
                task, model_name, data, engine, options, model_version, model_alias)
            logger.info(
                f"infer model={spec.model_name} v{spec.version} runtime={spec.runtime} task={spec.task_family}"
            )
            result = adapter.run(spec, data, opts)
            if contract is not None and isinstance(result, dict) and not result.get("tool_calls"):
                parsed = contract.check(result.get("response") or "")
                result = {**result, "response": json.dumps(parsed, ensure_ascii=False), "parsed": parsed}

            with self._stats_lock:
                self._stats["success"] += 1
            return {
                "result": result,
                "task": spec.task_family,
                "engine": spec.runtime,
                "model_name": spec.model_name,
                "model_version": spec.version,
                "usage": _usage_of(spec, result),
                "processing_time": time.time() - t0,
                "timestamp": time.time(),
            }

        except InferenceError:
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
        task: str | None,
        model_name: str,
        data: dict[str, Any],
        options: dict[str, Any] | None = None,
        engine: str | None = None,
        model_version: str | None = None,
        model_alias: str | None = None,
    ):
        """串流版 infer()：回傳逐段產出的 generator（str 增量 + 最後的 metadata dict）。

        驗證/載入/建立串流連線在「呼叫時」完成 — 失敗會直接拋錯（此時 API 層
        還能回正常錯誤碼）。runtime 不支援該 task 串流時拋 UnsupportedTaskError，
        呼叫端應 fallback 到非串流路徑。

        engine: 同 infer()/describe_model() 的 fallback 依據 -- 沒有這個，
        engine="ollama" 的呼叫端對未註冊模型走串流路徑會在這裡就 404，
        跟非串流路徑（infer()）不一致。

        最後的 metadata dict 另帶 "usage"（與非串流相同來源；無法提供時為 None）。

        要求結構化輸出（options["format"]）時拋 UnsupportedTaskError：輸出要整份驗證過才能交出，
        呼叫端照既有規則 fallback 到非串流路徑。
        """
        if output_contract((options or {}).get("format"), "format") is not None:
            raise UnsupportedTaskError("structured output is validated whole and is not streamed")
        with self._stats_lock:
            self._stats["total"] += 1
        try:
            spec, adapter, opts, _ = self._prepare(
                task, model_name, data, engine, options, model_version, model_alias)
            logger.info(
                f"infer(stream) model={spec.model_name} v{spec.version} runtime={spec.runtime} task={spec.task_family}"
            )
            inner = adapter.run_stream(spec, data, opts)
        except BaseException:
            with self._stats_lock:
                self._stats["failed"] += 1
            raise

        def gen():
            try:
                for piece in inner:
                    if isinstance(piece, dict) and "metadata" in piece:
                        piece = {**piece, "usage": _usage_of(spec, piece)}
                    yield piece
            except BaseException:
                with self._stats_lock:
                    self._stats["failed"] += 1
                raise
            else:
                with self._stats_lock:
                    self._stats["success"] += 1

        return gen()

    def _prepare(
        self,
        task: str | None,
        model_name: str,
        data: dict[str, Any],
        engine: str | None,
        options: dict[str, Any] | None,
        model_version: str | None,
        model_alias: str | None,
    ) -> tuple[ModelSpec, Any, dict[str, Any], OutputContract | None]:
        """解析版本 → 核對 task → 驗證 data / options / 上限 / 結構化輸出 → 物化 → 取得 adapter。"""
        spec = self._spec_from_registry(
            model_name, engine=engine, task=task, model_version=model_version, model_alias=model_alias
        )
        if task:
            task_family = canonicalize_task(task)
            if spec.task_family != task_family:
                raise ValidationError(
                    f"task mismatch: request task='{task_family}' but model "
                    f"'{model_name}' is registered as task='{spec.task_family}'. "
                    f"Omit 'task' to use the model's registered task family, or pick a different model."
                )
        self._validate_data(spec.task_family, data)
        opts = dict(options or {})
        contract = output_contract(opts.pop("format", None), "format")
        if contract is not None:
            if not _structured_output_supported(spec):
                raise UnsupportedCapabilityError(
                    f"model '{spec.model_name}' ({spec.runtime}, {spec.task_family}) cannot honour a requested "
                    f"output format; structured output needs an Ollama text-generation model",
                    capability="structured_output")
            opts["format"] = contract.runtime_format()
        if spec.task_family == "text-generation":
            opts = normalize_generation_options(opts)
            self._check_output_limit(spec, opts)
        self._materialize(spec)
        adapter = get_adapter(spec.runtime, configs=self._engine_configs)
        return spec, adapter, opts, contract

    @staticmethod
    def _check_output_limit(spec: ModelSpec, opts: dict[str, Any]) -> None:
        """max_tokens 超過本次已解析版本的已知上限 → 生成前拒絕（422）。上限未知時不判定。

        舊的 max_length-only 呼叫保留其 runtime 語意，不套用此檢查。
        """
        limit = spec.max_output_tokens
        requested = opts.get("max_tokens")
        if limit is not None and requested is not None and requested > limit:
            raise MaxTokensExceededError(requested, limit, spec.model_name, spec.version)

    # ===== 模型管理 =====

    def unload_model(self, model_name: str) -> dict[str, Any]:
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

    def unload_all(self) -> dict[str, Any]:
        total = 0
        for adapter in all_adapters().values():
            total += adapter.unload_all()
        return {
            "success": True,
            "total_unloaded": total,
            "gpu_memory_freed": _clear_gpu_cache(),
        }

    def loaded_models(self) -> dict[str, Any]:
        out: dict[str, list[str]] = {}
        details: dict[str, list] = {}
        for runtime, adapter in all_adapters().items():
            out[runtime] = adapter.loaded_models()
            details[runtime] = adapter.loaded_models_info()
        total = sum(len(v) for v in out.values())
        return {"total": total, "by_runtime": out, "details": details}

    def clear_cache(self) -> None:
        self.unload_all()

    # ===== 內省 =====

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            snap = dict(self._stats)
        snap["success_rate"] = snap["success"] / max(snap["total"], 1)
        snap["loaded"] = self.loaded_models()
        return snap

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "adapters_initialized": list(all_adapters().keys()),
            "stats": self.get_stats(),
            "timestamp": time.time(),
        }

    def get_supported_tasks(self) -> dict[str, Any]:
        from .spec import TASK_FAMILIES

        def _fmt(item):  # tuple = 擇一欄位，呈現為 "a | b"
            return " | ".join(item) if isinstance(item, tuple) else item

        return {
            t: {"required_fields": [_fmt(i) for i in _REQUIRED_DATA_FIELDS.get(t, [])]}
            for t in sorted(TASK_FAMILIES)
        }

    # ===== 內部 =====

    def _validate_data(self, task_family: str, data: dict[str, Any]) -> None:
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

    def describe_model(self, model_name: str, engine: str | None = None,
                       model_version: str | None = None, model_alias: str | None = None) -> ModelSpec:
        """讀取模型的推理規格（只讀 MLflow tag，不觸發 lakeFS 下載）。

        供 OpenAI 相容層等呼叫端在建構請求前查詢 task_family / runtime。

        engine: 與 infer() 同一個 fallback 依據 -- 沒有這個之前，/v1/chat/completions
        在呼叫這裡查 task_family 時就已經因為模型未註冊而 404，根本到不了
        infer() 那邊真正有 fallback 的路徑。呼叫端明確傳 engine="ollama" 才會
        觸發，不影響一般（未帶 engine）呼叫仍要求先在 MLflow 註冊過的行為。
        """
        return self._spec_from_registry(
            model_name, engine=engine, model_version=model_version, model_alias=model_alias
        )

    def model_info(self, model_name: str, model_version: str | None = None,
                   model_alias: str | None = None, engine: str | None = None) -> dict[str, Any]:
        """只讀的模型能力查詢：與推論共用 resolver，不觸發載入、pull 或權重下載。"""
        spec = self.describe_model(model_name, engine=engine, model_version=model_version, model_alias=model_alias)
        is_text = spec.task_family == "text-generation"
        return {
            "model_name": spec.model_name,
            "model_version": spec.version,
            "runtime": spec.runtime,
            "task_family": spec.task_family,
            "runtime_model": spec.ollama_model_name if spec.runtime == RUNTIME_OLLAMA else None,
            "supported_options": list(SHARED_TEXT_OPTIONS) if is_text else [],
            "limits": {
                "max_output_tokens": spec.max_output_tokens,
                "context_window_tokens": spec.context_window_tokens,
            },
            "usage_supported": is_text and not spec.custom_handler,
            "structured_output": _structured_output_supported(spec),
        }

    def _resolve(self, model_name: str, model_version: str | None,
                 model_alias: str | None) -> dict[str, Any]:
        if self._resolver is not None:
            return self._resolver(model_name, model_version=model_version, model_alias=model_alias)
        try:
            from ..core.model_manager import model_manager
        except ImportError as e:
            raise InferenceError(f"model_manager unavailable: {e}") from e
        return model_manager.resolve_model_version(
            model_name, model_version=model_version, model_alias=model_alias
        )

    def _spec_from_registry(
        self,
        model_name: str,
        engine: str | None = None,
        task: str | None = None,
        model_version: str | None = None,
        model_alias: str | None = None,
    ) -> ModelSpec:
        validate_selector(model_version, model_alias)
        try:
            info = self._resolve(model_name, model_version, model_alias)
        except ModelNotFoundError as e:
            # 只有「模型本身不存在」且未指定版本 selector 才走 ollama fallback；
            # 指定的版本 / alias 不存在、或 registry 無法連線時照實回報
            if (engine == RUNTIME_OLLAMA and type(e) is ModelNotFoundError
                    and model_version is None and model_alias is None):
                return self._ollama_fallback_spec(model_name, task)
            raise

        version = str(info.get("version", "0"))
        tags = info.get("tags", {}) or {}
        return ModelSpec.from_mlflow_tags(model_name=model_name, version=version, tags=tags)

    def _ollama_fallback_spec(self, model_name: str, task: str | None) -> ModelSpec:
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
        # Ollama-only 模型的限制來自 inference.yaml engines.ollama.model_limits（依 daemon tag）
        limits_cfg = ((self._engine_configs.get(RUNTIME_OLLAMA) or {}).get("model_limits") or {}).get(model_name) or {}
        limits = {k: parse_limit(limits_cfg.get(k), k, model_name, "unregistered") for k in LIMIT_TAGS}
        # inference.yaml didn't configure a limit for this daemon tag -- ask the daemon
        # itself for the model's real context window (POST /api/show) rather than
        # leaving /inference/infer's max_tokens check permanently unable to validate
        # anything for this model. Best-effort: a daemon that can't answer leaves the
        # limit None, same as today.
        if limits.get("max_output_tokens") is None:
            context_length = get_adapter(RUNTIME_OLLAMA, configs=self._engine_configs).get_context_length(model_name)
            if context_length is not None:
                limits["max_output_tokens"] = context_length
                if limits.get("context_window_tokens") is None:
                    limits["context_window_tokens"] = context_length
        return ModelSpec(
            model_name=model_name,
            version="unregistered",
            task_family=task_family,
            runtime=RUNTIME_OLLAMA,
            physical_path=model_name,
            ollama_model_name=model_name,
            **limits,
        )

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
                evicted: dict[str, list] = {}
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


def _structured_output_supported(spec: ModelSpec) -> bool:
    """能把 schema 交給 runtime 約束生成的路徑：Ollama 的 text-generation（/api/chat、/api/generate 的 format）。"""
    return spec.runtime == RUNTIME_OLLAMA and spec.task_family == "text-generation" and not spec.custom_handler


def _usage_of(spec: ModelSpec, result: Any) -> dict[str, int] | None:
    """text-generation 的統一 usage；其他 task 或缺真實計數時為 None。"""
    if spec.task_family != "text-generation" or not isinstance(result, dict):
        return None
    return usage_from_metadata(result.get("metadata"))


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


def _clear_gpu_cache() -> dict[str, Any]:
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
