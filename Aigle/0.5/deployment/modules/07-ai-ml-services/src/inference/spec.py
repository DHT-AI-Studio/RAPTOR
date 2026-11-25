# src/inference/spec.py
"""
ModelSpec — 模型推理規格

取代「靠字串匹配 + 硬編碼分支」的舊設計：
所有推理時所需的「該模型該怎麼跑」資訊，都從 MLflow tag 讀進來，
封裝為 ModelSpec，由 adapter 統一消費。

Spec 寫入時機：註冊模型到 MLflow 時（register_lakefs_to_mlflow / register_ollama_to_mlflow）
Spec 讀取時機：每次 InferenceService 收到請求時，先 resolve spec，再 dispatch 到 adapter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ModelNotFoundError, ValidationError

logger = logging.getLogger(__name__)


# ===== 規範化常數 =====

RUNTIME_OLLAMA = "ollama"
RUNTIME_HF = "hf-transformers"
SUPPORTED_RUNTIMES = {RUNTIME_OLLAMA, RUNTIME_HF}

TASK_FAMILIES = {
    "text-generation",
    "vlm",
    "asr",
    "ocr",
    "audio-classification",
    "video-analysis",
    "document-analysis",
    "image-captioning",
    "tts",
    "embedding",
    "rerank",
}


# 舊 task 名稱 → canonical task family（向下相容）
#
# 兩類來源都列在這裡：
#   1) v2 註冊端使用的連字號舊名（*-hf / *-ollama / scene-detection / ...）
#   2) inference.yaml 中 task_to_models 用的底線版舊名（text_generation / ...）
#
# VAD 走 audio-classification handler（speech / non-speech 二分類），不是 ASR。
_TASK_ALIAS = {
    # --- v2 連字號舊名 ---
    "text-generation-ollama": "text-generation",
    "text-generation-hf": "text-generation",
    "asr-hf": "asr",
    "vad-hf": "audio-classification",
    "ocr-hf": "ocr",
    "audio-transcription": "asr",
    "scene-detection": "video-analysis",
    "video-summary": "video-analysis",

    # --- inference.yaml 底線版舊名 ---
    "text_generation": "text-generation",
    "audio_transcription": "asr",
    "audio_classification": "audio-classification",
    "scene_detection": "video-analysis",
    "video_summary": "video-analysis",
    "document_analysis": "document-analysis",
    "image_captioning": "image-captioning",
    "vad": "audio-classification",

    # --- 服務化 task 的常見別名 ---
    "text-to-speech": "tts",
    "text_to_speech": "tts",
    "feature-extraction": "embedding",
    "feature_extraction": "embedding",
    "sentence-similarity": "embedding",
    "reranker": "rerank",
    "text-ranking": "rerank",
}


def canonicalize_task(task: str) -> str:
    """把 inference_api 收到的任意 task 字串轉為標準 task family。"""
    if task in TASK_FAMILIES:
        return task
    if task in _TASK_ALIAS:
        return _TASK_ALIAS[task]
    raise ValidationError(
        f"unknown task '{task}'. supported: {sorted(TASK_FAMILIES)}"
    )


# ===== ModelSpec 主結構 =====


@dataclass
class ModelSpec:
    """單一模型版本的完整推理規格。

    ModelSpec 是 adapter 與 service 之間的合約：
    adapter 看到 spec 就應該能完成 load_model/infer，不需要再讀任何 MLflow tag
    或對 model_name 做字串匹配。
    """

    # --- 識別 ---
    model_name: str            # MLflow registered name
    version: str               # MLflow version
    task_family: str           # canonical task (text-generation, vlm, ...)
    runtime: str               # ollama | hf-transformers | ...

    # --- 位址 ---
    physical_path: str         # lakefs://... | /local/path | hf-repo-id | ollama-model-name
    lakefs_commit_id: str | None = None

    # --- HF Transformers 專用 ---
    model_class: str | None = None       # e.g. "Qwen2_5_VLForConditionalGeneration"
    processor_class: str | None = None   # e.g. "AutoProcessor" / "AutoTokenizer"
    pipeline_task: str | None = None     # 若設了，走 pipeline shortcut
    trust_remote_code: bool = True
    quantization: str | None = None      # "4bit" | "8bit" | None
    torch_dtype: str | None = None       # "auto" | "fp16" | "bf16" | "fp32"

    # --- Ollama 專用 ---
    ollama_model_name: str | None = None  # 在 Ollama daemon 裡的真名

    # --- 自訂 handler 逃生口 ---
    custom_handler: str | None = None     # dotted import path 或 commit 內檔名

    # --- 模型／部署限制（未配置 = None，不以生成預設值代替）---
    max_output_tokens: int | None = None      # 單次生成 tokens 上限（max_tokens 驗證依據）
    context_window_tokens: int | None = None  # context window（輸入 + 輸出），與上者分開

    # --- 其他 ---
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 工廠方法：從 MLflow tag dict 建立 spec
    # ------------------------------------------------------------------

    @classmethod
    def from_mlflow_tags(
        cls,
        model_name: str,
        version: str,
        tags: dict[str, str],
    ) -> ModelSpec:
        """從 MLflow model version 的 tags 解析出 spec。

        新 schema 優先；缺欄位時退回舊 tag（向下相容）：
            - tags["runtime"]                  ↔ tags["inference_engine"]
            - tags["task_family"]              ↔ tags["inference_task"]
            - tags["model_class"] / ["processor_class"]  新增欄位
            - tags["ollama_model_name"]        Ollama 既有
            - tags["physical_path"]            既有
        """
        if not tags:
            raise ModelNotFoundError(
                f"model '{model_name}' v{version} has no MLflow tags; cannot build spec"
            )

        # runtime
        runtime = tags.get("runtime") or _legacy_runtime(tags)
        if runtime not in SUPPORTED_RUNTIMES:
            raise ValidationError(
                f"model '{model_name}' v{version}: unsupported runtime '{runtime}'. "
                f"set tag 'runtime' to one of {sorted(SUPPORTED_RUNTIMES)}"
            )

        # task family
        task_raw = tags.get("task_family") or tags.get("inference_task")
        if not task_raw:
            raise ValidationError(
                f"model '{model_name}' v{version}: missing 'task_family' tag"
            )
        task_family = canonicalize_task(task_raw)

        # path
        physical_path = (
            tags.get("physical_path")
            or tags.get("ollama_model_name")
            or ""
        )
        if not physical_path:
            raise ValidationError(
                f"model '{model_name}' v{version}: no physical_path/ollama_model_name tag"
            )

        return cls(
            model_name=model_name,
            version=str(version),
            task_family=task_family,
            runtime=runtime,
            physical_path=physical_path,
            lakefs_commit_id=tags.get("lakefs_commit_id"),
            model_class=tags.get("model_class"),
            processor_class=tags.get("processor_class"),
            pipeline_task=tags.get("pipeline_task"),
            trust_remote_code=_parse_bool(tags.get("trust_remote_code"), default=True),
            quantization=tags.get("quantization"),
            torch_dtype=tags.get("torch_dtype"),
            ollama_model_name=tags.get("ollama_model_name"),
            custom_handler=tags.get("custom_handler"),
            max_output_tokens=parse_limit(tags.get("max_output_tokens"), "max_output_tokens", model_name, version),
            context_window_tokens=parse_limit(
                tags.get("context_window_tokens"), "context_window_tokens", model_name, version
            ),
            extra={k: v for k, v in tags.items() if k not in _KNOWN_TAGS},
        )

    # ------------------------------------------------------------------
    # 快取識別：同名不同版本 / 不同實體位置 / 不同載入參數 不共用 adapter 實例
    # ------------------------------------------------------------------

    @property
    def cache_key(self) -> str:
        load_parts = [
            self.runtime,
            self.physical_path,
            self.ollama_model_name or "",
            self.lakefs_commit_id or "",
            self.model_class or "",
            self.processor_class or "",
            self.pipeline_task or "",
            self.quantization or "",
            self.torch_dtype or "",
            str(self.trust_remote_code),
        ]
        return f"{self.model_name}@{self.version}|" + "|".join(load_parts)

    # ------------------------------------------------------------------
    # 反向：把 spec 中的欄位整理回 tag dict（用於註冊時寫入 MLflow）
    # ------------------------------------------------------------------

    def to_mlflow_tags(self) -> dict[str, str]:
        """寫 MLflow tag — v3 schema 是唯一寫入端；不再回寫 inference_engine / inference_task。

        對既有模型（用舊 schema 註冊）的讀取相容，由 from_mlflow_tags 中的 _legacy_runtime
        與舊 inference_task tag 處理。
        """
        out: dict[str, str] = {
            "runtime": self.runtime,
            "task_family": self.task_family,
            "physical_path": self.physical_path,
            "trust_remote_code": str(self.trust_remote_code).lower(),
        }
        if self.lakefs_commit_id:
            out["lakefs_commit_id"] = self.lakefs_commit_id
        if self.model_class:
            out["model_class"] = self.model_class
        if self.processor_class:
            out["processor_class"] = self.processor_class
        if self.pipeline_task:
            out["pipeline_task"] = self.pipeline_task
        if self.quantization:
            out["quantization"] = self.quantization
        if self.torch_dtype:
            out["torch_dtype"] = self.torch_dtype
        if self.ollama_model_name:
            out["ollama_model_name"] = self.ollama_model_name
        if self.custom_handler:
            out["custom_handler"] = self.custom_handler
        if self.max_output_tokens is not None:
            out["max_output_tokens"] = str(self.max_output_tokens)
        if self.context_window_tokens is not None:
            out["context_window_tokens"] = str(self.context_window_tokens)
        return out


# ===== 內部 helpers =====


_KNOWN_TAGS = {
    "runtime", "task_family", "inference_task", "inference_engine",
    "physical_path", "lakefs_commit_id", "lakefs_repo",
    "model_class", "processor_class", "pipeline_task",
    "trust_remote_code", "quantization", "torch_dtype",
    "ollama_model_name", "custom_handler", "stage", "model_type",
    "max_output_tokens", "context_window_tokens",
}

LIMIT_TAGS = ("max_output_tokens", "context_window_tokens")


def parse_limit(value: Any, name: str, model_name: str = "", version: str = "") -> int | None:
    """limit tag → 正整數；未配置回 None。格式錯誤視同未配置並記 warning（不臆造上限）。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        n = int(str(value).strip())
    except ValueError:
        n = 0
    if n <= 0:
        logger.warning(f"model '{model_name}' v{version}: ignoring invalid {name}={value!r}")
        return None
    return n


def _legacy_runtime(tags: dict[str, str]) -> str | None:
    """從舊 tag 推斷 runtime（向下相容註冊在新 schema 之前的模型）。"""
    legacy = tags.get("inference_engine")
    if legacy == "ollama":
        return RUNTIME_OLLAMA
    if legacy in {"transformers", "huggingface", "hf"}:
        return RUNTIME_HF
    # 沒有 inference_engine 但有 ollama_model_name → Ollama
    if tags.get("ollama_model_name"):
        return RUNTIME_OLLAMA
    return None


def _parse_bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}
