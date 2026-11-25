"""INF-2: model discovery across Module 07's MLflow registry and the Ollama daemon.

Sources
  * Module 07 ``GET /models/registered_in_mlflow?show_all=true`` -- registered models and
    their version tags (stage, runtime, task_family, limits, ollama_model_name).  07 is used
    instead of MLflow's REST API directly so 35 does not depend on MLflow's API version.
  * Ollama ``/api/tags`` (models on disk) and ``/api/ps`` (models resident in memory),
    straight from ``MODULE07_OLLAMA_HOST`` so discovery still works when 07 is down.
  * Module 07 ``GET /inference/loaded-models`` -- which HF model versions 07 has in GPU memory.

Version policy: for each registered model, the highest ``production`` version is listed;
without one, the highest ``staging`` version.  Versions in neither stage are not listed.
The chosen version is sent explicitly to 07 on every inference call, so what GET /models
shows is what actually runs.

Status: ``loaded`` (resident now) > ``staged`` (chosen version is staging) > ``available``.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.module07_client import Module07Client

logger = logging.getLogger(__name__)

# 07 task_family -> Module 35 model type.  Families 35 does not serve (vlm, ocr, embedding,
# rerank, ...) are not listed.  07 has no clone / diarize family; a registered version can
# opt in with the tag ``inf_model_type`` (e.g. a TTS model registered for voice cloning).
_FAMILY_TO_TYPE = {"text-generation": "llm", "asr": "stt", "tts": "tts"}
MODEL_TYPES = ("llm", "stt", "tts", "clone", "diarize")
TYPE_CAPABILITIES = {
    "llm": ["translate", "summarize", "complete", "transform"],
    "stt": ["transcribe"],
    "tts": ["speech"],
    "clone": ["clone"],
    "diarize": ["diarize"],
}
_STAGE_RANK = {"production": 2, "staging": 1}
# Ollama families / names that are embedding models, not generators.
_EMBEDDING_FAMILIES = {"bert", "nomic-bert", "xlm-roberta"}
# Minimum age before a cache miss may force a refresh (protects 07 from unknown-name floods).
_MISS_REFRESH_MIN_AGE_S = 5.0


@dataclass
class ModelEntry:
    name: str
    type: str
    size_gb: float | None
    capabilities: list[str]
    source: str                 # mlflow | ollama | builtin
    status: str                 # loaded | available | staged
    version: str | None = None
    stage: str | None = None
    runtime: str | None = None
    runtime_model: str | None = None     # Ollama daemon tag backing this entry
    watermarked: bool = False            # clone-type builtin entries only

    @property
    def engine(self) -> str | None:
        """07 ``engine`` selector: only Ollama-only (unregistered) models need it."""
        return "ollama" if self.source == "ollama" else None

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegistrySnapshot:
    models: list[ModelEntry]
    fetched_at: float
    sources: dict[str, str] = field(default_factory=dict)

    def find(self, name: str) -> ModelEntry | None:
        for entry in self.models:
            if entry.name == name:
                return entry
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return round(float(match.group()), 2) if match else None


def _version_key(version: dict[str, Any]) -> int:
    try:
        return int(version.get("version"))
    except (TypeError, ValueError):
        return -1


def select_version(versions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest production version, else highest staging version, else None."""
    best: tuple[int, int, dict[str, Any]] | None = None
    for v in versions:
        stage = str(v.get("stage") or (v.get("tags") or {}).get("stage") or "").lower()
        rank = _STAGE_RANK.get(stage)
        if rank is None:
            continue
        key = (rank, _version_key(v))
        if best is None or key > best[:2]:
            best = (key[0], key[1], v)
    return best[2] if best else None


def _mlflow_entry(
    model: dict[str, Any],
    ollama_sizes: dict[str, float],
    ollama_loaded: set,
    hf_loaded: set,
) -> ModelEntry | None:
    version = select_version(model.get("versions") or [])
    if version is None:
        return None
    tags = version.get("tags") or {}
    model_type = tags.get("inf_model_type") or _FAMILY_TO_TYPE.get(
        tags.get("task_family") or tags.get("inference_task") or ""
    )
    if model_type not in MODEL_TYPES:
        return None
    name = model["name"]
    ver = str(version.get("version"))
    stage = str(version.get("stage") or tags.get("stage") or "").lower()
    runtime = tags.get("runtime") or tags.get("inference_engine")
    runtime_model = tags.get("ollama_model_name")

    size = _to_float(tags.get("size_gb")) or _to_float(tags.get("ollama_model_size_gb"))
    if size is None and runtime_model:
        size = ollama_sizes.get(runtime_model)
    if size is None:
        size = _to_float(tags.get("est_weights_vram_gb"))

    if (runtime_model and runtime_model in ollama_loaded) or (name, ver) in hf_loaded:
        status = "loaded"
    elif stage == "staging":
        status = "staged"
    else:
        status = "available"
    return ModelEntry(
        name=name, type=model_type, size_gb=size, capabilities=list(TYPE_CAPABILITIES[model_type]),
        source="mlflow", status=status, version=ver, stage=stage, runtime=runtime,
        runtime_model=runtime_model,
    )


def _ollama_entry(tag: dict[str, Any], ollama_loaded: set) -> ModelEntry | None:
    name = tag.get("name") or tag.get("model")
    if not name:
        return None
    details = tag.get("details") or {}
    families = set(details.get("families") or []) | {details.get("family") or ""}
    if "embed" in name.lower() or families & _EMBEDDING_FAMILIES:
        return None
    size = tag.get("size")
    return ModelEntry(
        name=name, type="llm",
        size_gb=round(size / (1024 ** 3), 2) if isinstance(size, (int, float)) else None,
        capabilities=list(TYPE_CAPABILITIES["llm"]), source="ollama",
        status="loaded" if name in ollama_loaded else "available",
        runtime="ollama", runtime_model=name,
    )


def _hf_loaded(loaded: dict[str, Any]) -> set:
    rows = ((loaded.get("details") or {}).get("hf-transformers")) or []
    return {
        (r.get("model_name"), str(r.get("model_version")))
        for r in rows
        if isinstance(r, dict) and not r.get("pending_unload")
    }


class ModelRegistry:
    def __init__(self, client: Module07Client, ttl_s: float = 60.0,
                 static_entries: Callable[[], list[ModelEntry]] | None = None) -> None:
        self._client = client
        self._ttl = ttl_s
        self._snapshot: RegistrySnapshot | None = None
        self._lock = asyncio.Lock()
        # Callable, not a frozen list: re-read on every refresh so a static
        # entry's "loaded" status reflects whichever backends are actually
        # resident right now (e.g. Cloning.builtin_model_entries).
        self._static_entries = static_entries or list

    def invalidate(self) -> None:
        self._snapshot = None

    async def snapshot(self, force: bool = False) -> RegistrySnapshot:
        snap = self._snapshot
        if not force and snap is not None and time.monotonic() - snap.fetched_at < self._ttl:
            return snap
        async with self._lock:
            if self._snapshot is not None and self._snapshot is not snap:
                return self._snapshot  # refreshed by another request while we waited
            self._snapshot = await self._fetch()
            return self._snapshot

    async def list_models(self) -> list[ModelEntry]:
        return (await self.snapshot()).models

    async def get(self, name: str) -> ModelEntry | None:
        """Look a model up; a miss on a snapshot older than a few seconds triggers one refresh."""
        snap = await self.snapshot()
        entry = snap.find(name)
        if entry is None and time.monotonic() - snap.fetched_at >= _MISS_REFRESH_MIN_AGE_S:
            entry = (await self.snapshot(force=True)).find(name)
        return entry

    async def _fetch(self) -> RegistrySnapshot:
        registered, loaded, tags, ps = await asyncio.gather(
            self._client.list_registered_models(),
            self._client.loaded_models(),
            self._client.ollama_tags(),
            self._client.ollama_ps(),
            return_exceptions=True,
        )
        sources = {"mlflow": "ok", "ollama": "ok"}
        for label, result in (("Module 07 registry", registered), ("Ollama /api/tags", tags)):
            if isinstance(result, BaseException):
                logger.warning("%s unavailable, listing without it: %s", label, result)
        for label, result in (("07 loaded-models", loaded), ("Ollama /api/ps", ps)):
            if isinstance(result, BaseException):
                logger.info("%s unavailable, loaded status may be incomplete: %s", label, result)

        ollama_tags = [] if isinstance(tags, BaseException) else tags
        ollama_loaded = set() if isinstance(ps, BaseException) else {
            m.get("name") or m.get("model") for m in ps
        }
        hf_loaded = set() if isinstance(loaded, BaseException) else _hf_loaded(loaded)
        ollama_sizes = {
            t.get("name"): round(t["size"] / (1024 ** 3), 2)
            for t in ollama_tags
            if isinstance(t.get("size"), (int, float))
        }

        models: dict[str, ModelEntry] = {}
        if isinstance(registered, BaseException):
            sources["mlflow"] = "unavailable"
        else:
            for model in registered:
                try:
                    entry = _mlflow_entry(model, ollama_sizes, ollama_loaded, hf_loaded)
                except Exception:  # one malformed registry row must not hide the rest
                    logger.exception("skipping malformed registry entry %r", model.get("name"))
                    continue
                if entry:
                    models[entry.name] = entry
        if isinstance(tags, BaseException):
            sources["ollama"] = "unavailable"
        for tag in ollama_tags:
            entry = _ollama_entry(tag, ollama_loaded)
            # Same name in both sources: the MLflow entry wins.
            if entry and entry.name not in models:
                models[entry.name] = entry
        for entry in self._static_entries():
            # Same "first source wins" precedent as mlflow > ollama above.
            if entry.name not in models:
                models[entry.name] = entry

        ordered = sorted(models.values(), key=lambda e: (e.type, e.name))
        return RegistrySnapshot(models=ordered, fetched_at=time.monotonic(), sources=sources)
