"""Dynamic GPU model resource manager for the in-process image guard models.

Owns every transformers model this service loads onto a GPU (Llama Guard 3
Vision, ShieldGemma 2).  It replaces the per-service ``_MODEL_CACHE`` dicts so
there is exactly one holder of loaded weights.

Phase-1 policy (conservative):
  * One exclusive lease per GPU: at most one managed model is loading,
    unloading or running inference on a device at any moment.  Requests queue
    FIFO, which also means "wait for the in-flight inference on the old model"
    falls out of the queue for free and a model that is in use can never be
    unloaded underneath a request.
  * A loaded model stays resident and is reused.  Before loading a model the
    manager probes the *live* free VRAM / RAM (the GPU is shared with other
    services, so nothing about available memory is hard-coded) and, if the
    target does not fit, unloads idle models LRU-first, re-probing after each.
    If it still does not fit, ``InsufficientResourcesError`` is raised — no
    retry loop.
  * Unloading drops the in-process references only.  The HuggingFace disk
    cache (HF_HOME) is never touched; reloading reads the same weights back.

Model specs are registered by the service modules; tests inject a fake
``ResourceProbe`` and fake loaders so none of this needs a real GPU.
"""
from __future__ import annotations

import gc
import itertools
import logging
import os
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

MB = 1024 * 1024


class ModelResourceError(RuntimeError):
    """Base class for resource-manager failures."""


class InsufficientResourcesError(ModelResourceError):
    """Even after unloading every idle model the target does not fit."""


class ModelBusyTimeoutError(ModelResourceError):
    """Timed out waiting for the GPU lease (another inference/load is running)."""


class ModelLoadError(ModelResourceError):
    """The model loader raised; manager state has been rolled back."""


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    IN_USE = "in_use"
    UNLOADING = "unloading"


@dataclass(frozen=True)
class ModelSpec:
    """How to load one model and how much memory it is expected to need.

    ``loader(device_index)`` returns a dict of the objects to keep resident
    (model, processor/tokenizer, ...).  Estimates are only the starting point;
    after the first successful load the measured footprint replaces
    ``vram_bytes``.
    """

    key: str
    loader: Callable[[int], dict]
    vram_bytes: int
    inference_reserve_bytes: int
    ram_bytes: int = 0


@dataclass
class _Entry:
    spec: ModelSpec
    state: ModelState = ModelState.UNLOADED
    objects: dict | None = None
    footprint_bytes: int | None = None
    last_used: float = 0.0
    in_use: int = 0
    last_error: str | None = None

    @property
    def vram_needed(self) -> int:
        return self.footprint_bytes if self.footprint_bytes is not None else self.spec.vram_bytes


@dataclass
class ResourceSnapshot:
    gpu_free_bytes: int
    gpu_total_bytes: int
    ram_available_bytes: int | None


class ResourceProbe:
    """Live GPU / RAM measurements.  Every call re-reads the device."""

    def __init__(
        self,
        meminfo_path: str = "/proc/meminfo",
        cgroup_dir: str = "/sys/fs/cgroup",
    ) -> None:
        self._meminfo_path = meminfo_path
        self._cgroup_dir = cgroup_dir

    # GPU ---------------------------------------------------------------------

    def cuda_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()

    def gpu_identity(self, device_index: int) -> str:
        import torch

        props = torch.cuda.get_device_properties(device_index)
        uuid = getattr(props, "uuid", None)
        return f"cuda:{device_index} {props.name}" + (f" GPU-{uuid}" if uuid else "")

    def gpu_free_bytes(self, device_index: int) -> int:
        """Device-wide free memory (other processes included) plus memory this
        process' caching allocator has reserved but is not using.

        On a unified-memory GPU (`is_integrated`, e.g. GX10/Grace-Blackwell,
        where CPU and GPU share one physical pool) `cudaMemGetInfo` reports
        the host's raw MemFree, not MemAvailable -- it doesn't know Linux's
        reclaimable buff/cache can be freed on demand. Confirmed live on
        GX10: `free -h` showed 2.1GiB free / 67GiB available while this same
        call agreed with the 2.1GiB figure, causing a real model load to be
        refused as "insufficient GPU memory" when 67GiB was actually usable.
        Doesn't affect a discrete GPU (`is_integrated` is False there; VRAM
        has no page-cache concept to begin with, so this never fires)."""
        import torch

        free, _total = torch.cuda.mem_get_info(device_index)
        if getattr(torch.cuda.get_device_properties(device_index), "is_integrated", False):
            host_available = self.ram_available_bytes()
            if host_available is not None:
                free = max(free, host_available)
        cached_unused = torch.cuda.memory_reserved(device_index) - torch.cuda.memory_allocated(device_index)
        return int(free + max(cached_unused, 0))

    def gpu_total_bytes(self, device_index: int) -> int:
        import torch

        return int(torch.cuda.mem_get_info(device_index)[1])

    def gpu_allocated_bytes(self, device_index: int) -> int:
        import torch

        return int(torch.cuda.memory_allocated(device_index))

    def release_cached(self, device_index: int) -> None:
        """Return the allocator's unused cache to the driver.  Only meaningful
        after every reference to the model's tensors has been dropped."""
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_index)
            torch.cuda.empty_cache()

    # RAM ---------------------------------------------------------------------

    def ram_available_bytes(self) -> int | None:
        """min(MemAvailable, cgroup v2 headroom); None when unreadable."""
        candidates = []
        try:
            with open(self._meminfo_path) as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        candidates.append(int(line.split()[1]) * 1024)
                        break
        except OSError:
            pass
        try:
            with open(os.path.join(self._cgroup_dir, "memory.max")) as f:
                limit = f.read().strip()
            if limit != "max":
                with open(os.path.join(self._cgroup_dir, "memory.current")) as f:
                    current = int(f.read().strip())
                candidates.append(max(int(limit) - current, 0))
        except (OSError, ValueError):
            pass
        return min(candidates) if candidates else None

    def snapshot(self, device_index: int) -> ResourceSnapshot:
        return ResourceSnapshot(
            gpu_free_bytes=self.gpu_free_bytes(device_index),
            gpu_total_bytes=self.gpu_total_bytes(device_index),
            ram_available_bytes=self.ram_available_bytes(),
        )


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


class ModelResourceManager:
    """Per-GPU owner of loaded models.  Thread-safe; callers run in worker threads."""

    def __init__(
        self,
        device_index: int = 0,
        probe: ResourceProbe | None = None,
        *,
        safety_margin_bytes: int = 256 * MB,
        allow_coexist: bool = True,
        acquire_timeout_s: float = 300.0,
        is_oom: Callable[[BaseException], bool] = _is_cuda_oom,
    ) -> None:
        self.device_index = device_index
        self.probe = probe or ResourceProbe()
        self.safety_margin_bytes = safety_margin_bytes
        self.allow_coexist = allow_coexist
        self.acquire_timeout_s = acquire_timeout_s
        self._is_oom = is_oom
        self._entries: dict[str, _Entry] = {}
        self._cond = threading.Condition()
        self._tickets = itertools.count()
        self._now_serving = 0
        self._abandoned: set[int] = set()
        self._identity_logged = False

    # registration / introspection --------------------------------------------

    def register(self, spec: ModelSpec) -> None:
        with self._cond:
            existing = self._entries.get(spec.key)
            if existing is not None:
                # Keep the resident model and measured footprint; a re-register
                # (every request does one) only refreshes the recipe.
                existing.spec = spec
                return
            self._entries[spec.key] = _Entry(spec=spec)

    def state(self, key: str) -> ModelState:
        with self._cond:
            return self._entries[key].state

    def status(self) -> dict[str, dict[str, Any]]:
        with self._cond:
            return {
                k: {
                    "state": e.state.value,
                    "footprint_bytes": e.footprint_bytes,
                    "estimate_bytes": e.spec.vram_bytes,
                    "last_error": e.last_error,
                }
                for k, e in self._entries.items()
            }

    # lease -------------------------------------------------------------------

    @contextmanager
    def lease(self, key: str, timeout: float | None = None) -> Iterator[dict]:
        """Exclusive GPU lease; yields the loaded objects dict for ``key``."""
        timeout = self.acquire_timeout_s if timeout is None else timeout
        with self._cond:
            if key not in self._entries:
                raise KeyError(f"model '{key}' is not registered")
            ticket = next(self._tickets)
            entry = self._entries[key]
        self._wait_for_turn(ticket, key, timeout)
        try:
            self._ensure_ready(entry)
            entry.state = ModelState.IN_USE
            entry.in_use += 1
            oom_text: str | None = None
            try:
                yield entry.objects  # type: ignore[misc]
            except Exception as exc:
                if not self._is_oom(exc):
                    raise
                # Activations are referenced by the traceback frames; free them
                # outside this block.  Weights are unaffected, so the model stays.
                oom_text = str(exc)
            finally:
                entry.in_use -= 1
                entry.state = ModelState.LOADED
                entry.last_used = time.monotonic()
            if oom_text is not None:
                self.probe.release_cached(self.device_index)
                raise InsufficientResourcesError(
                    f"CUDA out of memory during '{key}' inference on {self._device_label()}; "
                    f"model kept loaded: {oom_text}"
                )
        finally:
            self._release_turn()

    def _release_turn(self) -> None:
        with self._cond:
            self._now_serving += 1
            self._skip_abandoned()

    def _wait_for_turn(self, ticket: int, key: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._now_serving != ticket:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Give up our place without blocking the tickets behind us.
                    self._abandon(ticket)
                    raise ModelBusyTimeoutError(
                        f"timed out after {timeout:.0f}s waiting for {self._device_label()} "
                        f"to run '{key}' (another image-guard request holds the GPU)"
                    )
                self._cond.wait(remaining)

    def _abandon(self, ticket: int) -> None:
        # Lock held.  A timed-out ticket is skipped when the queue reaches it,
        # so the waiters behind it are not blocked.
        self._abandoned.add(ticket)
        self._skip_abandoned()

    def _skip_abandoned(self) -> None:
        # Lock held.
        while self._now_serving in self._abandoned:
            self._abandoned.discard(self._now_serving)
            self._now_serving += 1
        self._cond.notify_all()

    # resource decisions (lease held) -------------------------------------------

    def _device_label(self) -> str:
        return f"cuda:{self.device_index}"

    def _ensure_ready(self, entry: _Entry) -> None:
        if not self.probe.cuda_available():
            raise InsufficientResourcesError(
                f"no CUDA device available for '{entry.spec.key}' (expected {self._device_label()})"
            )
        if not self._identity_logged:
            logger.info("model-manager: managing %s", self.probe.gpu_identity(self.device_index))
            self._identity_logged = True

        if entry.state == ModelState.LOADED:
            need = entry.spec.inference_reserve_bytes + self.safety_margin_bytes
            self._make_room(entry, need, ram_need=0, purpose="inference")
            return

        need = entry.vram_needed + entry.spec.inference_reserve_bytes + self.safety_margin_bytes
        if not self.allow_coexist:
            for other in self._idle_others(entry):
                self._unload(other, reason=f"coexistence disabled, loading '{entry.spec.key}'")
        self._make_room(entry, need, ram_need=entry.spec.ram_bytes, purpose="load")
        self._load(entry)

    def _idle_others(self, entry: _Entry) -> list[_Entry]:
        others = [
            e for e in self._entries.values()
            if e is not entry and e.state == ModelState.LOADED and e.in_use == 0
        ]
        return sorted(others, key=lambda e: e.last_used)

    def _fits(self, vram_need: int, ram_need: int) -> tuple[bool, ResourceSnapshot]:
        snap = self.probe.snapshot(self.device_index)
        ram_ok = ram_need <= 0 or snap.ram_available_bytes is None or snap.ram_available_bytes >= ram_need
        return snap.gpu_free_bytes >= vram_need and ram_ok, snap

    def _make_room(self, entry: _Entry, vram_need: int, ram_need: int, purpose: str) -> None:
        unloaded: list[str] = []
        ok, snap = self._fits(vram_need, ram_need)
        for victim in self._idle_others(entry) if not ok else []:
            self._unload(victim, reason=f"need room to {purpose} '{entry.spec.key}'")
            unloaded.append(victim.spec.key)
            ok, snap = self._fits(vram_need, ram_need)
            if ok:
                break
        if ok:
            return
        ram = (
            f", RAM available {snap.ram_available_bytes / MB:.0f} MiB / needed {ram_need / MB:.0f} MiB"
            if ram_need and snap.ram_available_bytes is not None else ""
        )
        msg = (
            f"insufficient GPU memory to {purpose} '{entry.spec.key}' on {self._device_label()}: "
            f"needed {vram_need / MB:.0f} MiB (incl. inference reserve + safety margin), "
            f"free {snap.gpu_free_bytes / MB:.0f} MiB of {snap.gpu_total_bytes / MB:.0f} MiB{ram}; "
            f"unloaded idle models: {unloaded or 'none'}. "
            "The GPU is shared with other processes — free memory there or point "
            "GUARDRAIL_GPU_ID at a less loaded GPU."
        )
        logger.warning("model-manager: %s", msg)
        raise InsufficientResourcesError(msg)

    def _load(self, entry: _Entry) -> None:
        key = entry.spec.key
        entry.state = ModelState.LOADING
        before = self.probe.gpu_allocated_bytes(self.device_index)
        started = time.monotonic()
        error_text: str | None = None
        oom = False
        try:
            objects = entry.spec.loader(self.device_index)
        except Exception as exc:  # noqa: BLE001 -- the loader is an arbitrary caller-supplied callable; any failure must roll back state, logged
            # Keep only text: the traceback's frames are what still reference
            # the partially-built modules, so it must not outlive this block.
            error_text = f"{type(exc).__name__}: {exc}"
            oom = self._is_oom(exc)
            # Formatted text, not exc_info: a log handler that keeps records
            # would otherwise keep the frames (and the GPU tensors) alive.
            logger.error("model-manager: loading '%s' failed:\n%s", key, traceback.format_exc())
        if error_text is not None:
            entry.state = ModelState.UNLOADED
            entry.objects = None
            entry.last_error = error_text
            self.probe.release_cached(self.device_index)
            if oom:
                raise InsufficientResourcesError(
                    f"CUDA out of memory while loading '{key}' on {self._device_label()}: {error_text}"
                )
            raise ModelLoadError(f"failed to load '{key}': {error_text}")
        entry.objects = objects
        entry.footprint_bytes = max(self.probe.gpu_allocated_bytes(self.device_index) - before, 0) or None
        entry.state = ModelState.LOADED
        entry.last_error = None
        entry.last_used = time.monotonic()
        logger.info(
            "model-manager: loaded '%s' on %s in %.1fs (measured footprint %s MiB, estimate %.0f MiB)",
            key, self._device_label(), time.monotonic() - started,
            f"{entry.footprint_bytes / MB:.0f}" if entry.footprint_bytes else "n/a",
            entry.spec.vram_bytes / MB,
        )

    def _unload(self, entry: _Entry, reason: str) -> None:
        assert entry.in_use == 0, f"refusing to unload '{entry.spec.key}' while in use"
        key = entry.spec.key
        entry.state = ModelState.UNLOADING
        before = self.probe.gpu_allocated_bytes(self.device_index)
        objects, entry.objects = entry.objects, None
        if objects:
            model = objects.get("model")
            if model is not None:
                _remove_accelerate_hooks(model)
            objects.clear()
            del model
        del objects
        self.probe.release_cached(self.device_index)
        freed = before - self.probe.gpu_allocated_bytes(self.device_index)
        entry.state = ModelState.UNLOADED
        logger.info("model-manager: unloaded '%s' (%s); freed %.0f MiB", key, reason, freed / MB)
        if entry.footprint_bytes and freed < entry.footprint_bytes * 0.5:
            logger.warning(
                "model-manager: unloading '%s' freed only %.0f of %.0f MiB — "
                "something still references its tensors",
                key, freed / MB, entry.footprint_bytes / MB,
            )

    def unload_all(self, timeout: float | None = None) -> None:
        """Unload every idle model (used at shutdown)."""
        with self._cond:
            ticket = next(self._tickets)
        self._wait_for_turn(ticket, "<unload_all>", self.acquire_timeout_s if timeout is None else timeout)
        try:
            for entry in list(self._entries.values()):
                if entry.state == ModelState.LOADED and entry.in_use == 0:
                    self._unload(entry, reason="unload_all")
        finally:
            self._release_turn()


def _remove_accelerate_hooks(model: Any) -> None:
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return
    try:
        remove_hook_from_module(model, recurse=True)
    except Exception:  # best effort — hooks only hold references, never weights on disk
        logger.debug("model-manager: remove_hook_from_module failed", exc_info=True)


# ── process-wide singleton ────────────────────────────────────────────────────

def env_mb(name: str, default_mb: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(float(raw) * MB) if raw else default_mb * MB


_MANAGER: ModelResourceManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> ModelResourceManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = ModelResourceManager(
                device_index=int(os.environ.get("GUARDRAIL_CUDA_DEVICE", "0")),
                safety_margin_bytes=env_mb("GUARDRAIL_VRAM_SAFETY_MARGIN_MB", 256),
                allow_coexist=os.environ.get("GUARDRAIL_ALLOW_MODEL_COEXIST", "true").lower() == "true",
                acquire_timeout_s=float(os.environ.get("GUARDRAIL_MODEL_ACQUIRE_TIMEOUT_S", "300")),
            )
        return _MANAGER


def shutdown_manager(timeout: float = 30.0) -> None:
    """Unload resident models at service shutdown; no-op if none were ever used."""
    with _MANAGER_LOCK:
        manager = _MANAGER
    if manager is None:
        return
    try:
        manager.unload_all(timeout=timeout)
    except ModelBusyTimeoutError:
        logger.warning("model-manager: shutdown while an inference was still running; skipping unload")


def reset_manager_for_tests(manager: ModelResourceManager | None = None) -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager
