"""Unit tests for app/services/model_resource_manager.py — fake GPU probe and
fake loaders, no torch / CUDA needed.

The fake GPU counts a model's bytes as allocated for as long as the model
object is alive (WeakSet), so "unloaded" only shows up as free memory once the
manager has genuinely dropped every reference to it.
"""
import gc
import threading
import time
import weakref

import pytest
from app.services import model_resource_manager as mrm
from app.services.model_resource_manager import (
    InsufficientResourcesError,
    ModelBusyTimeoutError,
    ModelLoadError,
    ModelResourceManager,
    ModelSpec,
    ModelState,
    ResourceProbe,
)

MB = mrm.MB


class FakeOOM(Exception):
    pass


class FakeModel:
    def __init__(self, gpu: "FakeGPU", size: int, name: str) -> None:
        self.size = size
        self.name = name
        gpu.live.add(self)


class FakeGPU(ResourceProbe):
    def __init__(self, total_mb: int, other_mb: int = 0, ram_mb: int | None = 64_000) -> None:
        self.total = total_mb * MB
        self.other = other_mb * MB          # memory held by other processes
        self.ram = None if ram_mb is None else ram_mb * MB
        self.live: weakref.WeakSet[FakeModel] = weakref.WeakSet()
        self.release_calls = 0
        self.peak_used = 0

    def cuda_available(self) -> bool:
        return True

    def gpu_identity(self, device_index: int) -> str:
        return f"cuda:{device_index} FakeGPU"

    def allocated(self) -> int:
        gc.collect()
        used = sum(m.size for m in list(self.live))
        self.peak_used = max(self.peak_used, used + self.other)
        return used

    def gpu_free_bytes(self, device_index: int) -> int:
        return self.total - self.other - self.allocated()

    def gpu_total_bytes(self, device_index: int) -> int:
        return self.total

    def gpu_allocated_bytes(self, device_index: int) -> int:
        return self.allocated()

    def release_cached(self, device_index: int) -> None:
        self.release_calls += 1
        gc.collect()

    def ram_available_bytes(self):
        return self.ram


def make_spec(gpu, key, size_mb, *, reserve_mb=500, events=None, gate=None, fail=None, ram_mb=0):
    calls = {"n": 0}

    def loader(device_index):
        calls["n"] += 1
        if events is not None:
            events.append(f"load:{key}")
        if gate is not None:
            gate.wait(5)
        if fail is not None and fail():
            # Simulate a partially built model that only the traceback references.
            partial = FakeModel(gpu, size_mb * MB, key)  # noqa: F841
            raise RuntimeError(f"boom loading {key}")
        return {"model": FakeModel(gpu, size_mb * MB, key), "processor": object()}

    spec = ModelSpec(
        key=key, loader=loader, vram_bytes=size_mb * MB,
        inference_reserve_bytes=reserve_mb * MB, ram_bytes=ram_mb * MB,
    )
    return spec, calls


def make_manager(gpu, **kw):
    kw.setdefault("safety_margin_bytes", 100 * MB)
    kw.setdefault("acquire_timeout_s", 5)
    return ModelResourceManager(0, gpu, is_oom=lambda e: isinstance(e, FakeOOM), **kw)


def run_in_thread(fn):
    out = {}

    def target():
        try:
            out["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    return t, out


# ── reuse / residency ────────────────────────────────────────────────────────

def test_loaded_model_is_reused_not_reloaded():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    spec, calls = make_spec(gpu, "a", 5_000)
    mgr.register(spec)
    for _ in range(3):
        with mgr.lease("a") as objs:
            assert objs["model"].name == "a"
    assert calls["n"] == 1
    assert mgr.state("a") == ModelState.LOADED


def test_reregister_keeps_resident_model_and_footprint():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    spec, calls = make_spec(gpu, "a", 5_000)
    mgr.register(spec)
    with mgr.lease("a"):
        pass
    mgr.register(spec)
    with mgr.lease("a"):
        pass
    assert calls["n"] == 1
    assert mgr.status()["a"]["footprint_bytes"] == 5_000 * MB


def test_models_coexist_when_memory_allows():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    a, a_calls = make_spec(gpu, "a", 5_000)
    b, b_calls = make_spec(gpu, "b", 5_000)
    mgr.register(a)
    mgr.register(b)
    with mgr.lease("a"):
        pass
    with mgr.lease("b"):
        pass
    with mgr.lease("a"):
        pass
    assert mgr.state("a") == ModelState.LOADED
    assert mgr.state("b") == ModelState.LOADED
    assert (a_calls["n"], b_calls["n"]) == (1, 1)


def test_coexist_disabled_unloads_idle_model_first():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu, allow_coexist=False)
    a, _ = make_spec(gpu, "a", 5_000)
    b, _ = make_spec(gpu, "b", 5_000)
    mgr.register(a)
    mgr.register(b)
    with mgr.lease("a"):
        pass
    with mgr.lease("b"):
        pass
    assert mgr.state("a") == ModelState.UNLOADED
    assert mgr.state("b") == ModelState.LOADED


# ── switching under pressure ─────────────────────────────────────────────────

def test_switch_waits_for_inflight_inference_then_unloads_and_loads():
    # 16 GB card, 7 GB taken by another process: only one 6 GB model fits.
    gpu = FakeGPU(16_000, other_mb=7_000)
    mgr = make_manager(gpu)
    events: list[str] = []
    a, _ = make_spec(gpu, "a", 6_000, events=events)
    b, b_calls = make_spec(gpu, "b", 6_000, events=events)
    mgr.register(a)
    mgr.register(b)

    a_running = threading.Event()
    a_release = threading.Event()

    def use_a():
        with mgr.lease("a"):
            events.append("infer:a:start")
            a_running.set()
            a_release.wait(5)
            events.append("infer:a:end")

    def use_b():
        with mgr.lease("b") as objs:
            events.append(f"infer:b:a_state={mgr._entries['a'].state.value}")
            return objs["model"].name

    ta, out_a = run_in_thread(use_a)
    assert a_running.wait(5)
    tb, out_b = run_in_thread(use_b)
    time.sleep(0.2)
    # B must be queued, not loading, while A is still inferring.
    assert b_calls["n"] == 0
    assert mgr.state("a") == ModelState.IN_USE
    a_release.set()
    ta.join(5)
    tb.join(5)

    assert "error" not in out_a and "error" not in out_b
    assert out_b["result"] == "b"
    assert events.index("infer:a:end") < events.index("load:b")
    assert "infer:b:a_state=unloaded" in events
    assert mgr.state("a") == ModelState.UNLOADED
    assert gpu.peak_used <= gpu.total


def test_still_insufficient_after_unload_raises_clear_error_without_retry():
    gpu = FakeGPU(16_000, other_mb=7_000)
    mgr = make_manager(gpu)
    a, _ = make_spec(gpu, "a", 4_000)
    big, big_calls = make_spec(gpu, "big", 12_000)
    mgr.register(a)
    mgr.register(big)
    with mgr.lease("a"):
        pass

    with pytest.raises(InsufficientResourcesError) as ei:
        with mgr.lease("big"):
            pass
    msg = str(ei.value)
    assert "'big'" in msg and "needed" in msg and "free" in msg and "cuda:0" in msg
    assert "unloaded idle models: ['a']" in msg
    assert big_calls["n"] == 0
    assert mgr.state("big") == ModelState.UNLOADED
    # Manager is still usable afterwards.
    with mgr.lease("a"):
        pass


def test_insufficient_ram_is_reported():
    gpu = FakeGPU(16_000, ram_mb=1_000)
    mgr = make_manager(gpu)
    a, calls = make_spec(gpu, "a", 4_000, ram_mb=4_000)
    mgr.register(a)
    with pytest.raises(InsufficientResourcesError, match="RAM available"):
        with mgr.lease("a"):
            pass
    assert calls["n"] == 0


def test_loaded_model_without_inference_headroom_evicts_others_then_errors():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    a, _ = make_spec(gpu, "a", 6_000)
    b, _ = make_spec(gpu, "b", 6_000)
    mgr.register(a)
    mgr.register(b)
    with mgr.lease("a"):
        pass
    with mgr.lease("b"):
        pass
    # Another process grabs memory: 16000 - 12000 - 3800 = 200 MB free < 600 needed.
    gpu.other = 3_800 * MB
    with mgr.lease("b"):
        pass  # evicting idle 'a' makes room
    assert mgr.state("a") == ModelState.UNLOADED
    gpu.other = 9_900 * MB
    with pytest.raises(InsufficientResourcesError, match="inference"):
        with mgr.lease("b"):
            pass
    assert mgr.state("b") == ModelState.LOADED


# ── concurrency ──────────────────────────────────────────────────────────────

def test_concurrent_requests_for_same_model_load_once():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    gate = threading.Event()
    a, calls = make_spec(gpu, "a", 5_000, gate=gate)
    mgr.register(a)

    def use():
        with mgr.lease("a") as objs:
            return id(objs["model"])

    threads = [run_in_thread(use) for _ in range(10)]
    time.sleep(0.1)
    gate.set()
    for t, _ in threads:
        t.join(5)
    results = {out["result"] for _, out in threads}
    assert calls["n"] == 1
    assert len(results) == 1


def test_concurrent_different_models_never_unload_in_use_or_overcommit():
    gpu = FakeGPU(16_000, other_mb=7_000)
    # 16 queued leases that each swap a/b: the last ticket waits for all the
    # others: ~4.5s on an idle machine and ~10s on a busy runner, so the
    # default 5s test timeout made it flaky.
    mgr = make_manager(gpu, acquire_timeout_s=60)
    a, _ = make_spec(gpu, "a", 6_000)
    b, _ = make_spec(gpu, "b", 6_000)
    mgr.register(a)
    mgr.register(b)
    active = {"n": 0, "max": 0}
    lock = threading.Lock()
    violations = []

    def use(key):
        with mgr.lease(key) as objs:
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            model = objs["model"]
            time.sleep(0.01)
            if model not in gpu.live:
                violations.append(key)
            with lock:
                active["n"] -= 1

    threads = [run_in_thread(lambda k=k: use(k)) for k in ["a", "b"] * 8]
    for t, _ in threads:
        t.join(60)
    assert not any(t.is_alive() for t, _ in threads)
    assert all("error" not in out for _, out in threads)
    assert violations == []
    assert active["max"] == 1          # phase 1: one inference per GPU at a time
    assert gpu.peak_used <= gpu.total


def test_unload_refuses_model_in_use():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    a, _ = make_spec(gpu, "a", 5_000)
    mgr.register(a)
    with mgr.lease("a"):
        with pytest.raises(AssertionError):
            mgr._unload(mgr._entries["a"], reason="test")


def test_wait_timeout_raises_busy_and_queue_recovers():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    a, _ = make_spec(gpu, "a", 5_000)
    mgr.register(a)
    hold = threading.Event()
    started = threading.Event()

    def holder():
        with mgr.lease("a"):
            started.set()
            hold.wait(5)

    t, _ = run_in_thread(holder)
    assert started.wait(5)
    with pytest.raises(ModelBusyTimeoutError):
        with mgr.lease("a", timeout=0.1):
            pass
    hold.set()
    t.join(5)
    with mgr.lease("a", timeout=1):  # the abandoned ticket does not block the queue
        pass


# ── failures ─────────────────────────────────────────────────────────────────

def test_load_failure_releases_memory_and_resets_state():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    fail = {"on": True}
    a, calls = make_spec(gpu, "a", 5_000, fail=lambda: fail["on"])
    mgr.register(a)

    with pytest.raises(ModelLoadError, match="boom loading a"):
        with mgr.lease("a"):
            pass
    assert mgr.state("a") == ModelState.UNLOADED
    assert mgr.status()["a"]["last_error"].startswith("RuntimeError")
    assert gpu.allocated() == 0       # partial model freed
    assert gpu.release_calls >= 1

    fail["on"] = False
    with mgr.lease("a") as objs:
        assert objs["model"].name == "a"
    assert calls["n"] == 2
    assert mgr.status()["a"]["last_error"] is None


def test_oom_during_load_reports_insufficient_resources():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)

    def loader(_):
        raise FakeOOM("CUDA out of memory")

    mgr.register(ModelSpec("a", loader, 1_000 * MB, 100 * MB))
    with pytest.raises(InsufficientResourcesError, match="out of memory while loading"):
        with mgr.lease("a"):
            pass
    assert mgr.state("a") == ModelState.UNLOADED


def test_oom_during_inference_keeps_model_and_reports():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    a, calls = make_spec(gpu, "a", 5_000)
    mgr.register(a)
    with pytest.raises(InsufficientResourcesError, match="during 'a' inference"):
        with mgr.lease("a"):
            raise FakeOOM("CUDA out of memory")
    assert mgr.state("a") == ModelState.LOADED
    with mgr.lease("a"):
        pass
    assert calls["n"] == 1


def test_other_inference_errors_propagate_unchanged():
    gpu = FakeGPU(16_000)
    mgr = make_manager(gpu)
    a, _ = make_spec(gpu, "a", 5_000)
    mgr.register(a)
    with pytest.raises(ValueError, match="bad image"):
        with mgr.lease("a"):
            raise ValueError("bad image")
    assert mgr.state("a") == ModelState.LOADED


def test_unload_never_touches_disk_cache(monkeypatch):
    import os
    import shutil

    def forbidden(*a, **k):
        raise AssertionError("unload must not delete files")

    monkeypatch.setattr(shutil, "rmtree", forbidden)
    monkeypatch.setattr(os, "remove", forbidden)
    monkeypatch.setattr(os, "unlink", forbidden)
    gpu = FakeGPU(16_000, other_mb=7_000)
    mgr = make_manager(gpu)
    a, _ = make_spec(gpu, "a", 6_000)
    b, _ = make_spec(gpu, "b", 6_000)
    mgr.register(a)
    mgr.register(b)
    with mgr.lease("a"):
        pass
    with mgr.lease("b"):
        pass
    mgr.unload_all()
    assert gpu.allocated() == 0


def test_no_cuda_reports_insufficient_resources():
    gpu = FakeGPU(16_000)
    gpu.cuda_available = lambda: False
    mgr = make_manager(gpu)
    a, calls = make_spec(gpu, "a", 5_000)
    mgr.register(a)
    with pytest.raises(InsufficientResourcesError, match="no CUDA device"):
        with mgr.lease("a"):
            pass
    assert calls["n"] == 0


def test_unregistered_model_raises_keyerror():
    mgr = make_manager(FakeGPU(16_000))
    with pytest.raises(KeyError):
        with mgr.lease("missing"):
            pass


# ── real probe: RAM parsing ─────────────────────────────────────────────────

def test_probe_ram_uses_min_of_meminfo_and_cgroup(tmp_path):
    (tmp_path / "meminfo").write_text("MemTotal: 100000 kB\nMemAvailable:   8000000 kB\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(str(6 * 1024**3))
    (cg / "memory.current").write_text(str(2 * 1024**3))
    probe = ResourceProbe(meminfo_path=str(tmp_path / "meminfo"), cgroup_dir=str(cg))
    assert probe.ram_available_bytes() == 4 * 1024**3


def test_probe_ram_unlimited_cgroup_and_missing_files(tmp_path):
    (tmp_path / "meminfo").write_text("MemAvailable: 1024 kB\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text("max\n")
    probe = ResourceProbe(meminfo_path=str(tmp_path / "meminfo"), cgroup_dir=str(cg))
    assert probe.ram_available_bytes() == 1024 * 1024
    probe = ResourceProbe(meminfo_path=str(tmp_path / "nope"), cgroup_dir=str(tmp_path / "nope"))
    assert probe.ram_available_bytes() is None


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_CUDA_DEVICE", "1")
    monkeypatch.setenv("GUARDRAIL_VRAM_SAFETY_MARGIN_MB", "512")
    monkeypatch.setenv("GUARDRAIL_ALLOW_MODEL_COEXIST", "false")
    monkeypatch.setenv("GUARDRAIL_MODEL_ACQUIRE_TIMEOUT_S", "12")
    mrm.reset_manager_for_tests()
    try:
        mgr = mrm.get_manager()
        assert mgr.device_index == 1
        assert mgr.safety_margin_bytes == 512 * MB
        assert mgr.allow_coexist is False
        assert mgr.acquire_timeout_s == 12
        assert mrm.get_manager() is mgr
    finally:
        mrm.reset_manager_for_tests()
