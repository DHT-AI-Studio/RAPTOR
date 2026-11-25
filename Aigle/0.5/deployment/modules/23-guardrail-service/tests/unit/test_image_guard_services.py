"""Image-guard services and routers on top of the resource manager — model
loading is replaced by fakes, no torch / transformers / GPU needed."""
import io
import sys
import types
from typing import ClassVar

import pytest
from app.routers import infer_llama_guard_vision, infer_shieldgemma2
from app.services import llama_guard_vision_inference as lg
from app.services import model_resource_manager as mrm
from app.services import shieldgemma2_inference as sg
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.unit.test_model_resource_manager import FakeGPU, FakeModel, make_manager

MB = mrm.MB


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def gpu(monkeypatch):
    gpu = FakeGPU(16_000, other_mb=7_000)
    mrm.reset_manager_for_tests(make_manager(gpu))
    yield gpu
    mrm.reset_manager_for_tests()


class _FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, *_a, **_k):
        return self

    def __iter__(self):
        return iter(_Row(r) for r in self.rows)

    def __getitem__(self, item):
        return self.rows[item] if isinstance(item, int) else self


class _Row(list):
    def tolist(self):
        return list(self)


class _Inputs(dict):
    def to(self, *_a, **_k):
        return self


class FakeSGModel(FakeModel):
    device = "cuda:0"
    dtype = "bfloat16"

    def __call__(self, **inputs):
        return types.SimpleNamespace(probabilities=_FakeTensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]))


class FakeSGProcessor:
    policy_definitions: ClassVar[dict] = {"dangerous": "d", "sexual": "s", "violence": "v"}

    def __call__(self, images, return_tensors):
        return _Inputs(pixel_values=_FakeTensor([]), input_ids=_FakeTensor([]))


class FakeLGModel(FakeModel):
    device = "cuda:0"

    def generate(self, **inputs):
        return _FakeTensor([[1, 2, 3, 4]])


class FakeLGProcessor:
    def apply_chat_template(self, conversation, return_tensors):
        return "prompt"

    def __call__(self, text, images, return_tensors):
        return _Inputs(input_ids=[[1, 2]])

    def decode(self, tokens, skip_special_tokens):
        return "unsafe\nS10"


@pytest.fixture
def fake_torch(monkeypatch):
    import contextlib

    fake = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def _patch_loaders(monkeypatch, gpu, loads):
    def sg_loader(model_path, device_index):
        loads.append(("sg", model_path, device_index))
        return {"model": FakeSGModel(gpu, 4_900 * MB, "sg"), "processor": FakeSGProcessor()}

    def lg_loader(model_path, device_index):
        loads.append(("lg", model_path, device_index))
        return {"model": FakeLGModel(gpu, 7_000 * MB, "lg"), "processor": FakeLGProcessor()}

    monkeypatch.setattr(sg, "_load_model", sg_loader)
    monkeypatch.setattr(lg, "_load_model", lg_loader)


def test_shieldgemma_result_shape_unchanged_and_model_reused(monkeypatch, gpu, fake_torch):
    loads = []
    _patch_loaders(monkeypatch, gpu, loads)
    r1 = sg.run_shieldgemma2_inference("google/shieldgemma-2-4b-it", _png())
    r2 = sg.run_shieldgemma2_inference("google/shieldgemma-2-4b-it", _png())
    assert r1 == r2
    assert r1["overall_verdict"] == "unsafe"
    assert set(r1["policies"]) == {"dangerous", "sexual", "violence"}
    assert r1["policies"]["dangerous"] == {
        "unsafe_probability": 0.9, "safe_probability": 0.1,
        "verdict": "unsafe", "policy_definition": "d",
    }
    assert loads == [("sg", "google/shieldgemma-2-4b-it", 0)]


def test_switching_models_swaps_on_a_tight_gpu(monkeypatch, gpu, fake_torch):
    loads = []
    _patch_loaders(monkeypatch, gpu, loads)
    sg.run_shieldgemma2_inference("sg-path", _png())
    assert lg.run_llama_guard_vision_inference("lg-path", _png(), None) == "unsafe\nS10"
    sg.run_shieldgemma2_inference("sg-path", _png())
    assert [l[0] for l in loads] == ["sg", "lg", "sg"]
    assert gpu.peak_used <= gpu.total
    states = {k.split(":")[0]: v["state"] for k, v in mrm.get_manager().status().items()}
    assert states == {"shieldgemma-2-4b-it": "loaded", "llama-guard-3-11b-vision": "unloaded"}


def test_quantization_mode_is_part_of_the_cache_key(monkeypatch, gpu, fake_torch):
    loads = []
    _patch_loaders(monkeypatch, gpu, loads)
    sg.run_shieldgemma2_inference("sg-path", _png())
    monkeypatch.setenv("SHIELDGEMMA2_LOAD_IN_8BIT", "false")
    monkeypatch.setenv("SHIELDGEMMA2_VRAM_ESTIMATE_MB", "5000")
    sg.run_shieldgemma2_inference("sg-path", _png())
    assert len(loads) == 2
    assert any(k.endswith(":bf16") for k in mrm.get_manager().status())


def _client():
    app = FastAPI()
    app.include_router(infer_llama_guard_vision.router)
    app.include_router(infer_shieldgemma2.router)
    return TestClient(app)


@pytest.mark.parametrize("exc_cls", [mrm.InsufficientResourcesError, mrm.ModelBusyTimeoutError])
def test_routers_map_resource_errors_to_503(monkeypatch, exc_cls):
    def boom(*_a, **_k):
        raise exc_cls("needed 8000 MiB, free 1000 MiB")

    monkeypatch.setattr(infer_shieldgemma2, "run_shieldgemma2_inference", boom)
    monkeypatch.setattr(infer_llama_guard_vision, "run_llama_guard_vision_inference", boom)
    client = _client()
    files = {"image": ("x.png", _png(), "image/png")}
    r = client.post("/infer/google-shieldgemma-2", files=files)
    assert r.status_code == 503
    assert "free 1000 MiB" in r.json()["detail"]
    r = client.post("/infer/Llama-Guard-3-11B-Vision", files=files)
    assert r.status_code == 503


def test_routers_keep_502_for_other_failures(monkeypatch):
    def boom(*_a, **_k):
        raise mrm.ModelLoadError("failed to load")

    monkeypatch.setattr(infer_shieldgemma2, "run_shieldgemma2_inference", boom)
    monkeypatch.setattr(infer_llama_guard_vision, "run_llama_guard_vision_inference", boom)
    client = _client()
    files = {"image": ("x.png", _png(), "image/png")}
    assert client.post("/infer/google-shieldgemma-2", files=files).status_code == 502
    assert client.post("/infer/Llama-Guard-3-11B-Vision", files=files).status_code == 502


def test_llama_guard_router_response_shape_unchanged(monkeypatch):
    monkeypatch.setattr(
        infer_llama_guard_vision, "run_llama_guard_vision_inference",
        lambda *_a, **_k: "unsafe\nS1,S10",
    )
    r = _client().post(
        "/infer/Llama-Guard-3-11B-Vision", files={"image": ("x.png", _png(), "image/png")}
    )
    assert r.status_code == 200
    assert r.json() == {"label": "unsafe", "categories": ["S1", "S10"], "raw": "unsafe\nS1,S10"}
