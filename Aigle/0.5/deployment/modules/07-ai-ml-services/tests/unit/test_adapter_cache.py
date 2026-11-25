"""M07-FIX-04: version-aware adapter cache, in-flight accounting, unload semantics."""
import gc
import threading
import time
from unittest.mock import patch

from src.inference.adapters.base import BaseAdapter
from src.inference.adapters.ollama import OllamaAdapter
from src.inference.spec import ModelSpec


def _spec(version="1", name="m", path=None, runtime="hf-transformers", **kw):
    return ModelSpec(model_name=name, version=version, task_family="text-generation",
                     runtime=runtime, physical_path=path or f"/models/{name}/{version}", **kw)


class FakeAdapter(BaseAdapter):
    runtime = "fake"

    def __init__(self, **kw):
        super().__init__(config={}, max_cached_models=kw.pop("max_cached", 4), default_idle_timeout=300)
        self.loads = []
        self.unloaded = []
        self.load_delay = 0.0
        self.gate = None  # threading.Event to hold an inference in flight

    def load_model(self, spec):
        time.sleep(self.load_delay)
        self.loads.append(spec.version)
        return {"id": f"{spec.model_name}@{spec.version}#{len(self.loads)}"}

    def infer(self, model, spec, data, options):
        if self.gate is not None:
            self.gate.wait(5)
        return {"model_id": model["id"]}

    def infer_stream(self, model, spec, data, options):
        def gen():
            yield "a"
            yield {"metadata": {}}
        return gen()

    def _on_unload(self, model):
        self.unloaded.append(model["id"])


def test_same_name_different_versions_do_not_share_instances():
    a = FakeAdapter()
    r1 = a.run(_spec("1"), {}, {})
    r2 = a.run(_spec("2"), {}, {})
    r1b = a.run(_spec("1"), {}, {})
    assert r1["model_id"] != r2["model_id"]
    assert r1b["model_id"] == r1["model_id"]
    assert a.loads == ["1", "2"]
    info = {(i["model_name"], i["model_version"]) for i in a.loaded_models_info()}
    assert info == {("m", "1"), ("m", "2")}
    assert a.loaded_models() == ["m"]  # by-name view stays usable


def test_load_parameters_are_part_of_identity():
    a = FakeAdapter()
    a.run(_spec("1", quantization="4bit"), {}, {})
    a.run(_spec("1"), {}, {})
    assert a.loads == ["1", "1"]


def test_concurrent_requests_load_once_per_version():
    a = FakeAdapter()
    a.load_delay = 0.1
    results = []
    threads = [threading.Thread(target=lambda: results.append(a.run(_spec("1"), {}, {}))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert a.loads == ["1"] and len({r["model_id"] for r in results}) == 1


def test_unload_by_name_defers_release_while_in_flight_and_new_requests_reload():
    a = FakeAdapter()
    a.run(_spec("1"), {}, {})
    a.gate = threading.Event()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", a.run(_spec("1"), {}, {})))
    t.start()
    time.sleep(0.05)

    assert a.unload("m") is True
    assert a.unloaded == []  # still in flight -> not released yet
    assert any(i["pending_unload"] for i in a.loaded_models_info())

    a.gate.set()
    t.join()
    assert a.unloaded == ["m@1#1"]  # released when the last request finished
    assert out["r"]["model_id"] == "m@1#1"  # the running request completed on the old instance
    assert a.run(_spec("1"), {}, {})["model_id"] == "m@1#2"


def test_keep_alive_zero_unloads_only_that_version():
    a = FakeAdapter()
    a.run(_spec("2"), {}, {})
    a.run(_spec("1"), {}, {"keep_alive": 0})
    assert a.unloaded == ["m@1#2"]
    assert [i["model_version"] for i in a.loaded_models_info()] == ["2"]


def test_idle_eviction_skips_in_flight_and_uses_cache_keys():
    a = FakeAdapter()
    a.run(_spec("1"), {}, {"keep_alive": 1})
    evicted = a.evict_idle(now=time.time() + 5)
    assert len(evicted) == 1 and evicted[0].startswith("m@1|")
    assert a.unloaded == ["m@1#1"]


def test_stream_settles_on_completion_and_when_never_iterated():
    a = FakeAdapter()
    gen = a.run_stream(_spec("1"), {}, {})
    assert a.loaded_models_info()[0]["in_flight"] == 1
    assert list(gen) == ["a", {"metadata": {}}]
    assert a.loaded_models_info()[0]["in_flight"] == 0

    gen = a.run_stream(_spec("1"), {}, {})
    assert a.loaded_models_info()[0]["in_flight"] == 1
    del gen
    gc.collect()
    assert a.loaded_models_info()[0]["in_flight"] == 0


# ----- Ollama: several registry versions may point to the same daemon tag -----

class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_ollama_explicit_unload_keeps_daemon_model_used_by_other_version():
    ad = OllamaAdapter(config={"base_url": "http://ollama:11434", "keep_alive": 300})
    released = []
    gate = threading.Event()

    def fake_post(url, json=None, timeout=None, **kw):
        if json.get("keep_alive") == 0 and "prompt" not in json:
            released.append(json["model"])
            return _Resp({})
        gate.wait(5)
        return _Resp({"response": "ok", "done": True, "prompt_eval_count": 3, "eval_count": 1})

    tags = _Resp({"models": [{"name": "gemma4:12b"}]})
    v1 = _spec("1", path="gemma4:12b", runtime="ollama", ollama_model_name="gemma4:12b")
    v2 = _spec("2", path="gemma4:12b", runtime="ollama", ollama_model_name="gemma4:12b")
    with patch("src.inference.adapters.ollama.requests.get", return_value=tags), \
         patch("src.inference.adapters.ollama.requests.post", side_effect=fake_post):
        gate.set()
        ad.run(v1, {"inputs": "hi"}, {})
        gate.clear()
        t = threading.Thread(target=lambda: ad.run(v2, {"inputs": "hi"}, {}))
        t.start()
        time.sleep(0.05)
        ad.unload("m")  # unloads both cached versions of "m"
        assert released == []  # v2 still generating on the shared daemon model
        gate.set()
        t.join()
    # v2 was retired by the explicit unload; releasing it once idle asks the daemon to free the tag
    assert released == ["gemma4:12b"]
