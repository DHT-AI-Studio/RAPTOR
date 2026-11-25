"""M07-FIX-02/03: HF text generation (raw handler, pipeline route, streaming) with a
fake char-level tokenizer and a scripted model -- no weights, no GPU.

The fake model honours max_new_tokens, stopping_criteria and a streamer exactly
like transformers' generate() does, so these tests check what the helpers really
pass to generate() and how they count / trim the output.
"""
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.inference.adapters.hf_transformers import HFTransformersAdapter
from src.inference.exceptions import ResourceExhaustedError
from src.inference.handlers.text_generation import generation_kwargs
from src.inference.options import normalize_generation_options
from src.inference.spec import ModelSpec

BOS, EOS, OFFSET = 1, 2, 10


class FakeTokenizer:
    eos_token_id = EOS

    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        ids = ([BOS] if add_special_tokens else []) + [ord(c) + OFFSET for c in text]
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, skip_special_tokens=True, **kw):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = []
        for i in ids:
            if i < OFFSET:
                if not skip_special_tokens:
                    out.append("<s>" if i == BOS else "</s>")
                continue
            out.append(chr(i - OFFSET))
        return "".join(out)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.template_kwargs = kw
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return "<s>" + body + ("<assistant>" if add_generation_prompt else "")


class FakeModel:
    """Emits `script` then EOS, one token per step."""

    device = torch.device("cpu")

    def __init__(self, script, step_delay=0.0, error=None):
        self.script = script
        self.step_delay = step_delay
        self.error = error
        self.kwargs = None
        self.generated = 0
        self.finished = False

    def generate(self, input_ids, attention_mask=None, stopping_criteria=None, streamer=None, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        seq = input_ids
        if streamer is not None:
            streamer.put(input_ids)
        limit = kwargs.get("max_new_tokens", 10_000)
        for tok in [ord(c) + OFFSET for c in self.script] + [EOS]:
            if self.generated >= limit:
                break
            time.sleep(self.step_delay)
            seq = torch.cat([seq, torch.tensor([[tok]])], dim=-1)
            self.generated += 1
            if streamer is not None:
                streamer.put(torch.tensor([tok]))
            if tok == EOS:
                break
            if stopping_criteria is not None and bool(stopping_criteria(seq, None).all()):
                break
        if streamer is not None:
            streamer.end()
        self.finished = True
        return seq


def _spec(**kw):
    kw.setdefault("model_class", "AutoModelForCausalLM")
    return ModelSpec(model_name="hf", version="1", task_family="text-generation",
                     runtime="hf-transformers", physical_path="/models/hf", **kw)


def _loaded(model, tok=None):
    return {"kind": "model+processor", "model": model, "processor": tok or FakeTokenizer(), "device": "cpu"}


@pytest.fixture
def adapter():
    return HFTransformersAdapter(config={})


def _run(adapter, model, options, data=None, spec=None, tok=None):
    opts = normalize_generation_options(options)
    return adapter.infer(_loaded(model, tok), spec or _spec(), data or {"inputs": "Q:"}, opts)


def test_response_excludes_prompt_and_usage_counts_token_ids(adapter):
    model = FakeModel("Hi there")
    out = _run(adapter, model, {"max_tokens": 50})
    assert out["response"] == "Hi there"
    meta = out["metadata"]
    # prompt = BOS + "Q:" ; generated = 8 chars + EOS
    assert meta["prompt_tokens"] == 3 and meta["completion_tokens"] == 9
    assert meta["input_length"] == 3 and meta["output_length"] == 12  # legacy fields kept
    assert model.kwargs["max_new_tokens"] == 50


def test_max_tokens_actually_limits_generation(adapter):
    model = FakeModel("abcdefghij")
    out = _run(adapter, model, {"max_new_tokens": 4})
    assert out["response"] == "abcd" and out["metadata"]["completion_tokens"] == 4
    assert out["metadata"]["finish_reason"] == "length"


def test_stop_string_across_tokens_stops_generation(adapter):
    model = FakeModel("Hello END world and more")
    out = _run(adapter, model, {"stop_sequences": ["END"], "max_tokens": 100})
    assert out["response"] == "Hello "
    # generation really stopped right after "END" completed; the stop tokens are still counted
    assert model.generated == len("Hello END")
    assert out["metadata"]["completion_tokens"] == len("Hello END")
    assert out["metadata"]["finish_reason"] == "stop"


def test_chinese_and_multiple_stops(adapter):
    out = _run(adapter, FakeModel("你好。世界\n\n再見"), {"stop": ["\n\n", "。"]})
    assert out["response"] == "你好"


def test_stop_string_in_prompt_does_not_stop_early(adapter):
    model = FakeModel("ok END more")
    out = _run(adapter, model, {"stop": "END"}, data={"inputs": "say END"})
    assert out["response"] == "ok " and model.generated == len("ok END")


def test_temperature_zero_is_greedy(adapter):
    model = FakeModel("x")
    _run(adapter, model, {"temperature": 0, "top_p": 0.9})
    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs and "top_p" not in model.kwargs


def test_generation_kwargs_defaults_and_limit():
    assert generation_kwargs({}) == {"do_sample": True, "max_new_tokens": 512}
    assert generation_kwargs({}, max_output_tokens=100)["max_new_tokens"] == 100
    assert generation_kwargs({"max_length": 300})["max_length"] == 300  # legacy total-length meaning
    assert "max_new_tokens" not in generation_kwargs({"max_length": 300})
    assert generation_kwargs({"temperature": 0.7})["do_sample"] is True


def test_messages_use_chat_template_without_duplicate_special_tokens(adapter):
    tok = FakeTokenizer()
    model = FakeModel("fine")
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    out = _run(adapter, model, {}, data={"messages": msgs}, tok=tok)
    assert out["response"] == "fine"
    assert tok.calls[-1]["add_special_tokens"] is False
    assert tok.template_kwargs == {"enable_thinking": False}
    assert out["metadata"]["prompt_tokens"] == len("<s><system>be brief<user>hi<assistant>")


class FakePipe:
    def __init__(self, model, tok):
        self.model, self.tokenizer = model, tok

    def __call__(self, *a, **kw):  # the native pipeline call must not be used for text-generation
        raise AssertionError("pipeline __call__ used; expected shared generation helper")


def test_pipeline_route_uses_shared_helper_and_reports_usage(adapter):
    model = FakeModel("pipe out END x")
    spec = _spec(model_class=None, pipeline_task="text-generation")
    out = adapter.infer({"kind": "pipeline", "pipeline": FakePipe(model, FakeTokenizer())}, spec,
                        {"inputs": "prompt"}, normalize_generation_options({"stop": "END", "max_tokens": 64}))
    assert out["response"] == "pipe out "
    assert out["metadata"]["completion_tokens"] == len("pipe out END")


def test_oom_is_resource_exhausted(adapter):
    model = FakeModel("x", error=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    with pytest.raises(ResourceExhaustedError) as ei:
        _run(adapter, model, {})
    assert ei.value.code == "resource_exhausted"


# ----- streaming -----

def _stream(adapter, model, options, data=None):
    return adapter.infer_stream(_loaded(model), _spec(), data or {"inputs": "Q:"},
                                normalize_generation_options(options))


def test_stream_counts_generated_ids_not_chunks_and_hides_stop(adapter):
    model = FakeModel("one two three END four")
    items = list(_stream(adapter, model, {"stop": ["END"], "max_tokens": 100}))
    text = "".join(i for i in items if isinstance(i, str))
    assert text == "one two three "
    assert not any("E" in i for i in items if isinstance(i, str))
    meta = items[-1]["metadata"]
    assert meta["completion_tokens"] == len("one two three END") == model.generated
    assert meta["prompt_tokens"] == 3
    chunks = sum(1 for i in items if isinstance(i, str))
    assert chunks < meta["completion_tokens"]  # chunk count would have under-reported


def test_stream_and_non_stream_agree(adapter):
    opts = {"stop": ["。"], "max_tokens": 30}
    non_stream = _run(adapter, FakeModel("第一句。第二句"), opts)
    items = list(_stream(adapter, FakeModel("第一句。第二句"), opts))
    assert "".join(i for i in items if isinstance(i, str)) == non_stream["response"] == "第一句"
    assert items[-1]["metadata"]["completion_tokens"] == non_stream["metadata"]["completion_tokens"]


def test_stream_client_disconnect_cancels_generation_and_waits_for_worker(adapter):
    model = FakeModel("word " * 200, step_delay=0.005)
    gen = _stream(adapter, model, {"max_tokens": 1000})
    first = next(gen)
    assert isinstance(first, str)
    gen.close()  # client went away
    assert model.finished  # worker already joined -> safe to mark idle / unload
    assert model.generated < 1000


def test_stream_worker_error_is_raised(adapter):
    model = FakeModel("x", error=RuntimeError("CUDA out of memory"))
    gen = _stream(adapter, model, {})
    with pytest.raises(ResourceExhaustedError):
        list(gen)


def test_failed_load_releases_partially_loaded_weights(adapter, monkeypatch):
    """An OOM halfway through from_pretrained must not keep the partial weights alive
    (the exception traceback references them) -- otherwise the GPU stays full."""
    import gc
    import weakref

    class _Weights:
        pass

    refs = []

    class _OOMModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            partial = _Weights()  # stands in for tensors already moved to the GPU
            refs.append(weakref.ref(partial))
            raise RuntimeError("CUDA out of memory. Tried to allocate 170.00 MiB")

    monkeypatch.setattr(HFTransformersAdapter, "_resolve_class", staticmethod(lambda tx, name: _OOMModel))
    with pytest.raises(ResourceExhaustedError):
        adapter.load_model(_spec())
    gc.collect()
    assert refs and refs[0]() is None
