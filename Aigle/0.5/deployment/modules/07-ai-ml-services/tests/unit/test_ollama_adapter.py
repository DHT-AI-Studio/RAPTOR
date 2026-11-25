"""M07-FIX-02/03/06: what actually reaches the Ollama daemon, usage, error classes."""
import json
from unittest.mock import patch

import pytest
import requests
from src.inference.adapters.ollama import OllamaAdapter
from src.inference.exceptions import (
    EngineError,
    InferenceTimeoutError,
    ModelLoadError,
    ResourceExhaustedError,
    RuntimeUnavailableError,
)
from src.inference.options import normalize_generation_options
from src.inference.spec import ModelSpec

POST = "src.inference.adapters.ollama.requests.post"


class _Resp:
    def __init__(self, body=None, status=200, lines=None):
        self._body = body or {}
        self.status_code = status
        self.text = json.dumps(self._body)
        self._lines = lines or []
        self.closed = False

    def json(self):
        return self._body

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield json.dumps(line).encode()

    def close(self):
        self.closed = True


def _spec(limit=None):
    return ModelSpec(model_name="m", version="1", task_family="text-generation", runtime="ollama",
                     physical_path="gemma4:12b", ollama_model_name="gemma4:12b", max_output_tokens=limit)


@pytest.fixture
def ad():
    return OllamaAdapter(config={"base_url": "http://ollama:11434", "keep_alive": 300, "timeout": 5})


def _chat_body(content="hello", **kw):
    body = {"message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 12, "eval_count": 4}
    body.update(kw)
    return body


def test_normalized_options_reach_daemon(ad):
    opts = normalize_generation_options({"max_new_tokens": 50, "stop_sequences": ["###"], "temperature": 0})
    with patch(POST, return_value=_Resp(_chat_body())) as post:
        out = ad.infer("gemma4:12b", _spec(), {"messages": [{"role": "user", "content": "hi"}]}, opts)
    payload = post.call_args.kwargs["json"]
    assert payload["options"] == {"num_predict": 50, "stop": ["###"], "temperature": 0.0}
    assert payload["think"] is False and payload["keep_alive"] == 300
    assert out["metadata"]["prompt_tokens"] == 12 and out["metadata"]["completion_tokens"] == 4


def test_generate_path_maps_max_tokens_and_legacy_max_length(ad):
    with patch(POST, return_value=_Resp({"response": "x", "done": True, "prompt_eval_count": 1, "eval_count": 1})) as post:
        ad.infer("gemma4:12b", _spec(), {"inputs": "p"}, {"max_tokens": 7})
        assert post.call_args.kwargs["json"]["options"] == {"num_predict": 7}
        ad.infer("gemma4:12b", _spec(), {"inputs": "p"}, {"max_length": 9})
        assert post.call_args.kwargs["json"]["options"] == {"num_predict": 9}


def test_known_limit_caps_generation_when_caller_sets_no_length(ad):
    with patch(POST, return_value=_Resp(_chat_body())) as post:
        ad.infer("gemma4:12b", _spec(limit=256), {"messages": [{"role": "user", "content": "hi"}]}, {})
    assert post.call_args.kwargs["json"]["options"] == {"num_predict": 256}


def test_stop_text_is_trimmed_even_if_daemon_returns_it(ad):
    with patch(POST, return_value=_Resp(_chat_body("answer###junk"))):
        out = ad.infer("gemma4:12b", _spec(), {"messages": [{"role": "user", "content": "q"}]}, {"stop": ["###"]})
    assert out["response"] == "answer"


def test_missing_counts_are_not_faked(ad):
    body = _chat_body()
    del body["prompt_eval_count"]
    with patch(POST, return_value=_Resp(body)):
        out = ad.infer("gemma4:12b", _spec(), {"messages": [{"role": "user", "content": "q"}]}, {})
    assert "prompt_tokens" not in out["metadata"] and "completion_tokens" not in out["metadata"]


@pytest.mark.parametrize("exc,expected", [
    (requests.ReadTimeout("read timed out"), InferenceTimeoutError),
    (requests.ConnectTimeout("connect timed out"), RuntimeUnavailableError),
    (requests.ConnectionError("connection refused"), RuntimeUnavailableError),
])
def test_transport_errors_are_classified(ad, exc, expected):
    with patch(POST, side_effect=exc), pytest.raises(expected) as ei:
        ad.infer("gemma4:12b", _spec(), {"inputs": "p"}, {})
    assert ei.value.http_status == {InferenceTimeoutError: 504, RuntimeUnavailableError: 503}[expected]


@pytest.mark.parametrize("status,body,expected", [
    (500, {"error": "model requires more system memory (12 GiB) than is available (4 GiB)"}, ResourceExhaustedError),
    (500, {"error": "CUDA error: out of memory"}, ResourceExhaustedError),
    (404, {"error": "model 'gemma4:12b' not found"}, ModelLoadError),
    (500, {"error": "something else"}, EngineError),
])
def test_status_errors_are_classified(ad, status, body, expected):
    with patch(POST, return_value=_Resp(body, status=status)):
        with pytest.raises(expected):
            ad.infer("gemma4:12b", _spec(), {"inputs": "p"}, {})


def test_daemon_unreachable_during_load_is_runtime_unavailable(ad):
    with patch("src.inference.adapters.ollama.requests.get", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(RuntimeUnavailableError):
            ad.load_model(_spec())


def _stream_lines(pieces, **final):
    lines = [{"message": {"content": p}, "done": False} for p in pieces]
    lines.append({"message": {"content": ""}, "done": True, "prompt_eval_count": 9, "eval_count": len(pieces), **final})
    return lines


def test_stream_holds_back_stop_prefix_and_reports_usage(ad):
    resp = _Resp(lines=_stream_lines(["Hello", " wor", "ld#", "##", " never"]))
    with patch(POST, return_value=resp) as post:
        gen = ad.infer_stream("gemma4:12b", _spec(), {"messages": [{"role": "user", "content": "q"}]},
                              {"stop": ["###"], "max_tokens": 20})
        items = list(gen)
    assert post.call_args.kwargs["json"]["options"] == {"num_predict": 20, "stop": ["###"]}
    text = "".join(i for i in items if isinstance(i, str))
    assert text == "Hello world"
    assert all("#" not in i for i in items if isinstance(i, str))
    meta = items[-1]["metadata"]
    assert meta["prompt_tokens"] == 9 and meta["completion_tokens"] == 5
    assert resp.closed


def test_stream_mid_error_is_classified(ad):
    resp = _Resp(lines=[{"message": {"content": "a"}, "done": False}, requests.ConnectionError("read timed out")])
    with patch(POST, return_value=resp):
        gen = ad.infer_stream("gemma4:12b", _spec(), {"inputs": "q"}, {})
        with pytest.raises(InferenceTimeoutError):
            list(gen)
    assert resp.closed


def test_format_is_a_top_level_field_of_chat_and_generate(ad):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    with patch(POST, return_value=_Resp(_chat_body('{"a": 1}'))) as post:
        ad.infer("gemma4:12b", _spec(), {"messages": [{"role": "user", "content": "x"}]}, {"format": schema})
    payload = post.call_args.kwargs["json"]
    assert payload["format"] == schema and "format" not in payload["options"]
    with patch(POST, return_value=_Resp({"response": "{}", "done": True})) as post:
        ad.infer("gemma4:12b", _spec(), {"inputs": "x"}, {"format": "json"})
    assert post.call_args.kwargs["json"]["format"] == "json"
    with patch(POST, return_value=_Resp(_chat_body())) as post:
        ad.infer("gemma4:12b", _spec(), {"messages": [{"role": "user", "content": "x"}]}, {})
    assert "format" not in post.call_args.kwargs["json"]


def test_get_context_length_reads_the_architecture_prefixed_key(ad):
    body = {"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}}
    with patch(POST, return_value=_Resp(body)) as post:
        assert ad.get_context_length("qwen3:0.6b") == 40960
    assert post.call_args.args[0].endswith("/api/show")
    assert post.call_args.kwargs["json"] == {"name": "qwen3:0.6b"}


def test_get_context_length_is_none_on_any_failure(ad):
    with patch(POST, return_value=_Resp({}, status=404)):
        assert ad.get_context_length("missing:tag") is None
    with patch(POST, return_value=_Resp({"model_info": {}})):
        assert ad.get_context_length("no-arch-key:tag") is None
    with patch(POST, side_effect=requests.ConnectionError("boom")):
        assert ad.get_context_length("unreachable:tag") is None
