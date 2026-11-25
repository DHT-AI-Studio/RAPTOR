"""M07-FIX-01/04/05/06: /inference/infer, /inference/model-info and /v1/* contract.

The registry and the adapter are replaced by fakes via the service's injection
points (`_resolver`, `get_adapter`), so this exercises request validation,
version selection, limit checks, usage and error mapping end to end over HTTP
without MLflow, Ollama or a GPU.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api import inference_api, openai_api
from src.inference import service as service_mod
from src.inference.exceptions import (
    EngineError,
    InferenceTimeoutError,
    ModelAliasNotFoundError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    RegistryUnavailableError,
    ResourceExhaustedError,
    RuntimeUnavailableError,
)
from src.inference.service import inference_service


def _tags(**kw):
    tags = {"runtime": "ollama", "task_family": "text-generation", "ollama_model_name": "gemma4:12b",
            "physical_path": "gemma4:12b"}
    tags.update({k: str(v) for k, v in kw.items()})
    return tags


class FakeRegistry:
    """llm: v1 production (limit 100), v2 staging (limit 200), v3 unpromoted (no limit)."""

    def __init__(self):
        self.models = {
            "llm": {"aliases": {"production": "1"},
                    "versions": {"1": _tags(stage="production", max_output_tokens=100, context_window_tokens=4096),
                                 "2": _tags(stage="staging", max_output_tokens=200),
                                 "3": _tags(stage="none")}},
            "hf-llm": {"aliases": {},
                       "versions": {"1": {"runtime": "hf-transformers", "task_family": "text-generation",
                                          "physical_path": "/m/llm"}}},
            "whisper": {"aliases": {},
                        "versions": {"1": {"runtime": "hf-transformers", "task_family": "asr",
                                           "physical_path": "/m/whisper", "pipeline_task": "automatic-speech-recognition"}}},
        }
        self.down = False

    def resolve(self, name, model_version=None, model_alias=None):
        if self.down:
            raise RegistryUnavailableError("MLflow registry request failed: connection refused")
        m = self.models.get(name)
        if m is None:
            raise ModelNotFoundError(f"model '{name}' not in MLflow registry")
        if model_alias is not None:
            if model_alias not in m["aliases"]:
                raise ModelAliasNotFoundError(f"alias '{model_alias}' missing", model_name=name, model_alias=model_alias)
            v = m["aliases"][model_alias]
        elif model_version is not None:
            v = str(model_version)
            if v not in m["versions"]:
                raise ModelVersionNotFoundError(f"version {v} missing", model_name=name, model_version=v)
        else:
            v = max(m["versions"], key=int)
        return {"version": v, "tags": m["versions"][v]}


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.error = None
        self.stream_error = None
        self.response = "hello"
        self.context_length = None  # get_context_length() return value for the ollama-fallback auto-fill test

    def get_context_length(self, name):
        return self.context_length

    def run(self, spec, data, options):
        self.calls.append((spec.version, dict(options)))
        if self.error:
            raise self.error
        if spec.task_family != "text-generation":
            return {"text": "transcript", "metadata": {}}
        return {"response": self.response, "model": spec.ollama_model_name,
                "metadata": {"prompt_eval_count": 11, "eval_count": 3, "prompt_tokens": 11,
                             "completion_tokens": 3, "done_reason": "stop"}}

    def run_stream(self, spec, data, options):
        self.calls.append((spec.version, dict(options)))

        def gen():
            yield "hel"
            if self.stream_error:
                raise self.stream_error
            yield "lo"
            yield {"metadata": {"prompt_tokens": 11, "completion_tokens": 2, "done_reason": "length"}}
        return gen()


@pytest.fixture
def env(monkeypatch):
    reg, fake = FakeRegistry(), FakeAdapter()
    materialized = []
    monkeypatch.setattr(inference_service, "_resolver", reg.resolve)
    monkeypatch.setattr(inference_service, "_engine_configs",
                        {"ollama": {"model_limits": {"raw:tag": {"max_output_tokens": 50}}}})
    monkeypatch.setattr(inference_service, "_materialize", lambda spec: materialized.append(spec.version))
    monkeypatch.setattr(service_mod, "get_adapter", lambda runtime, configs=None: fake)
    app = FastAPI()
    app.include_router(inference_api.router)
    app.include_router(openai_api.router)
    return SimpleNamespace(client=TestClient(app), reg=reg, fake=fake, materialized=materialized)


def _infer(env, **body):
    body.setdefault("data", {"messages": [{"role": "user", "content": "hi"}]})
    body.setdefault("model_name", "llm")
    return env.client.post("/inference/infer", json=body)


# ----- success shape / usage -----

def test_success_keeps_legacy_fields_and_adds_usage(env):
    r = _infer(env, options={"max_tokens": 20})
    assert r.status_code == 200
    body = r.json()
    for key in ("success", "result", "task", "engine", "model_name", "model_version",
                "processing_time", "api_processing_time", "timestamp"):
        assert key in body
    assert body["result"]["response"] == "hello"
    assert body["result"]["metadata"]["eval_count"] == 3  # runtime metadata kept
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}


def test_non_text_task_has_null_usage_and_untouched_options(env):
    r = _infer(env, model_name="whisper", data={"audio": "/a.wav"}, options={"task": "transcribe", "max_length": 5})
    assert r.status_code == 200 and r.json()["usage"] is None
    assert env.fake.calls[-1][1] == {"task": "transcribe", "max_length": 5}


def test_options_are_normalized_before_the_adapter(env):
    _infer(env, options={"max_new_tokens": 30, "stop_sequences": "###", "temperature": 0, "num_ctx": 8192})
    assert env.fake.calls[-1][1] == {"max_tokens": 30, "stop": ["###"], "temperature": 0.0, "num_ctx": 8192}


# ----- version selection -----

@pytest.mark.parametrize("selector,version", [
    ({}, "3"), ({"model_version": "2"}, "2"), ({"model_alias": "production"}, "1"),
])
def test_infer_and_model_info_resolve_the_same_version(env, selector, version):
    r = _infer(env, **selector)
    assert r.json()["model_version"] == version == env.fake.calls[-1][0]
    info = env.client.get("/inference/model-info", params={"model_name": "llm", **selector}).json()
    assert info["model_version"] == version


def test_alias_move_is_picked_up_per_request(env):
    assert _infer(env, model_alias="production").json()["model_version"] == "1"
    env.reg.models["llm"]["aliases"]["production"] = "2"
    assert _infer(env, model_alias="production").json()["model_version"] == "2"


def test_selector_conflict_is_400(env):
    r = _infer(env, model_version="1", model_alias="production")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "conflicting_parameters"


@pytest.mark.parametrize("selector,code", [
    ({"model_version": "9"}, "model_version_not_found"),
    ({"model_alias": "staging"}, "model_alias_not_found"),
])
def test_missing_selector_target_is_404_without_fallback(env, selector, code):
    r = _infer(env, **selector)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == code and r.json()["detail"]["model_name"] == "llm"
    assert env.fake.calls == []


# ----- model-info -----

def test_model_info_shape_and_no_loading(env):
    r = env.client.get("/inference/model-info", params={"model_name": "llm", "model_alias": "production"})
    assert r.status_code == 200
    info = r.json()
    assert info == {
        "model_name": "llm", "model_version": "1", "runtime": "ollama", "task_family": "text-generation",
        "runtime_model": "gemma4:12b", "supported_options": info["supported_options"],
        "limits": {"max_output_tokens": 100, "context_window_tokens": 4096}, "usage_supported": True,
        "structured_output": True,
    }
    assert {"max_tokens", "stop", "stop_sequences", "temperature"} <= set(info["supported_options"])
    assert env.fake.calls == [] and env.materialized == []  # read-only: no load / download


def test_model_info_unknown_limits_are_null(env):
    info = env.client.get("/inference/model-info", params={"model_name": "llm"}).json()
    assert info["limits"] == {"max_output_tokens": None, "context_window_tokens": None}


def test_model_name_with_slash_via_query(env):
    env.reg.models["org/model"] = env.reg.models["llm"]
    assert env.client.get("/inference/model-info", params={"model_name": "org/model"}).status_code == 200


# ----- output-token limit -----

def test_limit_boundary(env):
    assert _infer(env, model_alias="production", options={"max_tokens": 100}).status_code == 200
    env.fake.calls.clear()
    r = _infer(env, model_alias="production", options={"max_tokens": 101})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "max_tokens_exceeded" and detail["model_limit"] == 100 and detail["requested"] == 101
    assert env.fake.calls == []  # rejected before generation


def test_limit_uses_resolved_version(env):
    assert _infer(env, model_version="2", options={"max_tokens": 150}).status_code == 200
    assert _infer(env, model_version="1", options={"max_tokens": 150}).status_code == 422


def test_unknown_limit_is_never_422(env):
    assert _infer(env, options={"max_tokens": 100000}).status_code == 200


def test_legacy_max_length_is_not_treated_as_max_tokens(env):
    assert _infer(env, model_alias="production", options={"max_length": 500}).status_code == 200


# ----- ollama-only (engine fallback) -----

def test_ollama_fallback_only_for_plain_not_found(env):
    r = _infer(env, model_name="raw:tag", engine="ollama", options={"max_tokens": 10})
    assert r.status_code == 200 and r.json()["model_version"] == "unregistered"
    info = env.client.get("/inference/model-info", params={"model_name": "raw:tag", "engine": "ollama"}).json()
    assert info["limits"]["max_output_tokens"] == 50 and info["runtime_model"] == "raw:tag"
    assert _infer(env, model_name="raw:tag", engine="ollama", options={"max_tokens": 51}).status_code == 422

    assert _infer(env, model_name="raw:tag").status_code == 404  # no engine -> no fallback
    r = _infer(env, model_name="raw:tag", engine="ollama", model_version="1")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "model_not_found"


def test_ollama_fallback_without_a_configured_limit_asks_the_daemon(env):
    """A daemon tag inference.yaml never configured a limit for still gets a real
    max_output_tokens -- from the daemon's own /api/show context length, not left None
    forever (which would make /complete-style max_tokens validation a no-op for it)."""
    env.fake.context_length = 8192
    info = env.client.get(
        "/inference/model-info", params={"model_name": "unconfigured:tag", "engine": "ollama"}
    ).json()
    assert info["limits"]["max_output_tokens"] == 8192
    assert info["limits"]["context_window_tokens"] == 8192
    assert _infer(env, model_name="unconfigured:tag", engine="ollama",
                  options={"max_tokens": 8193}).status_code == 422
    assert _infer(env, model_name="unconfigured:tag", engine="ollama",
                  options={"max_tokens": 100}).status_code == 200


def test_ollama_fallback_daemon_unreachable_leaves_limit_none(env):
    """get_context_length() is best-effort: if the daemon can't answer, the model-info
    limit stays None exactly like before this fallback existed, not a 500/503."""
    env.fake.context_length = None
    info = env.client.get(
        "/inference/model-info", params={"model_name": "still-unconfigured:tag", "engine": "ollama"}
    ).json()
    assert info["limits"]["max_output_tokens"] is None
    assert _infer(env, model_name="still-unconfigured:tag", engine="ollama").status_code == 200


def test_registry_down_is_503_and_never_falls_back(env):
    env.reg.down = True
    r = _infer(env, model_name="raw:tag", engine="ollama")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "registry_unavailable"
    assert env.fake.calls == []


# ----- error mapping -----

@pytest.mark.parametrize("error,status,code", [
    (InferenceTimeoutError("ollama timed out"), 504, "inference_timeout"),
    (RuntimeUnavailableError("daemon down"), 503, "runtime_unavailable"),
    (ResourceExhaustedError("oom"), 503, "resource_exhausted"),
    (EngineError("boom"), 500, "engine_error"),
])
def test_runtime_errors_keep_their_meaning(env, error, status, code):
    env.fake.error = error
    r = _infer(env)
    assert r.status_code == status
    detail = r.json()["detail"]
    assert detail["code"] == code and detail["error_type"] == type(error).__name__ and detail["message"]
    if code == "resource_exhausted":
        assert detail["retry_after"] == 60


def test_model_not_found(env):
    r = _infer(env, model_name="nope")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "model_not_found" and r.json()["detail"]["model_name"] == "nope"


def test_validation_errors_have_codes(env):
    r = _infer(env, options={"max_tokens": 5, "max_new_tokens": 6})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "conflicting_parameters"
    r = _infer(env, options={"temperature": 3})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid_parameter"
    r = _infer(env, data={"prompt": "x"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid_request"


# ----- /v1 shares usage and error codes -----

def _chat(env, **body):
    body.setdefault("model", "llm")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return env.client.post("/v1/chat/completions", json=body)


def test_v1_usage_matches_infer(env):
    r = _chat(env, max_tokens=20)
    assert r.status_code == 200
    assert r.json()["usage"] == _infer(env, options={"max_tokens": 20}).json()["usage"]
    assert env.fake.calls[0][1]["max_tokens"] == 20


def test_v1_error_envelope_uses_same_codes(env):
    env.reg.models["llm"]["aliases"]["production"] = "1"
    env.reg.models["llm"]["versions"]["3"]["max_output_tokens"] = "10"
    r = _chat(env, max_tokens=11)
    assert r.status_code == 422
    err = r.json()["detail"]["error"]
    assert err["code"] == "max_tokens_exceeded" and err["model_limit"] == 10
    env.fake.error = InferenceTimeoutError("slow")
    r = _chat(env)
    assert r.status_code == 504 and r.json()["detail"]["error"]["code"] == "inference_timeout"


def _sse_events(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_v1_stream_usage_and_finish_reason(env):
    r = _chat(env, stream=True)
    events = _sse_events(r.text)
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "hello"
    assert events[-1]["usage"] == {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
    assert events[-1]["choices"][0]["finish_reason"] == "length"


def test_v1_mid_stream_error_event_has_code(env):
    env.fake.stream_error = InferenceTimeoutError("read timed out")
    events = _sse_events(_chat(env, stream=True).text)
    assert events[-1]["error"]["code"] == "inference_timeout"


# ----- structured output (temporal plan §5.10, P10) -----

SCHEMA = {"type": "object", "properties": {"year": {"type": "integer", "minimum": 1900},
                                           "tags": {"type": "array", "items": {"type": "string"}}},
          "required": ["year"], "additionalProperties": False}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "event", "schema": SCHEMA}}


@pytest.mark.parametrize("fmt", [SCHEMA, RESPONSE_FORMAT])
def test_infer_format_reaches_the_runtime_and_the_output_is_validated(env, fmt):
    env.fake.response = '<think>x</think>```json\n{"year": 2003, "tags": ["a"]}\n```'
    r = _infer(env, options={"format": fmt, "temperature": 0})
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["parsed"] == {"year": 2003, "tags": ["a"]}
    assert json.loads(res["response"]) == res["parsed"]                 # canonical JSON text, no fences
    assert env.fake.calls[0][1]["format"] == SCHEMA                      # the schema Ollama constrains with


def test_infer_json_object(env):
    env.fake.response = 'Sure: {"a": 1}'
    r = _infer(env, options={"format": {"type": "json_object"}})
    assert r.status_code == 200 and r.json()["result"]["parsed"] == {"a": 1}
    assert env.fake.calls[0][1]["format"] == "json"
    env.fake.response = "[1, 2]"
    assert _infer(env, options={"format": "json"}).json()["detail"]["code"] == "structured_output_error"


@pytest.mark.parametrize("output, violation", [
    ('{"year": 1800}', "$.year: minimum 1900, got 1800"),
    ('{"year": 2003, "extra": 1}', "$: unexpected property 'extra'"),
    ('{"tags": []}', "$: missing required 'year'"),
    ("no json here", "$: not JSON"),
])
def test_infer_invalid_output_is_502_never_passed_on(env, output, violation):
    env.fake.response = output
    r = _infer(env, options={"format": RESPONSE_FORMAT})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["code"] == "structured_output_error" and violation in detail["violations"]


def test_infer_echoed_schema_wrapper_is_unwrapped_only_when_valid(env):
    env.fake.response = '{"type": "object", "properties": {"year": 2003}}'
    assert _infer(env, options={"format": RESPONSE_FORMAT}).json()["result"]["parsed"] == {"year": 2003}
    env.fake.response = '{"event": {"year": 2003}}'
    assert _infer(env, options={"format": RESPONSE_FORMAT}).json()["result"]["parsed"] == {"year": 2003}
    env.fake.response = '{"properties": {"year": "2003"}}'
    assert _infer(env, options={"format": RESPONSE_FORMAT}).status_code == 502


def test_infer_format_on_a_runtime_that_cannot_honour_it(env):
    r = _infer(env, model_name="hf-llm", options={"format": SCHEMA})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "unsupported_capability"
    assert env.fake.calls == []                                          # refused before running
    info = env.client.get("/inference/model-info", params={"model_name": "hf-llm"}).json()
    assert info["structured_output"] is False


@pytest.mark.parametrize("fmt, needle", [
    ({"type": "object", "properties": {"a": {"$ref": "#/x"}}}, "$ref"),
    ({"type": "object", "anyOf": []}, "anyOf"),
    ({"type": "json_schema", "json_schema": {"name": "x"}}, "json_schema.schema"),
    ({"type": "thing"}, "type"),
    (42, "format must be"),
])
def test_infer_bad_format_request_is_400(env, fmt, needle):
    r = _infer(env, options={"format": fmt})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "invalid_parameter" and needle in r.json()["detail"]["message"]
    assert env.fake.calls == []


def test_infer_text_format_is_plain_generation(env):
    r = _infer(env, options={"format": {"type": "text"}})
    assert r.status_code == 200 and "parsed" not in r.json()["result"]
    assert "format" not in env.fake.calls[0][1]


def test_v1_response_format(env):
    env.fake.response = '{"year": 2003}'
    r = _chat(env, response_format=RESPONSE_FORMAT)
    assert r.status_code == 200
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"year": 2003}
    assert env.fake.calls[0][1]["format"] == SCHEMA
    env.fake.response = '{"year": "x"}'
    r = _chat(env, response_format=RESPONSE_FORMAT)
    assert r.status_code == 502 and r.json()["detail"]["error"]["code"] == "structured_output_error"


def test_v1_response_format_with_stream_is_validated_whole(env):
    env.fake.response = '{"year": 2003}'
    events = _sse_events(_chat(env, stream=True, response_format=RESPONSE_FORMAT).text)
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert json.loads(content) == {"year": 2003}
    assert len(env.fake.calls) == 1                                      # run(), not run_stream()
