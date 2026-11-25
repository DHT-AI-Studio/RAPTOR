"""INF-5..8: request validation, what is sent to Module 07, post-processing and error mapping."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import httpx

from .conftest import error_response, make_settings


def _system(call):
    return next((m["content"] for m in call["data"]["messages"] if m["role"] == "system"), None)


def _user(call):
    return next(m["content"] for m in call["data"]["messages"] if m["role"] == "user")


# ---------------------------------------------------------------- shared contract

async def test_response_shape_and_upstream_payload(client, upstream):
    upstream.responses = ["Good morning"]
    resp = await client.post("/inference/text/translate",
                             json={"input": "早安", "options": {"target_language": "en", "source_language": "zh-TW"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "translate"
    assert body["model"] == "gemma-prod" and body["model_version"] == "1"
    assert body["output"] == "Good morning"
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert isinstance(body["latency_ms"], int)

    call = upstream.infer_calls[-1]
    assert call["task"] == "text-generation"
    assert call["model_name"] == "gemma-prod" and call["model_version"] == "1"
    assert [m["role"] for m in call["data"]["messages"]] == ["system", "user"]
    assert _user(call) == "早安"
    # prompt tasks use the model's configured output budget
    assert call["options"]["max_tokens"] == 4096


async def test_ollama_only_model_is_called_with_engine(client, upstream):
    await client.post("/inference/text/translate",
                      json={"model": "gemma3:4b", "input": "x", "options": {"target_language": "en"}})
    call = upstream.infer_calls[-1]
    assert call["engine"] == "ollama" and "model_version" not in call


async def test_think_blocks_are_stripped(client, upstream):
    upstream.responses = ["<think>\nhmm\n</think>\n\nHello"]
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.json()["output"] == "Hello"


async def test_null_usage_is_passed_through(client, upstream):
    def handler(request):
        if request.url.path == "/inference/infer":
            return httpx.Response(200, json={"success": True, "result": {"response": "x"}, "usage": None})
        return upstream.handler(request)

    client.app.state.text_inference._client._client._transport = httpx.MockTransport(handler)
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 200 and resp.json()["usage"] is None


async def test_http_timeout_is_504(client, upstream):
    upstream.infer_exception = httpx.ReadTimeout("slow")
    resp = await client.post("/inference/text/translate", json={"input": "x", "options": {"target_language": "en"}})
    assert resp.status_code == 504
    assert resp.json()["error"] == "upstream_timeout"


async def test_module07_inference_timeout_is_504(client, upstream):
    upstream.infer_error = error_response(504, "inference_timeout")
    resp = await client.post("/inference/text/translate", json={"input": "x", "options": {"target_language": "en"}})
    assert resp.status_code == 504


async def test_module07_unreachable_is_503(client, upstream):
    upstream.infer_exception = httpx.ConnectError("refused")
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 503 and resp.json()["error"] == "module07_unavailable"


async def test_resource_exhausted_keeps_retry_after(client, upstream):
    upstream.infer_error = error_response(503, "resource_exhausted", retry_after=60)
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "60"
    assert resp.json()["error"] == "resource_exhausted"


async def test_other_module07_500_is_502_not_504(client, upstream):
    upstream.infer_error = error_response(500, "inference_failed")
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 502
    assert resp.json()["upstream_code"] == "inference_failed"


async def test_guardrail_block_is_400(client, upstream):
    upstream.infer_error = error_response(400, "policy_violation", category="S1", direction="input")
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "policy_violation" and resp.json()["category"] == "S1"


async def test_stale_version_refreshes_registry_and_is_422(client, upstream):
    await client.get("/models")
    calls_before = upstream.list_calls
    upstream.infer_error = error_response(404, "model_version_not_found")
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "model_not_found" and "available" in resp.json()
    assert upstream.list_calls == calls_before + 1


async def test_unknown_fields_are_rejected(client):
    resp = await client.post("/inference/text/complete", json={"input": "x", "options": {"top_p": 0.9}})
    assert resp.status_code == 400
    resp = await client.post("/inference/text/complete", json={"input": "x", "extra": 1})
    assert resp.status_code == 400


async def test_empty_input_is_400(client):
    resp = await client.post("/inference/text/complete", json={"input": ""})
    assert resp.status_code == 400


# ---------------------------------------------------------------- INF-5 translate

async def test_translate_missing_target_language_is_400(client, upstream):
    for body in ({"input": "x", "options": {}}, {"input": "x"}):
        resp = await client.post("/inference/text/translate", json=body)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_request"
    assert upstream.infer_calls == []


async def test_translate_rejects_non_bcp47(client):
    resp = await client.post("/inference/text/translate",
                             json={"input": "x", "options": {"target_language": "Traditional Chinese"}})
    assert resp.status_code == 400


async def test_translate_detects_source_language(client, upstream):
    upstream.responses = ["DETECTED_LANGUAGE: zh\n\nGood morning"]
    resp = await client.post("/inference/text/translate", json={"input": "早安", "options": {"target_language": "en"}})
    body = resp.json()
    assert body["detected_language"] == "zh"
    assert body["output"] == "Good morning"
    assert "DETECTED_LANGUAGE" in _system(upstream.infer_calls[-1])


async def test_translate_detection_header_missing(client, upstream):
    upstream.responses = ["Good morning"]
    resp = await client.post("/inference/text/translate", json={"input": "早安", "options": {"target_language": "en"}})
    body = resp.json()
    assert body["detected_language"] is None and body["output"] == "Good morning"


async def test_translate_with_source_language_has_no_detection(client, upstream):
    upstream.responses = ["早安"]
    resp = await client.post("/inference/text/translate",
                             json={"input": "Good morning",
                                   "options": {"target_language": "zh-TW", "source_language": "en",
                                               "formality": "formal"}})
    body = resp.json()
    assert "detected_language" not in body and body["source_language"] == "en"
    system = _system(upstream.infer_calls[-1])
    assert "Traditional Chinese (Taiwan) [zh-TW]" in system
    assert "English [en]" in system
    assert "formal" in system and "DETECTED_LANGUAGE" not in system


async def test_translate_system_prompt_from_env(make_client, upstream):
    c = await make_client(make_settings(translate_system_prompt="Translate to {target_language}. {formality_instruction}"))
    await c.post("/inference/text/translate",
                 json={"input": "x", "options": {"target_language": "ja", "source_language": "en"}})
    system = _system(upstream.infer_calls[-1])
    assert system.startswith("Translate to Japanese [ja].")
    # the required source / output instructions are appended when the template omits them
    assert "The source language is English [en]." in system


# ---------------------------------------------------------------- INF-6 summarize

async def test_summarize_bullet_format_and_max_sentences(client, upstream):
    upstream.responses = ["Here is the summary:\n* one\n* two\n1. three\n- four"]
    resp = await client.post("/inference/text/summarize",
                             json={"input": "text", "options": {"style": "bullet", "max_sentences": 3}})
    body = resp.json()
    assert body["output"] == "- one\n- two\n- three"
    assert body["style"] == "bullet"
    assert "at most 3 bullet points" in _system(upstream.infer_calls[-1])


async def test_summarize_bullet_from_prose(client, upstream):
    upstream.responses = ["First point. Second point! 第三點。"]
    resp = await client.post("/inference/text/summarize", json={"input": "text"})
    assert resp.json()["output"] == "- First point.\n- Second point!\n- 第三點。"


async def test_summarize_paragraph_limits(client, upstream):
    upstream.responses = ["A one. A two.\n\nB one. B two.\n\nC one.\n\nD one."]
    resp = await client.post("/inference/text/summarize", json={"input": "t", "options": {"style": "paragraph"}})
    assert resp.json()["output"] == "A one. A two.\n\nB one. B two.\n\nC one."

    upstream.responses = ["A one. A two.\n\nB one. B two."]
    resp = await client.post("/inference/text/summarize",
                             json={"input": "t", "options": {"style": "paragraph", "max_sentences": 3}})
    assert resp.json()["output"] == "A one. A two.\n\nB one."


async def test_summarize_headline_word_limit(client, upstream):
    long_en = " ".join(f"w{i}" for i in range(30)) + "."
    upstream.responses = [f"# {long_en}\nsecond line"]
    resp = await client.post("/inference/text/summarize", json={"input": "t", "options": {"style": "headline"}})
    out = resp.json()["output"]
    assert out == " ".join(f"w{i}" for i in range(20))

    upstream.responses = ["「" + "字" * 30 + "」"]
    resp = await client.post("/inference/text/summarize", json={"input": "t", "options": {"style": "headline"}})
    assert resp.json()["output"] == "字" * 20

    upstream.responses = ["Raptor 平台集中模型管理於 Module 07，並透過 Module 35 提供多種任務 API"]
    resp = await client.post("/inference/text/summarize", json={"input": "t", "options": {"style": "headline"}})
    assert resp.json()["output"] == "Raptor 平台集中模型管理於 Module 07"

    upstream.responses = ["Markets rally. Then fall."]
    resp = await client.post("/inference/text/summarize", json={"input": "t", "options": {"style": "headline"}})
    assert resp.json()["output"] == "Markets rally."


async def test_summarize_language_instruction(client, upstream):
    await client.post("/inference/text/summarize", json={"input": "t", "options": {"language": "zh-TW"}})
    assert "Write the summary in Traditional Chinese (Taiwan) [zh-TW]." in _system(upstream.infer_calls[-1])
    await client.post("/inference/text/summarize", json={"input": "t"})
    assert "same language" in _system(upstream.infer_calls[-1])


async def test_summarize_long_input_is_chunked(make_client, upstream):
    c = await make_client(make_settings(summarize_chunk_chars=1000, summarize_max_chunks=8))
    text = "\n\n".join(f"Paragraph {i}. " * 40 for i in range(5))   # 5 x 560 chars
    upstream.responses = ["p1", "p2", "p3", "p4", "p5", "- final"]
    resp = await c.post("/inference/text/summarize", json={"input": text})
    body = resp.json()
    assert resp.status_code == 200
    assert body["chunks"] == 5 and len(upstream.infer_calls) == 6
    assert _user(upstream.infer_calls[-1]) == "p1\n\np2\n\np3\n\np4\n\np5"
    assert body["usage"]["total_tokens"] == 15 * len(upstream.infer_calls)
    assert all(len(_user(call)) <= 1000 for call in upstream.infer_calls[:-1])
    assert body["output"] == "- final"


async def test_summarize_too_long_is_413(make_client, upstream):
    c = await make_client(make_settings(summarize_chunk_chars=1000, summarize_max_chunks=2))
    resp = await c.post("/inference/text/summarize", json={"input": "x. " * 2000})
    assert resp.status_code == 413
    assert resp.json()["error"] == "input_too_long"
    assert upstream.infer_calls == []


async def test_summarize_system_prompt_from_env(make_client, upstream):
    c = await make_client(make_settings(summarize_system_prompt="SUM {style_instruction}"))
    await c.post("/inference/text/summarize", json={"input": "t", "options": {"style": "headline"}})
    system = _system(upstream.infer_calls[-1])
    assert system.startswith("SUM Write a single-sentence headline")
    assert "same language" in system   # omitted placeholder appended


# ---------------------------------------------------------------- INF-7 complete

async def test_complete_forwards_sampling_params_unchanged(client, upstream):
    options = {"system_prompt": "You are terse.", "max_tokens": 77, "temperature": 1.3,
               "stop_sequences": ["\n\n", "END"]}
    resp = await client.post("/inference/text/complete", json={"input": "Tell me", "options": options})
    assert resp.status_code == 200
    call = upstream.infer_calls[-1]
    assert call["options"] == {"max_tokens": 77, "temperature": 1.3, "stop_sequences": ["\n\n", "END"]}
    assert call["data"]["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Tell me"},
    ]
    assert resp.json()["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def test_complete_defaults(client, upstream):
    await client.post("/inference/text/complete", json={"input": "x"})
    call = upstream.infer_calls[-1]
    assert call["options"] == {"max_tokens": 1024, "temperature": 0.7}
    assert [m["role"] for m in call["data"]["messages"]] == ["user"]


async def test_complete_temperature_zero_and_bounds(client, upstream):
    resp = await client.post("/inference/text/complete", json={"input": "x", "options": {"temperature": 0}})
    assert resp.status_code == 200 and upstream.infer_calls[-1]["options"]["temperature"] == 0
    for bad in (-0.1, 2.1):
        resp = await client.post("/inference/text/complete", json={"input": "x", "options": {"temperature": bad}})
        assert resp.status_code == 400
    resp = await client.post("/inference/text/complete", json={"input": "x", "options": {"max_tokens": 0}})
    assert resp.status_code == 400
    resp = await client.post("/inference/text/complete", json={"input": "x", "options": {"stop_sequences": [""]}})
    assert resp.status_code == 400


async def test_complete_max_tokens_over_limit_is_422_before_generation(client, upstream):
    resp = await client.post("/inference/text/complete",
                             json={"model": "hf-staging", "input": "x", "options": {"max_tokens": 257}})
    assert resp.status_code == 422
    assert resp.json() == {"error": "max_tokens_exceeded", "model": "hf-staging", "model_limit": 256,
                           "requested": 257}
    assert upstream.infer_calls == []
    assert upstream.info_calls[-1] == {"model_name": "hf-staging", "model_version": "4"}

    resp = await client.post("/inference/text/complete",
                             json={"model": "hf-staging", "input": "x", "options": {"max_tokens": 256}})
    assert resp.status_code == 200


async def test_complete_default_max_tokens_checked_too(client, upstream):
    resp = await client.post("/inference/text/complete", json={"model": "hf-staging", "input": "x"})
    assert resp.status_code == 422 and resp.json()["requested"] == 1024


async def test_complete_unknown_limit_is_503(client, upstream):
    resp = await client.post("/inference/text/complete", json={"model": "llama3:8b", "input": "x"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_limits_not_configured"
    assert upstream.infer_calls == []


async def test_complete_module07_limit_error_is_mapped(client, upstream):
    upstream.infer_error = error_response(422, "max_tokens_exceeded", model_limit=100, requested=200)
    resp = await client.post("/inference/text/complete", json={"input": "x", "options": {"max_tokens": 200}})
    assert resp.status_code == 422
    assert resp.json()["error"] == "max_tokens_exceeded" and resp.json()["model_limit"] == 100


async def test_model_info_is_cached(client, upstream):
    await client.post("/inference/text/complete", json={"input": "x"})
    await client.post("/inference/text/complete", json={"input": "y"})
    assert len(upstream.info_calls) == 1


# ---------------------------------------------------------------- INF-8 transform

async def test_transform_requires_instruction(client, upstream):
    for options in ({}, {"instruction": ""}, {"instruction": "   "}):
        resp = await client.post("/inference/text/transform", json={"input": "x", "options": options})
        assert resp.status_code == 400
    resp = await client.post("/inference/text/transform", json={"input": "x"})
    assert resp.status_code == 400
    assert upstream.infer_calls == []


async def test_transform_instruction_and_structure(client, upstream):
    source = "# Title\n\nfirst para\n\nsecond para"
    upstream.responses = ["# 標題\n\n第一段\n\n第二段"]
    resp = await client.post("/inference/text/transform",
                             json={"input": source,
                                   "options": {"instruction": "改寫為正式繁體中文", "preserve_structure": True}})
    body = resp.json()
    assert body["structure"] == {"preserved": True, "input_paragraphs": 3, "output_paragraphs": 3,
                                 "input_headings": 1, "output_headings": 1}
    call = upstream.infer_calls[-1]
    assert "Instruction: 改寫為正式繁體中文" in _system(call)
    assert "Preserve the original structure" in _system(call)
    assert _user(call) == source


async def test_transform_structure_mismatch_reported(client, upstream):
    upstream.responses = ["merged into one paragraph"]
    resp = await client.post("/inference/text/transform",
                             json={"input": "a\n\nb", "options": {"instruction": "x", "preserve_structure": True}})
    assert resp.json()["structure"]["preserved"] is False


async def test_transform_without_preserve_structure(client, upstream):
    resp = await client.post("/inference/text/transform", json={"input": "a", "options": {"instruction": "shorter"}})
    assert "structure" not in resp.json()
    assert "Preserve the original structure" not in _system(upstream.infer_calls[-1])


async def test_transform_custom_template_keeps_instruction(make_client, upstream):
    c = await make_client(make_settings(transform_system_prompt="Custom rewriter."))
    await c.post("/inference/text/transform", json={"input": "a", "options": {"instruction": "make it formal"}})
    assert _system(upstream.infer_calls[-1]) == "Custom rewriter.\nInstruction: make it formal"


# ---------------------------------------------------------------- dispatcher

async def test_dispatcher_routes_every_task(client, upstream):
    bodies = {
        "translate": {"input": "x", "options": {"target_language": "en"}},
        "summarize": {"input": "x"},
        "complete": {"input": "x"},
        "transform": {"input": "x", "options": {"instruction": "y"}},
    }
    for task, body in bodies.items():
        resp = await client.post("/inference/text", json={"task": task, **body})
        assert resp.status_code == 200, resp.text
        assert resp.json()["task"] == task


async def test_dispatcher_unknown_or_missing_task_is_400(client):
    for body in ({"task": "dance", "input": "x"}, {"input": "x"}):
        resp = await client.post("/inference/text", json=body)
        assert resp.status_code == 400
        assert resp.json()["error"] == "unsupported_task"


async def test_dispatcher_validates_like_typed_route(client):
    resp = await client.post("/inference/text", json={"task": "transform", "input": "x", "options": {}})
    assert resp.status_code == 400
    assert resp.json()["details"][0]["field"] == "options.instruction"


async def test_health(client, monkeypatch):
    # Deployed text profile: no torch installed at all, unlike this shared test env.
    monkeypatch.setitem(sys.modules, "torch", None)
    resp = await client.get("/health")
    assert resp.json()["status"] == "ok"
    assert resp.json()["module"] == 35
    assert resp.json()["gpu_available"] is False
    assert resp.json()["voice"] == {"diarize_loaded": False, "clone_loaded": False,
                                     "clone_configured_backends": ["xtts-v2"]}
    assert resp.json()["warnings"] == []


async def test_health_reports_gpu_available(client, monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    resp = await client.get("/health")
    assert resp.json()["gpu_available"] is True
