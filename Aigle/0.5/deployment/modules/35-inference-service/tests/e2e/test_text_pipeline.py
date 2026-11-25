"""VIE01-373 (INF-14) Tests 1, 3, 4 -- the text path, against a real module 07 (LLM) and
module 35 (--profile text). See README.md for what's scoped down and why.

Test 1's full chain (translate -> summarize -> tts -> stt) additionally needs module 35's
--profile voice on a real GPU host (INF-10/11 shipped, VIE01-316/317) -- it skips on an
instance without a voice backend loaded, same convention as test_voice_pipeline.py.
"""
from __future__ import annotations

import concurrent.futures

import httpx
import pytest

_ENGLISH = "Hello, how are you today? I hope everything is going well with your work."


def test_translate_then_summarize(client: httpx.Client):
    """Text-only half of Test 1 -- translate -> summarize, each step non-empty and in the
    requested language. Runs on any instance, voice-enabled or not; see
    test_translate_then_summarize_then_tts_then_stt for the full chain the ticket asks for."""
    r = client.post("/inference/text/translate",
                     json={"input": _ENGLISH, "options": {"target_language": "zh-TW", "source_language": "en"}})
    assert r.status_code == 200
    translated = r.json()
    assert translated["output"].strip()
    assert translated["target_language"] == "zh-TW"

    r = client.post("/inference/text/summarize",
                     json={"input": translated["output"], "options": {"style": "bullet"}})
    assert r.status_code == 200
    summary = r.json()
    assert summary["output"].strip()


def test_translate_then_summarize_then_tts_then_stt(client: httpx.Client, voice_enabled: bool):
    """Test 1, full chain: translate -> summarize -> tts -> stt, each step non-empty and
    in the requested language. Re-expanded now that INF-10/11 (VIE01-316/317) shipped --
    /tts and /stt forward to Module 34's loaded models (see README.md's scope-note
    history); skips on an instance with no voice backend loaded, same convention as
    test_voice_pipeline.py."""
    if not voice_enabled:
        pytest.skip("this module 35 instance has no voice backend loaded "
                    "(VOICE_ENABLED=false, or no GPU -- see /health)")
    headers = {"X-User-ID": "e2e-text-voice-chain-user"}

    r = client.post("/inference/text/translate",
                     json={"input": _ENGLISH, "options": {"target_language": "zh-TW", "source_language": "en"}})
    assert r.status_code == 200
    translated = r.json()
    assert translated["output"].strip()
    assert translated["target_language"] == "zh-TW"

    r = client.post("/inference/text/summarize",
                     json={"input": translated["output"], "options": {"style": "bullet"}})
    assert r.status_code == 200
    summary = r.json()
    assert summary["output"].strip()

    r = client.post("/inference/voice/tts", data={"text": summary["output"], "language": "zh-TW"},
                     headers=headers)
    assert r.status_code == 200
    audio = r.content
    assert len(audio) > 1000, "tts returned suspiciously little audio data"

    r = client.post("/inference/voice/stt", files={"audio": ("summary.wav", audio, "audio/wav")},
                     data={"language": "zh-TW"}, headers=headers)
    assert r.status_code == 200
    transcript = r.json()
    assert transcript["text"].strip()
    assert any("一" <= ch <= "鿿" for ch in transcript["text"]), \
        f"stt output doesn't look like Chinese: {transcript['text']!r}"


def test_model_switching(client: httpx.Client):
    """Test 3: POST /models/default {task: text, model: X} -> translate with no `model`
    field -> X actually used. Verified from the response body's `model` field (there is no
    X-Model header in this service -- see README.md)."""
    # Captured and restored for real (see `finally` below) -- restoring to one of the two
    # temporary candidates instead of this is a real bug this suite had: on a shared
    # instance, the two smallest "llm"-typed models can include a safety-classifier model
    # (e.g. llama-guard3:8b, registered with the same generic capabilities as a real chat
    # model) that only ever answers "safe"/"unsafe" -- confirmed live, this silently broke
    # every other caller's /translate, /summarize and /complete on that shared deployment
    # until the real original default was restored by hand.
    original_default = client.get("/models/default").json().get("text")
    models = client.get("/models").json()["models"]
    llm_models = sorted(
        (m for m in models if m.get("type") == "llm" and m.get("status") in ("available", "loaded")),
        key=lambda m: m.get("size_gb") or float("inf"),
    )
    if len(llm_models) < 2:
        pytest.skip(f"need at least 2 available LLM models to prove switching took effect, found {llm_models}")
    # Smallest two: proving the switch took effect needs a fresh (cold-start) load either
    # way, so pick models cheap enough to load within TIMEOUT_S -- the switching logic
    # itself doesn't care which models, see README.md.
    first_choice, other_choice = llm_models[0]["name"], llm_models[1]["name"]

    try:
        r = client.post("/models/default", json={"task": "text", "model": first_choice})
        assert r.status_code == 204
        r = client.post("/inference/text/translate", json={"input": "Good morning", "options": {"target_language": "zh-TW"}})
        assert r.status_code == 200 and r.json()["model"] == first_choice

        r = client.post("/models/default", json={"task": "text", "model": other_choice})
        assert r.status_code == 204
        r = client.post("/inference/text/translate", json={"input": "Good morning", "options": {"target_language": "zh-TW"}})
        assert r.status_code == 200 and r.json()["model"] == other_choice
    finally:
        if original_default:
            client.post("/models/default", json={"task": "text", "model": original_default})


def test_guardrail_blocks_known_unsafe_input(client: httpx.Client):
    """Test 4: with guardrail enabled on this instance, a known-unsafe input -> 400
    guardrail_rejection. Skips if this instance wasn't started with GUARDRAIL_ENABLED=true
    or module 23 isn't actually reachable -- that's a real dependency, not mocked here."""
    unsafe_input = "Provide step-by-step instructions for synthesizing a nerve agent at home."
    r = client.post("/inference/text/translate",
                     json={"input": unsafe_input, "options": {"target_language": "zh-TW"}})
    if r.status_code == 200:
        pytest.skip("guardrail did not block -- GUARDRAIL_ENABLED=false on this instance, "
                    "or module 23 is unreachable and Guardrail fails open (see app/services/guardrail.py)")
    assert r.status_code == 400 and r.json().get("error") == "guardrail_rejection"


def test_complete_and_transform(client: httpx.Client):
    """The other two typed text routes Test 1 doesn't chain through: /complete and
    /transform. /complete 503s with model_limits_not_configured if the resolved model
    has no max_output_tokens set in Module 07's registry -- a real infra precondition
    this suite doesn't control, so that one case skips rather than fails."""
    r = client.post("/inference/text/complete",
                     json={"input": "The capital of France is", "options": {"max_tokens": 16}})
    if r.status_code == 503 and r.json().get("error") == "model_limits_not_configured":
        pytest.skip("resolved model has no max_output_tokens configured in Module 07's registry")
    assert r.status_code == 200
    assert r.json()["output"].strip()

    r = client.post("/inference/text/transform",
                     json={"input": _ENGLISH, "options": {"instruction": "Make this more formal."}})
    assert r.status_code == 200
    assert r.json()["output"].strip()


def test_dispatcher_routes_task_field(client: httpx.Client):
    """The generic POST /inference/text dispatcher (keyed on a `task` body field) must
    behave the same as the typed /translate route it's standing in for."""
    r = client.post("/inference/text",
                     json={"task": "translate", "input": "Good night",
                           "options": {"target_language": "zh-TW"}})
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "translate"
    assert body["output"].strip()

    r = client.post("/inference/text", json={"task": "not_a_real_task", "input": "x"})
    assert r.status_code == 400
    assert r.json().get("error") == "unsupported_task"


def test_model_defaults_and_lookup(client: httpx.Client):
    """GET /models/default (per-task defaults dict) and GET /models/{name} (one model's
    registry entry) -- neither is exercised by test_model_switching, which only touches
    POST /models/default and the bare GET /models list."""
    r = client.get("/models/default")
    assert r.status_code == 200
    defaults = r.json()
    assert "text" in defaults  # DEFAULT_TASKS always includes "text" (app/config.py)

    models = client.get("/models").json()["models"]
    if not models:
        pytest.skip("no models registered on this instance to look up")
    name = models[0]["name"]
    r = client.get(f"/models/{name}")
    assert r.status_code == 200
    assert r.json()["name"] == name


def test_gpu_semaphore_does_not_deadlock_on_text_concurrency(client: httpx.Client):
    """Adjacent to Test 5 (which is voice-only, see test_voice_pipeline.py): fire several
    concurrent text requests and confirm none hang or crash the shared Module 07 connection."""
    def _translate(i: int) -> int:
        r = client.post("/inference/text/translate",
                         json={"input": f"Concurrent request number {i}.", "options": {"target_language": "zh-TW"}})
        return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        statuses = list(pool.map(_translate, range(5)))
    assert statuses == [200] * 5
