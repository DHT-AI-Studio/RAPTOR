"""INF-13/VIE01-419 contract/cache tests; neural voice quality requires a real GPU run."""
import asyncio
import hashlib
import io
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from app.errors import ServiceError
from app.services.voice.backends.chatterbox import chatterbox_text
from app.services.voice.backends.xtts import XttsBackend, xtts_language
from app.services.voice.clone import CloneBackendSpec
from app.services.voice.speaker_cache import SpeakerCache

from tests.unit.test_diarize import wav


class Backend:
    """Stands in for XttsBackend: prepare_language mirrors xtts_language()'s
    real mapping (zh-TW -> zh-cn) so existing assertions keep their meaning."""

    name = "xtts-v2"

    def __init__(self):
        self.encodes = 0

    def prepare_language(self, language):
        return xtts_language(language)

    def prepare_text(self, text, language):
        return text

    def encode(self, audio):
        self.encodes += 1
        return b"conditioning"

    def synthesize(self, conditioning, text, language):
        assert conditioning == b"conditioning"
        assert language == "zh-cn"
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(np.arange(2400) * 0.1) * 0.2, 24000, format="WAV")
        return buffer.getvalue()


def attach_backend(client, backend, name="xtts-v2"):
    """Simulate an already-loaded backend without going through the real
    (torch/TTS-requiring) loader -- mirrors the old `cloning.backend = backend`."""
    spec = client.app.state.cloning.specs.setdefault(name, CloneBackendSpec(name=name, loader=lambda: backend))
    spec.instance = backend
    return spec


async def test_named_cache_skips_encoder_and_management_is_user_scoped(client):
    backend = Backend()
    attach_backend(client, backend)
    client.app.state.cloning.disabled_reason = None
    payload = {"text": "早安", "language": "zh-TW", "speaker_name": "demo"}
    for _ in range(2):
        response = await client.post("/inference/voice/clone", data=payload,
            files={"reference_audio": ("reference.wav", wav(10))}, headers={"x-user-id": "alice"})
        assert response.status_code == 200
        assert response.headers["x-speaker-id"] == "demo"
        assert response.headers["x-watermarked"] == "false"
        assert response.headers["content-type"] == "audio/wav"
        audio, rate = sf.read(io.BytesIO(response.content))
        assert rate == 24000 and np.max(np.abs(audio)) > 0
    assert backend.encodes == 1
    speakers = (await client.get("/inference/voice/clone/speakers", headers={"x-user-id": "alice"})).json()
    assert speakers["speakers"][0]["id"] == "demo"
    assert 0 < speakers["speakers"][0]["ttl_remaining_s"] <= 86400
    assert (await client.get("/inference/voice/clone/speakers", headers={"x-user-id": "bob"})).json() == {"speakers": []}
    await client.delete("/inference/voice/clone/speakers/demo", headers={"x-user-id": "bob"})
    assert await SpeakerCache(client.app.state.voice_redis, "alice").get("demo") is not None
    assert (await client.delete("/inference/voice/clone/speakers/demo", headers={"x-user-id": "alice"})).status_code == 204
    assert await SpeakerCache(client.app.state.voice_redis, "alice").get("demo") is None


async def test_hash_key_is_sha256_of_original_file(client):
    attach_backend(client, Backend())
    client.app.state.cloning.disabled_reason = None
    audio = wav(3)
    response = await client.post("/inference/voice/clone", data={"text": "你好", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", audio)}, headers={"x-user-id": "alice"})
    assert response.headers["x-speaker-id"] == hashlib.sha256(audio).hexdigest()


async def test_lru_capacity_and_expired_entries(redis):
    cache = SpeakerCache(redis, "alice", maximum=2)
    await cache.put("a", b"a")
    await cache.put("b", b"b")
    await cache.get("a")
    await cache.put("c", b"c")
    assert await cache.get("a") == b"a"
    assert await cache.get("b") is None
    assert await cache.get("c") == b"c"
    await redis.delete(cache.key("a"))
    await cache.put("d", b"d")
    assert {s["id"] for s in await cache.list()} == {"c", "d"}


@pytest.mark.parametrize("seconds,language,text,status", [
    (0.5, "zh-TW", "hi", 400), (61, "zh-TW", "hi", 400),
    (3, "xx", "hi", 422), (3, "zh-TW", "x" * 3001, 400), (3, "zh-TW", " ", 400),
])
async def test_clone_validation_before_encoder(client, seconds, language, text, status):
    backend = Backend()
    attach_backend(client, backend)
    client.app.state.cloning.disabled_reason = None
    response = await client.post("/inference/voice/clone", data={"text": text, "language": language},
        files={"reference_audio": ("a.wav", wav(seconds))}, headers={"x-user-id": "alice"})
    assert response.status_code == status
    assert backend.encodes == 0


def test_chatterbox_text_converts_traditional_to_simplified_for_zh():
    """Confirmed live: Chatterbox's "zh" model mispronounces Traditional-specific
    glyphs (they Cangjie-encode to token sequences it rarely saw in training) --
    converting to Simplified first fixed every observed case."""
    assert chatterbox_text("這是測試", "zh") == "这是测试"


def test_chatterbox_text_leaves_already_simplified_text_unchanged():
    assert chatterbox_text("这是测试", "zh") == "这是测试"


def test_chatterbox_text_is_a_noop_for_non_chinese_languages():
    assert chatterbox_text("Hello World", "en") == "Hello World"


async def test_clone_disabled_by_default_is_403_before_touching_the_backend(client):
    """VIE01-419 / INF-20: CLONE_ENABLED defaults to false (XTTS-v2 weights are
    Coqui Public Model License, non-commercial) -- 403, not the 503 a genuinely
    unloaded-but-enabled backend would give, and the backend is never called.
    Also proves 403 takes priority over the resolver, which would otherwise see
    zero registered clone entries while disabled and raise 400 first."""
    backend = Backend()
    attach_backend(client, backend)  # even if a backend somehow got attached
    response = await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "clone_disabled" and "CLONE_ENABLED" in body["reason"]
    assert backend.encodes == 0


async def test_clone_enabled_but_backend_not_loaded_is_still_503(make_client, settings):
    client = await make_client(replace(settings, clone_enabled=True))
    response = await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
    assert response.status_code == 503 and response.json()["error"] == "clone_unavailable"


async def test_clone_model_not_found_for_unconfigured_backend(make_client, settings):
    client = await make_client(replace(settings, clone_enabled=True))
    response = await client.post("/inference/voice/clone",
        data={"text": "早安", "language": "zh-TW", "model": "not-a-real-backend"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
    assert response.status_code == 422 and response.json()["error"] == "model_not_found"


async def test_clone_model_required_when_no_default_and_multiple_backends(make_client, settings):
    """With no default configured and 2+ backends enabled, an omitted `model`
    is a genuine ambiguity -- 400, not an arbitrary silent pick."""
    client = await make_client(replace(settings, clone_enabled=True,
                                        default_models={**settings.default_models, "clone": None}))
    cloning = client.app.state.cloning
    cloning.specs["chatterbox-multilingual-v3"] = CloneBackendSpec(
        name="chatterbox-multilingual-v3", loader=lambda: Backend(), watermarked=True,
    )
    response = await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
    assert response.status_code == 400 and response.json()["error"] == "model_required"


class ChatterboxLikeBackend:
    """A second, independent backend shape: different name/language/encoding --
    used to prove Cloning.clone() dispatches generically, not just to XTTS."""

    name = "chatterbox-multilingual-v3"

    def __init__(self):
        self.encodes = 0

    def prepare_language(self, language):
        return "zh" if language.split("-")[0].lower() == "zh" else language.split("-")[0].lower()

    def prepare_text(self, text, language):
        return text

    def encode(self, audio):
        self.encodes += 1
        return b"chatterbox-conditioning"

    def synthesize(self, conditioning, text, language):
        assert conditioning == b"chatterbox-conditioning"
        assert language == "zh"
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(np.arange(2400) * 0.1) * 0.2, 24000, format="WAV")
        return buffer.getvalue()


async def test_explicit_model_selects_the_right_backend_end_to_end(make_client, settings):
    client = await make_client(replace(settings, clone_enabled=True))
    xtts = Backend()
    chatterbox = ChatterboxLikeBackend()
    attach_backend(client, xtts, "xtts-v2")
    client.app.state.cloning.specs["chatterbox-multilingual-v3"] = CloneBackendSpec(
        name="chatterbox-multilingual-v3", loader=lambda: chatterbox, watermarked=True, instance=chatterbox,
    )
    for model, backend, watermarked in [("xtts-v2", xtts, "false"),
                                         ("chatterbox-multilingual-v3", chatterbox, "true")]:
        response = await client.post("/inference/voice/clone",
            data={"text": "早安", "language": "zh-TW", "model": model},
            files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
        assert response.status_code == 200
        assert response.headers["x-watermarked"] == watermarked
    assert xtts.encodes == 1 and chatterbox.encodes == 1


async def test_same_speaker_name_under_two_backends_does_not_collide(make_client, settings):
    """Caching a bare speaker_id across backends would feed one backend's
    conditioning bytes into another's deserializer -- a real format-confusion
    bug, not just quality loss. Confirms each backend gets its own cache entry."""
    client = await make_client(replace(settings, clone_enabled=True))
    xtts = Backend()
    chatterbox = ChatterboxLikeBackend()
    attach_backend(client, xtts, "xtts-v2")
    client.app.state.cloning.specs["chatterbox-multilingual-v3"] = CloneBackendSpec(
        name="chatterbox-multilingual-v3", loader=lambda: chatterbox, watermarked=True, instance=chatterbox,
    )
    payload = {"text": "早安", "language": "zh-TW", "speaker_name": "shared"}
    ids = set()
    for model in ("xtts-v2", "chatterbox-multilingual-v3"):
        response = await client.post("/inference/voice/clone", data={**payload, "model": model},
            files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
        assert response.status_code == 200
        ids.add(response.headers["x-speaker-id"])
    assert len(ids) == 2  # qualified, not the same cache entry
    assert xtts.encodes == 1 and chatterbox.encodes == 1


async def test_single_backend_speaker_id_format_is_unqualified(client):
    """Regression guard: with exactly one backend configured (the common
    case), X-Speaker-Id must stay byte-identical to the pre-multi-backend
    format -- no backend-name prefix when there's nothing to disambiguate."""
    attach_backend(client, Backend())
    client.app.state.cloning.disabled_reason = None
    response = await client.post("/inference/voice/clone",
        data={"text": "早安", "language": "zh-TW", "speaker_name": "demo"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
    assert response.headers["x-speaker-id"] == "demo"


async def test_lazy_backend_loads_on_first_use_not_before(make_client, settings):
    loads = []

    def loader():
        loads.append(1)
        return Backend()

    client = await make_client(replace(settings, clone_enabled=True))
    client.app.state.cloning.specs["xtts-v2"] = CloneBackendSpec(name="xtts-v2", loader=loader)
    assert loads == []  # not loaded merely by being configured
    assert client.app.state.cloning.loaded_backends() == []
    response = await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})
    assert response.status_code == 200
    assert loads == [1]
    assert client.app.state.cloning.loaded_backends() == ["xtts-v2"]


async def test_concurrent_first_use_loads_the_cold_backend_exactly_once(make_client, settings):
    load_count = 0

    def loader():
        nonlocal load_count
        load_count += 1
        time.sleep(0.05)  # runs in a thread via GpuLock.run; a race window
        return Backend()

    client = await make_client(replace(settings, clone_enabled=True))
    client.app.state.cloning.specs["xtts-v2"] = CloneBackendSpec(name="xtts-v2", loader=loader)

    async def one():
        return await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
            files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})

    responses = await asyncio.gather(one(), one())
    assert all(r.status_code == 200 for r in responses)
    assert load_count == 1


async def test_backend_construction_is_serialized_against_other_gpu_work(make_client, settings):
    """Construction must go through the same GpuLock as encode/synthesize --
    otherwise a lazy load could run concurrently with another backend's GPU
    call on the same CUDA context."""
    events = []

    def slow_loader(label):
        def _load():
            events.append(f"{label}-start")
            time.sleep(0.05)
            events.append(f"{label}-end")
            return Backend()
        return _load

    client = await make_client(replace(settings, clone_enabled=True))
    client.app.state.cloning.specs["xtts-v2"] = CloneBackendSpec(name="xtts-v2", loader=slow_loader("a"))
    client.app.state.cloning.specs["chatterbox-multilingual-v3"] = CloneBackendSpec(
        name="chatterbox-multilingual-v3", loader=slow_loader("b"), watermarked=True,
    )

    async def load(model):
        return await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW", "model": model},
            files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": "alice"})

    await asyncio.gather(load("xtts-v2"), load("chatterbox-multilingual-v3"))
    # Serialized (one fully starts and ends before the other starts) -- never
    # interleaved like ["a-start", "b-start", "a-end", "b-end"].
    assert events in (["a-start", "a-end", "b-start", "b-end"], ["b-start", "b-end", "a-start", "a-end"])


async def test_low_vram_warning_in_health(client, monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True, mem_get_info=lambda: (8 * 1024**3, 16 * 1024**3),
        get_device_properties=lambda index: SimpleNamespace(is_integrated=False),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    client.app.state.settings = replace(client.app.state.settings, voice_enabled=True)
    response = await client.get("/health")
    assert response.json()["warnings"] == ["Available GPU VRAM is below 12 GiB"]


async def test_low_vram_warning_on_unified_memory_gpu_uses_mem_available(client, monkeypatch):
    """GX10/Grace-Blackwell: cudaMemGetInfo under-reports free memory (host MemFree, not
    MemAvailable) on a unified-memory GPU -- don't warn when MemAvailable shows real headroom."""
    cuda = SimpleNamespace(
        is_available=lambda: True, mem_get_info=lambda: (2 * 1024**3, 16 * 1024**3),
        get_device_properties=lambda index: SimpleNamespace(is_integrated=True),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        "app.services.voice.clone._host_mem_available_bytes", lambda: 13 * 1024**3,
    )
    client.app.state.settings = replace(client.app.state.settings, voice_enabled=True)
    response = await client.get("/health")
    assert response.json()["warnings"] == []


async def test_health_reports_configured_and_loaded_clone_backends(client):
    response = await client.get("/health")
    body = response.json()
    assert body["voice"]["clone_configured_backends"] == ["xtts-v2"]
    assert body["loaded_voice_models"] == []  # nothing loaded yet -- lazy
    attach_backend(client, Backend())
    response = await client.get("/health")
    assert response.json()["loaded_voice_models"] == ["xtts-v2"]


@pytest.mark.parametrize("invalid", [False, True])
def test_xtts_float16_waveform_encodes_and_nonfinite_audio_is_rejected(monkeypatch, invalid):
    torch = SimpleNamespace(
        float32=np.float32, float16=np.float16, as_tensor=lambda value, **kwargs: value,
        inference_mode=nullcontext, autocast=lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    backend = object.__new__(XttsBackend)
    waveform = np.full(2400, np.nan, dtype=np.float16) if invalid else np.sin(np.arange(2400) * .1).astype(np.float16)
    backend.model = SimpleNamespace(inference=lambda *args, **kwargs: {"wav": waveform})
    buffer = io.BytesIO()
    np.savez(buffer, latent=np.ones((1, 2)), embedding=np.ones((1, 2)))
    if invalid:
        with pytest.raises(ServiceError, match="clone_unavailable"):
            backend.synthesize(buffer.getvalue(), "早安", "zh-cn")
    else:
        samples, rate = sf.read(io.BytesIO(backend.synthesize(buffer.getvalue(), "早安", "zh-cn")))
        assert rate == 24000 and np.std(samples) > .1


# ── GD-11: voice_clone consent ──────────────────────────────────────────────

async def _clone(client, user="alice"):
    return await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", wav(3))}, headers={"x-user-id": user})


async def test_clone_asks_module_33_for_the_callers_voice_clone_consent(client, upstream):
    backend = Backend()
    attach_backend(client, backend)
    client.app.state.cloning.disabled_reason = None
    assert (await _clone(client)).status_code == 200
    assert upstream.consent_calls == [("alice", "test-internal-token")]


async def test_clone_without_voice_clone_consent_is_403_before_the_reference_audio(client, upstream):
    backend = Backend()
    attach_backend(client, backend)
    client.app.state.cloning.disabled_reason = None
    upstream.voice_clone_consent = False
    response = await _clone(client)
    assert response.status_code == 403
    assert response.json() == {"error": "consent_required", "purpose": "voice_clone"}
    assert backend.encodes == 0
    # Nothing cached: the speaker embedding is never computed, let alone stored.
    assert (await client.get("/inference/voice/clone/speakers", headers={"x-user-id": "alice"})).json() == {"speakers": []}


async def test_clone_for_a_user_module_33_does_not_know_is_403(client, upstream):
    backend = Backend()
    attach_backend(client, backend)
    client.app.state.cloning.disabled_reason = None
    upstream.consent_status = 404
    response = await _clone(client)
    assert response.status_code == 403 and response.json()["error"] == "consent_required"
    assert backend.encodes == 0


@pytest.mark.parametrize("failure", ["down", "500"])
async def test_clone_fails_closed_when_consent_cant_be_read(client, upstream, failure):
    backend = Backend()
    attach_backend(client, backend)
    client.app.state.cloning.disabled_reason = None
    if failure == "down":
        upstream.consent_down = True
    else:
        upstream.consent_status = 500
    response = await _clone(client)
    assert response.status_code == 503
    assert response.json() == {"error": "consent_check_unavailable", "purpose": "voice_clone"}
    assert backend.encodes == 0


async def test_clone_disabled_still_answers_403_without_asking_module_33(client, upstream):
    """CLONE_ENABLED off stays the cheap, no-I/O first answer."""
    assert (await _clone(client)).json()["error"] == "clone_disabled"
    assert upstream.consent_calls == []


async def test_clone_without_user_identity_is_401_without_asking_module_33(client, upstream):
    client.app.state.cloning.disabled_reason = None
    response = await client.post("/inference/voice/clone", data={"text": "早安", "language": "zh-TW"},
        files={"reference_audio": ("a.wav", wav(3))})
    assert response.status_code == 401 and upstream.consent_calls == []


async def test_speaker_list_and_delete_need_no_consent(client, upstream):
    """A user who withdrew must still be able to see and delete what's cached."""
    upstream.voice_clone_consent = False
    assert (await client.get("/inference/voice/clone/speakers", headers={"x-user-id": "alice"})).status_code == 200
    assert (await client.delete("/inference/voice/clone/speakers/demo", headers={"x-user-id": "alice"})).status_code == 204
    assert upstream.consent_calls == []
