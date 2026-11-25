"""Diarization HTTP contract, missing models, count hints and cancelled GPU work."""
import asyncio
import io
import threading

import numpy as np
import pytest
import soundfile as sf
from app.services.voice.diarize import PyannoteWhisperX, _hf_chunks_to_segments
from app.services.voice.gpu_lock import GpuLock


def wav(seconds=2):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * 16000)), 16000, format="WAV")
    return buffer.getvalue()


async def test_missing_pyannote_returns_503(client):
    response = await client.post("/inference/voice/diarize", files={"audio": ("a.wav", wav())})
    assert response.status_code == 503
    assert response.json() == {"error": "diarize_unavailable", "reason": "pyannote model not loaded"}


@pytest.mark.parametrize("count", [None, 2, 5])
async def test_two_speaker_pipeline_and_count_hint(client, count):
    class Backend:
        def diarize(self, audio, num_speakers, language):
            assert num_speakers == count
            assert language == "zh-TW"
            assert audio.duration == 2
            return [{"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "早安"},
                    {"speaker": "SPEAKER_01", "start": 1, "end": 2, "text": "你好"}]
    client.app.state.diarization.backend = Backend()
    data = {"language": "zh-TW"}
    if count is not None:
        data["num_speakers"] = str(count)
    response = await client.post("/inference/voice/diarize", data=data,
                                 files={"audio": ("a.wav", wav())})
    assert response.status_code == 200
    assert response.json()["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert len(response.json()["segments"]) == 2


async def test_cancelled_caller_does_not_release_gpu_while_thread_runs():
    lock = GpuLock()
    started, finish = threading.Event(), threading.Event()
    def worker():
        started.set()
        assert finish.wait(3)
    task = asyncio.create_task(lock.run(worker))
    try:
        await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lock.semaphore.locked()
        next_task = asyncio.create_task(lock.run(lambda: "next"))
        await asyncio.sleep(0.02)
        assert not next_task.done()
        finish.set()
        assert await next_task == "next"
    finally:
        finish.set()


# ---------------------------------------------------------------- arm64 ASR swap (ctranslate2
# has no CUDA-enabled linux_aarch64 wheel; _IS_ARM64 routes through HF transformers instead)

def test_hf_chunks_to_segments_basic():
    chunks = [{"text": "你好", "timestamp": (0.0, 1.0)}, {"text": "世界", "timestamp": (1.0, 2.0)}]
    assert _hf_chunks_to_segments(chunks) == [
        {"start": 0.0, "end": 1.0, "text": "你好"}, {"start": 1.0, "end": 2.0, "text": "世界"},
    ]


def test_hf_chunks_to_segments_skips_empty_text():
    chunks = [{"text": "   ", "timestamp": (0.0, 1.0)}, {"text": "你好", "timestamp": (1.0, 2.0)}]
    assert _hf_chunks_to_segments(chunks) == [{"start": 1.0, "end": 2.0, "text": "你好"}]


def test_hf_chunks_to_segments_fills_missing_timestamps():
    chunks = [{"text": "你好", "timestamp": (0.0, 1.0)}, {"text": "世界", "timestamp": (None, None)}]
    assert _hf_chunks_to_segments(chunks)[1] == {"start": 1.0, "end": 1.0, "text": "世界"}


def test_transcribe_dispatches_to_hf_pipeline_on_arm64(monkeypatch):
    import app.services.voice.diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "_IS_ARM64", True)
    backend = object.__new__(PyannoteWhisperX)
    calls = []

    def fake_asr(inputs, **kwargs):
        calls.append((inputs, kwargs))
        return {"chunks": [{"text": "你好", "timestamp": (0.0, 1.0)}], "language": "zh"}

    backend.asr = fake_asr
    segments, language = backend._transcribe(np.zeros(16000, dtype="float32"), "zh")
    assert language == "zh"
    assert segments == [{"start": 0.0, "end": 1.0, "text": "你好"}]
    assert calls[0][1]["generate_kwargs"] == {"task": "transcribe", "num_beams": 1, "language": "zh"}


def test_transcribe_detects_language_on_arm64_when_none_is_requested(monkeypatch):
    import app.services.voice.diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "_IS_ARM64", True)
    backend = object.__new__(PyannoteWhisperX)
    backend.asr = lambda inputs, **kwargs: {"chunks": [{"text": "你好", "timestamp": (0.0, 1.0)}]}
    backend._detect_language = lambda samples: "zh"
    segments, language = backend._transcribe(np.zeros(16000, dtype="float32"), None)
    assert language == "zh"
    assert segments == [{"start": 0.0, "end": 1.0, "text": "你好"}]


def test_detect_language_reads_whispers_own_language_id_forward_pass():
    import torch

    backend = object.__new__(PyannoteWhisperX)

    class FakeFeatureExtractor:
        def __call__(self, samples, sampling_rate, return_tensors):
            return type("Inputs", (), {"input_features": torch.zeros(1, 80, 3000)})()

    class FakeModel:
        device, dtype = "cpu", torch.float32

        def detect_language(self, input_features):
            return torch.tensor([50260])

    class FakeTokenizer:
        def convert_ids_to_tokens(self, ids):
            return ["<|zh|>"]

    backend.asr = type("FakeAsr", (), {
        "feature_extractor": FakeFeatureExtractor(), "model": FakeModel(), "tokenizer": FakeTokenizer(),
    })()
    assert backend._detect_language(np.zeros(16000, dtype="float32")) == "zh"


def test_transcribe_dispatches_to_whisperx_on_amd64(monkeypatch):
    import app.services.voice.diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "_IS_ARM64", False)
    backend = object.__new__(PyannoteWhisperX)

    class FakeAsr:
        def transcribe(self, samples, *, batch_size, language):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "你好"}], "language": "zh"}

    backend.asr = FakeAsr()
    segments, language = backend._transcribe(np.zeros(16000, dtype="float32"), "zh")
    assert language == "zh"
    assert segments == [{"start": 0.0, "end": 1.0, "text": "你好"}]


def test_load_asr_uses_transformers_pipeline_on_arm64(monkeypatch):
    import sys
    import types

    import app.services.voice.diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "_IS_ARM64", True)
    calls = []

    class FakeTorch:
        float16 = "float16"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = lambda task, **kwargs: (calls.append((task, kwargs)), "fake-pipe")[1]
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = object.__new__(PyannoteWhisperX)
    backend.stt_path = "/opt/voice-models/whisper"
    assert backend._load_asr("zh") == "fake-pipe"
    assert calls[0][0] == "automatic-speech-recognition"
    assert calls[0][1]["model"] == "/opt/voice-models/whisper"
    assert calls[0][1]["device"] == "cuda"
    assert calls[0][1]["torch_dtype"] == "float16"


def test_load_asr_uses_whisperx_on_amd64(monkeypatch):
    import app.services.voice.diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "_IS_ARM64", False)
    backend = object.__new__(PyannoteWhisperX)
    backend.stt_path = "/opt/voice-models/whisper"
    backend.vad = "fake-vad"
    calls = []

    class FakeWhisperx:
        @staticmethod
        def load_model(path, device, **kwargs):
            calls.append((path, device, kwargs))
            return "fake-ct2-model"

    backend.whisperx = FakeWhisperx()
    assert backend._load_asr("zh") == "fake-ct2-model"
    assert calls[0] == ("/opt/voice-models/whisper", "cuda", {
        "compute_type": "float16", "language": "zh", "vad_model": "fake-vad", "local_files_only": True,
    })
