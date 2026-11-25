"""INF-12: offline pyannote 3.1 diarization and WhisperX aligned segment STT."""
from __future__ import annotations

import logging
import platform
import tempfile
import time
from pathlib import Path

from app.errors import ServiceError
from app.services.voice.audio import Audio, language_code
from app.services.voice.gpu_lock import GpuLock

logger = logging.getLogger(__name__)

# ctranslate2 (whisperx/faster-whisper's ASR engine) ships no CUDA-enabled linux_aarch64
# wheel on PyPI -- same root cause already fixed for Module 34's own STT
# (34-realtime-audio/app/stt.py's HFWhisperBackend) and Module 09's ASR. On arm64, the
# per-segment ASR step below runs through HF transformers' own Whisper pipeline instead,
# which talks to torch CUDA directly and never touches ctranslate2. prefetch_voice.py bakes
# the matching checkpoint format (HF safetensors vs CTranslate2) into root/"whisper" per arch,
# so this stays fully offline on both.
_IS_ARM64 = platform.machine() in ("aarch64", "arm64")


def _hf_chunks_to_segments(chunks: list[dict]) -> list[dict]:
    """HF transformers' ``return_timestamps=True`` chunks -> whisperx.align()'s input shape
    (just start/end/text -- whisperx redoes word-level alignment itself). A ``None``
    timestamp (HF can collapse a chunk's end to ``None``, see stt.py's own
    ``HFWhisperBackend`` note) carries the previous segment's end forward so align() never
    sees a null boundary."""
    segments: list[dict] = []
    last_end = 0.0
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        start, end = chunk.get("timestamp") or (None, None)
        start = last_end if start is None else start
        end = start if end is None else end
        segments.append({"start": start, "end": end, "text": text})
        last_end = end
    return segments


class PyannoteWhisperX:
    """Preloaded local models. Startup never downloads missing weights."""

    def __init__(self, root: str, stt_model: str = "large-v3") -> None:
        import torch
        import whisperx
        import yaml
        from pyannote.audio import Pipeline

        base = Path(root)
        config = base / "diarize.yaml"
        stt_path = base / "whisper"
        align_path = base / "align-zh"
        if not config.exists() or not stt_path.is_dir() or not align_path.is_dir():
            raise FileNotFoundError("voice model assets missing")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        # Resolve exported relative weights in the runtime mount, without
        # modifying a read-only NAS bundle or retaining host-specific paths.
        pipeline_config = yaml.safe_load(config.read_text())
        for key in ("segmentation", "embedding"):
            checkpoint = base / f"{key}.bin"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{key} model missing")
            pipeline_config["pipeline"]["params"][key] = str(checkpoint.resolve())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as local:
            yaml.safe_dump(pipeline_config, local)
            local.flush()
            self.pipeline = Pipeline.from_pretrained(local.name)
        if self.pipeline is None:
            raise RuntimeError("pyannote model not loaded")
        self.pipeline.to(torch.device("cuda"))
        # ASR is loaded for each requested language using the same local checkpoint.
        self.stt_path = str(stt_path)
        self.whisperx = whisperx
        self.asr = None
        self.asr_language = None
        self.align_path = str(align_path)
        from whisperx.vad import load_vad_model
        self.vad = load_vad_model("cuda", model_fp=str(base / "whisperx-vad.bin"))
        self.align, self.align_metadata = whisperx.load_align_model(
            language_code="zh", device="cuda", model_name=self.align_path,
        )

    def _load_asr(self, lang: str | None):
        if _IS_ARM64:
            import torch
            from transformers import pipeline as hf_pipeline

            return hf_pipeline("automatic-speech-recognition", model=self.stt_path,
                               torch_dtype=torch.float16, device="cuda")
        return self.whisperx.load_model(
            self.stt_path, "cuda", compute_type="float16", language=lang,
            vad_model=self.vad, local_files_only=True,
        )

    def _detect_language(self, samples) -> str | None:
        """Whisper's own lightweight language-ID forward pass (one forward pass, not a
        full decode). Confirmed empirically (openai/whisper-tiny, a short clip, auto-
        detect): the ASR pipeline's own ``return_language`` chunk field always comes back
        ``None`` here -- unlike ctranslate2/faster-whisper's ``info.language`` on amd64,
        which is always populated regardless of clip length."""
        import torch

        inputs = self.asr.feature_extractor(samples, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.asr.model.device, dtype=self.asr.model.dtype)
        with torch.no_grad():
            lang_id = self.asr.model.detect_language(input_features=input_features)[0].item()
        token = self.asr.tokenizer.convert_ids_to_tokens([lang_id])[0]
        return token.strip("<|>") or None

    def _transcribe(self, samples, lang: str | None) -> tuple[list[dict], str | None]:
        """Return (whisperx.align()-shaped segments, detected language)."""
        if _IS_ARM64:
            generate_kwargs: dict = {"task": "transcribe", "num_beams": 1}
            if lang:
                generate_kwargs["language"] = lang
            result = self.asr({"raw": samples, "sampling_rate": 16000}, return_timestamps=True,
                              generate_kwargs=generate_kwargs)
            # A forced language is never re-detected (same semantics as faster-whisper:
            # forcing skips detection); omitted, it needs the real check above.
            detected = lang or self._detect_language(samples)
            return _hf_chunks_to_segments(result.get("chunks") or []), detected
        transcript = self.asr.transcribe(samples, batch_size=1, language=lang)
        return transcript["segments"], transcript.get("language")

    def diarize(self, audio: Audio, num_speakers: int | None, language: str | None) -> list[dict]:
        """Use the same waveform for diarization, per-segment STT, and alignment."""
        import torch

        lang = language_code(language)
        if lang not in {None, "zh"}:
            raise ServiceError(422, "alignment_language_unavailable", language=language)
        kwargs = {"num_speakers": num_speakers} if num_speakers is not None else {}
        annotation = self.pipeline(
            {"waveform": torch.from_numpy(audio.samples).unsqueeze(0), "sample_rate": 16000},
            **kwargs,
        )
        if self.asr is None or self.asr_language != lang:
            self.asr = self._load_asr(lang)
            self.asr_language = lang
        output = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start, end = max(0.0, turn.start), min(audio.duration, turn.end)
            if end <= start:
                continue
            samples = audio.samples[int(start * 16000):int(end * 16000)]
            segments, detected_language = self._transcribe(samples, lang)
            if detected_language != "zh":
                raise ServiceError(422, "alignment_language_unavailable", language=detected_language)
            aligned = self.whisperx.align(segments, self.align, self.align_metadata, samples, "cuda")
            for segment in aligned["segments"]:
                output.append({"speaker": str(speaker), "start": start + segment["start"],
                               "end": start + segment["end"], "text": segment["text"].strip()})
        return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


class Diarization:
    """Expose a stable contract while allowing startup with absent models."""

    def __init__(self, gpu: GpuLock, backend=None) -> None:
        self.gpu = gpu
        self.backend = backend

    async def diarize(
        self, audio: Audio, model: str = "pyannote-3.1",
        num_speakers: int | None = None, language: str | None = None,
    ) -> dict:
        """Return speaker-labelled aligned segments; requested counts remain model hints."""
        if model != "pyannote-3.1":
            raise ServiceError(400, "unsupported_model", model=model)
        if num_speakers is not None and not 1 <= num_speakers <= 64:
            raise ServiceError(400, "invalid_num_speakers")
        language_code(language)
        if self.backend is None:
            raise ServiceError(503, "diarize_unavailable", reason="pyannote model not loaded")
        started = time.perf_counter()
        segments = await self.gpu.run(self.backend.diarize, audio, num_speakers, language)
        return {"speakers": sorted({seg["speaker"] for seg in segments}), "segments": segments,
                "latency_ms": round((time.perf_counter() - started) * 1000)}
