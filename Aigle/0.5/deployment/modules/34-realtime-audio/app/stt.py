"""RA-3 / RTA-X2: speech-to-text for finished utterances and for speech still in progress."""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

Send = Callable[[dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int
    probability: float


@dataclass(frozen=True)
class Transcript:
    text: str
    words: tuple[Word, ...]
    language: str | None
    language_probability: float | None = None

    @property
    def confidence(self) -> float | None:
        if not self.words:
            return None
        return round(sum(w.probability for w in self.words) / len(self.words), 3)


OnWords = Callable[[Sequence[Word], "str | None"], None]   # (words, detected language)


class SttBackend(Protocol):
    """A loaded speech-to-text model.  ``transcribe`` runs in a worker thread; ``on_words``
    (if given) is called from that thread once per decoded segment, with the language
    Whisper is decoding (detected or forced)."""

    name: str
    device: str
    compute_type: str

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        *,
        beam_size: int,
        prompt: str | None = None,
        on_words: OnWords | None = None,
        max_new_tokens: int | None = None,
        fallback: bool = True,
    ) -> Transcript: ...


def token_budget(samples: int) -> int:
    """Upper bound on decoded tokens for this much audio.  Fast speech is ~6 characters (about
    12 tokens) a second; the bound stops a decoder caught in a repetition loop from running to
    Whisper's 448-token limit while it holds a GPU slot."""
    return 16 + round(16 * samples / SAMPLE_RATE)


# ---------------------------------------------------------------- languages

_SIMPLIFIED = {"CN", "SG", "HANS"}


def whisper_language(tag: str | None) -> str | None:
    """BCP-47 tag -> Whisper language code (None = auto-detect)."""
    if tag is None:
        return None
    code = tag.split("-")[0].lower()
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
    except ImportError:            # unit tests run without faster-whisper
        return code
    if code not in _LANGUAGE_CODES:
        raise ValueError(f"language {tag!r} is not supported by speech recognition")
    return code


def wants_traditional(tag: str | None, detected: str | None) -> bool:
    """Whether Chinese output for this session should be converted to traditional script."""
    if tag is None:
        return detected == "zh"
    parts = [p.upper() for p in tag.split("-")]
    if parts[0] != "ZH":
        return False
    # zh-TW / zh-HK / zh-Hant and bare "zh" (this deployment serves Taiwan) convert; an explicit
    # Hans script or a simplified-script region does not, and Hans wins over a region.
    return not any(p in _SIMPLIFIED for p in parts[1:])


class ZhConverter:
    def __init__(self, config: str) -> None:
        self._cc = None
        if config:
            from opencc import OpenCC

            self._cc = OpenCC(config)

    def __call__(self, text: str) -> str:
        return self._cc.convert(text) if self._cc else text


# ---------------------------------------------------------------- faster-whisper

# ctranslate2's aarch64 wheel has no CUDA support at all (confirmed by
# extracting both PyPI wheels directly: the x86_64 build bundles libcudnn,
# the aarch64 one doesn't) -- faster-whisper is built on ctranslate2, so on
# GX10 it silently falls back to CPU instead of raising. Same root cause,
# same fix module 09's ASR step already applied (see its recognizer.py):
# swap to HF transformers' own Whisper pipeline on arm64, which runs on
# torch CUDA directly and never touches ctranslate2.
_IS_ARM64 = platform.machine() in ("aarch64", "arm64")


def resolve_model(name: str, model_dir: str) -> str:
    """A size name ("large-v3"), a directory under RT_STT_MODEL_DIR, or a path."""
    if os.path.isdir(name):
        return name
    if model_dir:
        for candidate in (name, f"faster-whisper-{name}"):
            path = os.path.join(model_dir, candidate)
            if os.path.isdir(path):
                return path
    return name    # faster-whisper resolves size names from the Hugging Face cache


class FasterWhisperBackend:
    def __init__(self, model: str, device: str, compute_type: str, workers: int) -> None:
        from faster_whisper import WhisperModel

        self.name = os.path.basename(model.rstrip("/")) or model
        self.device = device
        self.compute_type = compute_type
        self._model = WhisperModel(model, device=device, compute_type=compute_type, num_workers=max(1, workers))

    def transcribe(self, audio, language, *, beam_size, prompt=None, on_words=None,
                   max_new_tokens=None, fallback=True) -> Transcript:
        # Without fallback a single greedy/beam pass at temperature 0: no re-decoding of
        # low-confidence windows, which keeps rolling decodes short and predictable.
        temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if fallback else 0.0
        segments, info = self._model.transcribe(
            audio, language=language, beam_size=beam_size, word_timestamps=True,
            initial_prompt=prompt or None, condition_on_previous_text=False, vad_filter=False,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        texts: list[str] = []
        words: list[Word] = []
        for seg in segments:
            seg_words = [Word(w.word, round(w.start * 1000), round(w.end * 1000), float(w.probability))
                         for w in (seg.words or [])]
            texts.append(seg.text)
            words.extend(seg_words)
            if on_words and seg_words:
                on_words(seg_words, info.language)
        return Transcript("".join(texts).strip(), tuple(words), info.language,
                          round(float(info.language_probability), 3))


def cuda_available() -> bool:
    if _IS_ARM64:
        # ctranslate2.get_cuda_device_count() would always report 0 here
        # regardless of the GPU's own health (see _IS_ARM64's comment) --
        # ask torch instead, since HFWhisperBackend is what actually runs.
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:  # noqa: BLE001 -- no CUDA runtime at all means CPU
            return False
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 -- no CUDA runtime at all means CPU
        return False


def _hf_model_id(name: str) -> str:
    """A bare size name ("large-v3", "tiny") becomes an openai/whisper-*
    Hugging Face repo id -- the same convention module 09's own arm64 ASR
    swap uses (recognizer.py). Anything that already looks like a path or a
    full repo id (contains "/") is used as-is."""
    if "/" in name or os.path.isdir(name):
        return name
    return f"openai/whisper-{name}"


_HF_SEGMENT_PAUSE_GAP = 0.8          # seconds; splits HF's timestamped chunks into segments
_HF_SENTENCE_END_CHARS = ".!?。！？"


def _hf_chunks_to_segments(chunks: Sequence[dict[str, Any]]) -> list[list[Word]]:
    """Group HF's ``return_timestamps=True`` chunk list into Whisper-shaped
    segments (each a list of Word, really chunk-sized -- see
    HFWhisperBackend's own docstring) -- same pause-gap/sentence-end
    heuristic as module 09's ``_transcribe_hf`` (09-audio-processing/.../
    recognizer.py), kept standalone so it's unit-testable without a loaded
    model. HF gives no per-chunk confidence, so every Word gets
    probability=1.0, matching that same precedent."""
    segments: list[list[Word]] = []
    current: list[Word] = []
    last_end = 0.0
    for chunk in chunks:
        text = chunk.get("text") or ""
        if not text.strip():
            continue
        start, end = chunk.get("timestamp") or (None, None)
        start = last_end if start is None else start
        end = start if end is None else end
        if current and (start - last_end) > _HF_SEGMENT_PAUSE_GAP:
            segments.append(current)
            current = []
        current.append(Word(text, round(start * 1000), round(end * 1000), 1.0))
        last_end = end
        if text.strip() and text.strip()[-1] in _HF_SENTENCE_END_CHARS:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


class HFWhisperBackend:
    """arm64 (GX10) stand-in for FasterWhisperBackend -- see _IS_ARM64's own
    comment for why. Runs HF transformers' ASR pipeline on torch CUDA
    directly, no ctranslate2 involved.

    Requests return_timestamps=True, not "word": "word" forces
    output_attentions=True (needed for its DTW alignment), which disables
    SDPA and falls back to slow manual attention -- 13-21s per decode live
    on GX10, vs 0.4-1.2s with plain True. On this module's actual short
    utterances, "word" wasn't even giving real per-word output anyway
    (collapsed to one chunk with a None end timestamp); nothing here reads
    per-word timing downstream (Module 15 only sees the joined text).

    on_words fires once per chunk after the whole decode finishes, not
    interleaved with it like faster-whisper's own generator -- still
    correct, just without the provisional-decode latency benefit.

    fallback (FasterWhisperBackend's retry-on-low-confidence) has no HF
    equivalent and is ignored here.

    num_beams pinned to 1 (beam_size ignored): num_beams>1 crashed with
    word-level timestamps on transformers==4.51.3 (beam_indices shape
    mismatch); kept pinned since nothing here needs beam search badly
    enough to risk it again.
    """

    def __init__(self, model: str, device: str, compute_type: str) -> None:
        import torch
        from transformers import pipeline as hf_pipeline

        self.name = os.path.basename(model.rstrip("/")) or model
        self.device = device
        self.compute_type = compute_type
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = hf_pipeline("automatic-speech-recognition", model=model, torch_dtype=dtype, device=device)

    def transcribe(self, audio, language, *, beam_size, prompt=None, on_words=None,
                   max_new_tokens=None, fallback=True) -> Transcript:
        # num_beams pinned to 1, not max(1, beam_size) -- see the class docstring.
        generate_kwargs: dict[str, Any] = {"task": "transcribe", "num_beams": 1}
        if language:
            generate_kwargs["language"] = language
        if max_new_tokens:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        if prompt:
            try:
                generate_kwargs["prompt_ids"] = self._pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
            except Exception:  # prompting is best-effort; never block a decode over it
                logger.warning("HF Whisper prompt_ids failed, decoding without a prompt", exc_info=True)

        result = self._pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps=True, return_language=True, generate_kwargs=generate_kwargs,
        )
        chunks = result.get("chunks") or []
        detected = result.get("language") or (chunks[0].get("language") if chunks else None) or language

        words: list[Word] = []
        for seg_words in _hf_chunks_to_segments(chunks):
            words.extend(seg_words)
            if on_words:
                on_words(seg_words, detected)
        return Transcript("".join(w.text for w in words).strip(), tuple(words), detected, None)


@dataclass(frozen=True)
class SttOptions:
    model: str = "large-v3"
    model_dir: str = ""
    provisional_model: str = ""
    device: str = "auto"
    compute_type: str = ""
    cpu_model: str = "tiny"
    beam_size: int = 5
    provisional_beam_size: int = 1
    max_concurrent: int = 1
    zh_convert: str = "s2tw"


def faster_whisper_loader(opts: SttOptions) -> Callable[[str], SttBackend]:
    """Returns a loader ``model_name -> backend``.  ``RT_STT_DEVICE=auto`` uses CUDA float16
    when a GPU is visible; without one it falls back to RT_STT_CPU_MODEL (tiny) in int8 so a
    CPU-only CI runner can still exercise the pipeline."""

    def load(model: str) -> SttBackend:
        device = opts.device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
            if device == "cpu" and opts.cpu_model:
                logger.warning("no CUDA device: STT falls back to %s on CPU", opts.cpu_model)
                model = opts.cpu_model
        compute = opts.compute_type or ("float16" if device == "cuda" else "int8")
        if _IS_ARM64:
            backend: SttBackend = HFWhisperBackend(_hf_model_id(model), device, compute)
        else:
            path = resolve_model(model, opts.model_dir)
            backend = FasterWhisperBackend(path, device, compute, opts.max_concurrent)
        logger.info("STT model %s loaded (%s, %s)", backend.name, device, compute)
        return backend

    return load


# ---------------------------------------------------------------- service


class SttService:
    """Shared models plus the GPU admission semaphore (RT_STT_MAX_CONCURRENT)."""

    def __init__(self, loader: Callable[[str], SttBackend], opts: SttOptions) -> None:
        self._loader = loader
        self.opts = opts
        self.final: SttBackend | None = None
        self.provisional: SttBackend | None = None
        self.slots = max(1, opts.max_concurrent)
        self._sem = asyncio.Semaphore(self.slots)
        self._convert = ZhConverter(opts.zh_convert)
        self.skipped_provisional = 0
        self.busy = 0                                 # GPU slots in use (RTA-X4 metrics)
        self.waiting = 0                              # final decodes waiting for a slot

    async def start(self) -> None:
        self.final = await asyncio.to_thread(self._loader, self.opts.model)
        if self.opts.provisional_model and self.opts.provisional_model != self.opts.model:
            self.provisional = await asyncio.to_thread(self._loader, self.opts.provisional_model)
        else:
            self.provisional = self.final

    @property
    def ready(self) -> bool:
        return self.final is not None

    def info(self) -> dict[str, Any]:
        if self.final is None:
            return {"loaded": False}
        out = {"loaded": True, "model": self.final.name, "device": self.final.device,
               "compute_type": self.final.compute_type, "max_concurrent": self.opts.max_concurrent}
        if self.provisional is not self.final and self.provisional is not None:
            out["provisional_model"] = self.provisional.name
        return out

    def convert(self, text: str) -> str:
        return self._convert(text)

    async def transcribe(self, pcm: np.ndarray, language: str | None, on_words: OnWords | None = None) -> Transcript:
        """Final decode: waits for a GPU slot."""
        assert self.final is not None
        audio = pcm.astype(np.float32) / 32768.0
        async with self._slot():
            return await self._run(self.final, "final", audio, language, beam_size=self.opts.beam_size,
                                   on_words=on_words, max_new_tokens=token_budget(len(audio)))

    async def transcribe_provisional(self, pcm: np.ndarray, language: str | None) -> Transcript | None:
        """Best-effort decode; None when every slot is busy (or a final decode is waiting)."""
        assert self.provisional is not None
        if self._sem.locked():
            self.skipped_provisional += 1
            return None
        audio = pcm.astype(np.float32) / 32768.0
        async with self._slot():
            return await self._run(self.provisional, "provisional", audio, language,
                                   beam_size=self.opts.provisional_beam_size, max_new_tokens=token_budget(len(audio)), fallback=False)

    @asynccontextmanager
    async def _slot(self):
        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.busy += 1
        try:
            yield
        finally:
            self.busy -= 1
            self._sem.release()

    @staticmethod
    async def _run(backend: SttBackend, kind: str, audio: np.ndarray, language: str | None, **kw) -> Transcript:
        started = time.monotonic()
        result = await asyncio.to_thread(backend.transcribe, audio, language, **kw)
        elapsed = time.monotonic() - started
        if elapsed > 1.0 + len(audio) / SAMPLE_RATE / 4:
            logger.warning("slow %s decode: %.2f s for %.2f s of audio: %r", kind, elapsed,
                           len(audio) / SAMPLE_RATE, result.text[:120])
        return result


# ---------------------------------------------------------------- text helpers


def _keep(ch: str) -> bool:
    return unicodedata.category(ch)[0] in {"L", "N"}


def norm(text: str) -> str:
    """Comparison form: NFKC, lowercase, letters and digits only."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if _keep(c))


def _norm_index(text: str) -> tuple[str, list[int]]:
    """``norm`` plus, for each kept character, its index in ``text``."""
    out: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).lower():
            if _keep(c):
                out.append(c)
                index.append(i)
    return "".join(out), index


def join_words(words: Sequence[Word]) -> str:
    return "".join(w.text for w in words).strip()


def lcp(a: str, b: str) -> int:
    """Length of the longest common prefix."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---------------------------------------------------------------- per session


@dataclass
class _Rolling:
    """Rolling-decode state of the utterance in progress."""

    index: int                       # utterance index the provisional events belong to
    previous_norm: str = ""          # comparison form of the previous hypothesis
    stable_sent: str = ""            # stable text as the client has seen it
    stable_words: int = 0
    next_decode_ms: int = 0
    closed: bool = False


@dataclass(frozen=True)
class RollingParams:
    stride_s: float = 0.5
    first_decode_ms: int = 300
    enabled: bool = True


class SttSession:
    """STT for one WebSocket session.  ``send`` delivers JSON events to the client."""

    def __init__(self, service: SttService, language_tag: str | None, send: Send, params: RollingParams) -> None:
        self.svc = service
        self.tag = language_tag
        self.language = whisper_language(language_tag)
        self.send = send
        self.params = params
        self._rolling: _Rolling | None = None
        self._snapshot: tuple[int, int, np.ndarray] | None = None
        self._wake = asyncio.Event()
        self.last_final: Transcript | None = None

    # ----- text post-processing

    def _post(self, text: str, detected: str | None) -> str:
        return self.svc.convert(text) if wants_traditional(self.tag, detected) else text

    def _post_words(self, words: Sequence[Word], detected: str | None) -> list[Word]:
        """Convert the words as one text -- OpenCC picks a character by its phrase (头发 ->
        頭髮, not 頭發) -- and cut the result back into words.  Falls back to word by word when
        the conversion changes the length (s2twp vocabulary such as U盘 -> 隨身碟)."""
        if not words or not wants_traditional(self.tag, detected):
            return list(words)
        raw = "".join(w.text for w in words)
        converted = self.svc.convert(raw)
        if len(converted) != len(raw):
            return [Word(self.svc.convert(w.text), w.start_ms, w.end_ms, w.probability) for w in words]
        out, pos = [], 0
        for w in words:
            out.append(Word(converted[pos:pos + len(w.text)], w.start_ms, w.end_ms, w.probability))
            pos += len(w.text)
        return out

    # ----- rolling window (RTA-X2), driven by the VAD loop

    def speech_progress(self, index: int, speech_ms: int, audio: Callable[[], np.ndarray]) -> None:
        """Note the open utterance's progress; snapshot its audio when a rolling decode is due."""
        if not self.params.enabled:
            return
        r = self._rolling
        if r is None or r.index != index:
            r = self._rolling = _Rolling(index=index, next_decode_ms=self.params.first_decode_ms)
        if r.closed or speech_ms < r.next_decode_ms:
            return
        r.next_decode_ms = speech_ms + round(self.params.stride_s * 1000)
        self._snapshot = (index, speech_ms, audio())
        self._wake.set()

    def utterance_closed(self, index: int) -> None:
        """The VAD finished utterance ``index``: no provisional event may follow its FINAL."""
        r = self._rolling
        if r is not None and r.index == index:
            r.closed = True
        if self._snapshot is not None and self._snapshot[0] == index:
            self._snapshot = None

    def utterance_discarded(self) -> None:
        """Too short to be speech: the next utterance reuses the index, so start over."""
        self._rolling = None
        self._snapshot = None

    async def rolling_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            snap, self._snapshot = self._snapshot, None
            if snap is None:
                continue
            try:
                await self._decode_window(*snap)
            except Exception:
                logger.exception("provisional decode failed")

    async def _decode_window(self, index: int, audio_ms: int, pcm: np.ndarray) -> None:
        """Decode the whole open utterance; promote what this and the previous decode agree on."""
        r = self._rolling
        if r is None or r.index != index or r.closed:
            return
        result = await self.svc.transcribe_provisional(pcm, self.language)
        if result is None or r.closed or self._rolling is not r:
            return
        words = [w for w in self._post_words(result.words, result.language) if w.text.strip()]
        hypothesis = join_words(words)
        hyp_norm = norm(hypothesis)
        agreed = lcp(r.previous_norm, hyp_norm)
        r.previous_norm = hyp_norm
        stable_len = len(norm(r.stable_sent))

        newly: list[Word] = []
        pos = 0
        # Never the newest word of a decode: it may be cut off at the end of the audio.
        for w in words[:-1]:
            start, pos = pos, pos + len(norm(w.text))
            if pos <= stable_len:
                continue                            # already stable
            if start < stable_len or pos > agreed:
                break                               # straddles the stable text, or not agreed yet
            newly.append(w)
        if newly and hyp_norm.startswith(norm(r.stable_sent)):
            first = r.stable_words
            r.stable_words += len(newly)
            r.stable_sent = (r.stable_sent + join_words(newly)).strip()
            await self.send({"type": "TRANSCRIPT_STABLE", "utterance": index, "text": join_words(newly),
                             "stable_text": r.stable_sent, "word_index": first, "words": len(newly)})
        if not hypothesis and not r.stable_sent:
            return                                  # nothing recognised yet (noise or a breath)
        await self.send({"type": "TRANSCRIPT_PROVISIONAL", "utterance": index, "text": hypothesis,
                         "stable_text": r.stable_sent, "unstable_text": _after_prefix(hypothesis, r.stable_sent),
                         "audio_ms": audio_ms})

    # ----- final (RA-3)

    async def finalize(self, index: int, pcm: np.ndarray, duration_ms: int) -> Transcript:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        started = time.monotonic()

        def on_words(words: Sequence[Word], language: str | None) -> None:   # worker thread
            loop.call_soon_threadsafe(queue.put_nowait, (list(words), language))

        task = asyncio.create_task(self.svc.transcribe(pcm, self.language, on_words=on_words))
        word_index = 0
        first_word_ms: int | None = None

        async def emit(batch: list[Word], language: str | None) -> None:
            nonlocal word_index, first_word_ms
            for w in self._post_words(batch, language):
                if not w.text.strip():
                    continue
                if first_word_ms is None:
                    first_word_ms = round((time.monotonic() - started) * 1000)
                await self.send({"type": "TRANSCRIPT_PARTIAL", "utterance": index, "text": w.text,
                                 "confidence": round(w.probability, 3), "word_index": word_index,
                                 "start_ms": w.start_ms, "end_ms": w.end_ms})
                word_index += 1

        try:
            while True:
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    break
                await emit(*getter.result())
            # Callbacks were queued before the decode returned, so the rest is already here.
            while not queue.empty():
                await emit(*queue.get_nowait())
        except asyncio.CancelledError:
            task.cancel()
            raise
        result = task.result()

        text = self._post(result.text, result.language)
        transcript = Transcript(text, tuple(self._post_words(result.words, result.language)),
                                result.language, result.language_probability)
        await self._correct(index, text)
        self.last_final = transcript
        await self.send({"type": "TRANSCRIPT_FINAL", "utterance": index, "text": text,
                         "language": result.language, "confidence": transcript.confidence,
                         "duration_ms": duration_ms, "latency_ms": round((time.monotonic() - started) * 1000),
                         "first_word_ms": first_word_ms})
        return transcript

    async def _correct(self, index: int, final_text: str) -> None:
        """TRANSCRIPT_CORRECTION when the final text does not start with the stable text sent.
        ``stable_text[start:end]`` (to its end) is replaced by ``new_text``, giving the corrected
        stable text, which is also sent."""
        r = self._rolling
        if r is None or r.index != index or not r.stable_sent:
            return
        stable_norm, stable_pos = _norm_index(r.stable_sent)
        final_norm, final_pos = _norm_index(final_text)
        if final_norm.startswith(stable_norm):
            return
        same = lcp(stable_norm, final_norm)
        # Where the end of the stable text lands in the final text.
        mapped = same
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, stable_norm, final_norm, autojunk=False).get_opcodes():
            if i1 < i2 <= len(stable_norm):
                # A replacement at the end of the stable text also swallows the text that
                # follows it in the final; keep it to the replaced length.
                mapped = min(j2, j1 + i2 - i1) if tag == "replace" and i2 == len(stable_norm) else j2
        start = stable_pos[same] if same < len(stable_pos) else len(r.stable_sent)
        new_start = final_pos[same] if same < len(final_pos) else len(final_text)
        new_end = final_pos[mapped - 1] + 1 if mapped > same else new_start
        new_text = final_text[new_start:new_end]
        await self.send({"type": "TRANSCRIPT_CORRECTION", "utterance": index, "start": start,
                         "end": len(r.stable_sent), "old_text": r.stable_sent[start:], "new_text": new_text,
                         "stable_text": r.stable_sent[:start] + new_text})


def _after_prefix(text: str, prefix: str) -> str:
    """``text`` without the part matching ``prefix`` (compared in comparison form); all of
    ``text`` when it does not start with ``prefix``."""
    prefix_norm = norm(prefix)
    text_norm, pos = _norm_index(text)
    if not prefix_norm or not text_norm.startswith(prefix_norm):
        return text
    if len(prefix_norm) >= len(text_norm):
        return ""
    return text[pos[len(prefix_norm)]:]
