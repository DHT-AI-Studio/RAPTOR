"""RTA-X1 GPU runner using local STT/Kokoro weights and consented zh-TW fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.metrics import corpus_rates, percentile, text_normalize


def load_manifest(path: Path, minimum: int = 20) -> dict:
    """Validate consent/metadata/uniqueness before loading expensive models."""
    manifest = json.loads(path.read_text())
    samples = manifest.get("samples", [])
    if len(samples) < minimum:
        raise ValueError(f"at least {minimum} distinct zh-TW samples are required")
    ids, hashes = set(), set()
    for sample in samples:
        if sample.get("consented") is not True or sample.get("locale") != "zh-TW":
            raise ValueError("each fixture requires explicit consent and zh-TW locale")
        if any(not sample.get(key) for key in ("id", "reference", "speaker", "accent", "noise")):
            raise ValueError("missing fixture transcript or diversity metadata")
        if any(str(sample[key]).startswith("<") for key in ("speaker", "accent", "noise")):
            raise ValueError("fixture metadata still contains placeholders")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", sample["id"]):
            raise ValueError("fixture ID must be safe for an artifact filename")
        audio = path.parent / sample["audio"]
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest in hashes or sample["id"] in ids:
            raise ValueError("duplicate fixture IDs or audio content")
        hashes.add(digest)
        ids.add(sample["id"])
        sample["audio"] = str(audio.resolve())
        sample["sha256"] = digest
    if len({s["speaker"] for s in samples}) < 2:
        raise ValueError("varied speakers are required")
    return manifest


@contextmanager
def gpu_peak(gpu_index: int):
    """Sample total device memory, including CTranslate2/Paddle allocations, and this
    process's own share of it (visible when the container runs with --pid=host), which
    other jobs on a shared GPU do not disturb."""
    import os
    stop = threading.Event()
    def query(*args):
        return subprocess.run(["nvidia-smi", "-i", str(gpu_index), *args, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5, check=True).stdout
    def sample():
        return int(query("--query-gpu=memory.used").strip())
    def own():
        for line in query("--query-compute-apps=pid,used_memory").splitlines():
            pid, _, used = line.partition(",")
            if pid.strip() == str(os.getpid()):
                return int(used)
        return 0
    observations = [sample()]  # Fail before inference if the monitor is unavailable.
    process = [0]
    errors = []
    def poll():
        while not stop.is_set():
            try:
                observations.append(sample())
                process.append(own())
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                errors.append(exc)
                return
            stop.wait(.05)
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    record = {"device_baseline_mib": observations[0], "sampling_interval_ms": 50}
    inference_failed = False
    try:
        yield record
    except BaseException:
        inference_failed = True
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        if not errors:
            try:
                observations.append(sample())
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                errors.append(exc)
        record["device_peak_mib"] = max(observations)
        record["samples"] = len(observations)
        record["incremental_peak_mib"] = max(observations) - observations[0]
        record["process_peak_mib"] = max(process) or None
        if errors and not inference_failed:
            raise RuntimeError("GPU memory sampling failed; measurement is invalid") from errors[0]


def load_audio(path):
    """16 kHz mono float32 (faster-whisper's own file decoding does not work with PyAV >= 15);
    8 kHz G.711 fixtures are decoded and upsampled to 16 kHz, as a phone call is before STT."""
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio[:, 0]
    if rate != 16000:
        from scipy.signal import resample_poly

        gcd = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // gcd, rate // gcd).astype(np.float32)
    return np.ascontiguousarray(mono)


def stt_benchmark(samples: list[dict], model_path: str, tokenize, canonicalize, gpu_index: int) -> dict:
    """Score large-v3/medium CUDA float16; materialize the lazy segment iterator."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_path, device="cuda", device_index=gpu_index,
                         compute_type="float16", local_files_only=True)
    results = []
    for sample in samples:
        started = time.perf_counter()
        segments, _ = model.transcribe(load_audio(sample["audio"]), language="zh", beam_size=5)
        hypothesis = "".join(segment.text for segment in segments)
        elapsed = (time.perf_counter() - started) * 1000
        ref = canonicalize(sample["reference"])
        hyp = canonicalize(hypothesis)
        results.append({"id": sample["id"], "reference": ref, "hypothesis": hyp,
                        "reference_words": tokenize(ref), "hypothesis_words": tokenize(hyp),
                        "latency_ms": elapsed, "sha256": sample["sha256"],
                        **{key: sample.get(key) for key in ("consented", "locale", "speaker", "accent",
                                                           "noise", "code_switching", "codec", "sample_rate")}})
    return {"model_path": model_path, "compute_type": "float16", "samples": results,
            **corpus_rates(results), "latency_p50_ms": percentile([r["latency_ms"] for r in results], .5),
            "latency_p95_ms": percentile([r["latency_ms"] for r in results], .95)}


def main() -> None:
    """Write only measured evidence; do not select models without TTS human review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--stt-model", required=True)
    parser.add_argument("--fallback-model")
    parser.add_argument("--kokoro-dir", required=True)
    parser.add_argument("--voice", default="zf_001")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = load_manifest(args.manifest)
    if args.repeats < 20 or len(manifest.get("tts_sentences", [])) < 10:
        raise ValueError("need >=20 latency repeats and >=10 zh-TW TTS sentences")
    import jieba
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from kokoro import KModel, KPipeline
    from opencc import OpenCC
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU results cannot satisfy RTA-X1")
    torch.cuda.set_device(args.gpu)
    convert = OpenCC("s2tw").convert       # Taiwan forms: 裡/裏, 啟/啓 are the same character
    canonicalize = lambda text: text_normalize(convert(text))
    tokenize = lambda text: [token for token in jieba.lcut(text) if token.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "measured_quality_review_pending", "gpu": torch.cuda.get_device_name(args.gpu),
              "compute_capability": list(torch.cuda.get_device_capability(args.gpu)),
              "normalization": "NFKC+s2tw+lowercase+remove punctuation; WER=jieba 0.42.1",
              "fixture_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    with gpu_peak(args.gpu) as memory:
        report["stt"] = stt_benchmark(manifest["samples"], args.stt_model, tokenize, canonicalize, args.gpu)
    report["stt"]["memory"] = memory
    if report["stt"]["cer"] > .15:
        if args.fallback_model:
            with gpu_peak(args.gpu) as memory:
                report["fallback"] = stt_benchmark(manifest["samples"], args.fallback_model,
                                                   tokenize, canonicalize, args.gpu)
            report["fallback"]["memory"] = memory
        else:
            report["fallback"] = {"status": "required_medium_model_missing"}
    model = WhisperModel(args.stt_model, device="cuda", device_index=args.gpu,
                         compute_type="float16", local_files_only=True)
    latency_audio = args.manifest.parent / manifest["latency_audio"]
    info = sf.info(latency_audio)
    if abs(info.duration - 3.0) > .05:
        raise ValueError("latency fixture must be a 3-second utterance")
    latency_pcm = load_audio(latency_audio)

    def transcribe_latency():
        segments, _ = model.transcribe(latency_pcm, language="zh")
        list(segments)
    transcribe_latency()  # excluded warm-up
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        transcribe_latency()
        timings.append((time.perf_counter() - start) * 1000)
    report["stt_3s"] = {"p50_ms": percentile(timings, .5), "p95_ms": percentile(timings, .95),
                         "repeats": args.repeats}
    directory = Path(args.kokoro_dir)
    tts_model = KModel(config=str(directory / "config.json"),
                       model=str(next(directory.glob("kokoro-*.pth")))).to(f"cuda:{args.gpu}").eval()
    pipeline = KPipeline(lang_code="z", model=tts_model)
    voice = directory / "voices" / f"{args.voice}.pt"
    tts_results = []
    latency_sentence = manifest["tts_latency_sentence"]
    if len(tokenize(latency_sentence)) != 20:
        raise ValueError("TTS latency sentence must contain 20 words under the documented tokenizer")
    list(pipeline(latency_sentence, voice=str(voice)))  # excluded warm-up
    with gpu_peak(args.gpu) as memory:
        for sentence in manifest["tts_sentences"]:
            started = time.perf_counter()
            chunks = [audio.cpu().numpy() for _, _, audio in pipeline(sentence["text"], voice=str(voice))]
            import numpy as np
            audio = np.concatenate(chunks)
            elapsed = time.perf_counter() - started
            wav = args.output.parent / f"tts-{sentence['id']}.wav"
            sf.write(wav, audio, 24000)
            tts_results.append({"id": sentence["id"], "audio": wav.name, "duration_s": len(audio) / 24000,
                                "latency_s": elapsed, "rtf": elapsed / (len(audio) / 24000),
                                "word_count": len(tokenize(sentence["text"]))})
    tts_timings, rtfs = [], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        generated = list(pipeline(latency_sentence, voice=str(voice)))
        elapsed = time.perf_counter() - start
        duration = sum(len(chunk[2]) for chunk in generated) / 24000
        if duration <= 0:
            raise RuntimeError("TTS returned empty audio")
        tts_timings.append(elapsed * 1000)
        rtfs.append(elapsed / duration)
    report["tts"] = {"voice": args.voice, "sentences": tts_results, "memory": memory,
                     "20_word_latency_p50_ms": percentile(tts_timings, .5),
                     "20_word_latency_p95_ms": percentile(tts_timings, .95),
                     "rtf_p50": percentile(rtfs, .5), "rtf_p95": percentile(rtfs, .95)}
    report["combined_vram"] = {"status": "pending simultaneous WhisperX+PaddlePaddle profile"}
    report["final_model_selection"] = "pending human intelligibility/naturalness and combined VRAM review"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
