"""RTA-X12: 8 kHz G.711 degradation and evidence-only MOS/capacity evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmarks.metrics import percentile
from benchmarks.model_benchmark import load_manifest


def degrade(source: Path, destination: Path, codec: str = "PCMU", noise_snr_db: float | None = None) -> dict:
    """Downsample with antialiasing and save an actual G.711-companded WAV."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if codec not in {"PCMU", "PCMA"}:
        raise ValueError("codec must be PCMU or PCMA")
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate < 16000:
        raise ValueError("source must be wideband (>=16 kHz)")
    mono = audio.mean(axis=1)
    gcd = math.gcd(rate, 8000)
    narrow = resample_poly(mono, 8000 // gcd, rate // gcd)
    if noise_snr_db is not None:
        if not math.isfinite(noise_snr_db):
            raise ValueError("noise SNR must be finite")
        amplitude = np.sqrt(np.mean(narrow**2)) / 10**(noise_snr_db / 20)
        narrow += np.random.default_rng(20261005).normal(0, amplitude, len(narrow))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(narrow, -1, 1), 8000, format="WAV",
             subtype="ULAW" if codec == "PCMU" else "ALAW")
    return {"sample_rate": 8000, "codec": codec, "noise_snr_db": noise_snr_db,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "audio_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def mos_gate(ratings: list[dict], sample_ids: set[str]) -> dict:
    """Require >=3 unique Taiwanese evaluators for every encoded TTS sentence."""
    by_sample = {key: {} for key in sample_ids}
    for rating in ratings:
        sample, evaluator = rating["sample_id"], rating["evaluator_id"]
        score = float(rating["score"])
        if (sample not in by_sample or not evaluator or rating.get("evaluator_region") != "TW"
                or rating.get("sample_rate") != 8000 or rating.get("codec") not in {"PCMU", "PCMA"}
                or not math.isfinite(score) or not 1 <= score <= 5):
            raise ValueError("invalid narrowband Taiwanese MOS evidence")
        if evaluator in by_sample[sample]:
            raise ValueError("duplicate evaluator/sample rating")
        by_sample[sample][evaluator] = score
    if not by_sample or any(len(values) < 3 for values in by_sample.values()):
        return {"status": "pending", "reason": "need three Taiwanese evaluators per TTS sentence"}
    per_sentence = {key: sum(values.values()) / len(values) for key, values in by_sample.items()}
    mos = sum(per_sentence.values()) / len(per_sentence)
    return {"status": "pass" if mos >= 3.5 else "fail", "mos": mos, "per_sentence": per_sentence}


def capacity_gate(runs: list[dict], slos: dict[str, float], target: int = 20) -> dict:
    """Select GPUs only from measured external-brain runs that satisfy every p95 SLO."""
    if target < 1 or not slos or any(not math.isfinite(v) or v <= 0 for v in slos.values()):
        raise ValueError("positive target and actual RTA-X8 SLOs are required")
    accepted = []
    for run in runs:
        if run.get("mode") != "external-brain" or run.get("failures", 0) != 0:
            continue
        calls, gpus = run.get("concurrent_calls", 0), run.get("gpu_count", 0)
        observations = run.get("stage_latency_ms", {})
        if (not isinstance(calls, int) or calls < 1 or not isinstance(gpus, int) or gpus < 1
                or any(len(observations.get(stage, [])) < max(20, calls) for stage in slos)):
            continue
        p95 = {stage: percentile(observations[stage], .95) for stage in slos}
        if all(p95[stage] <= limit for stage, limit in slos.items()):
            accepted.append({"concurrent_calls": calls, "gpu_count": gpus, "p95_ms": p95})
    requirements = {}
    for count in {20, 50, 100, target}:
        candidates = [run["gpu_count"] for run in accepted if run["concurrent_calls"] >= count]
        requirements[str(count)] = min(candidates) if candidates else None
    return {"status": "pass" if requirements[str(target)] is not None else "pending_or_fail",
            "target_concurrent_calls": target, "measured_gpu_requirements": requirements,
            "accepted_runs": accepted}


def evaluate(evidence: dict, target: int = 20) -> dict:
    """Refuse a completed quality gate until acoustic, human and capacity evidence exist."""
    narrow, wide = evidence.get("narrowband", {}), evidence.get("wideband", {})
    samples = narrow.get("samples", [])
    acoustic_ready = (len(samples) >= 20 and all(s.get("sample_rate") == 8000
        and s.get("codec") in {"PCMU", "PCMA"} and s.get("consented") is True
        and s.get("locale") == "zh-TW" for s in samples)
        and len({s.get("sha256") for s in samples}) >= 20
        and len({s.get("speaker") for s in samples}) >= 2
        and any(s.get("code_switching") is True for s in samples))
    if acoustic_ready:
        cer = narrow.get("cer")
        acoustic_ready = isinstance(cer, (int, float)) and math.isfinite(cer) and cer >= 0
    stt = {"status": "pending"} if not acoustic_ready else {
        "status": "pass" if narrow["cer"] <= .15 else "fail", "cer": narrow["cer"],
        "wideband_cer": wide.get("cer"),
    }
    wide_cer = wide.get("cer")
    if not isinstance(wide_cer, (int, float)) or not math.isfinite(wide_cer) or wide_cer < 0:
        stt["status"] = "pending"
    tts_ids = set(evidence.get("encoded_tts_sample_ids", []))
    if len(tts_ids) < 10:
        # Checked before mos_gate(), not after: a rating for a sample_id not yet
        # in tts_ids (an in-progress evidence set, not just "missing") makes
        # mos_gate() raise -- this must report "pending" cleanly, never crash.
        mos = {"status": "pending", "reason": "need ten encoded TTS sentences"}
    else:
        mos = mos_gate(evidence.get("ratings", []), tts_ids)
    slos = evidence.get("rta_x8_slos_ms")
    capacity = capacity_gate(evidence.get("capacity_runs", []), slos, target) if slos else {
        "status": "pending", "reason": "RTA-X8 SLO contract missing",
    }
    complete = all(part["status"] == "pass" for part in (stt, mos, capacity))
    return {"status": "pass" if complete else "incomplete_or_fail", "stt": stt,
            "tts_mos": mos, "capacity": capacity}


def main() -> None:
    """Generate narrowband recordings or evaluate measured evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("manifest", type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--codec", default="PCMU", choices=["PCMU", "PCMA"])
    generate.add_argument("--noise-snr-db", type=float)
    gate = commands.add_parser("evaluate")
    gate.add_argument("evidence", type=Path)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--target-calls", type=int)
    encode_tts = commands.add_parser("encode-tts")
    encode_tts.add_argument("result", type=Path, help="model_benchmark result JSON with TTS WAVs")
    encode_tts.add_argument("--output-dir", type=Path, required=True)
    encode_tts.add_argument("--codec", default="PCMU", choices=["PCMU", "PCMA"])
    args = parser.parse_args()
    if args.command == "generate":
        manifest = load_manifest(args.manifest)
        for sample in manifest["samples"]:
            destination = args.output_dir / f"{sample['id']}.wav"
            sample.update(degrade(Path(sample["audio"]), destination, args.codec, args.noise_snr_db))
            sample["audio"] = destination.name
        latency_source = args.manifest.parent / manifest["latency_audio"]
        degrade(latency_source, args.output_dir / "latency-3s.wav", args.codec, args.noise_snr_db)
        manifest["latency_audio"] = "latency-3s.wav"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    elif args.command == "encode-tts":
        result = json.loads(args.result.read_text())
        sentences = result["tts"]["sentences"]
        encoded = []
        for sentence in sentences:
            filename = Path(sentence["audio"]).name
            destination = args.output_dir / filename
            details = degrade(args.result.parent / filename, destination, args.codec)
            encoded.append({"id": sentence["id"], "audio": filename, **details})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "encoded-tts.json").write_text(json.dumps({"samples": encoded,
            "ratings": [], "status": "human_MOS_pending"}, indent=2) + "\n")
    else:
        import os
        target = args.target_calls or int(os.getenv("RT_TARGET_CONCURRENT_CALLS", "20"))
        result = evaluate(json.loads(args.evidence.read_text()), target)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if result["status"] != "pass":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
