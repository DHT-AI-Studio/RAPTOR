"""RTA-X1 / RTA-X12: TTS clips for human MOS rating, made by the service's own TTS backends.

For every system (``backend:voice``) and every ``tts_sentences`` entry of the manifest:
  wideband    the backend's native output (24 kHz PCM WAV)                       RTA-X1 naturalness
  telephony   what a pcmu session receives: resampled straight to 8 kHz and G.711-encoded by
              app.codec (the service's own encoder), decoded back to an 8 kHz PCM WAV so a
              browser can play it                                                  RTA-X12 MOS gate

Clip ids are random (seeded) so raters cannot tell systems apart; manifest.json maps them back.
Breeze TTS 2 output is licensed for research and non-commercial use only -- rating it internally
is research use; do not publish its clips.

  python -m benchmarks.mos_samples benchmarks/fixtures/manifest.example.json --out /out/mos \\
      --system kokoro-82m:zf_001 --system breeze-tts-2:zh-female \\
      --kokoro-dir /models/tts/kokoro --breeze-url http://raptor-breeze-tts:7860
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np


def _wav(path: Path, pcm: np.ndarray, rate: int) -> dict:
    import soundfile as sf

    sf.write(path, pcm, rate, subtype="PCM_16")
    return {"file": path.name, "sample_rate": rate, "duration_s": round(len(pcm) / rate, 3),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def telephony(pcm: np.ndarray, rate: int, law: str = "pcmu") -> np.ndarray:
    """The 8 kHz G.711 audio a telephony session hears, as decoded 16-bit PCM."""
    from app.codec import g711_decode, make_encoder

    encoder = make_encoder(law, 8000)
    frames = encoder.encode(pcm, rate) + encoder.flush()
    return g711_decode(b"".join(frames), law)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--system", action="append", required=True, help="backend:voice, repeatable")
    parser.add_argument("--kokoro-dir", default="/models/tts/kokoro")
    parser.add_argument("--breeze-url", default="http://raptor-breeze-tts:7860")
    parser.add_argument("--language", default="zh-TW")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    from app.config import Settings
    from app.response import to_int16
    from app.tts_backends import load_backend

    sentences = json.loads(args.manifest.read_text(encoding="utf-8"))["tts_sentences"]
    base = Settings(tts_kokoro_dir=args.kokoro_dir, tts_breeze_url=args.breeze_url,
                    tts_breeze_enabled=any(s.startswith("breeze") for s in args.system))
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    clips = []
    for system in args.system:
        name, _, voice = system.partition(":")
        backend = load_backend(name, replace(base, tts_model=name))
        voice = voice or backend.default_voice
        for sentence in sentences:
            pcm = to_int16(backend.synthesize(sentence["text"], voice, args.language, 1.0))
            for condition, audio, rate, codec in (
                ("wideband", pcm, backend.sample_rate, "PCM"),
                ("telephony", telephony(pcm, backend.sample_rate), 8000, "PCMU"),
            ):
                clip_id = f"c{rng.getrandbits(32):08x}"
                info = _wav(args.out / f"{clip_id}.wav", audio, rate)
                clips.append({"clip_id": clip_id, "sentence_id": sentence["id"], "text": sentence["text"],
                              "system": name, "voice": voice, "condition": condition, "codec": codec, **info})
            print(f"{system} {sentence['id']}: {len(pcm) / backend.sample_rate:.2f} s", flush=True)
    rng.shuffle(clips)
    manifest = {"source": str(args.manifest), "language": args.language, "seed": args.seed,
                "systems": args.system, "sentences": sentences, "clips": clips}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
