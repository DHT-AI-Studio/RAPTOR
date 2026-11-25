"""Regenerate the RA-10 audio fixtures in this directory (committed; run only to change them).

    python tests/fixtures/make_fixtures.py --kokoro-dir /models/tts/kokoro [--voice zf_001]

Speech is synthesized with Kokoro-82M-v1.1-zh (Apache-2.0, the same weights Module 34 serves),
so the files carry no third-party recording licence.  The G.711 and Opus files are encoded by
libavcodec (PyAV), not by Module 34's own encoders, so the decoder tests check against an
independent implementation.

  zh_tw_5s.wav           16 kHz s16le mono, 5.000 s, one utterance  (WS test, codec reference)
  zh_tw_10s.wav          16 kHz s16le mono, 10.000 s, one utterance (gRPC test)
  zh_tw_5s_pcmu.wav      zh_tw_5s at 8 kHz, G.711 mu-law WAV (format tag 7)
  zh_tw_5s_opus48k.ogg   zh_tw_5s at 48 kHz, Ogg Opus, 20 ms packets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from dataclasses import replace

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

TEXTS = {
    "zh_tw_5s.wav": ("請問明天早上台北會不會下雨", 5.0),
    "zh_tw_10s.wav": ("我想預約下週三下午兩點到信義區的門市更換手機螢幕請問需要帶什麼證件", 10.0),
}
LEAD_S = 0.4
MAX_SPEECH_FRACTION = 0.92


def resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=dst)
    frame = av.AudioFrame.from_ndarray(pcm.astype(np.int16).reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = src
    out = [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    out += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(out).astype(np.int16)


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())


def encode(path: str, pcm16k: np.ndarray, fmt: str, codec: str, rate: int, frame_samples: int) -> None:
    with av.open(path, "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=rate, layout="mono")
        if codec == "libopus":
            stream.bit_rate = 32000
        pcm = resample(pcm16k, 16000, rate)
        pts = 0
        for i in range(0, len(pcm), frame_samples):
            chunk = pcm[i:i + frame_samples]
            if len(chunk) < frame_samples:
                chunk = np.concatenate([chunk, np.zeros(frame_samples - len(chunk), dtype=np.int16)])
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate, frame.pts = rate, pts
            pts += frame_samples
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kokoro-dir", default="/models/tts/kokoro")
    parser.add_argument("--voice", default="zf_001")
    args = parser.parse_args()

    from app.config import Settings
    from app.response import to_int16
    from app.tts_backends.kokoro import KokoroBackend

    settings = replace(Settings(), tts_kokoro_dir=args.kokoro_dir, tts_device="cpu")
    KokoroBackend.check(settings)
    tts = KokoroBackend(settings)
    tts.load()
    manifest = {"voice": args.voice, "model": "Kokoro-82M-v1.1-zh", "files": {}}
    for name, (text, total_s) in TEXTS.items():
        speed = 1.0
        while True:
            speech = to_int16(tts.synthesize(text, args.voice, "zh-TW", speed))
            speech = resample(speech, tts.sample_rate, 16000)
            if len(speech) / 16000 <= total_s * MAX_SPEECH_FRACTION - LEAD_S or speed >= 1.6:
                break
            speed = round(speed + 0.05, 2)
        total = int(total_s * 16000)
        pcm = np.zeros(total, dtype=np.int16)
        lead = int(LEAD_S * 16000)
        pcm[lead:lead + min(len(speech), total - lead)] = speech[: total - lead]
        write_wav(os.path.join(HERE, name), pcm, 16000)
        manifest["files"][name] = {"text": text, "seconds": total_s, "speech_s": round(len(speech) / 16000, 2),
                                   "lead_s": LEAD_S, "speed": speed, "sample_rate": 16000}
        print(f"{name}: speech {len(speech) / 16000:.2f} s at speed {speed}")
    with wave.open(os.path.join(HERE, "zh_tw_5s.wav")) as w:
        reference = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    encode(os.path.join(HERE, "zh_tw_5s_pcmu.wav"), reference, "wav", "pcm_mulaw", 8000, 160)
    encode(os.path.join(HERE, "zh_tw_5s_opus48k.ogg"), reference, "ogg", "libopus", 48000, 960)
    manifest["files"]["zh_tw_5s_pcmu.wav"] = {"from": "zh_tw_5s.wav", "codec": "pcm_mulaw", "sample_rate": 8000}
    manifest["files"]["zh_tw_5s_opus48k.ogg"] = {"from": "zh_tw_5s.wav", "codec": "libopus", "sample_rate": 48000,
                                                 "frame_ms": 20, "bit_rate": 32000}
    with open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
