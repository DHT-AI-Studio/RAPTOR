"""Stream WAV recordings to a running Module 34 in real time and score its transcripts.

Measures what RA-3 / RTA-X2 / RTA-X8 ask of the live service, end to end over the WebSocket:
  * corpus CER of the TRANSCRIPT_FINAL texts (OpenCC s2tw + NFKC, punctuation removed -- the
    same canonical form as model_benchmark.py)
  * first TRANSCRIPT_PROVISIONAL after speech start (RTA-X8: <= 600 ms)
  * TRANSCRIPT_FINAL after the VAD closes the utterance (UTTERANCE event) and after the last
    speech of the utterance (RTA-X8: <= 1.5 s)
  * first TRANSCRIPT_PARTIAL after the VAD closes the utterance (RA-3: <= 800 ms for <= 3 s)
  * voice turns (RA-4 / RA-5), when the service answers: TRANSCRIPT_FINAL -> RESPONSE_START
    (Module 15 + first sentence of TTS) and RESPONSE_TEXT -> first audio frame (TTS alone);
    with --save-audio the spoken answers (16 kHz PCM sessions) are written as WAV files

Audio is sent as 20 ms frames paced at real time -- 16 kHz PCM, or with --codec pcmu / pcma
resampled to 8 kHz and G.711-encoded on the client side (160-byte frames, the narrowband
telephony path of RTA-X11 / RTA-X12) -- followed by trailing silence so the
VAD ends the last utterance by silence rather than by END.  Wall-clock times are converted to
audio time with the pacing start, so latencies are relative to when that audio was sent.

The metadata file is a JSON list (or {"samples": [...]}) of {"id", "audio", "transcript"}
with paths relative to the metadata file, e.g. a dataset's approved_16k/metadata.json.

  python -m benchmarks.stream_eval --base-url http://rt-test-34:8034 --jwt "$JWT" \\
      --metadata /data/metadata.json --output /out/stream-eval.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.metrics import edit_distance, percentile, text_normalize

FRAME_MS = 20


def load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) else data
    out = []
    for s in samples:
        audio = path.parent / s["audio"]
        out.append({"id": s["id"], "audio": audio, "reference": s.get("transcript") or s.get("reference") or "",
                    "category": s.get("category")})
    return out


def read_pcm16k(path: Path) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(path, dtype="int16", always_2d=True)
    if rate != 16000:
        raise ValueError(f"{path}: {rate} Hz; stream_eval sends 16 kHz PCM, resample first")
    return data[:, 0]


def encode_frames(pcm16k: np.ndarray, codec: str) -> tuple[list[bytes], int]:
    """Client-side framing: (20 ms frames, sample rate announced in CONFIG)."""
    if codec == "pcm":
        step = 16 * FRAME_MS
        return [pcm16k[i:i + step].astype("<i2").tobytes() for i in range(0, len(pcm16k), step)], 16000
    from app.codec import Resampler, g711_encode

    r = Resampler(16000, 8000)
    narrow = np.concatenate([r(pcm16k), r.flush()])
    data = g711_encode(narrow, codec)
    return [data[i:i + 160] for i in range(0, len(data), 160)], 8000


async def run_one(base_url: str, jwt: str, sample: dict, language: str | None, tail_ms: int,
                  speed: float, codec: str = "pcm", save_audio: Path | None = None) -> dict:
    import httpx
    import websockets

    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.post(f"{base_url}/audio/session/token", headers={"Authorization": f"Bearer {jwt}"})
        resp.raise_for_status()
        ticket = resp.json()["token"]
    pcm = np.concatenate([read_pcm16k(sample["audio"]), np.zeros(16 * tail_ms, dtype=np.int16)])
    frames, rate = encode_frames(pcm, codec)
    ws_url = base_url.replace("http", "ws", 1) + f"/audio/stream?token={ticket}"
    events: list[tuple[float, dict]] = []
    audio: list[tuple[float, bytes]] = []
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "CONFIG", "codec": codec, "sample_rate": rate, "language": language}))
        while True:
            ready = json.loads(await ws.recv())
            if ready["type"] == "READY":
                break
        t0 = time.monotonic()

        async def sender():
            for i, frame in enumerate(frames):
                due = t0 + i * FRAME_MS / 1000 / speed
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(frame)
            await ws.send(json.dumps({"type": "END"}))

        async def receiver():
            async for raw in ws:
                if isinstance(raw, bytes):
                    audio.append((time.monotonic() - t0, raw))
                    continue
                ev = json.loads(raw)
                if ev["type"] == "PING":                       # RTA-X3 keep-alive
                    await ws.send(json.dumps({"type": "PONG", "ts": ev.get("ts")}))
                    continue
                events.append((time.monotonic() - t0, ev))
                if ev["type"] == "ENDED" or (ev["type"] == "ERROR" and ev.get("fatal", True)):
                    return

        await asyncio.gather(sender(), receiver())

    def audio_s(ms: int) -> float:          # wall time (since t0) at which that audio was sent
        return ms / 1000 / speed

    finals = [ev for _, ev in events if ev["type"] == "TRANSCRIPT_FINAL"]
    utterances = {ev["index"]: (t, ev) for t, ev in events if ev["type"] == "UTTERANCE"}
    starts = [ev["start_ms"] for _, ev in events if ev["type"] == "SPEECH_START"]
    first_prov = next((t for t, ev in events if ev["type"] == "TRANSCRIPT_PROVISIONAL"), None)
    per_utt = []
    for t, ev in events:
        if ev["type"] != "TRANSCRIPT_FINAL" or ev["utterance"] not in utterances:
            continue
        ut, u = utterances[ev["utterance"]]
        first_partial = next((pt for pt, pe in events if pe["type"] == "TRANSCRIPT_PARTIAL"
                              and pe["utterance"] == ev["utterance"]), None)
        per_utt.append({
            "utterance": ev["utterance"], "duration_ms": u["duration_ms"], "reason": u["reason"],
            "final_after_vad_ms": round((t - ut) * 1000),
            "final_after_speech_end_ms": round((t - audio_s(u["end_ms"] - 96)) * 1000),
            "partial_after_vad_ms": round((first_partial - ut) * 1000) if first_partial is not None else None,
            "stt_latency_ms": ev.get("latency_ms"),
        })
    # RESPONSE_TEXT precedes RESPONSE_START (the text goes out before its first frame), so a
    # response is everything from the first text after the previous RESPONSE_END to its END.
    turns, texts, start_t = [], [], None
    finals_t = [t for t, ev in events if ev["type"] == "TRANSCRIPT_FINAL" and ev["text"]]
    for t, ev in events:
        if ev["type"] == "RESPONSE_TEXT":
            texts.append((t, ev["text"]))
        elif ev["type"] == "RESPONSE_START":
            start_t = t
        elif ev["type"] == "RESPONSE_END" and start_t is not None:
            final_t = max((ft for ft in finals_t if ft <= start_t), default=None)
            first_audio = next((at for at, _ in audio if at >= start_t), None)
            turns.append({
                "request_id": ev["request_id"], "fallback": ev.get("fallback", False),
                "text": "".join(x for _, x in texts),
                "response_start_after_final_ms": round((start_t - final_t) * 1000)
                if final_t is not None and not ev.get("fallback") else None,
                "first_audio_after_text_ms": round((first_audio - texts[0][0]) * 1000)
                if texts and first_audio is not None else None,
                "duration_ms": ev.get("duration_ms"), "sentences": ev.get("sentences"),
            })
            texts, start_t = [], None
    if save_audio is not None and audio and codec == "pcm":
        import soundfile as sf

        save_audio.mkdir(parents=True, exist_ok=True)
        pcm_out = np.frombuffer(b"".join(chunk for _, chunk in audio), dtype="<i2")
        sf.write(save_audio / f"{sample['id']}.wav", pcm_out, 16000, subtype="PCM_16")
    return {
        "id": sample["id"], "category": sample["category"], "reference": sample["reference"],
        "responses": turns,
        "hypothesis": "".join(f["text"] for f in finals),
        "utterances": len(finals),
        "first_provisional_ms": round((first_prov - audio_s(starts[0])) * 1000)
        if first_prov is not None and starts else None,
        "provisional_events": sum(1 for _, ev in events if ev["type"] == "TRANSCRIPT_PROVISIONAL"),
        "stable_events": sum(1 for _, ev in events if ev["type"] == "TRANSCRIPT_STABLE"),
        "corrections": sum(1 for _, ev in events if ev["type"] == "TRANSCRIPT_CORRECTION"),
        "correction_details": [{k: ev[k] for k in ("utterance", "old_text", "new_text")}
                               for _, ev in events if ev["type"] == "TRANSCRIPT_CORRECTION"],
        "stable_texts": [ev["stable_text"] for _, ev in events if ev["type"] == "TRANSCRIPT_STABLE"],
        "warnings": [ev for _, ev in events if ev["type"] in ("WARNING", "ERROR")],
        "per_utterance": per_utt,
    }


def score(results: list[dict]) -> dict:
    from opencc import OpenCC

    to_trad = OpenCC("s2tw").convert
    chars = errors = 0
    for r in results:
        ref = "".join(text_normalize(to_trad(r.get("reference") or "")).split())
        if not ref:
            r["cer"] = None
            continue
        hyp = "".join(text_normalize(to_trad(r["hypothesis"])).split())
        r["cer"] = round(edit_distance(ref, hyp) / len(ref), 4)
        chars += len(ref)
        errors += edit_distance(ref, hyp)

    def pct(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {"p50": round(percentile(values, .5)), "p95": round(percentile(values, .95)),
                "max": max(values), "n": len(values)}

    utts = [u for r in results for u in r["per_utterance"]]
    return {
        "cer": round(errors / chars, 4) if chars else None, "reference_chars": chars, "char_errors": errors,
        "first_provisional_ms": pct([r["first_provisional_ms"] for r in results]),
        "final_after_vad_ms": pct([u["final_after_vad_ms"] for u in utts]),
        "final_after_speech_end_ms": pct([u["final_after_speech_end_ms"] for u in utts]),
        "partial_after_vad_ms": pct([u["partial_after_vad_ms"] for u in utts]),
        "partial_after_vad_ms_utterances_le_3s": pct([u["partial_after_vad_ms"] for u in utts
                                                      if u["duration_ms"] <= 3000]),
        "corrections": sum(r["corrections"] for r in results),
        "response_start_after_final_ms": pct([t["response_start_after_final_ms"]
                                              for r in results for t in r["responses"]]),
        "first_audio_after_text_ms": pct([t["first_audio_after_text_ms"] for r in results for t in r["responses"]]),
    }


async def main_async(args) -> dict:
    samples = load_samples(args.metadata)
    results = []
    if args.ids:
        samples = [s for s in samples if s["id"] in args.ids]
    if args.warmup:
        await run_one(args.base_url, args.jwt, samples[0], args.language, args.tail_ms, args.speed, args.codec)
    for s in samples:
        results.append(await run_one(args.base_url, args.jwt, s, args.language, args.tail_ms, args.speed,
                                     args.codec, args.save_audio))
        print(f"{s['id']}: {results[-1]['hypothesis']}", flush=True)
    return {"metadata": str(args.metadata),
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "language": args.language, "codec": args.codec, "speed": args.speed, "summary": score(results),
            "samples": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="http://host:port of Module 34")
    parser.add_argument("--jwt", required=True, help="Bearer JWT accepted by POST /audio/session/token")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--language", default="zh-TW")
    parser.add_argument("--codec", choices=["pcm", "pcmu", "pcma"], default="pcm")
    parser.add_argument("--ids", nargs="*", help="only these sample ids")
    parser.add_argument("--save-audio", type=Path, help="write each spoken answer here as <id>.wav")
    parser.add_argument("--tail-ms", type=int, default=800, help="silence appended to each recording")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
