"""Live checks of RTA-X5 / RTA-X3 / RTA-X4 against a running Module 34 (real VAD, STT and TTS).

  barge-in  ask a question and speak over the answer --after-s into it: BARGE_IN, how fast after
            the interrupting speech started, the reported vs. the client's playback position, how
            much answer audio the client must drop, and the interrupting utterance's transcript
  resume    ask a question, drop the connection --drop-after-ms into the answer, reconnect after
            --gap-s with RESUME: frames replayed, seq continuity, whether the whole answer arrived
  load      --sessions concurrent voice sessions asking --turns questions each: per question
            answered / fallback / error / nothing (no answer and no error event), latencies,
            peaks of the /metrics queue gauges
  external  an external-brain session (RTA-X9; the JWT needs the rt-external-brain role) that
            answers every TRANSCRIPT_FINAL with two SPEAK frames: first audio after the first
            SPEAK (SLO p95 <= 500 ms), answer order; --codec pcma|pcmu as a telephony bridge does

The client behaves like an open microphone: 20 ms frames at real time, silence between
questions, PONG for every PING, framing "seq" so every server frame can be accounted for.
The metadata file is the same as stream_eval's ({"id", "audio", "transcript"} per clip).

  python -m benchmarks.session_checks barge-in --base-url http://rt-test-34:8034 --jwt "$JWT" \\
      --metadata /data/metadata.json --question 01 --interrupt 05 --output /out/barge-in.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.metrics import edit_distance, percentile, text_normalize
from benchmarks.stream_eval import FRAME_MS, load_samples, read_pcm16k

SILENCE = np.zeros(16 * FRAME_MS, dtype=np.int16)


class LiveSession:
    """One Module 34 session over a real WebSocket, with a paced microphone."""

    def __init__(self, base_url: str, jwt: str, language: str = "zh-TW", **config) -> None:
        self.base_url = base_url.rstrip("/")
        self.jwt = jwt
        self.language = language
        self.config = config                                # extra CONFIG fields (brain, codec, ...)
        self.session_id: str | None = None
        self.ws = None
        self.frames: list[tuple[float, int, Any]] = []     # (wall time, seq, dict | bytes payload)
        self.last_seq = 0
        self.client_seq = 0
        self.mic: deque[np.ndarray] = deque()
        self.mic_frames = 0                                 # frames sent: audio time = 20 ms each
        self.t0 = 0.0
        self.changed = asyncio.Event()
        self.deaf = False                                   # frames arrive but are lost
        self._tasks: list[asyncio.Task] = []

    async def ticket(self) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.post(f"{self.base_url}/audio/session/token", headers={"Authorization": f"Bearer {self.jwt}"})
            r.raise_for_status()
            return r.json()["token"]

    async def connect(self, resume: bool = False) -> None:
        import websockets

        url = self.base_url.replace("http", "ws", 1) + f"/audio/stream?token={await self.ticket()}"
        if resume:
            url += f"&session_id={self.session_id}"
        self.ws = await websockets.connect(url, max_size=None)
        if resume:
            await self.ws.send(json.dumps({"type": "RESUME", "session_id": self.session_id, "last_seq": self.last_seq}))
        else:
            await self.ws.send(json.dumps({"type": "CONFIG", "codec": "pcm", "sample_rate": 16000,
                                           "language": self.language, "framing": "seq", **self.config}))
        self._tasks.append(asyncio.create_task(self._read(self.ws)))
        if not resume:
            ready = await self.wait("READY", timeout=30)
            self.session_id = ready["session_id"]
            self.t0 = time.monotonic()
            self._tasks.append(asyncio.create_task(self._speak()))

    def say(self, pcm: np.ndarray) -> float:
        """Queue audio after what is already queued; returns its start in audio time (ms)."""
        start = (self.mic_frames + len(self.mic)) * FRAME_MS
        step = 16 * FRAME_MS
        for i in range(0, len(pcm), step):
            frame = pcm[i:i + step]
            self.mic.append(np.pad(frame, (0, step - len(frame))) if len(frame) < step else frame)
        return start

    def _g711(self, frame: np.ndarray) -> bytes:
        from app.codec import Resampler, g711_encode

        if not hasattr(self, "_down"):
            self._down = Resampler(16000, 8000)
        return g711_encode(self._down(frame), self.config["codec"])

    def wall(self, audio_ms: float) -> float:
        return self.t0 + audio_ms / 1000

    async def _speak(self) -> None:
        while True:
            due = self.t0 + self.mic_frames * FRAME_MS / 1000
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            frame = self.mic.popleft() if self.mic else SILENCE
            self.mic_frames += 1
            ws = self.ws
            if ws is None:
                continue                                    # line down: this audio is lost
            self.client_seq += 1
            data = frame.astype("<i2").tobytes()
            if self.config.get("codec") in ("pcmu", "pcma"):
                data = self._g711(frame)
            try:
                await ws.send(struct.pack("<I", self.client_seq) + data)
            except Exception:  # noqa: BLE001 -- dropped connection; the caller reconnects
                self.ws = None

    async def _read(self, ws) -> None:
        try:
            async for raw in ws:
                if self.deaf:
                    continue
                now = time.monotonic()
                if isinstance(raw, bytes):
                    seq, payload = struct.unpack_from("<I", raw)[0], raw[4:]
                else:
                    payload = json.loads(raw)
                    seq = payload.get("seq", 0)
                    if payload["type"] == "PING":
                        await ws.send(json.dumps({"type": "PONG", "ts": payload.get("ts")}))
                if seq and seq <= self.last_seq:
                    continue                                # already have it (replay)
                self.last_seq = max(self.last_seq, seq)
                self.frames.append((now, seq, payload))
                self.changed.set()
        except Exception:  # noqa: BLE001, S110 -- connection closed or aborted
            pass
        self.changed.set()

    async def wait_for(self, pred: Callable[[float, Any], bool], timeout: float, after: float = 0.0):
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for t, _, f in self.frames[seen:]:
                if t >= after and pred(t, f):
                    return t, f
            seen = len(self.frames)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=left)
            except asyncio.TimeoutError:
                pass

    async def wait(self, kind: str, timeout: float = 60, after: float = 0.0, **match) -> dict:
        _, ev = await self.wait_for(lambda _t, f: isinstance(f, dict) and f["type"] == kind
                                    and all(f.get(k) == v for k, v in match.items()), timeout, after)
        return ev

    def time_of(self, ev: dict) -> float:
        return next(t for t, _, f in self.frames if f is ev)

    def events(self, kind: str) -> list[dict]:
        return [f for _, _, f in self.frames if isinstance(f, dict) and f["type"] == kind]

    def drop(self) -> None:
        """The line drops: abort the TCP connection, no close frame."""
        ws, self.ws = self.ws, None
        if ws is not None:
            ws.transport.abort()

    async def end(self, timeout: float = 300) -> dict | None:
        while self.mic:
            await asyncio.sleep(0.05)
        await self.ws.send(json.dumps({"type": "END"}))
        try:
            return await self.wait("ENDED", timeout=timeout)
        except TimeoutError:
            return None
        finally:
            await self.close()

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self.ws is not None:
            await self.ws.close()


def cer(reference: str, hypothesis: str) -> float:
    from opencc import OpenCC

    to_trad = OpenCC("s2tw").convert
    ref = "".join(text_normalize(to_trad(reference)).split())
    hyp = "".join(text_normalize(to_trad(hypothesis)).split())
    return round(edit_distance(ref, hyp) / max(1, len(ref)), 4)


def audio_ms(frames) -> float:
    return sum(len(f) for f in frames) / 32                 # 16 kHz s16le


# ---------------------------------------------------------------- barge-in


async def barge_in(args, clips: dict[str, dict]) -> dict:
    s = LiveSession(args.base_url, args.jwt, args.language)
    await s.connect()
    q, intr = clips[args.question], clips[args.interrupt]
    s.say(np.concatenate([read_pcm16k(q["audio"]), np.zeros(16 * 600, dtype=np.int16)]))
    start = await s.wait("RESPONSE_START", timeout=120)
    t_start = s.time_of(start)
    t_audio0, _ = await s.wait_for(lambda _t, f: isinstance(f, bytes), 30, after=t_start)
    await asyncio.sleep(max(0.0, t_audio0 + args.after_s - time.monotonic()))
    intr_ms = s.say(np.concatenate([read_pcm16k(intr["audio"]), np.zeros(16 * 800, dtype=np.int16)]))
    barge = await s.wait("BARGE_IN", timeout=30)
    t_barge = s.time_of(barge)
    onset = next(e for e in s.events("SPEECH_START") if e["start_ms"] >= intr_ms - 200)
    _, utt = await s.wait_for(lambda _t, f: isinstance(f, dict) and f["type"] == "UTTERANCE"
                              and f["end_ms"] > intr_ms, 60)
    nxt = await s.wait("RESPONSE_START", timeout=120, after=t_barge)
    ended = await s.end()
    frames = [(t, f) for t, _, f in s.frames]
    answer = [f for t, f in frames if isinstance(f, bytes) and t_start <= t < t_barge]
    after = [f for t, f in frames if isinstance(f, bytes) and t_barge <= t < s.time_of(nxt)]
    interrupt_text = "".join(e["text"] for e in s.events("TRANSCRIPT_FINAL") if e["utterance"] >= utt["index"])
    return {
        "question": args.question, "interrupt": args.interrupt, "after_s": args.after_s,
        "barge_in_after_speech_onset_ms": round((t_barge - s.wall(onset["start_ms"])) * 1000),
        "tts_position_ms": barge["tts_position_ms"],
        "client_playback_ms": round((t_barge - t_audio0) * 1000),
        "answer_audio_received_ms": round(audio_ms(answer)),
        "audio_to_discard_ms": round(audio_ms(answer) - barge["tts_position_ms"]),
        "answer_audio_after_barge_in_ms": round(audio_ms(after)),
        "interrupted_end": next((e for e in s.events("RESPONSE_END") if e.get("interrupted")), None),
        "interrupt_transcript": interrupt_text, "interrupt_reference": intr["reference"],
        "interrupt_cer": cer(intr["reference"], interrupt_text), "interrupt_utterance": utt,
        "states": [e["state"] for e in s.events("STATE")],
        "next_answer": nxt["request_id"] != start["request_id"], "ended": ended,
    }


# ---------------------------------------------------------------- resume


def answer_audio(frames: list[tuple[float, int, Any]], request_id: str) -> float:
    """Audio of one answer: binary frames between its RESPONSE_START and RESPONSE_END (seq order)."""
    inside, total = False, 0
    for _, _, f in sorted(frames, key=lambda x: x[1]):
        if isinstance(f, dict) and f.get("request_id") == request_id:
            inside = f["type"] != "RESPONSE_END" and (inside or f["type"] == "RESPONSE_START")
        elif inside and isinstance(f, bytes):
            total += len(f)
    return total / 32


async def resume(args, clips: dict[str, dict]) -> dict:
    s = LiveSession(args.base_url, args.jwt, args.language)
    await s.connect()
    q = clips[args.question]
    s.say(np.concatenate([read_pcm16k(q["audio"]), np.zeros(16 * 600, dtype=np.int16)]))
    start = await s.wait("RESPONSE_START", timeout=120)
    rid = start["request_id"]
    await s.wait_for(lambda _t, _f: answer_audio(s.frames, rid) >= args.drop_after_ms, 60)
    last = s.last_seq
    s.deaf = True                                       # the network stalls: frames are lost ...
    await asyncio.sleep(args.stall_s)
    s.drop()                                            # ... then the connection drops
    t_drop = time.monotonic()
    await asyncio.sleep(args.gap_s)
    s.deaf = False
    t_resume = time.monotonic()
    await s.connect(resume=True)
    resumed = await s.wait("RESUMED", timeout=10)
    end = await s.wait("RESPONSE_END", timeout=300, request_id=rid)
    ended = await s.end()
    seqs = [seq for _, seq, _ in s.frames]
    received = answer_audio(s.frames, rid)
    return {
        "question": args.question, "drop_after_ms": args.drop_after_ms, "stall_s": args.stall_s, "gap_s": args.gap_s,
        "last_seq_before_drop": last, "resumed": resumed,
        "resume_ms": round((s.time_of(resumed) - t_resume) * 1000),
        "seq_continuous": seqs == list(range(1, len(seqs) + 1)),
        "answer_audio_received_ms": round(received), "answer_duration_ms": end["duration_ms"],
        "whole_answer": abs(received - end["duration_ms"]) <= 130,         # one 4 KB PCM chunk
        "interrupted": bool(end.get("interrupted")), "barge_ins": len(s.events("BARGE_IN")),
        "offline_s": round(t_resume - t_drop + args.stall_s, 2), "ended": ended,
    }


# ---------------------------------------------------------------- external brain


async def external(args, clips: dict[str, dict]) -> dict:
    rate = 8000 if args.codec in ("pcmu", "pcma") else 16000
    s = LiveSession(args.base_url, args.jwt, args.language, brain="external", codec=args.codec, sample_rate=rate)
    await s.connect()
    turns = []
    for cid in (args.ids or sorted(clips))[: args.turns]:
        mark = time.monotonic()
        pcm = np.concatenate([read_pcm16k(clips[cid]["audio"]), np.zeros(16 * 700, dtype=np.int16)])
        done = s.wall(s.say(pcm) + len(pcm) / 16)
        # Answer once the whole question is said (a clip may be several utterances), like a brain would.
        await asyncio.sleep(max(0.0, done - time.monotonic()))
        await s.wait_for(lambda _t, f: isinstance(f, dict) and f["type"] == "TRANSCRIPT_FINAL" and f["text"], 60, mark)
        await asyncio.sleep(1.0)
        final = {"text": "".join(f["text"] for t, _, f in s.frames if t >= mark and isinstance(f, dict)
                                 and f["type"] == "TRANSCRIPT_FINAL")}
        uid = f"answer-{cid}"
        t_speak = time.monotonic()
        await s.ws.send(json.dumps({"type": "SPEAK", "utterance_id": uid, "text": "好的，我幫您查一下。", "final": False}))
        await s.ws.send(json.dumps({"type": "SPEAK", "utterance_id": uid, "text": "請稍等一下。", "final": True}))
        t_audio, _ = await s.wait_for(lambda _t, f: isinstance(f, bytes), 30, t_speak)
        end = await s.wait("RESPONSE_END", timeout=60, request_id=uid)
        texts = [f["text"] for _, _, f in s.frames if isinstance(f, dict) and f["type"] == "RESPONSE_TEXT"
                 and f.get("request_id") == uid]
        frames = [f for t, _, f in s.frames if isinstance(f, bytes) and t >= t_speak]
        turns.append({"clip": cid, "transcript": final["text"], "first_audio_after_speak_ms":
                      round((t_audio - t_speak) * 1000), "texts": texts, "duration_ms": end["duration_ms"],
                      "frame_bytes": sorted({len(f) for f in frames})})
        await asyncio.sleep(max(0.0, end["duration_ms"] / 1000 - (time.monotonic() - t_audio)) + 0.3)
    ended = await s.end()
    lat = [t["first_audio_after_speak_ms"] for t in turns]
    return {"codec": args.codec, "turns": len(turns), "first_audio_after_speak_ms":
            {"p50": round(percentile(lat, .5)), "p95": round(percentile(lat, .95)), "max": max(lat)},
            "in_order": all(t["texts"] == ["好的，我幫您查一下。", "請稍等一下。"] for t in turns),
            "guardrail_blocks": len(s.events("GUARDRAIL_BLOCK")), "barge_ins": len(s.events("BARGE_IN")),
            "ready": s.events("READY")[0],
            "ended": ended, "results": turns}


# ---------------------------------------------------------------- load


async def scrape(base_url: str, peaks: dict[str, float], stop: asyncio.Event) -> None:
    import httpx

    line = re.compile(r"^(rt_[a-z_]+(?:\{[^}]*\})?) ([0-9.e+-]+)$")
    async with httpx.AsyncClient(timeout=5) as http:
        while not stop.is_set():
            try:
                for m in filter(None, map(line.match, (await http.get(f"{base_url}/metrics")).text.splitlines())):
                    peaks[m[1]] = max(peaks.get(m[1], 0.0), float(m[2]))
            except Exception:  # noqa: BLE001, S110 -- best effort sampling
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def one_caller(args, n: int, clips: list[dict], results: list[dict]) -> None:
    await asyncio.sleep(n * args.ramp_s / max(1, args.sessions))
    s = LiveSession(args.base_url, args.jwt, args.language)
    try:
        await s.connect()
    except Exception as exc:  # noqa: BLE001 -- e.g. 503 too_many_sessions: an explicit refusal
        results.append({"session": n, "turn": -1, "outcome": f"refused: {exc}"})
        return
    for k in range(args.turns):
        clip = clips[(n + k) % len(clips)]
        mark = time.monotonic()
        pcm = np.concatenate([read_pcm16k(clip["audio"]), np.zeros(16 * 1500, dtype=np.int16)])
        start_ms = s.say(pcm)
        sent = s.wall(start_ms + len(pcm) / 16)
        outcome = "nothing"
        try:
            await s.wait_for(lambda _t, f: isinstance(f, dict) and f["type"] == "TRANSCRIPT_FINAL", 120, mark)
            await s.wait_for(lambda t, f, sent=sent: isinstance(f, dict) and f["type"] == "STATE"
                             and f["state"] == "LISTENING" and t >= sent, args.turn_timeout_s, mark)
        except TimeoutError:
            pass
        window = [(t, f) for t, _, f in s.frames if t >= mark and isinstance(f, dict)]
        ends = [f for _, f in window if f["type"] == "RESPONSE_END"]
        errors = [f["code"] for _, f in window if f["type"] == "ERROR"]
        if ends and not any(e.get("fallback") for e in ends) and not errors:
            outcome = "answered"
        elif ends:
            outcome = "fallback"
        elif errors:
            outcome = "error"
        finals = [(t, f) for t, f in window if f["type"] == "TRANSCRIPT_FINAL"]
        utts = {f["index"]: t for t, f in window if f["type"] == "UTTERANCE"}
        starts = [t for t, f in window if f["type"] == "RESPONSE_START"]
        results.append({
            "session": n, "turn": k, "clip": clip["id"], "outcome": outcome, "errors": errors,
            "final_after_vad_ms": [round((t - utts[f["utterance"]]) * 1000) for t, f in finals if f["utterance"] in utts],
            "response_start_after_final_ms": round((starts[0] - finals[0][0]) * 1000) if starts and finals else None,
            "cer": cer(clip["reference"], "".join(f["text"] for _, f in finals)) if finals else None,
        })
    ended = await s.end(timeout=args.turn_timeout_s)
    results.append({"session": n, "turn": "end", "outcome": "ended" if ended else "nothing"})


async def load(args, clips: dict[str, dict]) -> dict:
    pool = [clips[i] for i in (args.ids or sorted(clips))]
    results: list[dict] = []
    peaks: dict[str, float] = {}
    stop = asyncio.Event()
    sampler = asyncio.create_task(scrape(args.base_url, peaks, stop))
    started = time.monotonic()
    await asyncio.gather(*(one_caller(args, n, pool, results) for n in range(args.sessions)))
    stop.set()
    await sampler
    turns = [r for r in results if isinstance(r["turn"], int) and r["turn"] >= 0]
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    def pct(values):
        values = [v for v in values if v is not None]
        return {"p50": round(percentile(values, .5)), "p95": round(percentile(values, .95)), "max": max(values),
                "n": len(values)} if values else None

    return {
        "sessions": args.sessions, "turns": args.turns, "seconds": round(time.monotonic() - started),
        "outcomes": outcomes,
        "errors": sorted({c for r in turns for c in r["errors"]}),
        "final_after_vad_ms": pct([v for r in turns for v in r["final_after_vad_ms"]]),
        "response_start_after_final_ms": pct([r["response_start_after_final_ms"] for r in turns]),
        "cer_mean": round(float(np.mean([r["cer"] for r in turns if r["cer"] is not None])), 4)
        if any(r["cer"] is not None for r in turns) else None,
        "metrics_peaks": peaks, "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["barge-in", "resume", "load", "external"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--jwt", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--language", default="zh-TW")
    parser.add_argument("--question", default="01")
    parser.add_argument("--interrupt", default="05")
    parser.add_argument("--after-s", type=float, default=1.0, help="barge-in: seconds into the answer")
    parser.add_argument("--drop-after-ms", type=float, default=1500, help="resume: answer audio before the drop")
    parser.add_argument("--gap-s", type=float, default=3.0, help="resume: seconds without a connection")
    parser.add_argument("--stall-s", type=float, default=1.0, help="resume: frames lost before the drop")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--ramp-s", type=float, default=5.0, help="load: spread the session starts over this")
    parser.add_argument("--turn-timeout-s", type=float, default=180)
    parser.add_argument("--ids", nargs="*", help="load / external: clips to use")
    parser.add_argument("--codec", default="pcm", choices=["pcm", "pcmu", "pcma"], help="external: session codec")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clips = {c["id"]: c for c in load_samples(args.metadata)}
    check = {"barge-in": barge_in, "resume": resume, "load": load, "external": external}[args.check]
    report = asyncio.run(check(args, clips))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
