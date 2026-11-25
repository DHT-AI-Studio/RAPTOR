"""RTA-X12 driver for benchmarks/capacity.py: one external-brain call over Module 34's WebSocket.

config: base_url (one URL, or a list: one Module 34 instance per GPU, calls assigned round-robin),
jwt (realm role rt-external-brain), codec (pcma | pcmu | pcm), language.
The 16 kHz source clip is sent as 8 kHz G.711 on the wire, as a phone gateway would.
turn(sample) says the clip at real time, waits for its last TRANSCRIPT_FINAL, answers with two
SPEAK frames and waits until the service has played the answer (STATE back to LISTENING: a caller
who speaks earlier is talking over it); it returns the RTA-X8 stages in ms:
  first_provisional  speech onset -> first TRANSCRIPT_PROVISIONAL; when every GPU slot was busy
                     for the whole utterance there is none, and the first text is its FINAL
  final              UTTERANCE (VAD end) -> its TRANSCRIPT_FINAL, worst utterance of the clip
  first_audio        first SPEAK -> first answer audio frame
and the counts no_provisional (turns whose first text was a FINAL) and discarded (utterances
the service dropped as speech over its own answer).

  python -m benchmarks.capacity manifest.json capacity.json --driver benchmarks.external_driver:call \\
      --calls 20 --utterances 20 --output /out/capacity-20.json
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from benchmarks.session_checks import LiveSession
from benchmarks.stream_eval import read_pcm16k

ANSWER = ("好的，我幫您查一下。", "請稍等一下。")


class Call:
    def __init__(self, session: LiveSession, call_id: int) -> None:
        self.s = session
        self.call_id = call_id
        self.turns = 0

    def _events(self, kind: str, after: float) -> list[tuple[float, dict]]:
        return [(t, f) for t, _, f in self.s.frames if t >= after and isinstance(f, dict) and f["type"] == kind]

    async def turn(self, sample: dict) -> dict[str, float]:
        s = self.s
        mark = time.monotonic()
        pcm = np.concatenate([read_pcm16k(Path(sample["audio"])), np.zeros(16 * 700, dtype=np.int16)])
        start_ms = s.say(pcm)
        await asyncio.sleep(max(0.0, s.wall(start_ms + len(pcm) / 16) - time.monotonic()))
        await s.wait_for(lambda _t, f: isinstance(f, dict) and f["type"] == "TRANSCRIPT_FINAL", 60, mark)
        await asyncio.sleep(1.0)                           # the clip's later utterances
        onsets = [f["start_ms"] for _, f in self._events("SPEECH_START", mark) if f["start_ms"] >= start_ms - 200]
        provisional = self._events("TRANSCRIPT_PROVISIONAL", mark)
        final_events = self._events("TRANSCRIPT_FINAL", mark)
        utterances = {f["index"]: t for t, f in self._events("UTTERANCE", mark)}
        finals = [t - utterances[f["utterance"]] for t, f in final_events if f["utterance"] in utterances]
        if not onsets or not finals:
            raise RuntimeError(f"call {self.call_id}: speech start {len(onsets)}, utterances {len(utterances)}, "
                               f"finals {len(final_events)}")
        first_text = min(t for t, _ in provisional + final_events)
        self.turns += 1
        uid = f"c{self.call_id}-t{self.turns}"
        t_speak = time.monotonic()
        for i, text in enumerate(ANSWER):
            await s.ws.send(json.dumps(
                {"type": "SPEAK", "utterance_id": uid, "text": text, "final": i == len(ANSWER) - 1}))
        t_audio, _ = await s.wait_for(lambda _t, f: isinstance(f, bytes), 30, t_speak)
        end = await s.wait("RESPONSE_END", timeout=60, request_id=uid)
        states = self._events("STATE", t_audio)
        if not states or states[-1][1]["state"] != "LISTENING":   # an underrun is PROCESSING, not done
            await s.wait("STATE", timeout=60, after=s.time_of(end), state="LISTENING")
        await asyncio.sleep(0.3)                           # the caller's pause before the next question
        return {"first_provisional": (first_text - s.wall(onsets[0])) * 1000,
                "final": max(finals) * 1000, "first_audio": (t_audio - t_speak) * 1000,
                "no_provisional": int(not provisional or provisional[0][0] > first_text),
                "discarded": len(self._events("DISCARDED", mark))}


@asynccontextmanager
async def call(sample: dict, call_id: int, config: dict):
    codec = config.get("codec", "pcma")
    urls = config["base_url"] if isinstance(config["base_url"], list) else [config["base_url"]]
    s = LiveSession(urls[call_id % len(urls)], config["jwt"], config.get("language", "zh-TW"), brain="external",
                    codec=codec, sample_rate=8000 if codec in ("pcma", "pcmu") else 16000)
    await s.connect()
    try:
        yield Call(s, call_id)
    finally:
        await s.end(timeout=60)
