"""WebSocket frame protocol for /audio/stream (full description in README.md and
docs/CLIENT_GUIDE.md).

Client -> server
  text   {"type": "CONFIG", "codec": "pcm|opus|pcmu|pcma", "sample_rate": 16000, "language": "zh-TW",
          "framing": "raw|seq", "brain": "module15|external", "subject_id": "...",
          "record": bool, "index_transcripts": bool, "store_memory": bool}
         must be the first frame of a new session ("type" may be omitted there); any other first
         frame closes 4400.  "language": a BCP-47 tag forces the STT language, null auto-detects
         it, and leaving it out uses RT_STT_LANGUAGE.  "framing": "seq" puts a 4-byte
         little-endian sequence number in front of every binary frame, both ways (default "raw").
         "brain": "external" (realm role rt-external-brain) -- Module 34 does not ask Module 15, the
         client answers with SPEAK (RTA-X9).  "subject_id" (^[A-Za-z0-9_-]{1,128}$, realm role
         rt-delegate) -- the end user the session is for: consent, recordings, erasure, Module 25 and
         Module 26 use it (RTA-X10).  record / index_transcripts / store_memory: per-session side
         effects (default false for external sessions).
  text   {"type": "RESUME", "session_id": "...", "last_seq": 41}  instead of CONFIG, on a new
         connection (new ticket, same user) within RT_SESSION_RESUME_WINDOW_S of a drop: the
         session carries on and every frame after ``last_seq`` is sent again.
  binary AUDIO: pcm = s16le mono at sample_rate (8-48 kHz); opus = one raw Opus packet per frame;
         pcmu / pcma = G.711 at 8 kHz (any frame size; telephony uses 20 ms = 160 bytes)
  text   {"type": "SPEAK", "utterance_id": "...", "text": "...", "final": false|true}   external brain:
         speak this; SPEAK frames of one utterance_id play back to back, final=true closes it;
         utterances play in the order they arrive
  text   {"type": "SPEAK_CANCEL", "utterance_id": "..."}   drop its queued text, stop it if playing
  text   {"type": "PONG"} answers PING (within RT_PONG_TIMEOUT_S); {"type": "PING"} gets a PONG
  text   {"type": "END"}  (the bare string "END" is accepted too)
  JSON frames may carry "seq" (the client's own counter, shared with binary frames); the server
  reports the highest it has seen in RESUMED, and with "seq" framing drops audio it has seen.

Server -> client: JSON events, each with "seq" (one counter per session for everything the
server sends, binary frames included; no gaps) and "session_id"
  READY          {session_id, state, config, vad, stt, tts, output, recording, expires_in, resume_window_s}
  STATE          {state}
  SPEECH_START   {start_ms}
  UTTERANCE      {index, start_ms, end_ms, duration_ms, speech_ms, reason}
  DISCARDED      {start_ms, end_ms, speech_ms, reason}       too_short, or during_response (began over
                 an answer without barging in: echo)
  TRANSCRIPT_PROVISIONAL {utterance, text, stable_text, unstable_text, audio_ms}  while speaking
  TRANSCRIPT_STABLE      {utterance, text, stable_text, word_index, words}       agreed words
  TRANSCRIPT_PARTIAL     {utterance, text, confidence, word_index, start_ms, end_ms}  final decode
  TRANSCRIPT_CORRECTION  {utterance, start, end, old_text, new_text, stable_text}  stable text revised
  TRANSCRIPT_FINAL       {utterance, text, language, confidence, duration_ms, latency_ms, first_word_ms}
  RESPONSE_START / RESPONSE_TEXT / binary audio / RESPONSE_END     a spoken answer (app/response.py);
                 for SPEAK, request_id is the utterance_id and the binary frames are the
                 AUDIO_RESPONSE; RESPONSE_END {cancelled: true} after SPEAK_CANCEL
  GUARDRAIL_BLOCK {request_id, utterance, direction, reason}  Module 23 blocked the transcript (input:
                 Module 15 is not asked, a refusal is spoken) or an answer sentence (output: the
                 rest of the answer is replaced by the refusal)
  BARGE_IN       {request_id, tts_position_ms}   the user spoke over the answer: stop playback at
                 that position and drop the rest; RESPONSE_END {interrupted: true} follows
  PING           {ts}                            answer with PONG
  RESUMED        {session_id, replayed, gap, last_client_seq, state}   after the replayed frames
  WARNING        {code, message, ...}            QUEUE_OVERFLOW, invalid_audio, stt_failed, ...
  ERROR          {code, message, fatal?}         fatal unless "fatal": false (then the session goes on)
  ENDED          {session_id, utterances, audio_ms, dropped_chunks, responses}

An unknown codec -- or, with RT_CODEC_RESAMPLE=false, anything but 16 kHz PCM -- is ERROR
UNSUPPORTED_CODEC followed by close 4400.

Close codes: 1000 END handled; 4400 invalid CONFIG / protocol error; 4408 CONFIG timeout, session
lifetime (RT_SESSION_TTL_S) reached or no PONG; 4403 BRAIN_MODE_FORBIDDEN; 4409 the session was
resumed on another connection; 4410 SESSION_EXPIRED (nothing to resume); 1011 internal error.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.audio import CODECS, G711_RATES, OPUS_RATES, PCM_RATE_RANGE

CLOSE_NORMAL = 1000
CLOSE_INTERNAL = 1011
CLOSE_PROTOCOL = 4400
CLOSE_TIMEOUT = 4408
CLOSE_FORBIDDEN = 4403
CLOSE_REPLACED = 4409
CLOSE_EXPIRED = 4410

FRAMINGS = ("raw", "seq")
BRAINS = ("module15", "external")
_SUBJECT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SIDE_EFFECTS = ("record", "index_transcripts", "store_memory")

_BCP47 = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


class ProtocolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StreamConfig:
    codec: str
    sample_rate: int
    language: str | None             # None = auto-detect
    framing: str = "raw"
    brain: str = "module15"
    subject_id: str | None = None    # requested; effective only with the rt-delegate role
    record: bool | None = None       # None = the brain's default
    index_transcripts: bool | None = None
    store_memory: bool | None = None

    def public(self) -> dict[str, Any]:
        return {"codec": self.codec, "sample_rate": self.sample_rate, "language": self.language,
                "framing": self.framing, "brain": self.brain}


@dataclass(frozen=True)
class ResumeRequest:
    session_id: str
    last_seq: int


def parse_json_text(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_config(message: dict[str, Any], default_language: str | None = "zh-TW",
                 resample: bool = True) -> StreamConfig:
    """Validate the first frame (an ASGI websocket.receive message)."""
    text = message.get("text")
    if text is None:
        raise ProtocolError("config_required", "first frame must be a CONFIG JSON text frame")
    body = parse_json_text(text)
    if body is None:
        raise ProtocolError("invalid_config", "CONFIG must be a JSON object")
    if body.get("type", "CONFIG") != "CONFIG":
        raise ProtocolError("config_required", f"first frame must be CONFIG, got {body.get('type')!r}")

    codec = body.get("codec")
    if not isinstance(codec, str):
        raise ProtocolError("invalid_config", "codec is required")
    if codec not in CODECS:
        raise ProtocolError("UNSUPPORTED_CODEC", f"codec must be one of {', '.join(CODECS)}")
    rate = body.get("sample_rate")
    if not isinstance(rate, int) or isinstance(rate, bool):
        raise ProtocolError("invalid_config", "sample_rate must be an integer")
    if codec == "pcm":
        if not PCM_RATE_RANGE[0] <= rate <= PCM_RATE_RANGE[1]:
            low, high = PCM_RATE_RANGE
            raise ProtocolError("invalid_config", f"sample_rate for pcm must be within {low}-{high}")
    else:
        allowed = OPUS_RATES if codec == "opus" else G711_RATES
        if rate not in allowed:
            raise ProtocolError("invalid_config", f"sample_rate for {codec} must be one of {sorted(allowed)}")
    if not resample and (codec != "pcm" or rate != 16000):
        raise ProtocolError("UNSUPPORTED_CODEC", "this server only accepts 16 kHz pcm (RT_CODEC_RESAMPLE=false)")
    language = body.get("language", default_language)
    if language is not None and (not isinstance(language, str) or not _BCP47.match(language)):
        raise ProtocolError("invalid_config", "language must be a BCP-47 tag (e.g. zh-TW) or null")
    framing = body.get("framing", "raw")
    if framing not in FRAMINGS:
        raise ProtocolError("invalid_config", f"framing must be one of {', '.join(FRAMINGS)}")
    brain = body.get("brain", "module15")
    if brain not in BRAINS:
        raise ProtocolError("invalid_config", f"brain must be one of {', '.join(BRAINS)}")
    subject = body.get("subject_id")
    if subject is not None and (not isinstance(subject, str) or not _SUBJECT.match(subject)):
        raise ProtocolError("invalid_config", "subject_id must match ^[A-Za-z0-9_-]{1,128}$")
    flags = {}
    for name in SIDE_EFFECTS:
        value = body.get(name)
        if value is not None and not isinstance(value, bool):
            raise ProtocolError("invalid_config", f"{name} must be true or false")
        flags[name] = value
    return StreamConfig(codec=codec, sample_rate=rate, language=language, framing=framing, brain=brain,
                        subject_id=subject, **flags)


def parse_resume(message: dict[str, Any]) -> ResumeRequest | None:
    """The first frame as a RESUME request, or None when it is not one."""
    body = parse_json_text(message.get("text") or "")
    if body is None or body.get("type") != "RESUME":
        return None
    session_id, last_seq = body.get("session_id"), body.get("last_seq", 0)
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError("invalid_resume", "RESUME needs the session_id")
    if not isinstance(last_seq, int) or isinstance(last_seq, bool) or last_seq < 0:
        raise ProtocolError("invalid_resume", "last_seq must be a non-negative integer")
    return ResumeRequest(session_id, last_seq)


def is_end(text: str) -> bool:
    if text.strip() == "END":
        return True
    body = parse_json_text(text)
    return body is not None and body.get("type") == "END"
