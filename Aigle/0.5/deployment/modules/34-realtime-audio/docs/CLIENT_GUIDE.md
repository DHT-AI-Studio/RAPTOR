# Module 34 client guide — streaming, barge-in, resume

For clients of `WS /audio/stream` (browser apps, the telephony bridge) and of the gRPC
`AudioService.AudioChat` (native clients, §9). The full event list is in
`deployment/modules/34-realtime-audio/README.md`; this page covers what a client has to do.

## 1. Connect

1. `POST /audio/session/token` with `Authorization: Bearer <Keycloak JWT>` → `{"token", "expires_in": 30}`.
   The ticket is single use; get a new one for every connection, including reconnects.
2. Open `wss://<host>/audio/stream?token=<ticket>`. HTTP 401 = bad ticket, 503 = `RT_MAX_SESSIONS` reached
   (the ticket is not consumed; retry within 30 s).
3. Send CONFIG as the first frame:

```json
{"type": "CONFIG", "codec": "pcm", "sample_rate": 16000, "language": "zh-TW", "framing": "seq"}
```

   `codec`: `pcm` (s16le mono, 8–48 kHz), `opus` (one raw packet per frame), `pcmu` / `pcma` (8 kHz G.711,
   20 ms = 160 bytes per frame). Answer audio comes back in the same codec.
4. Wait for `READY` (it has `session_id`, `output`, `barge_in`, `resume_window_s`, `ping_interval_s`), then stream audio.
5. Send `{"type": "END"}` when done; the server finishes the pending transcripts and answers, sends `ENDED` and closes 1000.

## 2. Sequence numbers (`seq`)

- Every frame the server sends has a `seq`: one counter per session, starting at 1, no gaps, shared by JSON and
  binary frames. JSON events carry `seq` and `session_id`.
- `framing: "seq"`: every binary frame, both directions, starts with the seq as 4 bytes little-endian. The client
  numbers its own frames with its own counter (JSON frames may carry it as `"seq"`). After a resume the server drops
  client audio with a seq it has already seen, so a client may resend its unacknowledged audio safely.
- `framing: "raw"` (default): binary frames are bare. The client can still number them: a binary frame's seq is the
  previous JSON frame's seq plus the number of binary frames since then.
- Keep the last seq you have **processed**; it is what you send in RESUME.

## 3. Keep-alive

The server sends `{"type": "PING", "ts": …}` every `ping_interval_s` (30 s). Reply `{"type": "PONG"}` within
`RT_PONG_TIMEOUT_S` (10 s) or the server sends `ERROR pong_timeout`, closes 4408 and suspends the session (it can
still be resumed). A client may send `PING` too and gets `PONG`.

## 4. Answers, barge-in and echo

- An answer is `RESPONSE_START {request_id, output}`, then per sentence `RESPONSE_TEXT {index, text}` followed by its
  audio frames, then `RESPONSE_END {duration_ms, sentences}`. Play audio frames in order as they arrive. Audio is
  sent at playback speed, at most `RT_OUTPUT_LEAD_MS` (500 ms) ahead, so a small jitter buffer is enough.
- **Barge-in**: if the user speaks over the answer (speech probability > `RT_BARGE_IN_THRESHOLD` for
  `RT_BARGE_IN_MIN_MS`), the server cancels the answer and sends
  `{"type": "BARGE_IN", "request_id": …, "tts_position_ms": 1240}` followed by
  `RESPONSE_END {"interrupted": true}`. **Stop playback at once and drop every buffered audio frame of that answer.**
  `tts_position_ms` is the server's estimate of where playback was; use it for logs or captions. The user's speech
  becomes the next utterance, with its own transcripts and answer.
- Speech that ends while an answer is playing and did not barge in is ignored (`DISCARDED reason=during_response`):
  without echo cancellation it is the answer itself coming back through the microphone.
- **Echo is the client's job.** Module 34 has no server-side acoustic echo cancellation. Do not send the played answer
  back through the microphone:
  - browsers: `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}})`;
  - phones / SIP: the handset's or the gateway's AEC;
  - no AEC available (speakerphone into a raw microphone): ask the operator to set `RT_BARGE_IN_THRESHOLD=0.85`
    so the answer's own echo does not interrupt it. Headsets avoid the problem.

## 5. Dropped connections and RESUME

- A connection that drops without END keeps its session for `resume_window_s` (60 s). The pipeline keeps running:
  the open utterance, transcripts, the answer being generated and the Module 15 conversation are kept; answer audio
  waits on the server.
- To resume: get a **new ticket** (same user), connect to
  `/audio/stream?token=<ticket>&session_id=<session_id>` and send as the first frame

```json
{"type": "RESUME", "session_id": "<session_id>", "last_seq": 41}
```

  The server resends every frame after `last_seq` that it still has (original seqs), then sends
  `RESUMED {replayed, gap, last_client_seq, state}`, then carries on: the answer continues from where the client
  stopped. `gap: true` means some frames were too old to keep (`RT_RESUME_BUFFER_KB`). With `framing: "seq"`, resend
  your audio after `last_client_seq`.
- Not resumed in time, a different user, or an unknown id: `ERROR SESSION_EXPIRED`, close 4410. Start a new session.
- If the old connection is still open (half-open TCP), the new one takes over and the old one is closed with 4409.
- Close codes: 1000 END, 4400 protocol error, 4408 timeout / lifetime / no PONG, 4409 replaced, 4410 expired, 1011 error.

## 6. Errors that do not end the session

`ERROR` with `"fatal": false` — `LLM_TIMEOUT`, `LLM_UNAVAILABLE` (a canned answer is spoken), `STT_OVERLOADED`
(that utterance was dropped) — and `WARNING` (`QUEUE_OVERFLOW`: the client sends audio faster than real time or the
server is overloaded; the oldest audio was dropped). Any other `ERROR` is followed by close.

## 7. External brain and delegation (server-side clients)

For a product that keeps its own dialogue logic (a call-centre bridge, a kiosk). Needs a Keycloak service
account with realm roles `rt-external-brain` (and `rt-delegate` to act for end users).

```json
{"type": "CONFIG", "codec": "pcma", "sample_rate": 8000, "language": "zh-TW", "framing": "seq",
 "brain": "external", "subject_id": "cs-cust-0042", "record": false, "index_transcripts": false, "store_memory": false}
```

- `brain: "external"` without the role: `ERROR BRAIN_MODE_FORBIDDEN`, close 4403. Module 34 never asks Module 15;
  you get `TRANSCRIPT_FINAL` and decide what to say.
- Answer with `SPEAK {"utterance_id": "u7", "text": "好的，我幫您查一下。", "final": false}` and more `SPEAK`
  frames for the same `utterance_id`; the last one has `"final": true`. They play as one answer
  (`RESPONSE_START`/`RESPONSE_END` with `request_id` = `u7`, binary audio in the session codec in between).
  Different `utterance_id`s play in the order they arrive. First audio arrives about 0.3 s after the first `SPEAK`
  with Module 23 screening it (`RT_EXTERNAL_GUARDRAIL=output`), under 0.1 s without.
- `SPEAK_CANCEL {"utterance_id": "u7"}` drops what is not said yet; if it is playing you get
  `RESPONSE_END {"cancelled": true}`.
- `BARGE_IN` cancels the answer being played and every queued `SPEAK`: cancel your own generation too, and flush
  your playout buffer.
- `subject_id` (with `rt-delegate`): consent, recordings, erasure, the personal index and memory are the end user's,
  not the service account's. Without the role it is ignored (`WARNING subject_ignored` after `READY`).
- `record` / `index_transcripts` / `store_memory` default to `false` for external sessions; `READY.side_effects`
  says what is in effect.

## 8. Browser example

PCM 16 kHz with `framing: "seq"`, playback through Web Audio, PING/PONG, barge-in and reconnect with RESUME.
Microphone capture (an `AudioWorklet` that posts 20 ms `Int16Array` frames to `onMicFrame`) is left out.

```javascript
class RealtimeAudioClient {
  constructor({ baseUrl, getJwt }) {
    this.baseUrl = baseUrl;          // e.g. "https://raptor.example.com"
    this.getJwt = getJwt;            // async () => Keycloak access token
    this.sessionId = null;
    this.lastSeq = 0;                // last server seq processed
    this.sent = 0;                   // our own seq
    this.unacked = [];               // [seq, frame] sent but maybe not received
    this.ctx = new AudioContext({ sampleRate: 16000 });
    this.playAt = 0;
    this.sources = new Set();
    this.ended = false;
  }

  async ticket() {
    const r = await fetch(`${this.baseUrl}/audio/session/token`, {
      method: "POST", headers: { Authorization: `Bearer ${await this.getJwt()}` },
    });
    if (!r.ok) throw new Error(`ticket: ${r.status}`);
    return (await r.json()).token;
  }

  async start() {
    await this.open(false);
  }

  async open(resume) {
    const wsBase = this.baseUrl.replace(/^http/, "ws");
    let url = `${wsBase}/audio/stream?token=${await this.ticket()}`;
    if (resume) url += `&session_id=${this.sessionId}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify(resume
      ? { type: "RESUME", session_id: this.sessionId, last_seq: this.lastSeq }
      : { type: "CONFIG", codec: "pcm", sample_rate: 16000, language: "zh-TW", framing: "seq" }));
    ws.onmessage = (m) => this.onFrame(m.data);
    ws.onclose = (e) => this.onClose(e);
  }

  onFrame(data) {
    if (data instanceof ArrayBuffer) {
      const seq = new DataView(data).getUint32(0, true);
      if (seq <= this.lastSeq) return;                // replayed twice
      this.lastSeq = seq;
      this.play(new Int16Array(data, 4));
      return;
    }
    const ev = JSON.parse(data);
    if (ev.seq !== undefined) {
      if (ev.seq <= this.lastSeq) return;
      this.lastSeq = ev.seq;
    }
    switch (ev.type) {
      case "READY": this.sessionId = ev.session_id; break;
      case "PING": this.ws.send(JSON.stringify({ type: "PONG", ts: ev.ts })); break;
      case "BARGE_IN": this.stopPlayback(); break;
      case "RESUMED":
        // resend our audio the server has not seen
        for (const [seq, frame] of this.unacked) if (seq > ev.last_client_seq) this.ws.send(frame);
        this.unacked = this.unacked.filter(([seq]) => seq > ev.last_client_seq);
        break;
      case "ERROR":
        if (ev.code === "SESSION_EXPIRED") { this.sessionId = null; this.ended = true; }
        break;
      case "ENDED": this.ended = true; break;
      default: this.onEvent?.(ev);                      // transcripts, RESPONSE_TEXT, ...
    }
  }

  onMicFrame(int16) {                                   // 20 ms from the AudioWorklet
    const frame = new Uint8Array(4 + int16.byteLength);
    new DataView(frame.buffer).setUint32(0, ++this.sent, true);
    frame.set(new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength), 4);
    this.unacked.push([this.sent, frame]);
    if (this.unacked.length > 500) this.unacked.shift();  // ~10 s
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(frame);
  }

  play(int16) {
    const buf = this.ctx.createBuffer(1, int16.length, 16000);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    this.playAt = Math.max(this.playAt, this.ctx.currentTime + 0.05);
    src.start(this.playAt);
    this.playAt += buf.duration;
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  stopPlayback() {                                       // barge-in: drop what is buffered
    for (const src of this.sources) src.stop();
    this.sources.clear();
    this.playAt = 0;
  }

  async onClose(e) {
    if (this.ended || e.code === 1000 || e.code === 4409 || e.code === 4410 || !this.sessionId) return;
    for (let attempt = 0; attempt < 5 && !this.ended; attempt++) {   // within resume_window_s
      await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
      try { await this.open(true); return; } catch { /* ticket or network failed: retry */ }
    }
  }

  end() { this.ws?.send(JSON.stringify({ type: "END", seq: ++this.sent })); }
}
```

## 9. gRPC: `AudioService.AudioChat`

`deployment/modules/34-realtime-audio/proto/raptor/audio/v1/audio.proto`, served by Module 34 on port
8044 (`RT_GRPC_PORT`, host `PORT_REALTIME_AUDIO_GRPC`). One call is one session of the same pipeline as
the WebSocket; differences for the client:

- Authenticate with metadata `authorization: Bearer <Keycloak JWT>` -- no ticket. A bad, expired or
  missing token ends the call with `UNAUTHENTICATED`; a full server with `RESOURCE_EXHAUSTED`.
- The first `AudioChunk` sets `codec`, `sample_rate` and `language` (empty / 0 = pcm, the codec's usual
  rate, `RT_STT_LANGUAGE`); later chunks carry only `payload`. Send at playback speed.
- Half-close the request stream instead of END: the call ends with OK after the last answer has played.
  Cancelling the call ends the session; there is no RESUME, `seq` or PING/PONG.
- `TranscriptEvent.is_final` false = the rolling transcript, true = the utterance's final transcript.
  Play `AudioResponseChunk.payload` (the caller's codec; pcm = 16 kHz s16le) until the chunk with
  `is_last=true` for that `request_id`; `interrupted=true` there means the user spoke over the answer --
  drop what is still buffered, as for `BARGE_IN`.
- `ErrorEvent.fatal=true` is followed by a non-OK status; `fatal=false` is a warning.

```python
import asyncio
import grpc
from raptor.audio.v1 import audio_pb2, audio_pb2_grpc   # python scripts/gen_proto.py OUT_DIR

async def talk(pcm16k: bytes, jwt: str, play) -> None:
    async def microphone():
        for i in range(0, len(pcm16k), 640):            # 20 ms of 16 kHz s16le
            first = {"codec": "pcm", "sample_rate": 16000, "language": "zh-TW"} if i == 0 else {}
            yield audio_pb2.AudioChunk(payload=pcm16k[i:i + 640], **first)
            await asyncio.sleep(0.02)
        # returning half-closes: END

    async with grpc.aio.insecure_channel("raptor-realtime-audio:8044") as channel:
        call = audio_pb2_grpc.AudioServiceStub(channel).AudioChat(
            microphone(), metadata=(("authorization", f"Bearer {jwt}"),))
        async for event in call:
            kind = event.WhichOneof("event")
            if kind == "transcript" and event.transcript.is_final:
                print("you said:", event.transcript.text)
            elif kind == "audio" and not event.audio.is_last:
                play(event.audio.payload)
            elif kind == "error":
                print("server:", event.error.code, event.error.message)
```
