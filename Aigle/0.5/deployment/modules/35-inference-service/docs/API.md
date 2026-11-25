# Module 35 text and voice API

## Table of Contents

1. [Authentication & Routing](#authentication--routing)
2. [Guardrail](#guardrail)
3. [Error Envelope](#error-envelope)
4. [System — GET /health](#system--get-health)
5. [Models](#models)
6. [Text Inference](#text-inference)
7. [Voice Inference](#voice-inference)
8. [Internal GDPR Endpoints](#internal-gdpr-endpoints)
9. [Deployment Profiles](#deployment-profiles)

---

## Authentication & Routing

Every request must carry a `Bearer` JWT in the `Authorization` header.
Module 13 validates the token, injects `X-User-ID`, and proxies to Module 35.
Module 35 itself is not reachable from outside the Docker network.

---

## Guardrail

Guardrail is **disabled by default**.

| Environment variable     | Effect                                                  |
|--------------------------|---------------------------------------------------------|
| `GUARDRAIL_ENABLED`      | Enables input checks via Module 23 before inference     |
| `GUARDRAIL_CHECK_OUTPUT` | Additionally enables output checks after inference      |

Module 23 endpoint: `POST /guardrail/policy/decide` (field: `direction: input|output`)

| Verdict | Behaviour |
|---------|-----------|
| `block` | `400 guardrail_rejection` |
| `warn` | Request proceeds; warning logged |
| `redact` (input) | User text replaced with redacted version; if redaction reaches system prompt → `400 guardrail_rejection` |
| `redact` (output) | Redacted text returned to caller |
| Unavailable / invalid | **Fail open** — request proceeds |

---

## Error Envelope

All errors return a flat JSON body:

```json
{ "error": "<code>", "message": "<optional detail>", ...extra fields }
```

Validation errors return `400` with:

```json
{
  "error": "invalid_request",
  "message": "request validation failed",
  "details": [{ "field": "options.max_tokens", "message": "..." }]
}
```
### Error Code Reference

| **Status** | **Error** | **Meaning** |
|-----------|-----------|-------------|
| 400 | `invalid_request` | Request body failed schema validation |
| 400 | `guardrail_rejection` | Input blocked by Module 23 |
| 400 | `unsupported_task` | task not in `translate\|summarize\|complete\|transform` |
| 400 | `invalid_reference_duration` | Reference audio duration outside 1–60 s (clone) |
| 400 | `invalid_speaker_id` | speaker_name contains invalid characters, exceeds 128 chars, or starts with `_` |
| 400 | `invalid_text` | text empty or exceeds 3000 characters (clone) |
| 400 | `unsupported_model` | Requested model is not supported by this endpoint |
| 401 | `missing_user_identity` | X-User-ID header absent (voice/clone endpoints) |
| 401 | `invalid_internal_token` | X-Internal-Token missing or wrong (GDPR endpoints) |
| 403 | `clone_disabled` | CLONE_ENABLED not set; no clone backend loaded |
| 403 | `consent_required` | Caller hasn't granted `voice_clone` consent (`PUT /gdpr/consent/voice_clone`); body carries `purpose` (clone only) |
| 404 | `model_not_found` | Named model not in Module 07 registry |
| 413 | `input_too_long` | Input exceeds chunk limit (summarize) |
| 422 | `max_tokens_exceeded` | max_tokens exceeds the model's configured limit |
| 422 | `model_not_found` | `model` not one of the clone backends enabled on this deployment (clone only) |
| 422 | `unsupported_language` | Language not supported by the resolved clone backend (clone only) |
| 422 | `alignment_language_unavailable` | Language not in offline diarization bundle |
| 503 | `model_limits_not_configured` | Model has no max_output_tokens set in Module 07 |
| 503 | `diarize_unavailable` | pyannote model not loaded |
| 503 | `clone_unavailable` | Resolved clone backend's model not loaded |
| 503 | `redis_unavailable` | Redis connection failed |
| 503 | `consent_check_unavailable` | Module 33 couldn't be asked for the caller's consent -- denied, never assumed (clone only) |
| 5xx | `upstream_error` | Module 07 returned an unexpected error |

---

## System

### GET /health

Accessible via `/api/0.5/inference/health`.

**Response `200`**

```json
{
  "status": "ok",
  "module": 35,
  "gpu_available": false,
  "gpu_name": null,
  "module07_reachable": true,
  "module34_reachable": true,
  "loaded_voice_models": [],
  "voice": {
    "diarize_loaded": false,
    "clone_loaded": false,
    "clone_configured_backends": []
  },
  "warnings": []
}
```

| **Field** | **Type** | **Notes** |
|-----------|----------|-----------|
| `status` | `"ok"` \| `"degraded"` | `"degraded"` only when `VOICE_ENABLED=true` and no GPU |
| `gpu_available` | `bool` | Always `false` on text-only deployments |
| `gpu_name` | `string` \| `null` | CUDA device name, or `null` |
| `module07_reachable` | `bool` | Live `/health` probe to Module 07 |
| `module34_reachable` | `bool` | Live `/health` probe to Module 34 (backs `/inference/voice/stt` and `/tts`) |
| `loaded_voice_models` | `string[]` | `"pyannote-whisperx"`, 0–2 of `"xtts-v2"`/`"chatterbox-multilingual-v3"` (only backends actually loaded so far -- lazy), or empty |
| `voice.clone_loaded` | `bool` | Clone is configured and policy-enabled (`CLONE_ENABLED`), not necessarily resident yet |
| `voice.clone_configured_backends` | `string[]` | Backend names enabled via `CLONE_BACKENDS_RUNTIME` on this deployment |
| `warnings` | `string[]` | VRAM warnings when voice enabled and free VRAM < 12 GiB |

---

## Models

### `GET /models`
List all models from Module 07 (MLflow + Ollama), cached.

**Response `200`**
```json
{
  "models": [{ "name": "gemma4-12b", "version": "1", "source": "mlflow" }],
  "count": 1,
  "sources": ["mlflow"]
}
```

### `GET /models/default`
Current default model per task.

**Response `200`**
```json
{
  "text": "gemma4-12b",
  "stt": null,
  "tts": null,
  "clone": null,
  "diarize": null
}
```

### `POST /models/default`
Set the default model for a task (persisted in Redis). Returns `204` no body.

**Request Body**

| **Field** | **Type** | **Required** | **Notes** |
|-----------|----------|:------------:|-----------|
| `task` | `string` | ✅ | One of `text`, `stt`, `tts`, `clone`, `diarize` |
| `model` | `string` | ✅ | Must exist in Module 07 registry |

```json
{ "task": "text", "model": "gemma4-12b" }
```

### `GET /models/{name}`
One model's registry entry.

**Response `200`**
```json
{ "name": "gemma4-12b", "version": "1", "source": "mlflow" }
```

**Errors:** `404` `model_not_found`

---

## Text Inference

All text endpoints share the same base request fields:

| **Field** | **Type** | **Required** | **Notes** |
|-----------|----------|:------------:|-----------|
| `model` | `string` \| `null` | ❌ | Overrides the task default for this call |
| `input` | `string` | ✅ | Minimum 1 character |
| `options` | `object` | varies | Task-specific; see each endpoint |

---

### `POST /inference/text/translate`

**Options**

| **Field** | **Type** | **Required** | **Default** | **Notes** |
|-----------|----------|:------------:|-------------|-----------|
| `target_language` | `string` (BCP-47) | ✅ | — | e.g. `"zh-TW"`, `"en"` |
| `source_language` | `string` (BCP-47) \| `null` | ❌ | auto-detect | |
| `formality` | `"formal"` \| `"casual"` \| `null` | ❌ | `null` | |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/text/translate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "Good morning", "options": {"target_language": "zh-TW"}}'
```

**Response `200`**
```json
{ "output": "早安", "latency_ms": 312 }
```

---

### `POST /inference/text/summarize`

**Options**

| **Field** | **Type** | **Required** | **Default** | **Notes** |
|-----------|----------|:------------:|-------------|-----------|
| `style` | `"bullet"` \| `"paragraph"` \| `"headline"` | ❌ | `"bullet"` | |
| `max_sentences` | `integer ≥ 1` \| `null` | ❌ | `null` | Max bullets (`bullet`) or sentences (`paragraph`) |
| `language` | `string` (BCP-47) \| `null` | ❌ | `null` | Omit to keep source language |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/text/summarize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "Long article...", "options": {"style": "bullet", "max_sentences": 3}}'
```

**Response `200`**
```json
{ "output": "• Point one\n• Point two\n• Point three", "latency_ms": 540 }
```

**Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 413 | `input_too_long` | Input exceeds `summarize_max_chunks × summarize_chunk_chars` characters |

---

### `POST /inference/text/complete`

**Options**

| **Field** | **Type** | **Required** | **Default** | **Constraints** |
|-----------|----------|:------------:|-------------|-----------------|
| `system_prompt` | `string` \| `null` | ❌ | `null` | |
| `max_tokens` | `integer` | ❌ | `1024` | ≥ 1; ≤ model limit |
| `temperature` | `float` | ❌ | `0.7` | `0.0` – `2.0` |
| `stop_sequences` | `string[]` \| `null` | ❌ | `null` | No empty strings |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/text/complete \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "Explain quantum entanglement", "options": {"max_tokens": 200}}'
```

**Response `200`**
```json
{ "output": "Quantum entanglement is...", "latency_ms": 1820 }
```

**Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 400 | `invalid_request` | Request body failed schema validation |
| 422 | `max_tokens_exceeded` | max_tokens exceeds the model's configured limit |
| 503 | `model_limits_not_configured` | Model has no max_output_tokens in Module 07 |

---

### `POST /inference/text/complete/stream`

Accepts the same request body as `/inference/text/complete`. All validation
(schema, model limits, guardrail input check) runs before the stream opens.
Errors are returned as standard HTTP responses, not SSE events.

**Response Headers**
```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

**SSE Event Format**

Each generated token:
```
data: {"token": "<token_text>", "finish_reason": null}
```

Final frame:
```
data: {"token": "", "finish_reason": "stop|length"}
```

End sentinel:
```
data: [DONE]
```

Mid-stream upstream error:
```
data: {"error": "upstream_error", "message": "<details>"}
```

**Pre-stream Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 422 | `max_tokens_exceeded` | max_tokens exceeds the model's configured limit |
| 503 | `model_limits_not_configured` | Model has no max_output_tokens in Module 07 |
| 400 | `guardrail_rejection` | Input blocked by Module 23 (`GUARDRAIL_ENABLED`) |
| 400 | `invalid_request` | Invalid request body fields |

> **Note:** The native browser `EventSource` API does not support `POST`.
> Use `fetch()` with a `ReadableStream` reader instead.

**JavaScript Example**
```javascript
const resp = await fetch('/api/0.5/inference/text/complete/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  },
  body: JSON.stringify({ input: 'Hello', options: { max_tokens: 200 } }),
});
const reader = resp.body.getReader();
const dec = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of dec.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const payload = line.slice(6);
    if (payload === '[DONE]') return;
    const ev = JSON.parse(payload);
    if (ev.error) { console.error(ev.message); return; }
    if (ev.finish_reason === null) appendToken(ev.token);
  }
}
```

> Module 13 proxy forwards SSE headers and streams raw bytes via `aiter_raw()` —
> no nginx buffering config is required for this deployment.

---

### `POST /inference/text/transform`

**Options**

| **Field** | **Type** | **Required** | **Default** | **Notes** |
|-----------|----------|:------------:|-------------|-----------|
| `instruction` | `string` | ✅ | — | Non-blank rewrite instruction |
| `preserve_structure` | `bool` | ❌ | `false` | |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/text/transform \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "The cat sat on the mat.", "options": {"instruction": "Make it formal"}}'
```

**Response `200`**
```json
{ "output": "The feline was positioned upon the mat.", "latency_ms": 430 }
```

---

### `POST /inference/text`

Generic dispatcher. Routes on the `task` field; validated identically to the
matching typed route.

**Request Body**
```json
{ "task": "translate|summarize|complete|transform", "...task fields": "..." }
```

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/text \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "translate", "input": "早安", "options": {"target_language": "en"}}'
```

**Errors:** `400` `unsupported_task` if `task` is not one of the four values.

---

## Voice Inference

Voice endpoints require `VOICE_ENABLED=true` and the voice Docker profile.
All CUDA work serialises through a process-local GPU semaphore (`gpu_lock`).
Run **exactly one uvicorn worker** per voice instance.

---

### `POST /inference/voice/diarize`

Speaker diarization with WhisperX-aligned transcription.

**Request** — `multipart/form-data`

| **Field** | **Type** | **Required** | **Constraints** |
|-----------|----------|:------------:|-----------------|
| `audio` | file | ✅ | WAV / MP3 / FLAC; ≤ 25 MiB; ≤ 600 s |
| `model` | string | ❌ | Default: `"pyannote-3.1"` |
| `num_speakers` | integer | ❌ | ≥ 1, ≤ 64; clustering hint — overshooting will not crash |
| `language` | string | ❌ | BCP-47; offline bundle supports `zh` / `zh-TW` / `zh-CN` |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/voice/diarize \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio=@recording.wav" \
  -F "language=zh-TW"
```

**Response `200`**
```json
{
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    { "speaker": "SPEAKER_00", "text": "你好", "start": 0.0, "end": 1.2 }
  ],
  "latency_ms": 4200
}
```

**Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 400 | `invalid_num_speakers` | num_speakers outside 1–64 |
| 422 | `alignment_language_unavailable` | Language not in offline bundle |
| 400 | `unsupported_model` | `model` is not `"pyannote-3.1"` |
| 503 | `diarize_unavailable` | pyannote model not loaded (`reason` field included) |

---

### `POST /inference/voice/stt`

Speech-to-text. No local model: forwards to Module 34
(realtime-audio)'s already-loaded Whisper model, reusing the same
admission-controlled GPU slots its live audio sessions use — a batch call
here can add latency to a concurrent live session (see Module 34's own
README).

**Request** — `multipart/form-data`

| **Field** | **Type** | **Required** | **Constraints** |
|-----------|----------|:------------:|-----------------|
| `audio` | file | ✅ | WAV / MP3 / FLAC; decoded/normalized here before forwarding |
| `language` | string | ❌ | BCP-47; omit to auto-detect |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/voice/stt \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio=@clip.wav" \
  -F "language=zh-TW"
```

**Response `200`**
```json
{
  "text": "你好",
  "language": "zh",
  "language_probability": 0.99,
  "confidence": 0.95
}
```

**Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 400 | `unsupported_audio_format` / `invalid_audio` | Upload isn't a decodable WAV/MP3/FLAC |
| 413 | `audio_too_large` | Over the upload bound |
| 429 | `too_many_sessions` / `too_many_sessions_for_user` | Module 34's session quota exhausted — this call shares it with the caller's own live audio sessions |
| 503 | `module34_unavailable` | Module 34 unreachable |
| 504 | `upstream_timeout` | Module 34 didn't respond in time |

---

### `POST /inference/voice/tts`

Text-to-speech. No local model: forwards to Module 34's already-loaded
TTS model, same reasoning and GPU-slot caveat as `/stt` above. Returns the
whole synthesized clip as one WAV body, not a stream.

**Request** — `multipart/form-data`

| **Field** | **Type** | **Required** | **Constraints** |
|-----------|----------|:------------:|-----------------|
| `text` | string | ✅ | |
| `language` | string | ❌ | BCP-47 |

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/voice/tts \
  -H "Authorization: Bearer $TOKEN" \
  -F "text=你好" -F "language=zh-TW" \
  -o out.wav
```

**Response `200`** — `audio/wav` body (Module 34's configured sample rate).

**Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 429 | `too_many_sessions` / `too_many_sessions_for_user` | Module 34's session quota exhausted — this call shares it with the caller's own live audio sessions |
| 503 | `module34_unavailable` | Module 34 unreachable |
| 504 | `upstream_timeout` | Module 34 didn't respond in time |

---

### `POST /inference/voice/clone`

Synthesize speech in a cloned voice. Requires `CLONE_ENABLED=true` and the
caller's `voice_clone` consent (GDPR Art. 9 biometric data, off by default):
Module 35 asks Module 33's `GET /consent` on every call, before the reference audio
is read, and answers `403 consent_required` without it, `503
consent_check_unavailable` when Module 33 can't be asked. Withdrawing the consent
deletes the caller's cached speaker embeddings. Up to two
backends can be enabled side by side (`CLONE_BACKENDS_RUNTIME`): `xtts-v2` and
`chatterbox-multilingual-v3`. `GET /models?type=clone` lists which are
actually configured on this deployment.

> **License note:** XTTS-v2 weights use the Coqui Public Model License
> (non-commercial) — a deployment that never lists `xtts-v2` in
> `CLONE_BACKENDS_RUNTIME` never loads those weights at all. Chatterbox Multilingual
> V3 (MIT) carries no such restriction, but watermarks every output
> unconditionally (Resemble PerTh) — see `X-Watermarked` below.

**Request** — `multipart/form-data`

| **Field** | **Type** | **Required** | **Constraints** |
|-----------|----------|:------------:|-----------------|
| `reference_audio` | file | ✅ | WAV / MP3 / FLAC; recommended 3–30 s; accepted 1–60 s |
| `text` | string | ✅ | 1–3000 characters |
| `language` | string | ✅ | BCP-47; `zh-TW` maps to XTTS's `zh-cn` or Chatterbox's `zh` depending on the resolved backend (no Taiwanese accent guarantee either way) |
| `speaker_name` | string | ❌ | 1–128 chars; ASCII letters / digits / `.` / `-`; must not start with `_` |
| `model` | string | ❌ | `xtts-v2` or `chatterbox-multilingual-v3`. Omit to use the deployment's default (`DEFAULT_CLONE_MODEL` / `POST /models/default`) |

- **Unnamed speakers:** ID is SHA-256 of the uploaded audio bytes.
- **Named speakers:** conditioning is cached per user; reuse avoids re-encoding.
  When 2+ backends are enabled, the cache key is qualified by backend name
  internally (`X-Speaker-Id` for an unnamed speaker reflects this; a named
  speaker's `speaker_name` is unaffected either way) — the same
  `speaker_name` under two different backends is two separate cache entries.
- `X-User-ID` (injected by Module 13 from JWT) scopes the speaker cache.
- **Speaker cache:** TTL = `86400 s` (`SPEAKER_CACHE_TTL_S`), max = `100` per user (`SPEAKER_CACHE_MAX`).
- LRU eviction on admission; hits update recency but do not renew TTL.
- Delete before replacing reference audio under the same name.
- **First use of a backend is slower**: backends load lazily, on first
  request, not at startup (unless `CLONE_BACKENDS_WARM` preloads them) —
  this keeps VRAM usage proportional to what's actually been used, not what's
  merely configured.

**cURL**
```bash
curl -X POST http://localhost:8012/api/0.5/inference/voice/clone \
  -H "Authorization: Bearer $TOKEN" \
  -F "reference_audio=@voice_sample.wav" \
  -F "text=Hello world" \
  -F "language=en" \
  -F "speaker_name=alice" \
  -F "model=chatterbox-multilingual-v3"
```

**Response `200`** — raw WAV (`audio/wav`) with response headers:
```
X-Speaker-Id: alice
X-Watermarked: true
```
`X-Watermarked` reflects the resolved backend (`true` for Chatterbox, `false` for XTTS).

**Errors**

| **Status** | **Error** | **Condition** |
|-----------|-----------|---------------|
| 400 | `invalid_reference_duration` | Reference audio duration outside 1–60 s |
| 400 | `invalid_text` | text empty or > 3000 chars |
| 400 | `invalid_speaker_id` | `speaker_name` format invalid |
| 400 | `model_required` | No `model` given and no default configured for this deployment |
| 401 | `missing_user_identity` | `X-User-ID` header absent |
| 403 | `clone_disabled` | `CLONE_ENABLED` not set; `reason` field included |
| 422 | `model_not_found` | `model` not one of the backends enabled on this deployment |
| 422 | `unsupported_language` | Language not supported by the resolved backend |
| 503 | `clone_unavailable` | Resolved backend's model not loaded (`reason` field included) |

---

### `GET /inference/voice/clone/speakers`

List the caller's cached speaker embeddings and remaining TTL.

**Response `200`**
```json
{
  "speakers": [
    { "id": "alice", "ttl_remaining_s": 72400 }
  ]
}
```

---

### `DELETE /inference/voice/clone/speakers/{id}`

Evict a caller-owned speaker embedding.

**Response `204`** — no body.

---

## Internal GDPR Endpoints

Called only by Module 33 (privacy-compliance service). Not exposed through
Module 13's gateway. Authenticated via `X-Internal-Token` header
(`GD_INTERNAL_TOKEN` env var).

> Missing or wrong token → `401` `invalid_internal_token`.

---

### `DELETE /internal/gdpr/user/{user_id}`

Full-account erasure, and the cascade when the user withdraws `voice_clone`
consent. Deletes all speaker embeddings and the LRU index for the user.
Always `200` — a user with no embeddings returns `0`, which is not an error.

**Response `200`**
```json
{ "user_id": "user-123", "speaker_embeddings_deleted": 3 }
```

---

### `GET /internal/gdpr/user/{user_id}/summary`

Export manifest count. Returns embedding count only — never the tensors.
Read-only: does not prune the LRU index.

**Response `200`**
```json
{ "user_id": "user-123", "speaker_embeddings": 3 }
```

---

## Deployment Profiles

### Text-only (default)
```bash
docker compose up -d
```

### Voice
Accept pyannote and diarization model licence conditions, export your
HuggingFace token, then:

```bash
docker compose --profile voice build
docker compose --profile voice up -d
```

- HF token is a BuildKit secret — never a Docker `ARG`/`ENV` layer.
- All pyannote, WhisperX ASR, VAD and zh alignment assets are baked in at build time.
- Runtime sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`; no model downloads at runtime.
- `pyannote.audio 3.3.2` (WhisperX 3.3.1's exact dependency); diarization model `3.1`.
- Preserves `torch`/`torchaudio` `2.7.1` from the Module 08 CUDA 12.8 base.
- Full GPU image build pending gated model access and target GPU verification.
- Voice clone is additionally gated by `CLONE_ENABLED=true` (unified `.env`:
  `INF_CLONE_ENABLED`). With it unset, `/voice/clone` returns `403 clone_disabled`
  and no clone backend is ever loaded. Which backend(s) get loaded (XTTS-v2
  and/or Chatterbox Multilingual V3) is controlled separately by
  `CLONE_BACKENDS_RUNTIME` (unified `.env`: `INF_CLONE_BACKENDS_RUNTIME`), and must be a
  subset of `CLONE_BACKENDS_BUILD` (`INF_CLONE_BACKENDS_BUILD`), which
  decides what actually got baked into the image at build time.
  `CLONE_BACKENDS_RUNTIME` defaults to `CLONE_BACKENDS_BUILD`'s own value when unset,
  so enabling everything a build baked in needs only one var set.

**Minimum GPU target:** RTX 5060 Ti 16 GB or equivalent.
Health endpoint warns when free VRAM < 12 GiB.

`/voice/stt` and `/voice/tts` forward to Module 34 rather than loading a local
model, so they don't take this image's `gpu_lock`.
