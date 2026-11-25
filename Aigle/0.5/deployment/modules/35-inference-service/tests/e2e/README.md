# Module 35 end-to-end tests

Real HTTP calls against a real running stack -- no mocks, no `TestClient`. Not run in CI
(same category as this repo's `tests/integration/` suites: the CI unit-tests job only ever
targets `tests/unit/`, for every module, so this directory is never picked up automatically).
Run these by hand, or from a scheduled/manual workflow on a host with a real GPU.

## Scope note (2026-11-20)

The original design chains `translate -> summarize -> tts -> stt`. As of 2026-10-23, `/tts`
and `/stt` didn't exist in this service yet, so Test 1 was scoped down to
`translate -> summarize` only (`test_translate_then_summarize`). Both endpoints have since
shipped (forwarding to Module 34's already-loaded Whisper/TTS models) -- the full chain is
now covered by
`test_translate_then_summarize_then_tts_then_stt`, which needs a voice-enabled instance
(skips otherwise, same as `test_voice_pipeline.py`); the text-only version stays, since it
runs on any instance regardless of voice/GPU availability.

The ticket's Test 3 asks for an `X-Model` response header; no such header exists in this
service -- model selection is verified from the JSON body's `model` field instead (the only
place it's actually reported). It also names `llama3.2:8b` as the example model to switch to;
the test instead picks the two smallest currently-available LLM models at run time, so it
doesn't depend on one specific model being present on whatever instance it's run against.

Test 5 as written is voice-only (3 concurrent `/diarize` calls, no deadlock/OOM); covered
as-is in `test_voice_pipeline.py`. `test_text_pipeline.py` additionally has a text-path
concurrency test (5 concurrent `/translate` calls) -- that one is not asked for by the
ticket, added as the natural adjacent case for the text profile.

Beyond the ticket's 5 named tests, `test_text_pipeline.py` and `test_voice_pipeline.py` also
cover the rest of this module's existing endpoint surface that the 5 tests don't otherwise
exercise: `/complete` and `/transform`, the generic `/inference/text` dispatcher,
`GET /models/default` and `GET /models/{name}`, and the speaker-cache list/delete endpoints.

## Prerequisites

Bring up the real dependencies first (from `deployment/modules/`):

```bash
# Text path (Tests 1, 3): module 02 (Redis) + 07 (real LLM via Ollama) + 35 --profile text
python3 build.py -m 02,07,35 --build

# Test 4 (guardrail) additionally needs module 23 with GUARDRAIL_ENABLED=true for module 35
python3 build.py -m 23 --build
# ...and in deployment/modules/.env: GUARDRAIL_ENABLED=true (module 35 side)

# Voice path (Tests 2, 5) needs a real GPU host and the voice profile:
# deployment/modules/.env: VOICE_ENABLED=true, then
python3 build.py -m 35 --build
```

Point the tests at wherever module 35 actually landed (defaults assume the standard
`PORT_INFERENCE` mapping from `deployment/modules/.env.example`):

```bash
export E2E_INFERENCE_URL=http://localhost:8036   # default if unset
```

If module 35 isn't reachable at that URL, every test in this directory skips with a clear
message instead of failing on connection errors.

The clone-related tests in `test_voice_pipeline.py` (part of Test 2) additionally need
`voice_clone` GDPR consent actually granted for a real Keycloak user -- an arbitrary
made-up `X-User-ID` can never pass Module 33's check. `consented_user_id` (conftest.py) gets
this for free: it logs in as any ordinary user through the real gateway and grants that real
user's own `voice_clone` consent via `PUT /api/<version>/gdpr/consent/voice_clone` -- no
Keycloak admin credentials needed, since a user granting their own consent is always
self-service. Supply that user's credentials yourself (no default -- this is a credential,
unlike the other `E2E_*` vars below):

```bash
export E2E_GATEWAY_URL=http://localhost:8012      # default if unset
export E2E_GATEWAY_USER=<a real user on your realm>
export E2E_GATEWAY_PASSWORD=<their password>
# E2E_GATEWAY_REALM (default dhtsolution) / E2E_GATEWAY_CLIENT_ID (default raptor) --
# only needed if either differs on your instance.
```

Without `E2E_GATEWAY_USER`/`PASSWORD` set (or if the gateway isn't reachable, or that login
fails), only the clone-dependent tests skip with a clear message -- everything else in this
directory doesn't need the gateway at all (it talks to module 35 directly) and still runs.

## Audio fixtures (Tests 2, 5)

`fixtures/two_speakers_zh.wav` and `fixtures/reference_clip.wav` are checked in --
synthesized via `gtts-cli` (zh-TW) and `ffmpeg`: two short lines concatenated with a silence
gap, one line pitch-shifted (`asetrate`+`atempo`) so its voice is acoustically distinct from
the other, which is what lets diarization actually split them into 2 speakers rather than
merging a single synthetic voice back into one. `reference_clip.wav` is one clean line of
the un-shifted voice. If these are ever regenerated, a missing file still skips the affected
tests with a message naming the expected path, same as the service-reachability check above
-- it doesn't hard-fail the suite.

## Running

```bash
cd deployment/modules/35-inference-service
pytest tests/e2e -v
```
