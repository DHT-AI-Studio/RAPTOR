# Raptor gRPC Gateway

A gRPC service (Module 36, `raptor-grpc-gateway`, `:8825`): server bootstrap,
full `RaptorService` proto contract, standard health check, TLS/mTLS support,
and an 8-stage interceptor chain.

**Security status:** Bearer tokens are verified against Keycloak's JWKS,
`x-api-key` is checked against an allow-list, and each RPC is authorized
through module 06's UMA permission check. Talk to this repo's
maintainers before exposing the service beyond a trusted internal network --
a formal security review is still pending, transport is plaintext unless
`GRPC_GATEWAY_GRPC_SECURE=true`, `GRPC_GATEWAY_ALLOW_INSECURE_DEV_FALLBACK`
must stay `false` outside local development, and reaching a real external
caller needs its own setup (a public name, a real certificate, a
TLS-terminating proxy) that this module alone cannot provide.

RPC status:

| RPC | Status |
| --- | --- |
| `Ping` | real |
| `SearchByPerson` / `StreamSearchByPerson` | real — `VideoSearchService`, a copy of the Raptor 0.5 service kept under `vendor/` (module 13 on `main` no longer ships it; REST's `/search/video_search` calls module 25 directly, so the two paths are separate and have known differences, pinned by `tests/unit/test_rest_grpc_parity.py`) |
| `DownloadAsset` | real — resumable, constant-memory streaming download from module 04 |
| `UploadAsset` | real — client-streaming upload with idempotency, per-user concurrency limit, and idle timeout |
| `SubmitJob` / `GetJob` / `WatchJob` | real — see `services/job_service.py`; `SubmitJob` routes to the same Kafka pipeline as REST's `/processing/process-file` |
| `StreamVideoSegment` | real for the `asset` source only — FFmpeg-transcoded (MP4/MPEG-TS) or raw byte-range (`RAW_FILE_RANGE`) time-ranged video streaming; see `services/media_stream_service.py`. Its `video_id` source is `UNIMPLEMENTED` — no endpoint anywhere in Raptor resolves a bare `video_id` back to an `asset_path` today. |
| `JobSession` | real for the subscribe (event-relay) direction only — a bidirectional wrapper around `WatchJob`; see `services/job_service.py`. Its `cancel`/`control` directions are intent-only: acknowledged, recorded to Redis, but no processing worker actually stops or changes a running job. |
| `AnalyzeStream` | **off by default** (`GRPC_GATEWAY_ANALYZE_STREAM_ENABLED`; refused with `UNIMPLEMENTED` while off). When on, real for chunk-forwarding only — every `MediaChunk` sent is published to Kafka immediately, no buffering; see `services/media_stream_service.py`. Always ends `UNIMPLEMENTED` once the client's send side closes: no consumer anywhere in Raptor reads the analysis-requests topic or produces `AnalysisEvent`. |
| `Search` / `SearchBm25` / `SearchVector` | real — hybrid, keyword-only and semantic-only search over module 25's `/personal/search/{hybrid,bm25,vector}`; one `SearchResult` per matching chunk or whole-file summary (see `services/search_service.py`) |
| `StreamSearch` / `StreamSearchBm25` / `StreamSearchVector` | real — the same three searches delivered one `SearchResult` per stream item; buffered (module 25 has no incremental search) |
| `GraphRagSearch` / `TKGSearch` | real — module 25's `/personal/search/{graphrag,tkg}`; hits (matching moments, or dated facts for TKG) plus a `subgraph_json` string; see `services/search_service.py` |

`scripts/search_parity_check.py` is the live REST-vs-gRPC equivalence check for the search RPCs: against a deployed stack and one account it asks `/api/0.5/search/{hybrid,bm25,vector,graphrag,tkg}` and the matching gRPC calls the same questions and compares hits, order, scores and graph nodes. RPCs the server does not have yet are reported as SKIP. Read-only; credentials and addresses only via the environment (`PARITY_USER`, `PARITY_PW`, `PARITY_GRPC`, `PARITY_REST` — no default address is kept in the repo).
| `GetAsset`, `Chat`, `GetJobStatus`, `StreamVideo` | `UNIMPLEMENTED` — not built yet |

The goal of the still-unimplemented RPCs' transport/interceptor plumbing is to
let later feature phases build on it without reworking infrastructure — see
each interceptor/service module's docstring for the reasoning behind its design.

**`DownloadAsset` header notes** (`services/media_probe.py`):
- `duration_sec` is filled for audio and video files (by file name) by running ffprobe, which ships with the
  ffmpeg in the image, against the same presigned URL the download reads. It runs while the storage stream is
  being opened, so it usually adds a few tens of milliseconds to the header; `GRPC_GATEWAY_PROBE_DURATION_TIMEOUT_S`
  (3 s) caps the delay.
- Any failure (no ffprobe, an unreadable file, a timeout, a cancelled download) leaves `duration_sec` at 0
  ("unknown") and never fails the download; the subprocess is killed. `GRPC_GATEWAY_PROBE_DURATION=false` turns it
  off; `GRPC_GATEWAY_FFPROBE_BINARY` names the executable.

**`SubmitJob`/`GetJob`/`WatchJob` scope notes** (see `services/job_service.py`'s
module docstring for the full reasoning):
- `SubmitJob` accepts exactly one entry in `input_asset_ids` — the underlying
  Kafka pipeline (`KafkaService.send_processing_request`) dispatches one file
  per job; there's no multi-asset batch primitive anywhere in Raptor today.
- `WatchJob`'s event stream is synthesized by polling the same single mutable
  `step` field the processing workers already write to
  `{media_type}_orchestrator:{job_id}` — Raptor has no append-only job-event
  log upstream of this. Each observed state *transition* becomes one
  `JobEvent`, persisted to `grpc:job_events:{job_id}` (Redis list, TTL 7
  days) so a reconnecting client can replay past `last_sequence`.
- The WebSocket relay this RPC set was originally paired with (a
  `/api/v1/jobs/{job_id}/progress` endpoint relaying node-level
  `{"type": "executing", "node": ...}` / `{"type": "progress", "value": N,
  "max": N}` events) is **not implemented** — that endpoint doesn't exist
  anywhere in this codebase or in Content AI today, and its JSON shape
  describes per-node workflow execution progress that nothing in Raptor's
  job pipeline produces (Raptor's job model is the `JobState` enum in the proto,
  not node-level). Building it would mean either fabricating that data or
  instrumenting all four processing workers with new progress granularity —
  out of scope here; flagged for follow-up once the intended contract is
  confirmed.

> **Note:** Module 36 is registered in `build.py` (`python build.py -m 36`, which
> needs modules 02, 04, 05 and 25 running). To run it on its own, use
> `docker compose up -d` from this directory.

## Public test scope and what was verified live

What a caller can use, as of the Aigle 0.5 public test:

- **Works:** `Ping`, `Search` / `SearchBm25` / `SearchVector`, the three `StreamSearch*` (buffered, not
  incremental), `GraphRagSearch`, `TKGSearch`, `SearchByPerson` / `StreamSearchByPerson`, `UploadAsset`
  (optional CRC32C), `DownloadAsset`, `StreamVideoSegment` (`asset` source), `SubmitJob` / `GetJob` /
  `WatchJob`, `JobSession` (subscribe only).
- **`UNIMPLEMENTED`:** `GetAsset`, `Chat`, `GetJobStatus`, `StreamVideo`, `StreamVideoSegment` with `video_id`,
  and `AnalyzeStream` (off by default).
- Not provided: a browser WebSocket relay of job progress; cancelling a job through `JobSession` (only recorded).

Checked against a deployed test host (not in CI), with real accounts:

- every search RPC returns the same hits, order, scores and graph nodes as the REST API
  (`scripts/search_parity_check.py`: 24 checks, 0 differences);
- one account cannot read another's assets or jobs (`scripts/isolation_check.py`: 8 of 8);
- each RPC answers as documented, including `DATA_LOSS` on a wrong checksum, resume of `WatchJob`, and the
  `UNIMPLEMENTED` ones above; the `raptor-grpc-client` wheel installs in a clean environment and the
  README example runs;
- files of 10, 50, 100, 150 and 200 MB upload and download back byte-identical (see the `UNAVAILABLE` note below).

- cancelling an `UploadAsset` right after the last byte (20 MB files, cancel 20 ms to 0.8 s after the stream
  ended, 24 rounds): the commit survived the cancel in 15 rounds, each time a repeat with the same
  `idempotency_key` answered `ALREADY_EXISTS` with `asset-id` / `version-id` in the trailing metadata; in the other 9 the
  cancel won and nothing was stored, so the repeat created the asset. Every asset that existed (27) downloaded
  back complete and byte-identical; none was partial or duplicated. A repeat sent while the commit is still
  running can still produce a second answer (`ALREADY_EXISTS` without ids, once in 40 rounds): the known limit
  in E15.

Downloads right after an upload, or of a large file, answered `UNAVAILABLE` ("Downstream service unavailable") three
times in about 60 attempts and succeeded when repeated. The gateway now retries the asset lookup (metadata and
presigned URL, before any byte is sent) up to three times on a transient 502/503/504 or connection error; a
failure in the middle of the body is not retried, so a client should still retry `UNAVAILABLE` on `DownloadAsset`.
Whether this removes the failures seen on the test host is not yet checked.

Not verified live: several clients at the same time, files above 200 MB, and TLS (it is off by default, see below).

## Module layout (36-grpc-gateway)

- Build context is `deployment/modules/` (`docker-compose.yml` sets `context: ..`):
  the image reuses module 13's `app/services/{search,storage,personal_db,kafka}_service.py`
  and `app/core/config.py` instead of copying them.
- `vendor/app/services/video_search_service.py` is the one shared class module 13 no
  longer ships; it is copied verbatim from Raptor 0.5 (see `vendor/README.md`).
- `proto/buf.yaml` configures `buf lint` / `buf breaking`; `client/` and `scripts/build_client.sh`
  build the `raptor-grpc-client` wheel.
- `scripts/isolation_check.py` checks, against a deployed gateway and with two accounts, that one account
  cannot reach the other's assets or jobs (not part of CI; tracked in an internal security review). It needs
  `ISO_USER_A`, `ISO_PW_A`, `ISO_USER_B`, `ISO_PW_B`, `ISO_GRPC` and `ISO_REST` in the environment.
- Tests live in `tests/unit/` so CI discovers them. `dev_paths.py` sets up the same
  `app.*` import layout the image has, and generates `generated/` stubs on demand.
- `tests/unit/test_rest_grpc_parity.py` runs module 13's REST `video_search` route and
  `SearchByPerson` end to end against the same faked module 25 and requires identical clips
  (CI runs it). One known difference is pinned there on purpose: REST sends module 25 a
  hard-coded GraphRAG/TKG `score_threshold` of 0.3, the gateway forwards the caller's.

## Asset identifiers

`DownloadAsset`, `StreamVideoSegment` and `SubmitJob` take the `asset_path` the platform
reports everywhere else — REST, search results, `UploadAssetResponse.asset_id` — which has
**no file name** (`video/mp4/clip`). `DownloadAsset` and `StreamVideoSegment` also still accept
the full object key (`video/mp4/clip/clip.mp4`); `services/asset_resolver.py` tries the
platform form first (one metadata lookup, then the presign) and falls back to the key.
`DownloadAsset` and `StreamVideoSegment` need the `version_id` that came with the `asset_path`
(`UploadAssetResponse.version_id`); without it they answer `INVALID_ARGUMENT` ("version_id is required…").
`DownloadAsset`'s largest chunk is 4 MiB minus 4 KiB so that chunk plus framing fits gRPC's
default 4 MiB receive limit.

## Interceptor chain

```
AuthenticationInterceptor -> IdentityInterceptor -> AuthorizationInterceptor -> RequestIdInterceptor
  -> LoggingInterceptor -> MetricsInterceptor -> StreamLimitInterceptor -> [GuardrailInterceptor]
  -> ExceptionMappingInterceptor -> RPC handler
```

`GuardrailInterceptor` is in the chain only when `GRPC_GATEWAY_GUARDRAIL_ENABLED=true` and a
guardrail URL is set.

- **Auth**: requires `authorization: Bearer <jwt>` or `x-api-key: <key>` — returns
  `UNAUTHENTICATED` otherwise. Exempts `grpc.health.v1.Health` and reflection so
  health probes and `grpcurl` introspection work without credentials.
- **Identity**: verifies the Bearer token's signature against Keycloak's JWKS
  (matched by `kid`), the realm (`iss`) and expiry, and puts the token's `sub`
  into `user_var`, used by `SearchByPerson`/`DownloadAsset`/`UploadAsset`/
  `SubmitJob` to scope calls to the caller's own data. An invalid or expired
  token is `UNAUTHENTICATED`; an unreachable Keycloak is `UNAVAILABLE`. A call
  authenticated by `x-api-key` has no end user: it runs as one shared service
  identity (`dev-user`), so every API-key caller sees the same data. With
  `GRPC_GATEWAY_ALLOW_INSECURE_DEV_FALLBACK=true` (default `false`, local
  development only) a rejected token falls back to that identity instead.
- **Authz**: asks module 06's `/auth/permission` (the UMA check REST uses) with
  the RPC's full method name as the resource, for Bearer calls; an `x-api-key`
  call has no token to check.
- **Request ID**: propagates `x-request-id` from incoming metadata, or generates
  a UUID; echoed back in trailing metadata.
- **Logging**: one structured log line per call (method, request id, duration).
- **Metrics**: Prometheus `grpc_server_requests_total` / `grpc_server_handling_seconds`
  / `grpc_server_active_streams` / `grpc_server_stream_bytes_total` /
  `grpc_server_stream_cancellations_total` — see `interceptors/metrics.py`.
- **Stream limit**: per-user concurrent-stream cap shared across every
  streaming RPC (`GRPC_GATEWAY_MAX_STREAMS_PER_USER`, default 4; Redis key
  `grpc:streams:{user_id}`), rejects excess with `RESOURCE_EXHAUSTED`. See
  `interceptors/stream_limit.py`.
- **Guardrail** (off by default): sends the caller's text to module 23's
  `POST /guardrail/policy/decide` (`direction=input`, `module=36`) before the handler runs.

  | RPC | Field checked |
  |---|---|
  | `Search`, `SearchBm25`, `SearchVector` | `query` |
  | `StreamSearch`, `StreamSearchBm25`, `StreamSearchVector` | `query` |
  | `GraphRagSearch`, `TKGSearch` | `query` |
  | `SearchByPerson`, `StreamSearchByPerson` | `person_name` |
  | `Chat` | `message` |

  - `block` → `PERMISSION_DENIED`, `Policy violation: {category}` (the first guard
    category, e.g. `S1`).
  - `redact` → the handler gets the request with that field replaced by module 23's
    `redacted_content`.
  - `warn` → passes, logged. `allow` → passes.
  - Fail-open: a timeout (`GRPC_GATEWAY_GUARDRAIL_TIMEOUT_MS`, default 2000), connection error, non-2xx
    or malformed answer is logged as a warning and the call proceeds unchanged.
  - Every other RPC, including `DownloadAsset`, `UploadAsset` and the other streaming
    uploads, is not checked and makes no module 23 call. Responses are not checked.
  - A call already rejected by authentication or authorization is not checked.
  - Settings: `GRPC_GATEWAY_GUARDRAIL_ENABLED` (default `false`), the URL from
    `GRPC_GATEWAY_GUARDRAIL_URL` or `GUARDRAIL_URL`, and `GRPC_GATEWAY_GUARDRAIL_TIMEOUT_MS`.
    The older names `GR_GRPC_GUARDRAIL_ENABLED`, `GR_GRPC_GUARDRAIL_URL` and
    `GR_GRPC_TIMEOUT_MS` are accepted too. While it is on, startup registers `grpc_interceptor` as screened
    in module 23's coverage matrix; while it is off, startup clears that flag.
  - See `interceptors/guardrail.py`.
- **Exception mapping**: maps the `AppError` hierarchy in `interceptors/errors.py`
  to gRPC status codes (`ValidationError` → `INVALID_ARGUMENT`, `NotFoundError` →
  `NOT_FOUND`, etc. — see that file for the full table).

## StreamVideoSegment

Time-ranged video streaming, two families of format:

- **MP4_H264 / MPEG_TS** ("transcode formats"): an FFmpeg subprocess reads
  the asset's presigned URL directly (`-ss`/`-t` for the time range,
  `-c copy` remux — not a full re-encode, see `media_stream_service.py`'s
  module docstring for why and its codec-compatibility caveat), piped to
  stdout through the same bounded-queue backpressure pattern as
  `DownloadAsset`. Requires `ffmpeg`/`ffprobe` on the image (installed in
  the Dockerfile).
  **Accuracy of the cut:** because it is a stream copy, a segment starts at the
  keyframe at or before `start_time`, so it is usually longer than
  `end_time - start_time` (measured on a real 30 s clip: a requested 2 s came
  back as 4.1 to 5.6 s) and its timestamps restart at 0 — the actual start offset
  is not reported. A segment that starts at 0 is exact.
- **RAW_FILE_RANGE**: no FFmpeg — a direct ranged-GET proxy, structurally
  identical to `DownloadAsset`'s chunk loop.

Cancellation sends the FFmpeg subprocess SIGTERM, then SIGKILL after
`GRPC_GATEWAY_FFMPEG_TERMINATE_TIMEOUT_S` (default 5s) if it hasn't exited.
See `tests/test_media_stream_service.py` for a real-FFmpeg test suite
covering early-chunk-arrival timing, format correctness, and a 50-iteration
cancellation/orphan-process check.

**`video_id` is not resolvable.** `StreamVideoSegmentRequest.source` is a
`oneof asset_id | video_id` per its original spec, but only the `asset`
(asset_path + version_id) branch works — `video_id` (as returned by
`SearchByPerson`/`ClipHit`) is actually module 25's internal `version_id`,
and no endpoint anywhere in Raptor resolves a bare `version_id` back to its
owning `asset_path` (checked directly: `video_search_service.py` and module
25's `index.py` only ever return the pair together, inline, as part of a
search result). Requests using that branch get `UNIMPLEMENTED`.

## JobSession

A bidirectional RPC layering a control channel on top of `WatchJob`. The
subscribe direction (first message `subscribe=true`, then live relay) is a
thin, real wrapper around `watch_job()`. The `cancel`/`control` directions
are **intent-only**: no processing worker (modules 09-12) exposes any
cancel or control API today, so a `cancel=true` or `control` message is
recorded (a Redis key for cancel; nothing persisted for control, since
there's nowhere for it to take effect) and acknowledged with an ephemeral
`JobEvent` (`sequence=-1`, never written to the durable event log) rather
than silently no-op'd or reported as if the job actually stopped. See
`services/job_service.py`'s module docstring and `tests/test_job_session.py`
(which asserts the job's real state is untouched by a cancel message, not
just that an ack event was sent).

## AnalyzeStream

A bidirectional RPC intended for live chunk-level media analysis.

**Off by default.** With `GRPC_GATEWAY_ANALYZE_STREAM_ENABLED` unset (or `false`) every call is
refused with `UNIMPLEMENTED` ("AnalyzeStream is disabled on this gateway") before anything is
read, so nothing reaches Kafka or Redis: no service consumes what it would forward, and a client
should not be left believing its media is being analysed. Set it to `true` only once a consumer for
the analysis-requests topic exists. A client that keeps sending after the refusal may see
`INTERNAL` ("Internal error from Core") instead of `UNIMPLEMENTED`: the server has already ended the
call, and that is how gRPC reports a write to a finished stream. Treat either as "not available".
The rest of this section describes the enabled behaviour. Only the
chunk-forwarding half is real: the first message must be `start`
(`session_id` + `media_type`), then every subsequent `MediaChunk` is
published to Kafka (`GRPC_GATEWAY_KAFKA_ANALYSIS_REQUESTS_TOPIC`, default
`media-analysis-requests`) **immediately as it arrives** — no buffering,
verified by `tests/test_analyze_stream.py` asserting the Kafka message
count after each chunk individually. Idle timeout
(`GRPC_ANALYZE_IDLE_TIMEOUT_S`, default 120s) and cancellation cleanup
(session-state Redis key deleted in every exit path) both work.

**No `AnalysisEvent` is ever produced.** No consumer anywhere in Raptor
reads the analysis-requests topic — modules 09-12 are whole-file batch
pipelines (WhisperX/InternVL run against complete uploads, not incremental
chunks), and there is no streaming-capable inference anywhere in this
codebase. Rather than silently return an empty event stream, `AnalyzeStream`
forwards every chunk for real, then ends with `UNIMPLEMENTED` once the
client's send side closes.

## TLS / mTLS

Off by default (plaintext). Set `GRPC_GATEWAY_GRPC_SECURE=true` plus
`GRPC_GATEWAY_GRPC_TLS_CERT_PATH`/`GRPC_GATEWAY_GRPC_TLS_KEY_PATH` to serve
over TLS; additionally set `GRPC_GATEWAY_GRPC_MTLS_CA_CERT_PATH` to require
and verify a client certificate signed by that CA (mTLS, for internal
service-to-service paths). See `tests/test_tls.py` for a working example
(generates a throwaway self-signed CA/server/client cert set and drives a
real TLS/mTLS handshake against the server) and
`proxy-test/README.md` for terminating/forwarding TLS at a reverse proxy.

## Quick start

```bash
cp .env.example .env
docker compose up -d --build
```

### Verify health

```bash
grpc_health_probe -addr=:8825
# SERVING
```

(Or without the probe binary: any gRPC client calling
`grpc.health.v1.Health/Check` with no metadata — see `tests/conftest.py` for a
Python example.)

### Verify metrics

```bash
curl localhost:9825/metrics | grep grpc_server
```

### Manual RPC smoke test (reflection is enabled)

```bash
grpcurl -plaintext -H "authorization: Bearer x" \
  -d '{"message": "hello"}' \
  localhost:8825 raptor.v1.RaptorService/Ping
```

## Regenerating stubs

`generated/` is gitignored — it's produced by `scripts/gen_proto.sh`, both
locally and inside the Docker build, from `proto/raptor/v1/raptor.proto`.

```bash
pip install -r requirements.txt
bash scripts/gen_proto.sh
```

## Tests

```bash
pip install -r requirements.txt
bash scripts/gen_proto.sh
pytest tests/ -v
```

`tests/test_dummy_call.py` starts the real interceptor-wired server in-process
and asserts: an authenticated `Ping` call succeeds and gets back an
`x-request-id`; a call with no credentials gets `UNAUTHENTICATED`; an
`x-api-key` credential is accepted as an alternative to a Bearer token.

`tests/test_search_by_person.py`, `tests/test_download_asset.py`,
`tests/test_upload_asset.py` and `tests/test_job_lifecycle.py` cover the
real RPCs above — including error mapping, offset-resume, constant-memory
streaming (RSS), slow-client backpressure, idempotency, cancellation/cleanup,
and WatchJob's sequence-numbered reconnect/replay — against fake
storage/Redis/Kafka doubles, no live infrastructure required.

`tests/test_stream_limit.py` covers the per-user stream cap (shared across
RPC types, unary exemption, release-on-cancel). `tests/test_metrics.py`
covers the active-stream gauge and byte/cancellation counters.
`tests/test_tls.py` drives real TLS/mTLS handshakes against throwaway
certs. `tests/unit/test_rest_grpc_parity.py` and `tests/unit/test_check_reserved_fields.py`
are this repo's contract tests (see `scripts/check_reserved_fields.py`'s own docstring). `proxy-test/` holds
a real, verified nginx/Envoy HTTP/2 proxy configuration (see its README for
what was actually run and confirmed, not just written).

`tests/test_media_stream_service.py` covers `StreamVideoSegment` against a
**real FFmpeg subprocess** and a real local HTTP server serving a tiny
synthetic test video (generated with `ffmpeg -f lavfi ...` at test time) —
early-chunk-arrival timing (<3s) plus `ffprobe`-verified structural validity
of the streamed prefix, MPEG-TS sync-byte correctness, time-range
correctness, `RAW_FILE_RANGE`'s no-FFmpeg-spawned guarantee, and a
50-iteration cancellation test asserting no orphaned FFmpeg processes.
`tests/test_job_session.py` covers `JobSession`'s subscribe/cancel/control
directions, including a test proving cancel is acknowledged but never
changes the job's real state. `tests/test_analyze_stream.py` covers
`AnalyzeStream`'s real chunk-forwarding (including a test proving each
chunk is published individually, not batched), idle timeout, cancellation
cleanup, the shared stream limit, and the honest end-of-stream
`UNIMPLEMENTED` — against a fake Kafka producer, since no live Kafka broker
or GPU/ML stack exists in this development environment.
