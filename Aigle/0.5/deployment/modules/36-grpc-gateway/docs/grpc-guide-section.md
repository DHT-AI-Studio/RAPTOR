# gRPC interface: guide section for `raptor_guide_0.5.md`

Every Python example below was run, in this order, against a deployed gateway.

## 1. What it is

`RaptorService` is a gRPC API (module 36, container `raptor-grpc-gateway`) in front of the same
services REST uses: module 25 (search), module 04 (assets) and the media workers (jobs). It
listens on **:8825** (gRPC, plaintext unless TLS is switched on) and serves Prometheus metrics on
**:9825**. The contract is `proto/raptor/v1/raptor.proto`; the generated Python client is the
`raptor-grpc-client` package (section 2).

| Role | Interface | Port |
| --- | --- | --- |
| Request/response, browsers, existing integrations | REST (module 13) | 8012 |
| Typed clients, streaming transfers, job event streams | **gRPC** (module 36) | 8825 |
| gateway metrics | Prometheus `/metrics` | 9825 |

## 2. Quick start (Python)

The client is a set of generated stubs. It is not published to a package index yet; build the
wheel from the repository (`bash scripts/build_client.sh`, result in `client/dist/`) and install it:

```bash
pip install client/dist/raptor_grpc_client-*.whl     # needs grpcio >= 1.68.0, protobuf >= 5.28.1
```

**Authenticate.** Every call needs `authorization: Bearer <access token>` (the same Keycloak token REST
uses) or `x-api-key: <key>` from the gateway's allow-list. Health checks and server reflection need
neither. A Bearer token is verified (signature, realm, expiry); an invalid one is `UNAUTHENTICATED`.
An `x-api-key` call has no end user and runs as one shared service identity, so every API-key
caller sees the same data.

```python
# example 1: connect and say hello. TOKEN is an access token, TARGET is "host:8825".
import grpc
from raptor.v1 import raptor_pb2 as pb, raptor_pb2_grpc as pbg

channel = grpc.insecure_channel(TARGET, options=[("grpc.max_receive_message_length", 16 * 1024 * 1024)])
stub = pbg.RaptorServiceStub(channel)
AUTH = (("authorization", f"Bearer {TOKEN}"),)
print(stub.Ping(pb.PingRequest(message="hello"), metadata=AUTH, timeout=10).message)
```

Raise `grpc.max_receive_message_length` when you ask for chunks larger than about 4 MiB (the default
limit); the gateway never sends more than 4 MiB minus 4 KiB per chunk.

**Search for a person** (a list of `ClipHit`, one per matching segment):

```python
# example 2: unary search, then the same search as a stream
reply = stub.SearchByPerson(pb.SearchByPersonRequest(person_name="黃仁勳", top_k=3), metadata=AUTH, timeout=60)
for hit in reply.hits:
    print(hit.filename, round(hit.score, 3), hit.start_time, hit.end_time)
streamed = list(stub.StreamSearchByPerson(pb.SearchByPersonRequest(person_name="黃仁勳", top_k=3), metadata=AUTH, timeout=60))
print(len(reply.hits), len(streamed))
```

**Search by text** with `Search` (keyword and meaning together, the one to use by default),
`SearchBm25` (keyword only) or `SearchVector` (meaning only). All three take a `SearchRequest`:

```python
# example 7: the same query three ways
req = pb.SearchRequest(query="黃仁勳", top_k=5, media_type="videos")
for name in ("Search", "SearchBm25", "SearchVector"):
    reply = getattr(stub, name)(req, metadata=AUTH, timeout=60)
    for r in reply.results:
        print(name, r.filename, round(r.score, 3), r.start_time, r.end_time, r.version_id[:8])
```

`media_type` is module 25's value (`videos`, `audios`, `images`, `documents`); empty means every type.
`top_k` 0 means 10, and more than 100 is `INVALID_ARGUMENT`, as is an empty `query`. Scores are not
comparable between the three: BM25 scores are unbounded (a few units for a good match), the other two are
0 to 1. `SearchVector` always returns the nearest `top_k` even when nothing is relevant, so judge by
`score`; `SearchBm25` returns nothing when no word matches. Each result carries `asset_path` and
`version_id`, which `DownloadAsset`, `StreamVideoSegment` and `SubmitJob` take. A list mixes two kinds
of result: a chunk of a file (`chunk_id` set, `start_time` and `end_time` give the range, 10 seconds for
videos today) and the file's whole-file summary (`chunk_id` empty, no times), so check
`HasField("start_time")` before using the range. Results are per user: a call made with an `x-api-key` runs as the shared dev user and searches that user's (empty) database.

`SearchRequest` also takes `asset_types` (several types at once; combined with `media_type` if both are
set) and `speaker_filter` (only chunks spoken by one of these speakers). `StreamSearch`, `StreamSearchBm25`
and `StreamSearchVector` are the same three searches delivered as a stream, one `SearchResult` per item in
rank order (buffered, see 5).

**Search the knowledge graph** with `GraphRagSearch` (the moments that match, plus the entities and graph
around them) and `TKGSearch` (dated facts about the entities a question names, optionally inside a time
window with `time_start` / `time_end`). Both return `results` plus `subgraph_json`, one JSON object with
`matched_entities`, `nodes` and `edges`, and `citations` (GraphRAG) or `moments` (TKG); the records inside
are module 25's own. For `GraphRagSearch` each result is a matching moment, like a `Search` chunk. For
`TKGSearch` each result is a *fact*, not a piece of a file: `snippet` is the fact in words, `score` is
module 25's confidence, `version_id` is the source it came from, and `asset_path` and the times are empty.
`top_k` 0 means module 25's own default; `max_depth` is 0 (default 2) to 4; `GraphRagSearch.strategy` is
`literal`, `semantic` or `hybrid`. A user with nothing indexed gets an empty list (an empty `{}` subgraph),
not an error.

**Upload an asset** (client-streaming: first message is the metadata, then the bytes):

```python
# example 3: upload a small video with an idempotency key, then repeat the same upload
import time
data = open("clip.mp4", "rb").read()          # any small .mp4 you have
name = f"guide-example-{int(time.time())}.mp4"
key = f"guide-{name}"

def upload_messages():
    yield pb.UploadAssetRequest(metadata=pb.UploadAssetMetadata(
        filename=name, content_type="video/mp4", total_size=len(data), idempotency_key=key))
    for i in range(0, len(data), 64 * 1024):
        yield pb.UploadAssetRequest(chunk_data=data[i:i + 64 * 1024])

done = stub.UploadAsset(upload_messages(), metadata=AUTH, timeout=120)
print(done.asset_id, done.version_id, done.bytes_received)
try:
    stub.UploadAsset(upload_messages(), metadata=AUTH, timeout=120)
except grpc.RpcError as err:
    print(err.code().name, dict(err.trailing_metadata()).get("asset-id") == done.asset_id)
```

**Download it back**, whole or from an offset (resume), checking each chunk's CRC32C:

```python
# example 4: download, then resume from the middle of the file
import google_crc32c   # pip install google-crc32c

def fetch(**kw):
    out = bytearray()
    for msg in stub.DownloadAsset(
            pb.DownloadAssetRequest(asset_path=done.asset_id, version_id=done.version_id, **kw),
            metadata=AUTH, timeout=120):
        if msg.HasField("chunk"):
            assert google_crc32c.value(msg.chunk.data) == msg.chunk.crc32c
            out += msg.chunk.data
    return bytes(out)

mid = len(data) // 2
print(fetch() == data, fetch(offset=mid) == data[mid:], fetch(offset=mid, length=5_000) == data[mid:mid + 5_000])
```

**Run a job and watch it.** `WatchJob` ends on its own when the job reaches a final state, and a
client that reconnects passes the last sequence number it saw to get the missed events replayed:

```python
# example 5: submit a processing job for the uploaded video, watch it, then replay its events
# (the job runs on the real workers, so this takes from a few seconds to about half a minute)
job = stub.SubmitJob(pb.SubmitJobRequest(
    job_type="guide", request_id=f"guide-job-{name}",
    input_asset_ids=[pb.AssetRef(asset_path=done.asset_id, version_id=done.version_id)]), metadata=AUTH, timeout=30)
print(job.job_id, pb.JobState.Name(job.state), job.media_type)
seen = []
for event in stub.WatchJob(pb.WatchJobRequest(job_id=job.job_id), metadata=AUTH, timeout=300):
    seen.append(event.sequence)
    print(event.sequence, pb.JobState.Name(event.state))
print([e.sequence for e in stub.WatchJob(pb.WatchJobRequest(job_id=job.job_id, last_sequence=seen[0]), metadata=AUTH, timeout=30)])
print(pb.JobState.Name(stub.GetJob(pb.GetJobRequest(job_id=job.job_id), metadata=AUTH, timeout=10).state))
```

**Stream a time range of a video** as playable fragmented MP4 (or MPEG-TS):

```python
# example 6: the first 2 seconds of the uploaded video
segment = b"".join(m.data for m in stub.StreamVideoSegment(pb.StreamVideoSegmentRequest(
    asset=pb.AssetRef(asset_path=done.asset_id, version_id=done.version_id),
    start_time=0, end_time=2, format=pb.VIDEO_STREAM_FORMAT_MP4_H264), metadata=AUTH, timeout=60))
print(len(segment) > 0)
```

## 3. Service reference

| RPC | Shape | Status |
| --- | --- | --- |
| `Ping` | unary | real (a connectivity and auth check) |
| `SearchByPerson` | unary | real; returns `ClipHit`s, one per segment |
| `StreamSearchByPerson` | server stream | real; **buffered** (see 5) |
| `Search` / `SearchBm25` / `SearchVector` | unary | real; hybrid, keyword-only, semantic-only; one `SearchResult` per matching chunk or whole-file summary |
| `StreamSearch` / `StreamSearchBm25` / `StreamSearchVector` | server stream | real; the same searches, **buffered** (see 5) |
| `GraphRagSearch` / `TKGSearch` | unary | real; hits plus a `subgraph_json` |
| `DownloadAsset` | server stream | real; `MediaHeader` first, then `MediaChunk`s |
| `UploadAsset` | client stream | real; metadata first, then bytes |
| `SubmitJob` / `GetJob` / `WatchJob` | unary / unary / server stream | real |
| `StreamVideoSegment` | server stream | real for the `asset` source; `video_id` source is `UNIMPLEMENTED` |
| `JobSession` | bidirectional | subscribe direction is real; `cancel` and `control` only record intent |
| `AnalyzeStream` | bidirectional | **off by default**: every call ends `UNIMPLEMENTED` ("disabled on this gateway"). Nothing consumes what it would forward, so it never returns an `AnalysisEvent`. A client that keeps sending may see `INTERNAL` instead; treat both as "not available" |
| `GetAsset`, `Chat`, `GetJobStatus`, `StreamVideo` | | `UNIMPLEMENTED`, not built |
| `grpc.health.v1.Health` (`Check`, `Watch`), server reflection | | real, no credentials needed |

**Identifying an asset.** Use the `asset_path` the platform reports (REST, search results, and the
`asset_id` that `UploadAsset` returns): it has no file name, for example `video/mp4/clip`. The full object
key (`video/mp4/clip/clip.mp4`) is accepted too. Always send the `version_id` with it.

**Chunk size.** `preferred_chunk_size` is clamped to between 64 KiB and 4 MiB minus 4 KiB; 0 means the
default (1 MiB). A chunk carries `offset`, `sequence`, a per-chunk `crc32c`, and `end_of_stream` on the last one.

**`DownloadAsset` header.** `content_length` is the size (or -1 if unknown), `seekable` says whether a range
was honoured. `duration_sec` is the length in seconds of an audio or video file, read with ffprobe; 0 means unknown (not a
media file, probing switched off, or ffprobe failed or timed out). A slow ffprobe delays the header by at most
`GRPC_GATEWAY_PROBE_DURATION_TIMEOUT_S` (3 s); it never fails the download.

**Uploads.** `filename` is required; `total_size` (optional) is checked against the bytes received and a
mismatch is `INVALID_ARGUMENT`; an upload that is cancelled midway stores nothing (a cancel that arrives after the
last byte, while the gateway is finishing the upload, does not stop it: the finished asset is kept, and if you
sent an `idempotency_key` you get its `asset-id` by repeating the upload). A repeated
`idempotency_key` within 24 hours returns `ALREADY_EXISTS` and puts the original `asset-id` and
`version-id` in the trailing metadata.

*Checksum (optional).* Send the file's CRC32C (Castagnoli, the same one `DownloadAsset` puts on each chunk) either
in `metadata.crc32c` when you know it up front, or in a last `final_crc32c` message when you compute it while
streaming (nothing may follow it); if you send both, both are checked. The gateway computes the CRC32C of the bytes
it received and, on a mismatch, answers `DATA_LOSS` **before** handing the file to the asset service, so no asset
is created. Without a checksum, only `total_size` is checked, as before. `0` is a real value, not "absent".

**Jobs.** The media type (document, image, video, audio) is detected from the file name, the same as REST.
`SubmitJob` takes exactly one input asset and is idempotent on `request_id`. States:

| State | Meaning |
| --- | --- |
| `QUEUED` | accepted, not picked up yet |
| `RUNNING` | a worker is processing it (the worker's own step is in `JobEvent.detail`) |
| `INDEXING` | results are being indexed |
| `COMPLETE` | finished (final) |
| `FAILED` | failed; `detail` carries the error (final) |
| `CANCELLED` | reserved: no worker supports cancellation today, so it is never reached |
| `TRANSCRIBING`, `EXTRACTING` | reserved, not produced |

Events have increasing `sequence` numbers starting at 1 and are kept for 7 days, so a job can be replayed
with `last_sequence`. Once a job's final state has been seen (by `GetJob` or `WatchJob`, within 24 hours of the
worker finishing) `GetJob` keeps answering with it for the same 7 days. A job that was
submitted through REST is not visible to these RPCs.

**Errors.**

| Status | Typical cause |
| --- | --- |
| `UNAUTHENTICATED` | no credentials, an invalid or expired token, an unknown `x-api-key` |
| `PERMISSION_DENIED` | the caller may not use this RPC |
| `DATA_LOSS` | an upload's `crc32c` / `final_crc32c` did not match the bytes received; nothing was stored |
| `INVALID_ARGUMENT` | malformed request (bad time range, unsupported format, negative offset, `total_size` mismatch, first upload message not metadata) |
| `NOT_FOUND` | no such asset or job |
| `ALREADY_EXISTS` | repeated `idempotency_key` |
| `RESOURCE_EXHAUSTED` | the caller already has the maximum number of concurrent streams (default 4, shared by all streaming RPCs); a stream that was just cancelled can take a few milliseconds to free its slot |
| `UNAVAILABLE` | a service behind the gateway (module 25, module 04, Keycloak) cannot be reached; Kafka is not configured for `SubmitJob` |
| `DEADLINE_EXCEEDED` | a downstream timeout, or an upload with no data for 60 s |
| `UNIMPLEMENTED` | the RPC or branch is not built (table above) |

Cancelling a call from the client stops the work on the server (file reads, the FFmpeg process) and frees the
stream slot.

## 4. Choosing REST or gRPC

Use **gRPC** when you move large files and want resume, a checksum per chunk and bounded memory on both
sides; when you want to follow a job's progress as a stream with replay; or when you want generated,
typed clients. Use **REST** for anything a browser calls directly (browsers cannot speak native gRPC), for
one-off scripts, and for the operations gRPC does not offer yet. The two interfaces share the same
back-end services but are separate code paths, so results can differ in small ways (one known case: REST
fixes the GraphRAG/TKG `score_threshold` at 0.3 while gRPC forwards the caller's value, default 0.52).

Content AI's `RaptorClient` (outside this repository) chooses between them with its
`raptor_transport` setting, `"rest"` or `"grpc"`.

## 5. Streaming search is buffered

`StreamSearchByPerson`, `StreamSearch`, `StreamSearchBm25` and `StreamSearchVector` wait for the complete result from module 25 and then sends the hits one by one.
It is chunked delivery of a finished result, not incremental search: it does not make a large result
arrive sooner. (An internal design note has the full reasoning behind this choice.)

## 6. Progress for browsers

A browser cannot call `WatchJob` directly. The WebSocket progress endpoint that was planned to relay these
events (`/api/v1/jobs/{job_id}/progress`) is **not built**. Until it is, put a small service between the
browser and the gateway: it calls `WatchJob` (passing the last `sequence` it relayed when it reconnects) and
forwards each event to the browser over a WebSocket or server-sent events. The simplest alternative that
needs no relay is for the browser's backend to poll `GetJob`.

## 7. Operating it

- **Health:** `grpc_health_probe -addr=host:8825` (also `-service=raptor.v1.RaptorService`) returns `SERVING`.
- **TLS:** off by default. Set `GRPC_GATEWAY_GRPC_SECURE=true` with `GRPC_GATEWAY_GRPC_TLS_CERT_PATH` and
  `GRPC_GATEWAY_GRPC_TLS_KEY_PATH`; add `GRPC_GATEWAY_GRPC_MTLS_CA_CERT_PATH` to require client certificates.
- **Limits:** `GRPC_GATEWAY_MAX_STREAMS_PER_USER` (4), `GRPC_GATEWAY_UPLOAD_IDLE_TIMEOUT_S` (60),
  `GRPC_GATEWAY_UPLOAD_TIMEOUT` (300, the call to the asset service).
- **Behind a proxy:** both nginx and Envoy need HTTP/2 to the gateway; see `proxy-test/` (the nginx example
  must disable `client_max_body_size`, or uploads over 1 MiB fail with HTTP 413).
- **Metrics and dashboard:** `grpc_server_*` on :9825; a Grafana dashboard for them is `grafana/raptor-grpc-gateway.json`
  (added separately).
- **Known issues:** `KNOWN_ISSUES.md`.

## 8. Not covered here

The client wrapper in Content AI (`RaptorClient`), the WebSocket relay, and the AudioService on :8044 (a
separate server and proto package, `raptor.audio.v1`, still to be merged from module 34).
