# Raptor — Setup, Build & Configuration Guide

The complete guide for deploying Raptor: prerequisites (hardware,
network, external servers), dependency map, `.env` configuration, build/deploy
commands, single-host and multi-host setups, verification, and source-maintenance
rules.

The single build entry point is `deployment/modules/build.py`; `deploy.sh` at the
repo root is a thin wrapper — every flag is identical.

---

## 1. Prerequisites

Read this section **before** starting any setup.

### 1.1 Hardware / Infrastructure Requirements

| Host role                                                                       | Minimum requirement                                                                                    | Notes                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CPU core host** (infrastructure + APIs: modules 01–06, 13–15, 21–22, 24, 27, 29, 33, 36) | 16 CPU cores, 64 GB RAM, 500 GB+ SSD                                                                   | Runs Kafka (6 containers), Redis standalone (cluster mode is opt-in, see below), PostgreSQL, SeaweedFS+LakeFS, Keycloak, API/reasoning services, ArcadeDB, MCP, doc-processing-agent, privacy-compliance-service, grpc-gateway. **Guardrail (23), Personal DB (25) and Memory (26) are GPU-reserving — see the GPU host row, not this one** (a prior version of this table miscategorized them here). Module 22 is still a placeholder — don't start it. Modules 19/20 are deprecated (§Deprecated Modules in the root README) — don't start them either. Module 18 needs a GPU — see the row below, not this one. |
| **GPU host(s)** (modules 04¹, 07–12, 16, 18, 23, 25, 26, 34)                                  | NVIDIA GPU with**CUDA 12.8+ driver**, 24 GB+ VRAM; 64 GB RAM                                     | GPU stack targets sm_120 (Blackwell / RTX 50-series): torch 2.7.1 cu128, PaddlePaddle 3.3.0 cu129. Older GPUs (sm_8x/sm_90) work with the same wheels. (Module 17 also reserves a GPU but is deprecated — don't start it.) Module 35 is a special case: its default `--profile text` has no GPU reservation at all, only `--profile voice` does — see §2's module-35 audit. |
| — module 11 (video)                                                            | **36 GB+ VRAM recommended**                                                                      | InternVL frame description;`MAX_MEMORY_PER_GPU=36GiB` cap (shared with module 10)                                                                                                                                            |
| **NFS server**                                                            | Any host/appliance exporting NFSv4 (port 2049), 1 TB+ recommended                                      | Module 01 provides a containerized NFS server, or use a native/appliance NFS                                                                                                                                                   |
| **Ollama inference server** (required by default)                         | GPU host sized for your configured models (`INFERENCE_MODEL`, `LLM_MODEL`, `QO_INFERENCE_MODEL`) | **May be an existing external server outside the RAPTOR cluster** — RAPTOR only needs HTTP reachability to port 11434. Direct consumers: 07, 13, 15, 21, 23 (guardrail's own `OLLAMA_URL`, disabled by default), 35 (`MODULE07_OLLAMA_HOST`, kept so model discovery still works when 07 is down). Indirect (proxied through module 07's AI Lifecycle API via `INFERENCE_URL`, not a direct Ollama connection): 09, 11, 12, 22, 26                  |
| **vLLM inference server** (optional)                                      | GPU host for the served model                                                                          | vLLM is**not bundled** — it is an optional external OpenAI-compatible backend. Any `LLM_BASE_URL` consumer (15 chat) can point at it instead of Ollama; module 07 can register models with `engine: vllm` |

¹ Module 04's `thumbnail-worker` sidecar carries an unconditional GPU reservation
(no `profiles:` gate exists yet) — a known, open gap, not a deliberate design
choice (the rest of module 04 — SeaweedFS/LakeFS/Asset Management — is
genuinely CPU-only). Until that gate is added, module 04 as a whole needs a
GPU-capable host.

All host classes can be combined onto one machine for evaluation (single-host
deployment, §5) or split across machines (multi-host deployment, §6).

### 1.2 Software Requirements (every module host)

- Docker Engine 24.0+ and Docker Compose v2.20+
- NVIDIA Container Toolkit (GPU hosts); verify with
  `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
- Python 3.10+ on the host (for `build.py`)
- `nfs-common` (Debian/Ubuntu) or `nfs-utils` (RHEL) on NFS client hosts

### 1.3 Network Requirements

- All hosts on one routable network; static IPs (or DHCP reservations) —
  `HOST_IP` is baked into Kafka advertised listeners and Redis cluster announce
  addresses at first start.
- Clocks synchronized (chrony/ntp) — Kafka and Keycloak are time-sensitive.
- Firewall openings (minimum):

| From → To                                | Ports                                                                                                                                                                       |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| all module hosts → NFS server            | 2049/tcp,udp (NFSv4)                                                                                                                                                        |
| core + GPU hosts → Ollama server         | 11434/tcp                                                                                                                                                                   |
| core + GPU hosts → vLLM server (if used) | 8000/tcp                                                                                                                                                                    |
| GPU hosts → core host                    | 19092–19094 (Kafka EXTERNAL), 6379 (Redis standalone; + 7000–7005 only if cluster mode is enabled via `COMPOSE_PROFILES=cluster`, off by default), 5433 (PostgreSQL — published host port, container listens on 5432), 8333 (SeaweedFS S3), 8001 (LakeFS — published host port, container listens on 8000), 8080 (Keycloak), 8000 (Asset Management API, module 04) |
| any host → whichever host runs Guardrail (23) / Personal DB (25) / Memory (26) / Privacy Compliance (33) | 8023, 8025, 8026, 8033 — these modules are **GPU-reserving (23/25/26) or CPU (33)**, and can end up on either the core host or a GPU host depending on topology; only reachable cross-host after setting `INTERNAL_BIND_IP` to that host's private LAN IP (see §6.1 rule 4) |
| core host → AI-lifecycle GPU host        | 8010 (AI Lifecycle API)                                                                                                                                                     |
| core host → search GPU host              | 8843 (Query Orchestrator — published host port, container listens on 8000)                                                                                                             |
| clients → core host                      | 8012 (API Gateway), 8021 (chat), 8843 (query orchestrator), 8030 (A2A), 8027 (MCP Server), 8290 (doc-processing-agent), 3000 (demo frontend — **note**: Grafana on module 14 also defaults to 3000; running both on the same host needs one of `PORT_GRAFANA` / `DEMO_PORT` changed) |
| clients → whichever host runs module 34/35/36 | 8035 (realtime-audio WebSocket) + 8044 (realtime-audio gRPC `AudioChat`), 8036 (inference-service), 8825 (grpc-gateway) + 9825 (grpc-gateway metrics) — all publish on all interfaces by default (§3.1), same as the core-host gateway modules above |

Deprecated modules' ports (6333/6334 Qdrant, OpenSearch/hybrid-search-API, 8844 graph-service) are omitted above — see the root README's Deprecated Modules section. They're still technically reachable if you start those containers, but nothing in the live pipeline calls them.

### 1.4 External Inference Servers — summary

| Server                               | Status                                                                     | RAPTOR consumers                                                                                                                                                                                                                                                | Key env vars                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Ollama** (`:11434`)        | Default LLM backend; may be a pre-existing external server                 | Direct: 07 (`OLLAMA_API_BASE` — MLflow registration + inference routing), 13 (`GATEWAY_SMOLAGENTS_MODEL=ollama/<model>`), 15 (`LLM_BASE_URL`), 21 (`OLLAMA_HOST`), 23 (`OLLAMA_URL`, guardrail's own guard-model calls, disabled by default). Indirect, proxied through module 07's AI Lifecycle API via `INFERENCE_URL` (module 12's own `OLLAMA_URL` var is dead/unused code — don't set it expecting an effect): 09, 11, 12, 22, 26 | `OLLAMA_HOST` → derives `LLM_BASE_URL=${OLLAMA_HOST}/v1`, `OLLAMA_API_BASE=${OLLAMA_HOST}`; indirect consumers use `INFERENCE_URL` instead |
| **vLLM** (`:8000`, optional) | Not bundled (no in-tree runtime); external OpenAI-compatible`/v1` server | Any`LLM_BASE_URL` consumer (15); module 07 model registration accepts `engine: vllm`                                                                                                                                                                    | `LLM_BASE_URL=http://<vllm_host>:8000/v1` per consuming module                                  |

Model names are configured via `LLM_MODEL`, `INFERENCE_MODEL` (+ per-modality
`AUDIO/VIDEO/DOCUMENT_INFERENCE_MODEL`), `QO_INFERENCE_MODEL`, `CHAT_LLM_MODEL`;
pull/serve those models on the inference server before first use. All default
to `gemma4:12b`.

## 2. Dependency Map (NFS · GPU)

### NFS

| Role                 | Module                                | Detail                                                                                                      |
| -------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Server               | 01 nfs-server                          | Containerized NFS (port`2049`/`PORT_NFS`), exports `/nfs-share` backed by host path `NFS_BASE_PATH` |
| Docker-volume client | 04, 07, 09–13, 16, 22, 26 | Every volume on these modules mounts straight over the network via the Docker NFS volume driver (`type: "nfs"`, `addr=${NFS_SERVER}`) — no host-level `mount` step; e.g. 07's `tmp`/`data` volumes use `${NFS_SERVER}:${NFS_AIML_TMP_PATH}`, 04's SeaweedFS volumes use `${NFS_SERVER}:${SEAWEEDFS_BASE_DIR}/...`, 22 shares 07/16's `NFS_AIML_TMP_PATH` (read-only), 26 uses its own `NFS_MEMVID_PATH` |
| Shared media scratch | 09–12 workers + 13 gateway             | `NFS_MEDIA_TMP_PATH=/media/processing` on the shared export, used to exchange temporary media             |

### GPU modules

| Module                 | GPU use                                                           | Device-selection env                                                          |
| ---------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 04 object-storage      | `thumbnail-worker` sidecar only (ffmpeg-based); unconditional, no `profiles:` gate yet — see §1.1's footnote | `THUMBNAIL_GPU`                                            |
| 07 ai-ml-services      | AI Lifecycle inference                                            | `AI_LIFECYCLE_GPU`                                                          |
| 08 media-worker        | **Build-time base image** for 09–12 (no runtime container) | —                                                                            |
| 09 audio-processing    | WhisperX STT / diarization / classifier                           | `AUDIO_RECOGNIZER_GPU`, `AUDIO_DIARIZATION_GPU`, `AUDIO_CLASSIFIER_GPU` |
| 10 image-processing    | InternVL image analysis                                           | `IMAGE_PROCESSING_GPU`                                                      |
| 11 video-processing    | Chunking / frame description / OCR (highest VRAM)                 | `VIDEO_CHUNKING_GPU`, `VIDEO_OCR_GPU`, `VIDEO_FRAME_DESC_GPU`           |
| 12 document-processing | PDF/Office OCR + analysis                                         | `DOCUMENT_ANALYSIS_GPU`                                                     |
| 16 training-service    | Training orchestration                                            | (compose reservation)                                                         |
| 18 query-orchestrator  | Intent classification / rerank                                    | `QUERY_ORCHESTRATOR_GPU`                                                    |
| 23 guardrail-service   | Image guard: Llama Guard 3 11B Vision (NF4 4-bit) + ShieldGemma 2 4B (8-bit) | `GUARDRAIL_GPU_ID`                                         |
| 25 personal-db-service | bge-m3 embedder + bge-reranker-v2-m3                               | `PERSONAL_DB_GPU`                                                            |
| 26 memory-service      | MemVID embedding/retrieval                                        | `MEMORY_SERVICE_GPU`                                                         |
| 34 realtime-audio      | Two separate GPU-reserving services: `realtime_audio` (faster-whisper STT) and `breeze_tts` | `REALTIME_AUDIO_GPU`, `REALTIME_AUDIO_BREEZE_GPU`           |
| 35 inference-service   | Voice only (`--profile voice`): XTTS clone + pyannote/WhisperX diarization. The default `--profile text` has no GPU reservation at all | `INFERENCE_VOICE_GPU`                        |

Module 17 (deprecated) also reserves a GPU (`HYBRID_SEARCH_GPU`) for embedding + cross-encoder rerank — omitted from the table above since it shouldn't be started; listed here only so its GPU reservation isn't mistaken for unused capacity if you do see it in a compose file.

Device ids are per-service: multiple services can pin different GPUs on one host,
or modules can be split across GPU hosts (§6).

### Module 35: text vs. voice profile

Module 35 ships one `docker-compose.yml` with two profiles: the default
`--profile text` has no GPU reservation at all; `--profile voice` reserves
one (`INFERENCE_VOICE_GPU`) for XTTS clone + pyannote/WhisperX diarization.
`VOICE_ENABLED` in `.env` picks which profile `build.py` starts. Module 35's
`build.py` entry depends on module 08's shared base image
(`deps=["02","07","08"]`, `extends_shared_image=True`).

`docker compose down` needs `--profile "*"` to actually stop every
profile-gated container — this also applies to module 02's `cluster` profile
and module 14's `gpu` profile, not just module 35's `text`/`voice`. `build.py
--delete` passes this; a manual `docker compose down` without it can
silently leave a profile-gated container running.

## 3. Configuration (`.env`)

Every module directory under `deployment/modules/` — plus the `modules/` root —
carries a committed **`.env.example`** template and expects a local, uncommitted
**`.env`**.

### 3.1 Create the `.env` files

Pick **one** of the two options below — both end up at the same place (every `.env` created,
root's placeholders filled in). Either way, create every module's `.env`, even ones you don't
plan to touch — several modules' `docker-compose.yml` require their own local `.env` to exist on
disk (`env_file: - .env` with no `required: false`); `docker compose` fails immediately if that
file is missing, regardless of whether the root `.env` already has everything the module needs.

**Option A — recommended: `./deploy.sh --init-env`** (from the repo root; `deploy.sh` is the thin
wrapper around `deployment/modules/build.py`, see the top of this guide). It creates every
missing `.env` from its `.env.example` (root + each module; an existing `.env` is left untouched),
then walks the root `.env` interactively over just its unfilled `<placeholder>` values -- a
password/secret/token/key-shaped one gets an auto-generated random value, an IP-shaped one gets
this host's detected IP, a path-shaped one gets a suggested path, each offered as `[Enter=accept]`
so you only type something when you actually need a different value. A handful of
genuinely-your-own-info fields (`HF_TOKEN`, `DOCKER_NETWORK`, `MLFLOW_AUTH_ADMIN_USERNAME`, …)
have no sensible default and just prompt for free text. Ctrl-C/EOF stops cleanly and keeps
whatever was already answered. Module-local `.env` files are copied as-is, not interactively
filled -- see why below.

**Option B — by hand** (same end result as Option A, spelled out — use this if you want to see
exactly what that does, or need to do this on a host without this repo's Python available):

```bash
cd deployment/modules                     # from the repo root
cp .env.example .env                      # global settings
for m in */; do
  [ -f "$m/.env.example" ] && cp "$m/.env.example" "$m/.env"
done
```

This only copies files -- it doesn't fill anything in, so you still need to replace every
`<placeholder>` value yourself afterward (the grep command further down finds them all).

**Where to actually edit values — root first, module-local only for what's
genuinely module-specific:**

- For modules whose `docker-compose.yml` lists `env_file: [.env, ../.env]`
  (local first, root second — check the module's own compose file), Compose
  merges both and **the later file wins on duplicate keys** — so root's
  value overrides the module-local one. In practice this means most shared
  credentials (`REDIS_PASSWORD`, `NFS_SERVER`, `HF_TOKEN`, `NEO4J_PASSWORD`,
  `POSTGRES_PASSWORD`, `OPENSEARCH_PASSWORD`, the Keycloak admin vars, …)
  only need editing once, in the **root** `.env`. Leaving the module-local
  copy's placeholder value untouched is fine for these.
- A few modules (e.g. `06-authentication`) don't use `env_file:` at all —
  their `docker-compose.yml` reads `${VAR}` directly from whatever
  `--env-file` `build.py` resolves (root, if it exists — see
  `_env_file_args()`). For these, the module-local `.env` file's *content*
  is never read at all; only root matters.
- What's left in each module's local `.env` after that is genuinely
  module-specific and has no single correct value to share (per-host GPU
  device selection, a module's own S3 bucket name, Kafka cluster ID, its
  own alerting email, internal tuning knobs like log level or worker
  count). Most of these already ship with safe, usable defaults in
  `.env.example` — you only need to touch the ones still showing a
  `<placeholder>`-style value:
  ```bash
  # from deployment/modules/ — lists every var (root + all module-local
  # files) still left as a placeholder: wholly or partly wrapped in <...>
  # (e.g. OLLAMA_HOST=http://<host_ip>:11434 counts too), or plain empty
  grep -rEn '^[A-Z_][A-Z0-9_]*=$|^[A-Z_][A-Z0-9_]*=""$|^[A-Z_][A-Z0-9_]*=.*<.*>.*$' \
    .env.example */.env.example
  ```

**Port variable naming convention.** `PORT_*` (prefix) names a host-published
port — it's the one you're meant to change if you need to avoid a conflict
on the deployment machine; every module's `docker-compose.yml` maps it as
`${PORT_X:-default}:<fixed-container-port>`, so changing it only affects
what's reachable from outside that container, never what the service
listens on internally. `*_PORT` (suffix) names a fixed, container-internal
port used to build connection strings between services on the same Docker
network (e.g. `POSTGRES_PORT`, `REDIS_PORT`) — these are deliberately not
tied to any host-published value and shouldn't be treated as something a
deployer needs to tune.

**Internal services bind to `INTERNAL_BIND_IP`.** Modules 23
(Guardrail), 25 (Personal DB), 26 (Memory) and 33 (Privacy Compliance) are
internal-only: their compose files map
`${INTERNAL_BIND_IP:-127.0.0.1}:${PORT_X}:<container-port>`, so by default
their `PORT_*` is reachable from the docker host itself (curl, debugging) but
not from other machines. Other containers never use these host ports — they
call the service over the `raptor` network (e.g.
`http://raptor-guardrail-service:8026`). Set `INTERNAL_BIND_IP` only for a
multi-host deployment (§6.1). Gateway-facing modules (13, and 22/27/29/34/35/36
for now) still publish on all interfaces. For development, `EXPOSE_INTERNAL_PORTS=true`
in `.env` switches 23/25/26/33 back to the original (pre-internal-binding) mode — published
on all interfaces, so `http://<host-ip>:<PORT_X>` works from other machines.
`build.py` applies it by exporting `INTERNAL_BIND_IP=0.0.0.0` to its compose
calls; for a manual `docker compose` run, set `INTERNAL_BIND_IP=0.0.0.0`
yourself. Keep it `false` (the default) on anything reachable from an
untrusted network, and re-run `compose up` after changing it.

**One exception**: Module 02's Redis Cluster ports (`PORT_REDIS_CLUSTER_*`,
`PORT_REDIS_CLUSTER_BUS_*`) use the `PORT_*` prefix but, unlike everywhere
else, the same value drives both the host-published and the container-
internal port together. This is intentional, not an inconsistency to "fix" —
Redis Cluster's own protocol requires each node's `cluster-announce-port` to
equal the port other nodes/clients can actually reach it on. Decoupling the
two here would make a node announce a port nothing is listening on and break
cluster gossip/routing.

### 3.2 Edit the root `modules/.env` — required values

This table groups values by purpose for orientation — it is not guaranteed to
name every single placeholder (it drifts as modules are added). The `grep`
command in §3.1 against the root `.env.example` is the authoritative, always-
current list of what still needs a real value.

| Group             | Variables                                                                                                                                                                                                                                                                                                   | What to set                                                                                                                                                                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Host & network    | `HOST_IP`                                                                                                                                                                                                                                                                                                 | **This host's own IP** — drives Kafka `EXTERNAL` advertised listeners (`:19092-19094`), Redis `--cluster-announce-ip`, Keycloak `KC_HOSTNAME`. Set **before first start**                                                                                                     |
| NFS               | `NFS_SERVER`, `NFS_BASE_PATH`, `NFS_EXPORT`, `NFS_AIML_TMP_PATH`, `NFS_AIML_DATA_PATH`, `NFS_MEDIA_TMP_PATH`                                                                                                                                                                                    | NFS server IP and export paths (§2)                                                                                                                                                                                                                                                               |
| Inference         | `OLLAMA_HOST=http://<ollama_ip>:11434`                                                                                                                                                                                                                                                                    | Ollama server URL;`LLM_BASE_URL` / `OLLAMA_API_BASE` derive from it. Point `LLM_BASE_URL` at vLLM `/v1` instead where desired                                                                                                                                                              |
| Models            | `LLM_MODEL`, `INFERENCE_MODEL`, `VIDEO/AUDIO/DOCUMENT_INFERENCE_MODEL`, `IMAGE/VIDEO_VLM_MODEL_PATH`, `CHAT_LLM_MODEL`, `QO_INFERENCE_MODEL`, `CONTEXTUALIZE_MODEL_NAME`, `RERANKER_TEMPERATURE`                                                                                            | Model selection (must exist on the inference server)                                                                                                                                                                                                                                               |
| Credentials       | `REDIS_PASSWORD`, `POSTGRES_PASSWORD`, `OPENSEARCH_PASSWORD`, `NEO4J_PASSWORD`, `LAKEFS_*_KEY`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, Keycloak admin vars, `MCP_KEYCLOAK_USERNAME`/`MCP_KEYCLOAK_PASSWORD` (module 27's service account), `GRAFANA_ADMIN_PASSWORD`, `HF_TOKEN`, `GD_INTERNAL_TOKEN` (shared GDPR-erasure token across 04/13/25/26/33/35), `GD_CONTROLLER_NAME` (legal org name, module 33's GDPR records) | Replace every`<your_...>` placeholder                                                                                                                                                                                                                                                            |
| Cross-module URLs | `KAFKA_BOOTSTRAP_SERVERS`, `REDIS_HOST`, `POSTGRES_HOST`, `S3_ENDPOINT`/`S3_PUBLIC_URL`, `LAKEFS_ENDPOINT`, `ASSET_MANAGEMENT_URL`, `INFERENCE_URL` (module 07's AI Lifecycle API — consumed indirectly by 09/11/12/22/26; **not** `AI_MODEL_LIFECYCLE_URL`, which is a dead/unused leftover key), `QUERY_ORCHESTRATOR_URL`, `GUARDRAIL_URL` (module 07/13 → module 23, disabled by default), `KEYCLOAK_URL`, `GATEWAY_TRAINING_SERVICE_URL`        | Single host: keep docker-alias defaults**except** `S3_PUBLIC_URL` and `GATEWAY_TRAINING_SERVICE_URL` — both ship with a literal `<host_ip>` baked into the URL and need editing even for a single-host deploy. Multi-host: replace the rest with host IPs + published ports too (§6) — remember `LAKEFS_ENDPOINT`'s published port is `8001` and `QUERY_ORCHESTRATOR_URL`'s is `8843`, not the containers' internal `8000`. (`QDRANT_HOST`/`GRAPH_SERVICE_URL` still exist as vars but are unused — modules 17/19/20 are deprecated, see the root README.) |
| Mail (module 06)  | `SMTP_HOST/PORT/USER/PASSWORD/FROM/...`                                                                                                                                                                                                                                                                   | Optional e-mail notifications                                                                                                                                                                                                                                                                      |
| Tuning            | `MEMORY_TTL`, `MEMORY_CONTEXT_WINDOW`, `REQUEST_TIMEOUT`, `CROSS_SERVICE_TIMEOUT`, `DOCUMENT_ANALYSIS_CONTEXTUAL_BATCH_SIZE`, `SUMMARY_MAX_TOKENS`, `DOCUMENT_ANALYSIS_MAX_CHUNK_TOKENS`, `VIDEO_GRAPH_CPU_LIMIT/MEMORY_LIMIT`, `MAX_MEMORY_PER_GPU`                                      | Defaults are sane; adjust per hardware                                                                                                                                                                                                                                                             |

**Redis defaults to standalone, not cluster.** Every module connects to
`raptor-redis-standalone` (`REDIS_HOST`) by default. Module 02's 6-node
cluster (services `redis1`–`redis6`, ports 7000–7005) only starts if you set
`COMPOSE_PROFILES=cluster` before bringing module 02 up — nothing in this
repo sets that automatically. Don't provision hardware/firewall rules for the
cluster unless you're deliberately enabling it.

### 3.3 Per-module `.env` files

**Ten modules actually read their own local `.env` when the root `.env` is
also present** — `04-object-storage`, `07-ai-ml-services`, `13-api-services`,
`16-training-service`, `17-hybrid-search` (deprecated, but the directive is
still real), `18-query-orchestrator`, `22-benchmark-service`,
`26-memory-service`, `27-mcp-server` and `36-grpc-gateway` all have an
`env_file:` directive in their `docker-compose.yml` — confirmed by grep
against every module's compose file, not just the two this section used to
name. For all ten, `env_file:` lists the module's own `.env` first and root's
`../.env` second (`required: false`) — since Compose lets a later file
override a duplicate key, **root wins whenever the same key exists in
both**; the module-local copy only matters for module-specific keys root
doesn't define (07: GPU device IDs, `NVIDIA_VISIBLE_DEVICES`; 04: its own
`S3_BUCKET`/Kafka-unrelated settings; the others similarly — see the
placeholder-scan command above for the definitive per-module list).

`09`–`12`, `15-chat-service`, `20-graph-service` and `21-agent-protocol` have
**no** `env_file:` directive at all (confirmed by grep against every
`docker-compose.yml`), and no working `load_dotenv()` path either (21's own
`load_dotenv()` call is a no-op: its Dockerfile only `COPY app/ .`, so no
`.env` file ever exists inside that container to find). For these six
modules, the `environment:` block's `${VAR}` placeholders are resolved
entirely from whichever single `--env-file` `build.py` passes to `docker
compose` — and `_env_file_args()` picks that file with a strict priority
order: the root `deployment/modules/.env` wins outright if it exists at all
(its branch `return`s immediately, without even checking whether the module
has its own `.env`); only when root's `.env` is absent does it fall back to
the module's own local `.env`.

In practice, since §3.1 has you create the root `.env` before anything else,
**root's `.env` is what actually gets read for these modules — editing their
own local `.env` has no effect once root exists.** The module-local `.env`
only becomes live again if you deploy that module completely standalone,
with no root `.env` present at all.

Keep every `.env.example` in sync when adding/removing keys (see §9).

## 4. Build & Deploy Commands

```bash
cd RAPTOR/Aigle/0.5                # release root, wherever you cloned the repo

# 1. Build the shared GPU base image FIRST (used by modules 09–12)
bash deploy.sh -m 08 --build

# 2. Start everything in dependency order
bash deploy.sh                       # all modules
bash deploy.sh --cpu-only            # skip GPU modules
bash deploy.sh -m 02,03,04           # specific modules
bash deploy.sh -e 11                 # all except one

# Lifecycle
bash deploy.sh -m 05 --stop          # stop (keep volumes)
bash deploy.sh -m 05 --delete        # stop + remove volumes
bash deploy.sh -m 05 --restart
bash deploy.sh -m 18 --build         # rebuild one image then start
bash deploy.sh -m 18 --build-only    # rebuild the image only, don't start the container

# Validation
bash deploy.sh --check-ports         # detect host-port conflicts + verify container-port comments
bash deploy.sh --check-env           # find unfilled <placeholder> values in the env file

# Info
bash deploy.sh --list
bash deploy.sh --status
bash deploy.sh -m 18 --logs
bash deploy.sh -l build.log          # tee output to a log file
```

Notes:

- Module 07 builds a custom MLflow image (`Dockerfile.mlflow` → `raptor/mlflow:${RAPTOR_VERSION}`).
- Module 08 produces `raptor/media-worker:${RAPTOR_VERSION}`; modules 09–12 `FROM` it. It is
  not pushed to a registry — **build it locally on every host that runs 09–12**.
- All stacks join the external `raptor` Docker bridge network; `build.py` creates
  it if missing.

## 5. Single-Host Deployment (evaluation)

```bash
cd RAPTOR/Aigle/0.5                # release root, wherever you cloned the repo
# .env: HOST_IP=<this host IP>, OLLAMA_HOST=http://<ollama_ip>:11434, NFS vars
bash deploy.sh -m 08 --build
bash deploy.sh                       # or --cpu-only without GPUs
```

No manual `mount -t nfs4` step needed here — modules 04/07/09-13/16/22/26 mount
NFS directly as a network client via the Docker volume driver (`NFS_SERVER` +
`PORT_NFS` in `.env` is enough); nothing needs a host-level mount first.

<a id="first-login"></a>

### First login — the seeded users have no password yet

Module 06's `dhtsolution` realm import (`06-authentication/realm-import/dhtsolution-realm.json`)
creates 6 users — `test_basicuser`, `test_standarduser`, `dht_developer`,
`test_training`, `test_nontraining`, `service-account-raptor` — but every one
of them is imported with **no credentials set**. `POST /api/{version}/sso/login`
fails for all of them until an admin sets a password; nothing else about the
deployment is broken.

**Set a password for one of the seeded users** (Keycloak Admin Console —
there's no REST endpoint for this in module 06's own API, only for creating a
brand-new user, see below):

1. Open `http://<host>:${PORT_KEYCLOAK:-8080}` and log in with
   `PERMANENT_MASTER_ADMIN_USER` / `PERMANENT_ADMIN_PASSWORD` (from module 06's
   `.env` — this account is auto-created by `create-permanent-admin.sh` on
   first start, in the `master` realm).
2. Switch realm from `master` to `dhtsolution` (top-left realm switcher).
3. **Users** → click the username → **Credentials** tab → **Set password** →
   enter a password, toggle **Temporary** off (leaving it on forces a
   password-change flow the API-only login doesn't handle) → **Save**.

**Create a new user instead** — module 06 exposes its own admin API for this
(`app/routers/creat_user.py`), on its own port (`PORT_KEYCLOAK_API`, default
`8800`), separate from Keycloak's own `8080`:

```bash
# 1. Get an admin token (this calls module 06's own /SSO/login, not Keycloak's
#    token endpoint directly — realm=master, client_id=admin-cli). The
#    response is a JSON string literal (quoted) -- strip the quotes or the
#    Bearer header below is malformed.
ADMIN_TOKEN=$(curl -s -X POST http://<host>:8800/SSO/login \
  -d "username=<PERMANENT_MASTER_ADMIN_USER>&password=<PERMANENT_ADMIN_PASSWORD>&realm_name=master&client_id=admin-cli" \
  | tr -d '"')

# 2. Create the user (returns a temporaryPassword to hand to them). The
#    credential itself is never a Keycloak "temporary" credential (always
#    `"temporary": false`, regardless of SMTP) -- but whether the new user can
#    actually log in with it right away depends on whether this deployment's
#    module 06 .env has SMTP_* configured:
#      - SMTP_HOST unset (most dev/test deployments): emailVerified=true, no
#        requiredActions set. The returned password logs in immediately via
#        POST /sso/login, exactly like any other account.
#      - SMTP_HOST set: emailVerified=false, and the user is created with
#        requiredActions=["VERIFY_EMAIL","UPDATE_PASSWORD"] (app/keycloak/user.py's
#        user_create_api_basic). A direct-grant (password) login then fails with
#        400 "Account is not fully set up" until those are cleared -- there is
#        no REST endpoint for that; it has to go through Keycloak's own
#        email-verification link + the account console's forced password
#        change, or be cleared by hand in the admin console (Users → the user →
#        clear the two chips under Required user actions).
#      - Either way, add "skip_verification": true to the body below to force
#        emailVerified=true and skip sending the verification email outright,
#        regardless of SMTP -- for an admin who already knows the
#        username/password being issued and doesn't need Keycloak's
#        self-registration-style verification flow in the way. Default is
#        false (the SMTP-dependent behavior above, unchanged for existing
#        callers).
curl -s -X POST "http://<host>:8800/admin/keycloak/user?realm_name=dhtsolution" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "newuser@example.com",
    "firstName": "New",
    "lastName": "User",
    "groups": ["/default_group"],
    "realmRoles": ["Developer"],
    "clientRoles": {"raptor": ["user_basic", "user_standard"]},
    "skip_verification": true
  }'

# 3. Look up or remove a user (same Bearer token). Query params are `keyword`
#    (GET, substring match on username/email/name) and `user_id` (DELETE, the
#    UUID GET returns) -- not `username` for either, despite what you'd guess.
curl -s -G "http://<host>:8800/admin/keycloak/user" \
  --data-urlencode "realm_name=dhtsolution" --data-urlencode "keyword=newuser" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

curl -s -X DELETE -G "http://<host>:8800/admin/keycloak/user" \
  --data-urlencode "realm_name=dhtsolution" --data-urlencode "user_id=<uuid from GET>" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

On an SMTP-configured deployment, the returned `temporaryPassword` does not log in right away --
Keycloak requires completing its own email-verification link and forced password-change flow
first (`emailSent: true` only means the send call didn't error, not that the password already
works). Once that flow is done, `/sso/login` with the new password returns a normal token with
`email_verified: true`. On a deployment with no SMTP configured, the `temporaryPassword` itself
logs in immediately instead, same as any other account.

**Granting an admin realm role at creation:** `realmRoles` isn't limited to ordinary roles like
`Developer` -- it also works for the four admin roles the gateway's own routers check for
(`deployment/modules/06-authentication/realm-import/dhtsolution-realm.json` defines all four, so
they already exist in the realm and don't need to be created first):

| Realm role | Unlocks |
|---|---|
| `gdpr-admin` | `/gdpr/*`'s admin-only endpoints (audit-log, jobs, retention policy, processing-records, the `/{user_id}` admin routes on erasure/export/dsar, legal-hold) |
| `guardrail-admin` | every `/guardrail/*` endpoint (module 23's own role gate covers both the policy CRUD/system-toggle writes **and** the plain reads like `GET /guardrail/policies`) |
| `moderation-admin` | `/moderation/flags*` |
| `compliance-auditor` | `GET /compliance/report` |

`api_smoke_test.py`'s "Admin" section (`RAPTOR_ADMIN_USERNAME`/`RAPTOR_ADMIN_PASSWORD`) runs
against an account created this way with all four roles, and exercises one endpoint from each
group -- a repeatable way to check this creation path still grants working permissions, not just
a Keycloak role label.

### Demo frontend (optional)

```bash
cd raptor-demo-frontend
cp .env.example .env        # set API_TARGET / DEMO_PORT
docker compose up -d --build
```

## 6. Multi-Host Deployment

Deploy vLLM, Ollama, and NFS on **separate hosts (different IPs)** and split the
GPU modules across multiple GPU hosts.

### 6.1 Reference topology

Example IP plan (replace with your own):

| Host            | Example IP    | Role                                        | Modules / services                           |
| --------------- | ------------- | ------------------------------------------- | -------------------------------------------- |
| `nfs-host`    | `10.0.0.10` | NFS server                                  | 01 (or native NFS)                           |
| `ollama-host` | `10.0.0.11` | Ollama (may be an existing external server) | Ollama`:11434`                             |
| `vllm-host`   | `10.0.0.12` | vLLM (GPU, optional)                        | vLLM OpenAI server`:8000`                  |
| `core-host`   | `10.0.0.20` | CPU infrastructure + APIs                   | 02, 03, 04¹, 05, 06, 13, (14), 15, 21, 24, 27, 29, 33, 36 (skip 22 — placeholder; skip 19/20 — deprecated) |
| `gpu-host-1`  | `10.0.0.31` | GPU — media A                              | 07, 08 (build), 09, 10                       |
| `gpu-host-2`  | `10.0.0.32` | GPU — media B                              | 08 (build), 11, 12                           |
| `gpu-host-3`  | `10.0.0.33` | GPU — training & search                    | 16, 18                                       |
| `gpu-host-4`  | `10.0.0.34` | GPU — guardrail / personal-db / memory / voice | 23, 25, 26, 34, 35                       |

¹ module 04's `thumbnail-worker` sidecar needs a GPU host today (§1.1's
footnote) — if that matters for your topology, run module 04 on a GPU host
instead, or wait for the pending `profiles:` gate fix.

Ground rules:

1. Every module host clones the same repo and fills in all `.env`
   files; each host starts **only its own modules** (`bash deploy.sh -m <ids>`).
2. Same-host modules use the local `raptor` docker network aliases; **cross-host
   references must use the target host IP + published port**.
3. `HOST_IP` on each host = that host's own IP.
4. On whichever host actually runs Guardrail (23) / Personal DB (25) / Memory
   (26) / Privacy Compliance (33) — `gpu-host-4` in the table above, or
   `core-host` if you keep them on CPU hardware instead — set
   `INTERNAL_BIND_IP` to that host's **private LAN IP** so other hosts can
   reach them; its default `127.0.0.1` keeps 23/25/26/33 host-only.
   Never use `0.0.0.0` on an untrusted network, and firewall these ports to
   the hosts that actually need them.

### 6.2 NFS host

```bash
cd RAPTOR/Aigle/0.5                # release root, wherever you cloned the repo
# .env: HOST_IP=10.0.0.10, NFS_BASE_PATH=/srv/raptor-nfs, NFS_EXPORT=/nfs-share, PORT_NFS=2049
sudo mkdir -p /srv/raptor-nfs
bash deploy.sh -m 01

# shared directory layout
sudo mkdir -p /srv/raptor-nfs/aiml/{tmp,data} \
             /srv/raptor-nfs/media/processing \
             /srv/raptor-nfs/seaweedfs/{master1,master2,master3,vol1,vol2,vol3,vol4,filer} \
             /srv/raptor-nfs/lakefs
```

Open 2049/tcp,udp. Verify reachability from each client host — no host-level
`mount` needed, every client module (04/07/09-13/16/22/26) mounts NFS directly
via the Docker volume driver:

```bash
showmount -e 10.0.0.10
```

### 6.3 Ollama host (separate / external)

If an external Ollama server already exists, only verify reachability (below).

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl edit ollama       # [Service] Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
ollama pull <INFERENCE_MODEL>; ollama pull <LLM_MODEL>; ollama pull <QO_INFERENCE_MODEL>
```

Open 11434/tcp. Verify from any module host, then set in **every** host's env:

```bash
curl http://10.0.0.11:11434/api/tags
# deployment/modules/.env on all hosts:
OLLAMA_HOST=http://10.0.0.11:11434
# plus the per-module copies (§3.3): 15 LLM_BASE_URL, 21 OLLAMA_HOST, 07 OLLAMA_API_BASE, 23 OLLAMA_URL (if enabled)
```

### 6.4 vLLM host (optional, separate GPU host)

```bash
docker run -d --name raptor-vllm --gpus all -p 8000:8000 \
  -v /opt/models:/models vllm/vllm-openai:latest \
  --model /models/<served-model> --served-model-name <model-name> \
  --host 0.0.0.0 --port 8000
curl http://10.0.0.12:8000/v1/models
```

Point selected consumers at it (must match `--served-model-name`):

```bash
# e.g. 15-chat-service/.env
LLM_BASE_URL=http://10.0.0.12:8000/v1
CHAT_LLM_MODEL=<model-name>
```

### 6.5 Core host

```bash
# deployment/modules/.env:
HOST_IP=10.0.0.20
NFS_SERVER=10.0.0.10
OLLAMA_HOST=http://10.0.0.11:11434
bash deploy.sh -m 02,03,04,05,06,13,15,21,24,27,29,33,36         # add 14 for monitoring; skip 16/18/23/25/26/34/35 (GPU, see §6.1/§6.6), 19/20/22
```

Values other hosts use to reach this host:
`KAFKA_BOOTSTRAP_SERVERS=10.0.0.20:19092,10.0.0.20:19093,10.0.0.20:19094`
(**EXTERNAL listener ports** — the `kafka-brokerN:9092` aliases only resolve on
this host), `REDIS_HOST=10.0.0.20`, `POSTGRES_HOST=10.0.0.20`,
`S3_ENDPOINT=http://10.0.0.20:8333`,
`LAKEFS_ENDPOINT=http://10.0.0.20:8001` (published port, not the container's
internal 8000), `KEYCLOAK_URL=http://10.0.0.20:8080`,
`ASSET_MANAGEMENT_URL=http://10.0.0.20:8000` (module 04 — `QDRANT_HOST` is
unused, module 17 is deprecated).

### 6.6 GPU hosts

Common steps on every GPU host:

```bash
nvidia-smi                                             # driver check (CUDA 12.8+)
# deployment/modules/.env:
HOST_IP=<this host's own IP>
NFS_SERVER=10.0.0.10
OLLAMA_HOST=http://10.0.0.11:11434
KAFKA_BOOTSTRAP_SERVERS=10.0.0.20:19092,10.0.0.20:19093,10.0.0.20:19094
REDIS_HOST=10.0.0.20
POSTGRES_HOST=10.0.0.20
```

Not every GPU-hosted module reads every one of those five -- setting all of them on every GPU
host is harmless (an unused var is simply ignored), and simpler than tracking which subset each
host's modules individually need. `S3_ENDPOINT`, `LAKEFS_ENDPOINT` and `KEYCLOAK_URL` are
different: none of the modules below read `LAKEFS_ENDPOINT` at all, and only one module each
reads `S3_ENDPOINT` (07) and `KEYCLOAK_URL` (34) -- those two are set in that module's own block
below, not here.

Per host:

```bash
# gpu-host-1 (10.0.0.31): AI lifecycle + audio + image
bash deploy.sh -m 08 --build                           # local base image, used by 09/10
# .env (this host): S3_ENDPOINT=http://10.0.0.20:8333
bash deploy.sh -m 07 --build && bash deploy.sh -m 09,10
#   pin GPUs: AI_LIFECYCLE_GPU, AUDIO_*_GPU, IMAGE_PROCESSING_GPU
#   publishes AI Lifecycle API for others: http://10.0.0.31:8010

# gpu-host-2 (10.0.0.32): video + document
bash deploy.sh -m 08 --build                           # local base image, used by 11/12
bash deploy.sh -m 11,12
#   INFERENCE_URL=http://10.0.0.31:8010   (summaries call module 07 — NOT
#   AI_MODEL_LIFECYCLE_URL, that key is a dead leftover nothing reads)
#   VIDEO_*_GPU on the largest-VRAM GPU; DOCUMENT_ANALYSIS_GPU for 12

# gpu-host-3 (10.0.0.33): training + search
bash deploy.sh -m 16,18
#   QUERY_ORCHESTRATOR_GPU
#   publishes for core-host: QUERY_ORCHESTRATOR_URL=http://10.0.0.33:8843
#   (published port — the container's own internal port is 8000)
#   (module 17/HYBRID_SEARCH_GPU omitted — deprecated, don't start)

# gpu-host-4 (10.0.0.34): guardrail + personal-db + memory + voice
# .env (this host): KEYCLOAK_URL=http://10.0.0.20:8080
bash deploy.sh -m 23,25,26,34,35
#   GUARDRAIL_GPU_ID, PERSONAL_DB_GPU, MEMORY_SERVICE_GPU,
#   REALTIME_AUDIO_GPU + REALTIME_AUDIO_BREEZE_GPU (34 has two), INFERENCE_VOICE_GPU (35, voice profile only)
#   set INTERNAL_BIND_IP=10.0.0.34 so other hosts can reach 23/25/26 (§6.1 rule 4)
#   34/35 publish on all interfaces by default (§3.1) — no INTERNAL_BIND_IP needed for them
```

## 7. Verification

```bash
bash deploy.sh --status                                 # per host

# NFS: write from a container on one host, read from a container on another
# (no host-level mount to touch directly — go through a running container's
# NFS-backed volume instead, e.g. any 09-12 worker's shared media-scratch mount)
docker exec <container-on-host-a> touch /media/processing/.probe
docker exec <container-on-host-b> ls /media/processing/.probe

# Inference endpoints
curl http://10.0.0.11:11434/api/tags                    # Ollama
curl http://10.0.0.12:8000/v1/models                    # vLLM (if used)

# Kafka from a GPU host (EXTERNAL listener)
docker run --rm edenhill/kcat:1.7.1 -b 10.0.0.20:19092 -L | head

# API smoke test against the gateway (all groups except /aiml, /training --
# see the script's own docstring for exactly what's covered and why some
# individual endpoints within each group are skipped)
export RAPTOR_USERNAME=<test account username>
export RAPTOR_PASSWORD=<test account password>
export RAPTOR_TEST_VIDEO_DIR=<a directory containing at least one .mp4>
# Optional:
#   RAPTOR_BASE_URL              default http://raptor_open_0_5_api.dhtsolution.com:8012
#   RAPTOR_TEST_AUDIO_WAV         a real WAV file -- without it, voice/diarize,
#                                 voice/stt, voice/tts and voice/clone are skipped
#   RAPTOR_ADMIN_USERNAME         a second account with the gdpr-admin/guardrail-admin/
#   RAPTOR_ADMIN_PASSWORD         moderation-admin/compliance-auditor realm roles (see
#                                 the admin-realm-role section above) -- without both,
#                                 the "Admin" section is skipped
python3 api_smoke_test.py
# prints a ✓/✗ line per endpoint, ends with "PASS n / n   FAIL m" and exits 1 if m > 0
```

## 8. Troubleshooting

| Symptom                                       | Likely cause / fix                                                                                                                                       |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker compose up` hangs (not errors) on any of 04/07/09-13/16/22/26 | `NFS_SERVER`/`PORT_NFS` wrong or the port is unreachable. The `type: "nfs"` Docker volume driver calls `mount -t nfs4` with no timeout/retry bound, so a port nothing answers on (typo, firewall, NFS server down) blocks that container's startup **indefinitely** instead of failing fast — there is no preflight check for this today. Diagnose with a bounded-timeout manual mount from the affected host: `timeout 15 sudo mount -t nfs4 -o port=$PORT_NFS $NFS_SERVER:/ /mnt/test && sudo umount /mnt/test`; if that times out, fix `NFS_SERVER`/`PORT_NFS` (or whatever's blocking the port) before retrying `docker compose up`. A TCP port that answers but isn't actually NFS (e.g. `PORT_NFS` pointed at some other live service) can still hang the same way — the manual mount test above is the only way to be sure, a plain `nc`/reachability check on the port is not sufficient |
| GPU-host workers can't reach Kafka            | Using`kafka-brokerN:9092` aliases off-host — use `<core_ip>:19092-19094`; `HOST_IP` must be set on core-host **before** Kafka's first start |
| Redis clients get`MOVED` to unreachable IPs | Cluster created before`HOST_IP` was set — recreate with correct announce IP                                                                           |
| LLM calls time out                            | `OLLAMA_HOST`/`LLM_BASE_URL` still a docker alias or localhost — use the inference host IP                                                          |
| 09–12 images fail to build                   | `raptor/media-worker:${RAPTOR_VERSION}` missing on that host — `bash deploy.sh -m 08 --build` locally                                                             |
| InternVL OOM on module 11                     | Assign a larger GPU (`VIDEO_FRAME_DESC_GPU`); check `MAX_MEMORY_PER_GPU`                                                                             |
| Files missing between gateway and workers     | `NFS_MEDIA_TMP_PATH` not on the shared mount on some host (13 + 09–12 must see the same export)                                                       |
| `/sso/login` fails for every seeded user (`test_basicuser` etc.) right after first start | Expected on a fresh deploy — the realm import creates those users with no password set. See [First login](#first-login) in §5 |

## 9. Source Maintenance Process

Rules for keeping the release tree publishable:

1. **Never commit `.env`** — real credentials and internal addresses live only in
   local `.env` files. This repo's `.gitignore` excludes `.env`, key files,
   `__pycache__/`, `logs/`, `checkpoints/`, model caches and Docker artifacts.
   Do not delete this file.
2. **`.env.example` is the contract.** Whenever a key is added to / removed from
   any `.env`, mirror the change in the module's `.env.example` with a
   placeholder value (`<your_...>` for secrets, `<host_ip>` for internal
   addresses). Quick sync check for a module `$d`:
   ```bash
   diff <(grep -E '^[A-Z_]+=' $d/.env.example | cut -d= -f1 | sort) \
        <(grep -E '^[A-Z_]+=' $d/.env         | cut -d= -f1 | sort)
   ```
3. **No internal IPs or hostnames** (e.g. `192.168.*`) in committed files —
   including comments. Documentation uses placeholder addressing (`10.0.0.x`,
   `<host_ip>`).
4. **Model defaults in templates** follow the model policy with one grandfathered
   exception kept until the v1.0 production release: the **InternVL** VLM
   (the Qwen inference/LLM exception was retired early — all LLM
   defaults are now `gemma4:12b`). All other model defaults must
   be non-mainland-China models (embedding/reranker templates use
   `intfloat/multilingual-e5-large` and
   `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). Changing the embedding model
   requires rebuilding existing vector indexes.
5. **Deployment-specific mounts stay local.** Bind-mount overrides (e.g. module
   07 pointing at local disk instead of NFS) are host-specific; committed compose
   files keep the parameterized NFS form.
6. **Runtime artifacts** (`logs/`, `checkpoints/`, `__pycache__/`, `.DS_Store`)
   are never committed.
7. **Before tagging a release:** run the sync check above for every module,
   `git status` must show no unexpected deletions of `.env.example` /
   `.gitignore`, and `api_smoke_test.py` must pass on a reference host.
