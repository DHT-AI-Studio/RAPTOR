# Raptor 0.3 — Setup, Build & Configuration Guide

The complete guide for deploying RAPTOR Aigle 0.3: prerequisites (hardware,
network, external servers), dependency map, `.env` configuration, build/deploy
commands, single-host and multi-host setups, verification, and source-maintenance
rules.

The single build entry point is `deployment/modules/build.py`; `deploy.sh` at the
release root (`Aigle/0.3/deploy.sh`) is a thin wrapper — every flag is identical.

---

## 1. Prerequisites

Read this section **before** starting any setup.

### 1.1 Hardware / Infrastructure Requirements

| Host role | Minimum requirement | Notes |
| --- | --- | --- |
| **CPU core host** (infrastructure + APIs: modules 01–06, 13–15, 19–21) | 16 CPU cores, 64 GB RAM, 500 GB+ SSD | Runs Kafka (6 containers), Redis cluster (8), PostgreSQL, Qdrant, SeaweedFS+LakeFS, Keycloak, API/reasoning services |
| **GPU host(s)** (modules 07–12, 16–18) | NVIDIA GPU with **CUDA 12.8+ driver**, 24 GB+ VRAM; 64 GB RAM | 0.3 GPU stack targets sm_120 (Blackwell / RTX 50-series): torch 2.7.1 cu128, PaddlePaddle 3.3.0 cu129. Older GPUs (sm_8x/sm_90) work with the same wheels |
| — module 11 (video) | **36 GB+ VRAM recommended** | InternVL frame description; `MAX_MEMORY_PER_GPU=36GiB` cap (shared with module 10) |
| **NFS server** | Any host/appliance exporting NFSv4 (port 2049), 1 TB+ recommended | Module 01 provides a containerized NFS server, or use a native/appliance NFS |
| **Ollama inference server** (required by default) | GPU host sized for your configured models (`INFERENCE_MODEL`, `LLM_MODEL`, `QO_INFERENCE_MODEL`) | **May be an existing external server outside the RAPTOR cluster** — RAPTOR only needs HTTP reachability to port 11434. Consumed by modules 07, 12, 13, 15, 20, 21 (and 09/11 indirectly via module 07) |
| **vLLM inference server** (optional) | GPU host for the served model | vLLM is **not bundled** — it is an optional external OpenAI-compatible backend. Any `LLM_BASE_URL` consumer (15 chat, 20 graph) can point at it instead of Ollama; module 07 can register models with `engine: vllm` |

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

| From → To | Ports |
| --- | --- |
| all module hosts → NFS server | 2049/tcp,udp (NFSv4) |
| core + GPU hosts → Ollama server | 11434/tcp |
| core + GPU hosts → vLLM server (if used) | 8000/tcp |
| GPU hosts → core host | 19092–19094 (Kafka EXTERNAL), 6379 + 7000–7005 (Redis), 5432 (PostgreSQL), 6333/6334 (Qdrant), 8333 (SeaweedFS S3), 8000 (LakeFS), 8080 (Keycloak), asset-management port |
| core host → AI-lifecycle GPU host | 8010 (AI Lifecycle API) |
| core host → search GPU host | 8000 (Query Orchestrator), OpenSearch + hybrid-search API ports |
| clients → core host | 8012 (API Gateway), 8021 (chat), 8843/8844 (graph), 8030 (A2A), 3000 (demo frontend) |

### 1.4 External Inference Servers — summary

| Server | Status | RAPTOR consumers | Key env vars |
| --- | --- | --- | --- |
| **Ollama** (`:11434`) | Default LLM backend; may be a pre-existing external server | 07 (`OLLAMA_API_BASE` — MLflow registration + inference routing), 12 (`OLLAMA_URL`), 13 (`GATEWAY_SMOLAGENTS_MODEL=ollama/<model>`), 15 (`LLM_BASE_URL`), 20 (`LLM_BASE_URL`), 21 (`OLLAMA_HOST`); 09/11 indirectly via `AI_MODEL_LIFECYCLE_URL` | `OLLAMA_HOST` → derives `LLM_BASE_URL=${OLLAMA_HOST}/v1`, `OLLAMA_API_BASE=${OLLAMA_HOST}` |
| **vLLM** (`:8000`, optional) | Not bundled (no in-tree runtime); external OpenAI-compatible `/v1` server | Any `LLM_BASE_URL` consumer (15, 20); module 07 model registration accepts `engine: vllm` | `LLM_BASE_URL=http://<vllm_host>:8000/v1` per consuming module |

Model names are configured via `LLM_MODEL`, `INFERENCE_MODEL` (+ per-modality
`AUDIO/VIDEO/DOCUMENT_INFERENCE_MODEL`), `QO_INFERENCE_MODEL`, `CHAT_LLM_MODEL`;
pull/serve those models on the inference server before first use. Per the model
policy, Qwen defaults are retained until v1.0.

## 2. Dependency Map (NFS · GPU)

### NFS

| Role | Module | Detail |
| --- | --- | --- |
| Server | 01 nfs-server | Containerized NFS (port `2049`/`PORT_NFS`), exports `/nfs-share` backed by host path `NFS_BASE_PATH` |
| Docker-volume client | 07 ai-ml-services | `tmp`/`data` volumes mount `${NFS_SERVER}:${NFS_AIML_TMP_PATH|NFS_AIML_DATA_PATH}` over NFSv4 |
| Host-mount client | 04 object-storage | SeaweedFS/LakeFS bind mounts under `/mnt/disk1/nfs/...` — mount the export there on the docker host |
| Shared media scratch | 09–12 workers + 13 gateway | `NFS_MEDIA_TMP_PATH=/media/processing` on the shared export, used to exchange temporary media |

### GPU modules

| Module | GPU use | Device-selection env |
| --- | --- | --- |
| 07 ai-ml-services | AI Lifecycle inference | `AI_LIFECYCLE_GPU` |
| 08 media-worker | **Build-time base image** for 09–12 (no runtime container) | — |
| 09 audio-processing | WhisperX STT / diarization / classifier | `AUDIO_RECOGNIZER_GPU`, `AUDIO_DIARIZATION_GPU`, `AUDIO_CLASSIFIER_GPU` |
| 10 image-processing | InternVL image analysis | `IMAGE_PROCESSING_GPU` |
| 11 video-processing | Chunking / frame description / OCR (highest VRAM) | `VIDEO_CHUNKING_GPU`, `VIDEO_OCR_GPU`, `VIDEO_FRAME_DESC_GPU` |
| 12 document-processing | PDF/Office OCR + analysis | `DOCUMENT_ANALYSIS_GPU` |
| 16 training-service | Training orchestration | (compose reservation) |
| 17 hybrid-search | Embedding + cross-encoder rerank | `HYBRID_SEARCH_GPU` |
| 18 query-orchestrator | Intent classification / rerank | `QUERY_ORCHESTRATOR_GPU` |

Device ids are per-service: multiple services can pin different GPUs on one host,
or modules can be split across GPU hosts (§6).

## 3. Configuration (`.env`)

Every module directory under `deployment/modules/` — plus the `modules/` root —
carries a committed **`.env.example`** template and expects a local, uncommitted
**`.env`**.

### 3.1 Create the `.env` files

```bash
cd Aigle/0.3/deployment/modules
cp .env.example .env                      # global settings
for m in */; do
  [ -f "$m/.env.example" ] && cp "$m/.env.example" "$m/.env"
done
```

### 3.2 Edit the root `modules/.env` — required values

| Group | Variables | What to set |
| --- | --- | --- |
| Host & network | `HOST_IP` | **This host's own IP** — drives Kafka `EXTERNAL` advertised listeners (`:19092-19094`), Redis `--cluster-announce-ip`, Keycloak `KC_HOSTNAME`. Set **before first start** |
| NFS | `NFS_SERVER`, `NFS_BASE_PATH`, `NFS_EXPORT`, `NFS_AIML_TMP_PATH`, `NFS_AIML_DATA_PATH`, `NFS_MEDIA_TMP_PATH` | NFS server IP and export paths (§2) |
| Inference | `OLLAMA_HOST=http://<ollama_ip>:11434` | Ollama server URL; `LLM_BASE_URL` / `OLLAMA_API_BASE` derive from it. Point `LLM_BASE_URL` at vLLM `/v1` instead where desired |
| Models | `LLM_MODEL`, `INFERENCE_MODEL`, `VIDEO/AUDIO/DOCUMENT_INFERENCE_MODEL`, `IMAGE/VIDEO_VLM_MODEL_PATH`, `CHAT_LLM_MODEL`, `QO_INFERENCE_MODEL`, `CONTEXTUALIZE_MODEL_NAME`, `RERANKER_TEMPERATURE` | Model selection (must exist on the inference server) |
| Credentials | `REDIS_PASSWORD`, `POSTGRES_PASSWORD`, `OPENSEARCH_PASSWORD`, `NEO4J_PASSWORD`, `LAKEFS_*_KEY`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, Keycloak admin vars, `HF_TOKEN` | Replace every `<your_...>` placeholder |
| Cross-module URLs | `KAFKA_BOOTSTRAP_SERVERS`, `REDIS_HOST`, `POSTGRES_HOST`, `QDRANT_HOST`, `S3_ENDPOINT`/`S3_PUBLIC_URL`, `LAKEFS_ENDPOINT`, `ASSET_MANAGEMENT_URL`, `AI_MODEL_LIFECYCLE_URL`, `QUERY_ORCHESTRATOR_URL`, `GRAPH_SERVICE_URL`, `KEYCLOAK_URL` | Single host: keep docker-alias defaults. Multi-host: replace with host IPs + published ports (§6) |
| Mail (module 06) | `SMTP_HOST/PORT/USER/PASSWORD/FROM/...` | Optional e-mail notifications |
| Tuning | `MEMORY_TTL`, `MEMORY_CONTEXT_WINDOW`, `REQUEST_TIMEOUT`, `CROSS_SERVICE_TIMEOUT`, `DOCUMENT_ANALYSIS_CONTEXTUAL_BATCH_SIZE`, `VIDEO_GRAPH_CPU_LIMIT/MEMORY_LIMIT`, `MAX_MEMORY_PER_GPU` | Defaults are sane; adjust per hardware |

### 3.3 Per-module `.env` files

Several modules keep local copies of shared values — keep them consistent with
the root `.env`:

- `07-ai-ml-services/.env`: `OLLAMA_API_BASE`, `NFS_SERVER`, `NFS_AIML_*_PATH`
- `15-chat-service/.env`: `LLM_BASE_URL`, `CHAT_LLM_MODEL`
- `20-graph-service/.env`: `LLM_BASE_URL`, `CHAT_MODEL_NAME`
- `21-agent-protocol/.env`: `OLLAMA_HOST`, `OLLAMA_MODEL`
- `04-object-storage/.env`: `NFS_SERVER`, `NFS_EXPORT`
- `09–12/.env`: `NFS_MEDIA_TMP_PATH`, `AI_MODEL_LIFECYCLE_URL`, per-service `*_GPU`

Keep every `.env.example` in sync when adding/removing keys (see §9).

## 4. Build & Deploy Commands

```bash
cd Aigle/0.3

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

# Info
bash deploy.sh --list
bash deploy.sh --status
bash deploy.sh -m 18 --logs
bash deploy.sh -l build.log          # tee output to a log file
```

Notes:

- Module 07 builds a custom MLflow image (`Dockerfile.mlflow` → `raptor/mlflow:0.3`).
- Module 08 produces `raptor/media-worker:0.3`; modules 09–12 `FROM` it. It is
  not pushed to a registry — **build it locally on every host that runs 09–12**.
- All stacks join the external `raptor` Docker bridge network; `build.py` creates
  it if missing.

## 5. Single-Host Deployment (evaluation)

```bash
cd Aigle/0.3
# .env: HOST_IP=<this host IP>, OLLAMA_HOST=http://<ollama_ip>:11434, NFS vars
sudo mkdir -p /mnt/disk1/nfs && sudo mount -t nfs4 <nfs_ip>:/ /mnt/disk1/nfs   # or local dirs
bash deploy.sh -m 08 --build
bash deploy.sh                       # or --cpu-only without GPUs
```

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

| Host | Example IP | Role | Modules / services |
| --- | --- | --- | --- |
| `nfs-host` | `10.0.0.10` | NFS server | 01 (or native NFS) |
| `ollama-host` | `10.0.0.11` | Ollama (may be an existing external server) | Ollama `:11434` |
| `vllm-host` | `10.0.0.12` | vLLM (GPU, optional) | vLLM OpenAI server `:8000` |
| `core-host` | `10.0.0.20` | CPU infrastructure + APIs | 02, 03, 04, 05, 06, 13, (14), 15, 19, 20, 21 |
| `gpu-host-1` | `10.0.0.31` | GPU — media A | 07, 08 (build), 09, 10 |
| `gpu-host-2` | `10.0.0.32` | GPU — media B | 08 (build), 11, 12 |
| `gpu-host-3` | `10.0.0.33` | GPU — search & training | 16, 17, 18 |

Ground rules:

1. Every module host clones the same `Aigle/0.3/` tree and fills in all `.env`
   files; each host starts **only its own modules** (`bash deploy.sh -m <ids>`).
2. Same-host modules use the local `raptor` docker network aliases; **cross-host
   references must use the target host IP + published port**.
3. `HOST_IP` on each host = that host's own IP.

### 6.2 NFS host

```bash
cd Aigle/0.3
# .env: HOST_IP=10.0.0.10, NFS_BASE_PATH=/srv/raptor-nfs, NFS_EXPORT=/nfs-share, PORT_NFS=2049
sudo mkdir -p /srv/raptor-nfs
bash deploy.sh -m 01

# shared directory layout
sudo mkdir -p /srv/raptor-nfs/aiml/{tmp,data} \
             /srv/raptor-nfs/media/processing \
             /srv/raptor-nfs/seaweedfs/{master1,master2,master3,vol1,vol2,vol3,vol4,filer} \
             /srv/raptor-nfs/lakefs
```

Open 2049/tcp,udp. Verify and mount from each client host:

```bash
showmount -e 10.0.0.10
sudo mkdir -p /mnt/disk1/nfs
sudo mount -t nfs4 10.0.0.10:/ /mnt/disk1/nfs
# /etc/fstab:  10.0.0.10:/  /mnt/disk1/nfs  nfs4  defaults,_netdev  0  0
```

Mount on **core-host** (module 04 bind mounts) and every GPU host running 09–13.

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
# plus the per-module copies (§3.3): 15/20 LLM_BASE_URL, 21 OLLAMA_HOST, 07 OLLAMA_API_BASE
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
sudo mount -t nfs4 10.0.0.10:/ /mnt/disk1/nfs      # module 04 bind mounts
bash deploy.sh -m 02,03,04,05,06,13,15,19,20,21     # add 14 for monitoring
```

Values other hosts use to reach this host:
`KAFKA_BOOTSTRAP_SERVERS=10.0.0.20:19092,10.0.0.20:19093,10.0.0.20:19094`
(**EXTERNAL listener ports** — the `kafka-brokerN:9092` aliases only resolve on
this host), `REDIS_HOST=10.0.0.20`, `POSTGRES_HOST=10.0.0.20`,
`QDRANT_HOST=10.0.0.20`, `S3_ENDPOINT=http://10.0.0.20:8333`,
`LAKEFS_ENDPOINT=http://10.0.0.20:8000`, `KEYCLOAK_URL=http://10.0.0.20:8080`,
`ASSET_MANAGEMENT_URL=http://10.0.0.20:<asset_port>`.

### 6.6 GPU hosts

Common steps on every GPU host:

```bash
nvidia-smi                                             # driver check (CUDA 12.8+)
sudo mount -t nfs4 10.0.0.10:/ /mnt/disk1/nfs          # if running 09–13
# deployment/modules/.env:
HOST_IP=<this host's own IP>
NFS_SERVER=10.0.0.10
OLLAMA_HOST=http://10.0.0.11:11434
KAFKA_BOOTSTRAP_SERVERS=10.0.0.20:19092,10.0.0.20:19093,10.0.0.20:19094
REDIS_HOST=10.0.0.20
POSTGRES_HOST=10.0.0.20
QDRANT_HOST=10.0.0.20
S3_ENDPOINT=http://10.0.0.20:8333
LAKEFS_ENDPOINT=http://10.0.0.20:8000
KEYCLOAK_URL=http://10.0.0.20:8080
bash deploy.sh -m 08 --build                           # local base image (hosts running 09–12)
```

Per host:

```bash
# gpu-host-1 (10.0.0.31): AI lifecycle + audio + image
bash deploy.sh -m 07 --build && bash deploy.sh -m 09,10
#   pin GPUs: AI_LIFECYCLE_GPU, AUDIO_*_GPU, IMAGE_PROCESSING_GPU
#   publishes AI Lifecycle API for others: http://10.0.0.31:8010

# gpu-host-2 (10.0.0.32): video + document
bash deploy.sh -m 11,12
#   AI_MODEL_LIFECYCLE_URL=http://10.0.0.31:8010   (summaries call module 07)
#   VIDEO_*_GPU on the largest-VRAM GPU; DOCUMENT_ANALYSIS_GPU for 12

# gpu-host-3 (10.0.0.33): training + search
bash deploy.sh -m 16,17,18
#   HYBRID_SEARCH_GPU, QUERY_ORCHESTRATOR_GPU
#   publishes for core-host: QUERY_ORCHESTRATOR_URL=http://10.0.0.33:8000, hybrid-search API
```

## 7. Verification

```bash
bash deploy.sh --status                                 # per host

# NFS: write on one host, read on another
touch /mnt/disk1/nfs/media/processing/.probe

# Inference endpoints
curl http://10.0.0.11:11434/api/tags                    # Ollama
curl http://10.0.0.12:8000/v1/models                    # vLLM (if used)

# Kafka from a GPU host (EXTERNAL listener)
docker run --rm edenhill/kcat:1.7.1 -b 10.0.0.20:19092 -L | head

# End-to-end API suite against the gateway
python3 test_all_apis.py                                # base URL http://<core_ip>:8012
```

## 8. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| Module 07 volumes fail to create | `NFS_SERVER`/`NFS_AIML_*_PATH` wrong or 2049 blocked; test a manual `mount -t nfs4` |
| GPU-host workers can't reach Kafka | Using `kafka-brokerN:9092` aliases off-host — use `<core_ip>:19092-19094`; `HOST_IP` must be set on core-host **before** Kafka's first start |
| Redis clients get `MOVED` to unreachable IPs | Cluster created before `HOST_IP` was set — recreate with correct announce IP |
| LLM calls time out | `OLLAMA_HOST`/`LLM_BASE_URL` still a docker alias or localhost — use the inference host IP |
| 09–12 images fail to build | `raptor/media-worker:0.3` missing on that host — `bash deploy.sh -m 08 --build` locally |
| InternVL OOM on module 11 | Assign a larger GPU (`VIDEO_FRAME_DESC_GPU`); check `MAX_MEMORY_PER_GPU` |
| Files missing between gateway and workers | `NFS_MEDIA_TMP_PATH` not on the shared mount on some host (13 + 09–12 must see the same export) |

## 9. Source Maintenance Process

Rules for keeping the release tree publishable:

1. **Never commit `.env`** — real credentials and internal addresses live only in
   local `.env` files. `Aigle/0.3/.gitignore` excludes `.env`, key files,
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
4. **Model defaults in templates** follow the model policy with two grandfathered
   exceptions kept until the v1.0 production release: the **Qwen**
   inference/LLM defaults and the **InternVL** VLM. All other model defaults must
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
   `.gitignore`, and `test_all_apis.py` must pass on a reference host.
