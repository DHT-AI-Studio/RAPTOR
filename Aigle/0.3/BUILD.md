# Raptor 0.3 — Build & Source Maintenance Guide

This release replaces the 0.2 monolithic `docker-compose.yaml` + shell-script workflow with a **module-based build system**. The single entry point is:

```
deployment/modules/build.py
```

`deploy.sh` at the release root (`Aigle/0.3/deploy.sh`) is a thin wrapper — every flag is identical.

---

## 1. Prerequisites

- Docker Engine 24.0+ and Docker Compose v2.20+
- NVIDIA Container Toolkit (GPU modules 07–12, 16–17)
- **GPU driver supporting CUDA 12.8+** — the 0.3 media stack targets sm_120 (Blackwell / RTX 50-series): torch 2.7.1 cu128, paddlepaddle-gpu 3.3.0 (cu129). Older GPUs (sm_8x/sm_90) still work with the same wheels.
- Python 3.10+ on the host (for `build.py`)

## 2. Configuration (`.env` / `.env.example`)

Every module directory under `deployment/modules/` — plus the `modules/` root itself — carries a committed **`.env.example`** template and expects a local, uncommitted **`.env`**:

```bash
cd deployment/modules
cp .env.example .env                      # global settings
for m in */; do
  [ -f "$m/.env.example" ] && cp "$m/.env.example" "$m/.env"
done
# then edit the .env files: hosts, credentials, model names, GPU limits
```

Key parameter groups in the root `modules/.env`:

| Group | Examples |
| --- | --- |
| Host & network | `HOST_IP`, `NFS_SERVER`, per-module `PORT_*` |
| Credentials | `REDIS_PASSWORD`, `POSTGRES_PASSWORD`, `OPENSEARCH_PASSWORD`, `NEO4J_PASSWORD`, LakeFS / AWS keys, Keycloak admin, `HF_TOKEN` |
| Model selection | `LLM_MODEL`, `INFERENCE_MODEL` + per-modality overrides (`VIDEO_INFERENCE_MODEL`, `AUDIO_INFERENCE_MODEL`, `DOCUMENT_INFERENCE_MODEL`), `IMAGE_VLM_MODEL_PATH`, `VIDEO_VLM_MODEL_PATH`, `CHAT_LLM_MODEL`, `CONTEXTUALIZE_MODEL_NAME`, `RERANKER_TEMPERATURE` |
| Mail (module 06) | `SMTP_HOST/PORT/USER/PASSWORD/FROM/...` |
| Tuning | `MEMORY_TTL`, `MEMORY_CONTEXT_WINDOW`, `REQUEST_TIMEOUT`, `CROSS_SERVICE_TIMEOUT`, `DOCUMENT_ANALYSIS_CONTEXTUAL_BATCH_SIZE`, `VIDEO_GRAPH_CPU_LIMIT/MEMORY_LIMIT` |

Changes vs 0.2-era templates: per-service `*_GPU_COUNT` variables were removed (GPU allocation is handled by module 07), `GATEWAY_KEYCLOAK_CLIENT_ID/REALM` were removed (permission checks now go through module 06 `/auth/permission`), and `TKG_AGENT_URL` / `RERANKER_AGENT_URL` were removed from module 21.

## 3. Build & Deploy

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
- Module 08 produces `raptor/media-worker:0.3`; modules 09–12 `FROM` it.
- All stacks join the external `raptor` Docker bridge network; `build.py` creates it if missing.

### Demo frontend (optional)

```bash
cd raptor-demo-frontend
cp .env.example .env        # set API_TARGET / DEMO_PORT
docker compose up -d --build
```

## 4. Verification

```bash
bash deploy.sh --status                 # every requested module healthy
python3 test_all_apis.py                # end-to-end API suite (see API_REFERENCE.md)
```

---

## 5. Source Maintenance Process

Rules for keeping the release tree publishable:

1. **Never commit `.env`** — real credentials and internal addresses live only in local `.env` files. `Aigle/0.3/.gitignore` excludes `.env`, `*.key/pem/p12/pfx`, `__pycache__/`, `logs/`, model caches and Docker artifacts. Do not delete this file.
2. **`.env.example` is the contract.** Whenever a key is added to / removed from any `.env`, mirror the change in the module's `.env.example` with a placeholder value (`<your_...>` for secrets, `<host_ip>` for internal addresses). Quick sync check for a module `$d`:
   ```bash
   diff <(grep -E '^[A-Z_]+=' $d/.env.example | cut -d= -f1 | sort) \
        <(grep -E '^[A-Z_]+=' $d/.env         | cut -d= -f1 | sort)
   ```
3. **No internal IPs or hostnames** (e.g. `192.168.*`) in committed files — including comments.
4. **Model defaults in templates** follow the model policy with two grandfathered exceptions kept until the v1.0 production release: the **Qwen** inference/LLM defaults and the **InternVL** VLM (functionally integrated in the media pipeline). All other model defaults must be non-mainland-China models (embedding/reranker templates use `intfloat/multilingual-e5-large` and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). At v1.0, migrate the Qwen defaults to Gemma / Llama / Mistral. Changing the embedding model requires rebuilding existing vector indexes.
5. **Deployment-specific mounts stay local.** Bind-mount overrides such as module 07's `/mnt/disk1/...` paths are host-specific; the committed compose files should keep the parameterized NFS form.
6. **Runtime artifacts** (`logs/`, `checkpoints/`, `__pycache__/`, `.DS_Store`) are never committed.
7. **Before tagging a release:** run the sync check above for every module, `git status` must show no unexpected deletions of `.env.example` / `.gitignore`, and `test_all_apis.py` must pass on a reference host.
