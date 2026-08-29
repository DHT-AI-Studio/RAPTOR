# 07-ai-ml-services Standalone Test Guide (Dev / Standalone)

This document explains how to bring up this module **on its own** for testing while the full
raptor stack is already running locally, without conflicting with the running raptor containers,
ports, or Docker network.

> Isolation key: `docker-compose.dev.yml` uses a separate compose project name via the top-level
> `name: aiml-test`, fully separated from the live `07-ai-ml-services` project (its own containers,
> volumes, and network). **Always use the commands given in this file — never run `down` on the dev
> compose using the live project's name, or you risk deleting the live module's containers and data.**

---

## 1. Isolation from the live raptor stack

| Item | Live raptor module | This standalone test |
| --- | --- | --- |
| Compose project | `07-ai-ml-services` | `aiml-test` |
| Containers | `raptor-mlflow` / `raptor-ai-lifecycle-api` | `aiml-test-mlflow` / `aiml-test-api` / `aiml-dev-postgres` |
| Network | swarm `raptor` (overlay) | `aiml-test` (bridge) |
| Volumes | `07-ai-ml-services_*` | `aiml-test_*` |
| GPU | Decided by the live configuration | Uses only GPU #4 (physical index 3) |
| API port | 8010 | 9997 |
| MLflow port | 5556 | 5555 |
| Postgres host port | — | 15432 |

Published ports (host side):

- AI Lifecycle API: `http://localhost:9997`
- MLflow UI: `http://localhost:5555`
- PostgreSQL: `localhost:15432` (still 5432 inside the container)

---

## 2. Prerequisites

- The NVIDIA Container Toolkit is installed on the host, and GPU #4 (index 3) is available.
- The NFS server (`.env.dev`'s `NFS_SERVER`) is reachable — the `tmp` and `data` volumes mount onto it.
- If inference needs to download models from lakeFS: make sure `.env.dev`'s
  `MLFLOW_S3_ENDPOINT_URL` and the lakeFS endpoint in the config both point at a host-reachable
  address (the standalone network can't resolve the `lakefs` service name — use the
  host-published `http://<host-ip>:8003` instead).

> GPU config: `AI_LIFECYCLE_GPU=3` makes compose reserve only physical index 3 for the container;
> inside the container, `CUDA_VISIBLE_DEVICES=0` (the container only sees that one GPU, renumbered
> to 0) — do not change it to 3.

---

## 3. Starting up

```bash
cd deployment/modules/07-ai-ml-services
docker compose --env-file .env.dev -f docker-compose.dev.yml up -d --build
```

> Every subsequent command must include `--env-file .env.dev -f docker-compose.dev.yml`,
> or compose won't find the right variables and project.

### Checking status

```bash
# Container status
docker compose --env-file .env.dev -f docker-compose.dev.yml ps

# Health checks
curl -s http://localhost:9997/health    # API (adjust to the actual route)
curl -s http://localhost:5555/health     # MLflow

# Confirm only GPU #4 is visible
docker exec aiml-test-api python -c "import torch; print(torch.cuda.device_count())"   # expect 1
```

### Viewing logs

```bash
# All services
docker compose --env-file .env.dev -f docker-compose.dev.yml logs -f

# A single service
docker compose --env-file .env.dev -f docker-compose.dev.yml logs -f api
```

---

## 4. Common operations

```bash
# Changes under src/: uvicorn --reload hot-reloads automatically in most cases — no restart needed.

# Release lingering GPU CUDA context (a few hundred MB still showing in nvidia-smi after unload
# is normal — that's the process's CUDA context, and only fully clears when the process restarts)
docker restart aiml-test-api

# Rebuild and restart just one service
docker compose --env-file .env.dev -f docker-compose.dev.yml up -d --build api
```

---

## 5. Shutting down / cleanup (compose down)

> ⚠️ Every `down` command must include `--env-file .env.dev -f docker-compose.dev.yml`
> to guarantee it operates on the `aiml-test` project, not the live `07-ai-ml-services`.

### 5.1 Stop and remove containers (**keeps** volumes and data)

The most common case. Postgres, downloaded models, and caches all survive until the next `up`.

```bash
cd deployment/modules/07-ai-ml-services
docker compose --env-file .env.dev -f docker-compose.dev.yml down
```

This removes `aiml-test`'s containers and the `aiml-test` bridge network it created, but keeps all volumes.

### 5.2 Stop only, don't remove containers

```bash
docker compose --env-file .env.dev -f docker-compose.dev.yml stop
# Bring it back later with start
docker compose --env-file .env.dev -f docker-compose.dev.yml start
```

### 5.3 Remove volumes too (**this deletes data**: Postgres, downloaded models, caches)

Use this when you need a clean slate (e.g., rebuilding the database, clearing downloaded models).

```bash
docker compose --env-file .env.dev -f docker-compose.dev.yml down -v
```

> `-v` only deletes this project's volumes (`aiml-test_*`) — it never touches the live
> `07-ai-ml-services_*` ones.
> Note that `tmp` / `data` are NFS volumes: deleting the volume object does not delete the actual
> files on the NFS server — a full cleanup needs a separate pass on the NFS side.

### 5.4 Remove the built images as well

```bash
docker compose --env-file .env.dev -f docker-compose.dev.yml down --rmi local
# Or clear volumes and images together
docker compose --env-file .env.dev -f docker-compose.dev.yml down -v --rmi local
```

### 5.5 Confirming the cleanup

```bash
docker compose --env-file .env.dev -f docker-compose.dev.yml ps        # should show no containers
docker network ls   | grep aiml-test                                    # should be gone after down
docker volume  ls   | grep aiml-test                                    # should be gone after -v
```

---

## 6. Troubleshooting

- **Port already in use**: change `PORT_AI_LIFECYCLE_API` / `PORT_MLFLOW` / `PORT_POSTGRES` in `.env.dev`.
  (5433 was already taken by another local service, hence Postgres's host port being 15432.)
- **lakeFS model download fails / unreachable**: make sure `MLFLOW_S3_ENDPOINT_URL` and the lakeFS
  endpoint point at the host-published address, not `http://lakefs:8000` (unresolvable on the
  standalone network).
- **GPU still shows residual usage after unload**: this is the normal CUDA context —
  `docker restart aiml-test-api` clears it.
- **Want zero impact on the live raptor stack**: always use this file's full commands
  (including `--env-file` and `-f`) — never omit them.
