# Running Tests

This document outlines the steps required to run the tests for the `asset_management` API.

Tests are split into two kinds:

- **`tests/unit/`** — pure logic, hand-written fakes, no live infra required. Runs anywhere.
- **`tests/integration/`** — hits a real running Module 04 + LakeFS + Postgres + SeaweedFS. Requires the local docker-compose stack to be up (`bash deploy.sh -m 04` from the repo root, or `-m 04 --build` after code changes). Split by concern, not one big file: `test_asset_lifecycle.py` (full asset lifecycle), `test_claim_processing.py` (claim atomicity), `test_commits_status_filter.py` (status filtering) -- all share this directory's `conftest.py`.

## Prerequisites

1. **Activate a conda environment with this module's dependencies installed**

```bash
conda activate <your-env>
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-mock python-dotenv
```

(There may be stale environments on a shared machine that were never `pip install -r requirements.txt`'d against this module — check with `pip show lakefs asyncpg` before assuming an existing env works.)

2. **`tests/integration/` only: no manual setup needed for credentials/ports.**
   This module's own `.env` (used by the running containers) has placeholder
   credentials and docker-internal hostnames (`lakefs`, `postgres`) that
   aren't valid from a host-side test process. `tests/integration/conftest.py`
   automatically reads the real credentials, `HOST_IP`, and the host-forwarded
   `PORT_*` values from the unified `deployment/modules/.env` instead —
   nothing to edit by hand. If that file is missing (e.g. CI without a live
   docker-compose stack), it falls back to `localhost` and today's known ports.

## Running the Tests

Run from the `04-object-storage/` directory (the parent of `tests/`), not from inside `tests/` itself — relative imports assume this.

Use `python -m pytest`, not a bare `pytest` — on a machine with a `~/.local/bin/pytest` (a `pip install --user pytest` outside any conda env) ahead of conda envs on `PATH`, a bare `pytest` silently resolves to that one instead of your activated env's, and fails with `ModuleNotFoundError` for this module's own dependencies (`lakefs`, `asyncpg`, ...) even though `python -c "import asyncpg"` works fine in the same shell. `python -m pytest` always uses the currently-active `python`, sidestepping this.

```bash
# unit only — no live infra needed
python -m pytest tests/unit/ -v

# integration only — requires Module 04 + LakeFS + Postgres + SeaweedFS running locally
python -m pytest tests/integration/ -s

# both
python -m pytest tests/ -s
```

### Notes

* Integration tests generate randomized content/filenames per run (`uuid.uuid4()`) so repeated runs don't collide with prior runs' leftovers, and a session-scoped `autouse` teardown fixture in `conftest.py` deletes the test branch's data (Module 25 index, `commit_history`/`audit_log` rows, LakeFS branch) after the run.
* If Module 04's code changed, rebuild the running container first (`bash deploy.sh -m 04 --build` from the repo root) — the integration tests hit the container over HTTP, not the local Python source directly.
