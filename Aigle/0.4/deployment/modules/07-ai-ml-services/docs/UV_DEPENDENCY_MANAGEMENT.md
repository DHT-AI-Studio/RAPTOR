# UV Dependency Management Best Practices

## Problem analysis

### Why do version mismatches still happen even when using `uv`?

Even with `uv` as a fast pip replacement, without a version-locking mechanism you can still hit these problems:

1. **No version locking**
   - `requirements.txt` only lists package names, no pinned versions
   - Installing at different times yields different versions
   - Transitive dependency (dependency-of-a-dependency) versions aren't fixed either

2. **Environment drift**
   - Local dev environment: whatever version combination was installed at some point in time (works fine)
   - Docker build: uses whatever's latest at build time, may be incompatible
   - CI/CD environment: yet another version combination

3. **Docker layer caching**
   - Even after updating requirements.txt, Docker may reuse a cached old layer
   - Leading to installed dependencies that don't match what's expected

### The specific incident

```
Local environment (working):    Docker environment (failing):
datasets 4.3.0                  datasets 2.14.4
pyarrow 21.0.0                  pyarrow 21.0.0
✅ compatible                   ❌ incompatible
```

## Solution

### Option 1: Full project management with `uv` (recommended)

#### 1. Create `pyproject.toml`
```toml
[project]
name = "ai-model-lifecycle"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi",
    "pyarrow>=14.0.0,<18.0.0",
    "datasets>=2.19.0",
    # ... other dependencies
]
```

#### 2. Generate the lock file
```bash
uv lock
```
This produces a `uv.lock` file recording the exact resolved version of every dependency.

#### 3. Export requirements from the lock file
```bash
uv export --no-hashes --no-dev | grep -v "^-e " > requirements.lock.txt
```

#### 4. Use the lock file in the Dockerfile
```dockerfile
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.lock.txt
```

### Option 2: Simple version pinning (stopgap)

Pin the critical dependency versions directly in `requirements.txt`:
```txt
pyarrow>=14.0.0,<18.0.0
datasets>=2.19.0
```

## Why this actually fixes it

### 1. **Deterministic builds**
```
uv.lock → requirements.lock.txt → Docker image
Exact versions recorded → reproducible install → matches local
```

### 2. **Version-compatibility guarantee**
- `uv lock` resolves a mutually-compatible version for every dependency
- Guarantees every package is compatible with every other
- Resolve once, use everywhere

### 3. **No dependence on install time**
```
Traditional way:
Install at time T1 → version A (works)
Install at time T2 → version B (may fail)

With a lock file:
Install at any time → the locked version (always the same)
```

## Workflow

### Development

1. Change a dependency: edit `pyproject.toml`
2. Update the lock: `uv lock`
3. Sync the environment: `uv sync`
4. Export requirements: `uv export --no-hashes --no-dev | grep -v "^-e " > requirements.lock.txt`

### CI/CD

```dockerfile
COPY requirements.lock.txt ./
RUN uv pip install --system -r requirements.lock.txt
```

### Other team members

```bash
# Use the exact same versions
uv sync --frozen
```

## Best practices summary

### ✅ Do
1. Manage project dependencies with `pyproject.toml`
2. Commit `uv.lock` to version control
3. Export and use `requirements.lock.txt` in Docker
4. Use `--no-cache` in the Dockerfile to avoid cache issues
5. Add version constraints for critical dependencies

### ❌ Don't
1. Leave every dependency in `requirements.txt` unversioned
2. Ignore the `uv.lock` file
3. Rely on Docker layer caching to install dependencies
4. Assume "the latest version" is always compatible
5. Use different dependency-management approaches locally vs. in Docker

## Verification

### Check version consistency

```bash
# Local environment
pip list | grep -E "pyarrow|datasets"

# Docker environment
docker exec <container> pip list | grep -E "pyarrow|datasets"
```

### You should see matching versions
```
datasets    4.0.0
pyarrow     17.0.0
```

## Related files

- `pyproject.toml` - project configuration and dependency declarations
- `uv.lock` - the full locked dependency tree (version-controlled)
- `requirements.lock.txt` - pip-format export from uv.lock (used by Docker)
- `requirements.txt` - the original simple dependency list (kept for reference)

## References

- [UV Official Docs](https://github.com/astral-sh/uv)
- [Python Packaging Best Practices](https://packaging.python.org/)
- [Reproducible Builds Guide](https://reproducible-builds.org/)
