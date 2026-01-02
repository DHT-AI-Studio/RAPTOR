# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## Versioning Policy `[MAJOR.MINOR.PATCH]`

### Before `1.0.0` (`0.x.x`)
- **API breaking changes** → **MINOR +1**  
- **New features (backward-compatible)** → **MINOR +1**  
- **Bug fixes / small internal changes** → **PATCH +1**

### After `1.0.0`
- **Backward-incompatible API changes or major architectural redesigns.** → **MAJOR +1**
- **New features or enhancements, backward-compatible.** → **MINOR +1**   
- **Bug fixes, documentation updates, dependency bumps, small internal changes.** → **PATCH +1**

---

## [0.0.0] - 2025-08-06

### Added
- Initial project upload and setup of the basic Redis caching structure.

### Changed
- N/A

### Fixed
- N/A

---

## [0.1.0] - 2025-10-09

### Added
- **Ollama Support for Semantic Embeddings.**
  - Introduced the `ollama_url` parameter to **`SemanticRedisCache`**, allowing the use of an external Ollama server for generating vector embeddings.
  - The system will **fall back to `SentenceTransformer`** when `ollama_url` is not provided.

### Changed
- **Redis Cluster Configuration Hardening.**
  - Updated default Redis Cluster configuration files to include **memory-related settings** (e.g., `maxmemory` and `maxmemory-policy`), enhancing stability and performance in production environments.
- **Docker Image Version Pinning.**
  - All project Docker images now specify explicit version tags instead of using `:latest`, improving build **reproducibility and stability**.
- **Docker Compose Secrets Management.**
  - Refactored the Redis Cluster Docker Compose YAML to move sensitive data (such as passwords) into the **`.env`** file.
- **Centralized Secrets Management in Docker Configs.**
  - Hardcoded passwords across Docker configurations (both in `./master_slave_replication` and `./redis_cluster` directories) have been replaced with the **`<YOUR_REDIS_PASSWORD>` placeholder**. This centralizes password management to the project's `.env` files, eliminating secrets from version-controlled configuration files.

### Fixed
- **Codebase Clean-up.**
  - Removed **unnecessary Python `import` statements** across several files, improving code clarity and reducing clutter.

---

## [0.2.0] - 2025-10-16

### Added

* **Distributed Lock Mechanism via Redis.**

  * Introduced a new module **`distributed_lock.py`** implementing:

    * `RedisLock` — synchronous Redis-based distributed lock.
    * `AsyncRedisLock` — asynchronous variant supporting asyncio-based workflows.
  * Both ensure **atomic acquisition** and **safe release** using Lua scripting.
* **`CacheManagerDistLock` Implementation.**

  * Added a new cache manager class that leverages `RedisLock`/`AsyncRedisLock` for distributed environments.
  * Functionally equivalent to the original `CacheManager`, but supports **multi-instance coordination** across containers or servers.
  * Enables Redis to act as both a cache and a temporary lock registry.

### Changed

* **CacheManager Double-Check Enhancement.**

  * Updated the existing `CacheManager` to include a **second cache existence check** after acquiring the lock.
  * Prevents unnecessary recomputation and **mitigates cache breakdown** under high concurrency.

### Fixed

* N/A

---

## [0.3.0] - 2025-10-31 

### Added
* **Dual Mode Docker Compose Configuration in `./redis_cluster`.**
    * Introduced two separate Docker Compose files (`docker-compose.public.yml` and `docker-compose.internal.yml`) to support distinct networking strategies.
    * Added corresponding environment variable templates (`.env.public.example`, `.env.internal.example`) for deployment configuration.
* **Dedicated Environment Variable Loading.**
    * Deployment now explicitly uses the `--env-file` flag (`.env.public` or `.env.internal`) to ensure correct variable substitution during YAML parsing.

### Changed
* **Refactored Redis Cluster Port Mapping.**
    * The configuration is now split to manage external accessibility:
        * **Public Mode:** Maps all Redis data/cluster ports (`7000-7005` and `17000-17005`) to the host.
        * **Internal Mode:** **Removes mapping** for Redis data/cluster ports, isolating them within the Docker network.
* **Standardized RedisInsight Port Exposure.**
    * `RedisInsight` port (`5540`) is **now consistently mapped** to the host in *both* Public and Internal modes, ensuring monitoring is always accessible.
* **Updated Documentation (`README.md`).**
    * Completely revised the README to detail the **two deployment modes**, provide clear **setup instructions**, and explain the specific **connection requirements for RedisInsight** in each mode.
* **Simplified Cluster Testing.**
    * Updated cluster verification instructions to primarily use the **RedisInsight CLI and Browser** instead of relying on the host-installed `redis-cli`.

### Fixed
* N/A

---

## [0.4.0] - 2026-01-02

### Added
- **Integrated Distributed Locking.**
    - Merged the distributed lock logic directly into the main `CacheManager` class. Users can now toggle this feature via the `use_distributed_lock` parameter without switching classes.
- **Model Instance Injection.**
    - Added the `embedding_model_instance` parameter to `CacheManager` and `SemanticRedisCache`, allowing the reuse of pre-initialized `SentenceTransformer` objects to save memory.
- **Resource Management Lifecycle.**
    - Introduced `close()` and `aclose()` methods to gracefully shut down Redis connection pools and background cleanup tasks.
- **Lock Watchdog (Auto-Extension).**
    - Distributed locks now feature an automatic extension mechanism (Watchdog) to prevent locks from expiring during long-running cache-miss functions.
- **Namespace Isolation for Semantic Caching.**
    - Implemented a hashing mechanism that separates cache entries based on non-query arguments, preventing "Apple" in a `fruit` category from hitting an "Apple" entry in a `tech` category.
- **Dynamic Semantic Query Mapping.**
    - Introduced the `query_param_name` parameter to the `@cache` decorator and `CacheManager`. This allows users to designate any function argument as the source for semantic embedding, rather than being restricted to the name `"query"`.
    - Integrated `inspect.signature` for robust argument binding. The system now accurately identifies the designated query value whether it is passed as a positional or keyword argument.
    
### Changed
- **Unified Cache Manager Logic.**
    - Simplified the codebase by removing the separate `CacheManagerDistLock` class and unifying sync/async wrapper logic.
- **Robust Semantic Index Management.**
    - `SemanticRedisCache` now automatically detects vector dimension mismatches (e.g., when switching models) and recreates the RediSearch index if necessary.
- **Enhanced Background Cleanup.**
    - The background task now cleans up both `hit_counter` and `local_locks` for expired keys, preventing memory bloat in long-running processes.
- **Improved Metadata Handling.**
    - Key generation now filters out non-serializable parameters to prevent `pickle` errors during metadata hashing.

### Fixed
- **Resource Leaks.**
    - Fixed potential memory leaks by using `weakref.WeakValueDictionary` for tracking in-progress tasks.
- **Thread Safety.**
    - Refactored internal locking mechanisms to ensure consistent behavior between synchronous and asynchronous environments.

---