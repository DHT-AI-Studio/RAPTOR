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

## [0.0.0] - 2025-09-15
### Added
- Initial release of **Asset Management Service**.
- Multi-format asset upload with versioning support.
- JWT-based authentication with role-based access control (RBAC).
- Asset lifecycle management: automatic archive and destroy tasks.
- RESTful API endpoints:
  - `/token` – Authenticate and obtain JWT token.
  - `/users` – Create new user (admin only).
  - `/fileupload` – Upload primary and associated files.
  - `/add-associated-files/{asset_path}` – Add associated files to existing assets.
  - `/filedownload/{asset_path}/{version_id}` – Download asset metadata and optionally file content.
  - `/filearchive/{asset_path}/{version_id}` – Archive an asset.
  - `/delfile/{asset_path}/{version_id}` – Permanently destroy an archived asset.
  - `/fileversions/{asset_path}/{filename}` – List all versions of a file.
  - `/set-access-policy/{asset_path}/{version_id}` – Set access policy for an asset.
- Integration with **SeaweedFS** cluster for object storage (3 masters, 4 volumes, replication `011`).
- S3-compatible API via `seaweedfs-s3`.
- Integration with **lakeFS** for versioned object storage.
- Vector search support via **Qdrant**.
- MySQL database for storing metadata and user roles.
- Full observability with Prometheus, Grafana, Alertmanager, Node Exporter.
- NFS volume mounting with automated directory creation (`ensure_nfs_dirs.sh`).
- Docker Compose orchestration for all services.
- Automated daily scheduled jobs:
  - `auto_archive_assets` at 00:00.
  - `auto_destroy_assets` at 01:00.
  - `lakefs_gc` at 02:00.
- Centralized backup of SeaweedFS filer data.
- Implemented MCP server for asset management API, providing a unified interface for agents to interact with the service.

### Changed
- N/A

### Fixed
- N/A

---

## [0.1.0] - 2025-09-19
### Added
- Support for multi-user, branch-based asset management.
- Added functionality to separate branches based on owner name (e.g., username: `Alice` → branch: `Alice_space`).
- Support for two usage scenarios: individual usage or collaborative usage (admin creates a branch space and can create shared users with configurable permissions).
- Added ability to manage shared users (create, update, delete) under a branch.
- Added `TIMEZONE` in `.env` to unify time definition across services.

### Changed
- Adjusted internal service initialization process and refactored part of the logic to support multi-user / multi-branch functionality.
- Updated API logic:
  
  | Endpoint        | Method | JWT Auth | Description |
  |-----------------|--------|------|-------------|
  | `/users`        | POST   | ❌   | Create a new user (initial admin / branch owner). Branch is automatically assigned. |
  | `/shared-users` | POST   | ✅   | Create a shared user under the current user's branch. |
  | `/shared-users` | DELETE | ✅   | Delete a shared user under the current user's branch. |
  | `/shared-users` | PUT    | ✅   | Update a shared user's permissions under the current user's branch. |

- Removed unused `/set-access-policy` endpoint as file-level access is now governed by user permissions instead of per-file policies.
- Docker Compose now pins explicit image versions instead of using `latest`, ensuring reproducible deployments and preventing unexpected upgrades:
  - `chrislusf/seaweedfs:3.96`
  - `qdrant/qdrant:v1.15.1`
  - `mysql:8.0`
  - `prom/prometheus:2.48.0`
  - `prom/alertmanager:v0.28.1`
  - `prom/node-exporter:v1.9.1`
  - `grafana/grafana:12.0.2`
  - `treeverse/lakefs:1.65.2`
  - `app` (built from local Dockerfile)
  - `lakefs-gc-cron` (built from `./docker_compose_settings/lakefs/gc-runner`)
- `.env` no longer includes `ADMIN_USERNAME` / `ADMIN_PASSWORD`. Admin users are not auto-created; they must now be created explicitly via the `/users` API.

### Fixed
- N/A

---

## [0.1.1] - 2025-10-09
### Added
- Added custom Dockerfile and entrypoint script for Alertmanager to support environment-variable-based configuration.

### Changed
- Changed the healthcheck method in docker compose for lakefs.
- Removed unused imports for cleaner code.
- Updated `lakefs-gc-cron` Dockerfile to use `apache/spark:3.5.7` instead of deprecated `bitnami/spark:3.5`.
- Alertmanager configuration now uses environment variables instead of static YAML values for SMTP credentials and email recipients.
- Updated docker-compose for Alertmanager to use build context instead of image.
- Auto-daily archive and destroy times are now configurable via environment variables and read through `config.py`, instead of being hardcoded to `00:00` and `01:00`.
- Temporarily disabled Qdrant write operations in `client.py`. Only `archive` and `destroy` actions remain active, as real-world scenarios typically delay Qdrant storage until additional metadata is available.
- Commented out the `qdrant` service in `docker-compose.yaml`; in real deployments, the application should connect to an **external Qdrant instance** instead.  
  For testing Qdrant connectivity, the service can still be started locally, and the write operations in `client.py` can be re-enabled.

### Fixed
- N/A

---

## [0.1.2] - 2025-10-17

### Added
- **Documentation:** Added a **System Architecture Diagram** to `README.md` to provide a high-level overview of the microservices, data flow, and technologies used (e.g., FastAPI, LakeFS, Qdrant, MySQL, SeaweedFS).

### Changed
- **Documentation:** Updated the **API Data Flow Diagrams** in `README.md`.

### Fixed
- N/A

---

## [0.1.3] - 2025-10-22

### Added
- N/A

### Changed
- **Documentation:** Updated the **User Management Data Flow Diagram** in `README.md`.

### Fixed
- N/A

---