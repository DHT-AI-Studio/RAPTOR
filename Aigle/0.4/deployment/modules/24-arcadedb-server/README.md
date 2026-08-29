# Module 24 — arcadedb-server

**Status:** Merged from `feature/v0.4-personal-db-24-25` (VIE01-188).

ArcadeDB multi-model DB server — physical per-user database isolation (vector + graph + BM25 in one engine). One database per user, named `user_{branch_id}`, created and managed by Module 25 (Personal DB Service).

**Key dependencies:** —

---

## Deploy

```bash
bash deploy.sh -m 24
```

`ARCADEDB_ROOT_PASSWORD` must be set in `deployment/modules/.env` first — the compose file has no default and will refuse to start without it.

| Variable | Default | Purpose |
|---|---|---|
| `PORT_ARCADEDB_HTTP` | `2480` | HTTP API — Module 25 connects here; also serves ArcadeDB Studio |
| `PORT_ARCADEDB_BINARY` | `2424` | Binary protocol (optional) |
| `ARCADEDB_ROOT_PASSWORD` | *(none)* | **Required.** No default by design |

Verify:

```bash
curl -i http://localhost:2480/api/v1/ready     # expect 204
docker exec raptor-arcadedb ls /home/arcadedb/databases
```

---

## Four things the upstream docs get wrong

All four were verified against `arcadedata/arcadedb:latest` (**26.6.1**). Each one costs an hour if you hit it cold, so they are recorded here rather than only in the compose file.

### 1. The database directory is `/home/arcadedb/databases`

Not `/arcadedb-ce/databases`. That path belongs to an older community-edition image layout and **does not exist** in the current image:

```
$ docker exec raptor-arcadedb ls -d /arcadedb-ce/databases
ls: /arcadedb-ce/databases: No such file or directory
```

Mounting the volume at the wrong path is silently destructive: the container happily creates an empty directory there, ArcadeDB writes to the real path inside the container layer, and **every database is lost on restart**.

### 2. The root password must go through `JAVA_OPTS`

```yaml
JAVA_OPTS: -Darcadedb.server.rootPassword=${ARCADEDB_ROOT_PASSWORD}
```

The documented `ARCADEDB_SERVER_ROOTPASSWORD` environment variable is **not honoured** by this image. Setting it instead of `JAVA_OPTS` leaves the server waiting for a password on stdin, so **startup hangs** with no error — the container stays up but never becomes ready.

### 3. `GET /api/v1/ready` returns **204**, not 200

```
HTTP/1.1 204 No Content
Content-Type: application/json
```

204 is a success (`curl -f` and `wget` both treat any 2xx as OK), so the healthcheck works. But a check written as `status == 200` will report a healthy server as down.

### 4. The image ships `wget`, not `curl`

The healthcheck uses `wget -q -O /dev/null`. A `curl`-based healthcheck fails with "executable not found" and the container never reports healthy.

---

## Notes

- **Development mode** is intentional: it keeps ArcadeDB Studio available at `:2480`. Production mode disables Studio and `LOAD CSV`; switch if you need those durability/security defaults.
- **Data volume** `arcadedb_data` is declared in this module's compose file and persists across `deploy.sh --stop`. `deploy.sh --delete` removes it along with every user database.
