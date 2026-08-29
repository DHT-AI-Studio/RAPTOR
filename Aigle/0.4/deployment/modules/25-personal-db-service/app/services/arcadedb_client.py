"""
Async ArcadeDB HTTP API client wrapper (Module 24).

Verified against arcadedata/arcadedb:latest (26.6.1):
  - create/drop database : POST /api/v1/server  {"command": "create database <db>"}
  - list databases       : GET  /api/v1/databases  -> {"result": [<names>]}
  - SQL query (read)     : POST /api/v1/query/<db>    {"language":"sql","command": "..."}
  - SQL command (write)  : POST /api/v1/command/<db>  {"language":"sql","command": "..."}
  - creating an existing database returns an error -> callers check existence first.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import httpx

from app.core.config import settings

logger = logging.getLogger("personal_db.arcadedb")


def db_name_for(branch_id: str) -> str:
    """Map a user id (the JWT `sub`, often a UUID) to its per-user database name.

    Alphanumeric + underscore only. The database name is interpolated into SQL /
    Gremlin statements and into the ArcadeDB REST path, so anything outside
    [A-Za-z0-9_] — including the dashes a UUID `sub` carries — is collapsed to
    `_`. That leaves no character that could terminate a string, a statement, or
    a path segment, so no injection is possible via the identifier.

    The mapping is not injective (`a-b` and `a_b` both give `user_a_b`), which is
    safe only because `sub` is a Keycloak UUID — fixed-position dashes, hex
    elsewhere, so no two real subs collide.
    """
    s = re.sub(r"[^A-Za-z0-9_]", "_", (branch_id or "").strip())
    s = s.strip("_") or "anon"
    return f"user_{s}"


class ArcadeDBClient:
    def __init__(self) -> None:
        self.base = settings.arcadedb_url.rstrip("/")
        self.auth = (settings.arcadedb_user, settings.arcadedb_password)
        self.timeout = settings.http_timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(auth=self.auth, timeout=self.timeout)

    # ---------------------------------------------------------------- server
    async def list_databases(self) -> List[str]:
        async with self._client() as c:
            r = await c.get(f"{self.base}/api/v1/databases")
            r.raise_for_status()
            return r.json().get("result", [])

    async def database_exists(self, db: str) -> bool:
        return db in await self.list_databases()

    async def create_database(self, db: str) -> None:
        async with self._client() as c:
            r = await c.post(f"{self.base}/api/v1/server",
                             json={"command": f"create database {db}"})
            if r.status_code >= 400 and "already exists" not in r.text.lower():
                r.raise_for_status()

    async def drop_database(self, db: str) -> None:
        async with self._client() as c:
            r = await c.post(f"{self.base}/api/v1/server",
                             json={"command": f"drop database {db}"})
            low = r.text.lower()
            if r.status_code >= 400 and "not found" not in low and "does not exist" not in low:
                r.raise_for_status()

    # ---------------------------------------------------------------- db ops
    async def command(self, db: str, sql: str,
                      params: Dict[str, Any] | None = None,
                      ignore_exists: bool = False) -> List[Dict[str, Any]]:
        """Run a write SQL statement.

        Pass ``params`` to use named bindings (``:name`` in the SQL) instead of
        interpolating values into the string — this is injection-safe and the
        required way to store untrusted content (document text, ASR transcripts).
        """
        payload: Dict[str, Any] = {"language": "sql", "command": sql}
        if params:
            payload["params"] = params
        async with self._client() as c:
            r = await c.post(f"{self.base}/api/v1/command/{db}", json=payload)
            if r.status_code >= 400:
                if ignore_exists and "already exists" in r.text.lower():
                    return []
                logger.warning(f"[arcade] command failed on {db}: {sql[:60]}… → {r.text[:160]}")
                r.raise_for_status()
            return r.json().get("result", [])

    async def query(self, db: str, sql: str,
                    params: Dict[str, Any] | None = None,
                    language: str = "sql") -> List[Dict[str, Any]]:
        """Run a read query. Supports the same named-binding mechanism for both
        languages -- SQL uses ``:name`` in the query text, Cypher uses ``$name``
        -- either way, bindings go in ``params`` (e.g. for vector-search query
        vectors, or Cypher subgraph traversal in graph_query.py).

        language="cypher": ArcadeDB's own Cypher dialect, used for multi-edge-
        type variable-length subgraph patterns (verified live: `[:A|B*1..n]`,
        `path`/`relationships(path)`, `OPTIONAL MATCH` + `collect(DISTINCT ...)`
        all work) -- no APOC-equivalent SQL rewrite needed for that."""
        payload: Dict[str, Any] = {"language": language, "command": sql}
        if params:
            payload["params"] = params
        async with self._client() as c:
            r = await c.post(f"{self.base}/api/v1/query/{db}", json=payload)
            r.raise_for_status()
            return r.json().get("result", [])
