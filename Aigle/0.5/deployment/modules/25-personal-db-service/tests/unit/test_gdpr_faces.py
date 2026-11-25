"""GD-11: DELETE /internal/gdpr/faces/user/{user_id} -- Module 33's cascade when a
user withdraws face_detection consent. Deletes the face layer only, never the
database or its content index.

Pure unit test: ArcadeDB is replaced by a fake that holds a vertex count per type
and applies DELETE VERTEX FROM <type> to it, so it runs anywhere.
"""
from __future__ import annotations

import pytest
from app.core.config import settings
from app.routers import gdpr as gdpr_router
from app.services import identity_store
from fastapi import FastAPI
from fastapi.testclient import TestClient

TOKEN = "test-internal-token"
FACE_TYPES = {"Face": 7, "FaceTrace": 5, "FaceGroup": 3, "Identity": 2, "IdentityHistory": 1}
CONTENT_TYPES = {"Chunk": 40, "Source": 4, "Entity": 9}


class _FakeArcade:
    def __init__(self, exists: bool = True, face_layer: bool = True):
        self.exists = exists
        self.vertices = {**CONTENT_TYPES, **(FACE_TYPES if face_layer else {})}
        self.commands: list[str] = []
        self.dropped = False

    async def database_exists(self, db):
        return self.exists

    async def drop_database(self, db):
        self.dropped = True

    async def query(self, db, sql, params=None):
        if "FROM schema:types" in sql:
            return [{"name": "FaceTrace"}] if "FaceTrace" in self.vertices else []
        raise AssertionError(f"unexpected query: {sql}")

    async def command(self, db, sql, params=None):
        self.commands.append(sql)
        vtype = sql.removeprefix("DELETE VERTEX FROM ").strip()
        assert sql.startswith("DELETE VERTEX FROM ") and " " not in vtype, sql
        n, self.vertices[vtype] = self.vertices.get(vtype, 0), 0
        return [{"count": n}]


@pytest.fixture
def client_for(monkeypatch):
    monkeypatch.setattr(settings, "gdpr_internal_token", TOKEN)
    monkeypatch.setattr(identity_store, "_face_schema_ready", set())

    def _make(fake: _FakeArcade) -> TestClient:
        monkeypatch.setattr(gdpr_router, "client", fake)
        app = FastAPI()
        app.include_router(gdpr_router.router)
        return TestClient(app)

    return _make


def _delete(client: TestClient, user_id: str = "u1", token: str | None = TOKEN):
    headers = {"X-Internal-Token": token} if token else {}
    return client.delete(f"/internal/gdpr/faces/user/{user_id}", headers=headers)


@pytest.mark.parametrize("token", [None, "wrong"])
def test_requires_internal_token(client_for, token):
    fake = _FakeArcade()
    assert _delete(client_for(fake), token=token).status_code == 401
    assert fake.commands == []


def test_deletes_every_face_vertex_and_nothing_else(client_for):
    fake = _FakeArcade()
    r = _delete(client_for(fake))
    assert r.status_code == 200
    assert r.json() == {
        "user_id": "u1", "database": "user_u1", "db_exists": True,
        "deleted": {"identity_history": 1, "identities": 2, "face_groups": 3, "face_traces": 5, "faces": 7},
    }
    assert all(fake.vertices[t] == 0 for t in FACE_TYPES)
    # The content index stays, and the database is never dropped (that's GD-1).
    assert {t: fake.vertices[t] for t in CONTENT_TYPES} == CONTENT_TYPES
    assert fake.dropped is False


def test_is_idempotent_so_module_33_can_retry(client_for):
    fake = _FakeArcade()
    client = client_for(fake)
    _delete(client)
    r = _delete(client)
    assert r.status_code == 200 and set(r.json()["deleted"].values()) == {0}


def test_database_without_face_layer_deletes_nothing(client_for):
    """A database created before the face layer has no Face types to delete from --
    a DELETE naming one would fail."""
    fake = _FakeArcade(face_layer=False)
    r = _delete(client_for(fake))
    assert r.status_code == 200 and set(r.json()["deleted"].values()) == {0}
    assert fake.commands == []


def test_unprovisioned_user_is_zero_not_404(client_for):
    fake = _FakeArcade(exists=False)
    r = _delete(client_for(fake), "ghost")
    assert r.status_code == 200
    assert r.json()["db_exists"] is False and set(r.json()["deleted"].values()) == {0}
    assert fake.commands == []
