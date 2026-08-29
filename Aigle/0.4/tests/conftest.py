"""Shared fixtures for the Personal DB contract and isolation tests (VIE01-192).

These are integration tests: they run against a real ArcadeDB, a real PostgreSQL
and (for the Kafka cases) a real broker, because what they verify is that two
users' data is actually in separate places — a mocked store would prove nothing
about isolation.

Point them at running services with env vars and run from the repo root:

    PD_ARCADEDB_URL=http://localhost:2480 \
    PD_ARCADEDB_PASSWORD=$ARCADEDB_ROOT_PASSWORD \
    PD_POSTGRES_DSN=postgresql://raptor:$POSTGRES_PASSWORD@localhost:5433/personal_db \
    PD_KAFKA_BOOTSTRAP=localhost:19092 \
    pytest tests/ -v

Anything unreachable causes the tests that need it to skip, so the suite still
runs (and still means something) on a machine with only part of the stack up.
"""
from __future__ import annotations

import os
import sys
import pathlib

import pytest
import pytest_asyncio

# Module 25's app package — the service under test. Added to the path rather than
# installed because the modules are deployed as containers, not as libraries.
MODULE_25 = pathlib.Path(__file__).resolve().parents[1] / "deployment/modules/25-personal-db-service"
sys.path.insert(0, str(MODULE_25))

os.environ.setdefault("PD_KAFKA_ENABLED", "0")      # tests drive the consumer themselves
os.environ.setdefault("PD_ARCADEDB_PASSWORD", "changeme")

from app.services.arcadedb_client import ArcadeDBClient, db_name_for   # noqa: E402
from app.services.schema_init import initialize_schema                 # noqa: E402

USER_A = "contract-user-a"
USER_B = "contract-user-b"


async def _arcadedb_reachable(client: ArcadeDBClient) -> bool:
    try:
        await client.list_databases()
        return True
    except Exception:
        return False


@pytest.fixture
def client() -> ArcadeDBClient:
    return ArcadeDBClient()


@pytest_asyncio.fixture
async def arcade(client: ArcadeDBClient) -> ArcadeDBClient:
    if not await _arcadedb_reachable(client):
        pytest.skip("ArcadeDB not reachable (set PD_ARCADEDB_URL / PD_ARCADEDB_PASSWORD)")
    return client


@pytest_asyncio.fixture
async def two_users(arcade: ArcadeDBClient):
    """Two provisioned, empty databases — dropped afterwards whatever happens.

    Dropped on entry as well as exit: a previous run that died mid-test would
    otherwise leave data behind and turn "B cannot see A's document" into a
    result that depends on history.
    """
    for user in (USER_A, USER_B):
        name = db_name_for(user)
        if await arcade.database_exists(name):
            await arcade.drop_database(name)
        await arcade.create_database(name)
        await initialize_schema(arcade, name)

    yield USER_A, USER_B

    for user in (USER_A, USER_B):
        try:
            await arcade.drop_database(db_name_for(user))
        except Exception:
            pass


@pytest_asyncio.fixture
async def pg_pool():
    """PostgreSQL pool for the dedup/DLQ tables, or skip."""
    try:
        from app.services.audit import get_pool, close_pool
        pool = await get_pool()
        await pool.execute("SELECT 1")
    except Exception:
        pytest.skip("PostgreSQL not reachable (set PD_POSTGRES_DSN)")
    yield pool
    await close_pool()
