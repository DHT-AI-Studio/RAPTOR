import logging
import os

import lakefs
import pytest_asyncio
from asset_management.config import settings


# This module's own .env carries placeholder credentials
# (POSTGRES_USER=admin, 10-char dummy LakeFS keys) and docker-internal
# hostnames/ports (postgres:5432, lakefs:8000) -- none of that is valid
# from a host-side test process. The real credentials, and the ports
# docker-compose actually forwards to the host, live in the unified
# deployment/modules/.env that build.py passes to every container's
# environment. Parsed once here instead of hardcoding magic port numbers
# or re-deriving credentials per test file.
def _load_unified_env():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()  # strip inline comments
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    # Resolve ${VAR}-style self-references (e.g. AWS_ACCESS_KEY=${AWS_ACCESS_KEY_ID}),
    # which docker-compose expands at container-start time but a plain line
    # parser doesn't. One pass is enough -- nothing in this file chains
    # references more than one level deep.
    for k, v in env.items():
        if v.startswith("${") and v.endswith("}"):
            env[k] = env.get(v[2:-1], v)
    return env


_unified = _load_unified_env()

# Host-visible forwarded ports (falls back to today's known-good values if
# the unified .env isn't present, e.g. CI without a live docker-compose stack)
_host = _unified.get("HOST_IP", "localhost")
settings.postgres_host = _host
settings.postgres_port = int(_unified.get("PORT_POSTGRES", settings.postgres_port))
settings.lakefs_endpoint = f"http://{_host}:{_unified.get('PORT_LAKEFS', '8001')}"
settings.s3_endpoint = f"http://{_host}:{_unified.get('PORT_SEAWEEDFS_S3', '8333')}"
# Defaults to the docker-internal hostname (raptor-personal-db-service),
# unreachable from this host-side test process -- same class of mismatch
# as MODULE04_BASE_URL below, just for Module 25 instead of Module 04.
settings.personal_db_service_url = f"http://{_host}:{_unified.get('PORT_PERSONAL_DB', '8025')}"

# Module 04's own FastAPI service -- test_asset_lifecycle.py's real HTTP
# calls need this too, not just the direct-client tests above. Was
# hardcoded to localhost:8000, so a run with HOST_IP pointed at a remote
# deploy host still silently hit whatever Module 04 happens to be running
# on localhost instead -- real upload succeeds there, but any assertion
# that then reads the result back via a direct client (LakeFS commit log,
# ArcadeDB, etc., which DO follow _host) is comparing two different hosts
# and fails with a confusing "not found", not an HTTP/connection error.
MODULE04_BASE_URL = f"http://{_host}:{_unified.get('PORT_ASSET_MANAGEMENT', '8000')}"

# Module 04's own presign-cache Redis -- test_presign_batch_many_files.py
# connects directly (not through settings.redis_host, the docker-internal
# hostname) to clear presign:* keys before each case, same reason as
# MODULE04_BASE_URL above.
REDIS_URL = f"redis://:{_unified.get('REDIS_PASSWORD', '')}@{_host}:{_unified.get('PORT_REDIS_STANDALONE', '6379')}/{_unified.get('REDIS_DB', '0')}"

# Direct ArcadeDB access -- Module 25's own HTTP API has no endpoint that
# returns a Chunk/Source's actual status string (only /exists, a bool), so
# verifying clone_point()'s status="cloned" vs register_asset()'s flip to
# "active" needs to bypass Module 25 and query ArcadeDB directly, the same
# way a human would via its Studio UI on this same port.
ARCADEDB_URL = f"http://{_host}:{_unified.get('PORT_ARCADEDB_HTTP', '2480')}"
ARCADEDB_PASSWORD = _unified.get("ARCADEDB_ROOT_PASSWORD", "")

# Real credentials -- postgres_db intentionally NOT overridden, this module
# owns its own "asset_management" database on the shared postgres server,
# not the unified file's generic POSTGRES_DB=raptor.
if "POSTGRES_USER" in _unified:
    settings.postgres_user = _unified["POSTGRES_USER"]
if "POSTGRES_PASSWORD" in _unified:
    settings.postgres_password = _unified["POSTGRES_PASSWORD"]
if "LAKEFS_ACCESS_KEY" in _unified:
    settings.lakefs_access_key = _unified["LAKEFS_ACCESS_KEY"]
if "LAKEFS_SECRET_KEY" in _unified:
    settings.lakefs_secret_key = _unified["LAKEFS_SECRET_KEY"]
if "S3_BUCKET" in _unified:
    settings.s3_bucket = _unified["S3_BUCKET"]
if "AWS_ACCESS_KEY" in _unified:
    settings.aws_access_key = _unified["AWS_ACCESS_KEY"]
if "AWS_SECRET_KEY" in _unified:
    settings.aws_secret_key = _unified["AWS_SECRET_KEY"]

from asset_management.database import Database
from asset_management.object_store import ObjectStore
from asset_management.search_sync import SearchSync

# 設定測試用的分支名稱，必須與 test_asset_lifecycle.py 一致
TEST_BRANCH = "test_api_user"
# test_commits_status_filter.py's test_status_filter_reflects_real_mtr3_writeback
# drives a real upload through the live API under this same branch name --
# unlike its other tests, which only insert directly into Postgres and clean
# up via their own db fixture (and never touch LakeFS), this one leaves a
# real LakeFS branch behind (its own commit_history rows are already
# self-cleaned in the test's own finally block), so it needs to be swept
# here too.
COMMITS_FILTER_BRANCH = "test_commits_status_filter_branch"
# test_clone_point_real_arcadedb.py's real clone_point()/register_asset()
# round trip -- its own fixture only tears down the Module 25/ArcadeDB side,
# the Postgres/LakeFS/SeaweedFS side is swept here like every other branch.
CLONE_ARCADEDB_BRANCH = "test_clone_point_real_arcadedb_branch"
# test_presign_batch_many_files.py's own branch -- uploads an asset with
# many associated files, left for this global teardown to sweep like
# every other branch above.
PRESIGN_BATCH_BRANCH = "test_presign_batch_many_files_branch"

logger = logging.getLogger(__name__)

@pytest_asyncio.fixture(scope="session", autouse=True)
async def global_teardown():
    """
    全域清理 Fixture。
    scope="session": 整個測試期間只執行一次。
    autouse=True: 自動執行，不需在測試函式中呼叫。
    """
    yield

    # 測試結束後執行 (Teardown)
    branches_to_clean = [TEST_BRANCH, COMMITS_FILTER_BRANCH, CLONE_ARCADEDB_BRANCH, PRESIGN_BATCH_BRANCH]
    print(f"\n\n[Cleanup] >>> Starting cleanup for branches: {branches_to_clean}...")

    # 初始化連線物件
    db = Database()
    obj = ObjectStore()
    sync = SearchSync()

    try:
        # 初始化所有連線
        await db.init_db()
        await obj.initialize()
        await sync.initialize()

        for branch in branches_to_clean:
            # --- 1. 清理 Module 25 (ArcadeDB) 索引 (依 asset 逐一刪除) ---
            # 沒有 branch 欄位，改從 PostgreSQL 查出該 branch 的 assets 再刪除
            try:
                assets = await db.get_assets_by_branch(branch)
                for asset in assets:
                    await sync.delete_metadata(asset.asset_path, asset.version_id, branch)
                print(f"[Cleanup] SearchSync: Removed {len(assets)} assets from Module 25 index for {branch}.")
            except Exception as se:  # noqa: BLE001 -- test teardown; one subsystem's cleanup failure must not block the rest, printed
                print(f"[Cleanup] SearchSync Error for {branch}: {se}")

            # --- 2. 清理 PostgreSQL ---
            async with db.pool.acquire() as conn:
                await conn.execute("DELETE FROM commit_history WHERE branch = $1", branch)
                await conn.execute("DELETE FROM audit_log WHERE branch = $1", branch)
            print(f"[Cleanup] PostgreSQL: Removed commit_history and audit_log entries for {branch}.")

            # --- 3. Hard-delete SeaweedFS physical objects ---
            # LakeFS branch delete only removes the ref -- reclaiming the
            # underlying physical object is normally GC's job, but this
            # environment's GC (needs Spark) doesn't clean up SeaweedFS
            # correctly, so it can't be relied on. Before deleting the
            # branch, resolve each file still on it to its physical address
            # (same as client.py's destroy_v2() / gdpr_erase_user() and
            # object_store.py's bulk_delete_files_v2()) and hard-delete them in one chunked bulk
            # DeleteObjects call -- the same batching lakeFS's own GC sweep
            # step uses, instead of one delete_object round-trip per key.
            try:
                branch_ref = lakefs.repository(repository_id=settings.lakefs_repository, client=obj.client).branch(branch)
                listed = list(branch_ref.objects())
                physical_addresses = []
                for o in listed:
                    try:
                        physical_addresses.append(
                            branch_ref.object(path=o.path).stat().physical_address.replace(f"s3://{settings.s3_bucket}/", "")
                        )
                    except Exception as fe:  # noqa: BLE001 -- test teardown; one object's resolve failure must not block the rest, printed
                        print(f"[Cleanup] SeaweedFS Error resolving {o.path} for {branch}: {fe}")
                result = await obj._bulk_delete_from_s3_bucket(physical_addresses)
                print(f"[Cleanup] SeaweedFS: Bulk-deleted {result['deleted']}/{len(listed)} physical objects for {branch} ({result['errors']} errors).")
            except Exception as le:  # noqa: BLE001 -- test teardown; one subsystem's cleanup failure must not block the rest, printed
                print(f"[Cleanup] SeaweedFS Error listing objects for {branch}: {le}")

            # --- 4. 清理 LakeFS branch (Object Store) ---
            try:
                obj.delete_branch(repository_id=settings.lakefs_repository, branch=branch)
                print(f"[Cleanup] LakeFS: Branch {branch} deleted.")
            except Exception as le:  # noqa: BLE001 -- test teardown; one subsystem's cleanup failure must not block the rest, printed
                print(f"[Cleanup] LakeFS Error for {branch}: {le}")

    except Exception as e:  # noqa: BLE001 -- test teardown; outer net so the session fixture never errors out of cleanup, printed
        print(f"[Cleanup Failed] Error during teardown: {e}")
    finally:
        await db.close()
        await obj.close()
        await sync.close()
        print("[Cleanup] <<< Cleanup process finished.")