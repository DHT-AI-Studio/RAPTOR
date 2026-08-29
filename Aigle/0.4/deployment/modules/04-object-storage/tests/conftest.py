import pytest_asyncio
import logging


from asset_management.config import settings

# 更改為你的測試用設定
settings.postgres_host = "localhost"
settings.postgres_port = 5433
settings.lakefs_endpoint = "http://localhost:8001"
settings.s3_endpoint = "http://localhost:8333"
settings.qdrant_host = "localhost"
settings.qdrant_port = 6333
settings.qdrant_collection = "raptor"
settings.opensearch_host = "localhost"
settings.opensearch_port = 9200
settings.opensearch_user = "admin"
# opensearch_password intentionally NOT overridden here -- settings already
# loaded it from .env (Settings.model_config env_file=".env" in config.py)
# before this module's overrides run above, so it's already the real value.
# Hardcoding it here would just be a second, gitignore-blind copy of the
# same production secret.
settings.opensearch_index = "hybrid_index"
settings.opensearch_verify_certs = False

from asset_management.database import Database
from asset_management.object_store import ObjectStore
from asset_management.search_sync import SearchSync

# 設定測試用的分支名稱，必須與 test_api.py 一致
TEST_BRANCH = "test_api_user"

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
    print(f"\n\n[Cleanup] >>> Starting cleanup for branch: {TEST_BRANCH}...")

    # 初始化連線物件
    db = Database()
    obj = ObjectStore()
    sync = SearchSync()

    try:
        # 初始化所有連線
        await db.init_db()
        await obj.initialize()
        await sync.initialize()

        # --- 1. 清理 Qdrant + OpenSearch (依 asset 逐一刪除) ---
        # 新架構沒有 branch 欄位，改從 PostgreSQL 查出 TEST_BRANCH 的 assets 再刪除
        try:
            assets = await db.get_assets_by_branch(TEST_BRANCH)
            for asset in assets:
                await sync.delete_metadata(asset.asset_path, asset.version_id, TEST_BRANCH)
            print(f"[Cleanup] SearchSync: Removed {len(assets)} assets from Qdrant + OpenSearch.")
        except Exception as se:
            print(f"[Cleanup] SearchSync Error: {se}")

        # --- 2. 清理 PostgreSQL ---
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM commit_history WHERE branch = $1", TEST_BRANCH)
            await conn.execute("DELETE FROM audit_log WHERE branch = $1", TEST_BRANCH)
        print("[Cleanup] PostgreSQL: Removed commit_history and audit_log entries.")

        # --- 3. 清理 LakeFS branch (Object Store) ---
        try:
            obj.delete_branch(repository_id=settings.lakefs_repository, branch=TEST_BRANCH)
            print("[Cleanup] LakeFS: Test branch deleted.")
        except Exception as le:
            print(f"[Cleanup] LakeFS Error: {le}")

    except Exception as e:
        print(f"[Cleanup Failed] Error during teardown: {e}")
    finally:
        await db.close()
        await obj.close()
        await sync.close()
        print("[Cleanup] <<< Cleanup process finished.")