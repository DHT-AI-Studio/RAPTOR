import pytest_asyncio
import logging


from asset_management.config import settings

# 更改為你的測試用設定
settings.postgres_host = "localhost"
settings.postgres_port = 5433
settings.lakefs_endpoint = "http://localhost:8001"
settings.s3_endpoint = "http://localhost:8333"
settings.qdrant_host = "localhost"
settings.qdrant_port = 6334

from asset_management.database import Database
from asset_management.object_store import ObjectStore
from asset_management.vector_store import VectorStore

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
    vec = VectorStore()

    try:
        # 初始化所有連線 (會自動讀取剛剛 load_dotenv 載入的變數)
        await db.init_db()
        await obj.initialize()
        await vec.initialize()


        # --- 1. 清理 PostgreSQL ---
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM commit_history WHERE branch = $1", TEST_BRANCH)
            await conn.execute("DELETE FROM audit_log WHERE branch = $1", TEST_BRANCH)
        print("[Cleanup] PostgreSQL: Removed commit_history and audit_log entries.")

        # --- 2. 清理 Qdrant (Vector Store) ---
        # 根據 branch 標籤刪除點
        try:
            from qdrant_client import models
            
            for collection in vec.collection_name:
                await vec.client.delete(
                    collection_name=collection,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="branch",
                                match={"value": TEST_BRANCH}
                            )
                        ]
                    )
                )

            print("[Cleanup] Qdrant: Removed vector points.")
        except Exception as ve:
            print(f"[Cleanup] Qdrant Error: {ve}")

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
        await vec.close()
        print("[Cleanup] <<< Cleanup process finished.")