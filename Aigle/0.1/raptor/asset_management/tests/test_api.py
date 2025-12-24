import pytest
import httpx
import urllib.parse

# 配置區
BASE_URL = "http://localhost:8000"
USER_ID = "test_api_user"
BRANCH_ID = "test_api_user"

@pytest.fixture
def auth_headers():
    return {
        "X-User-ID": USER_ID,
        "X-Branch-ID": BRANCH_ID
    }

@pytest.mark.asyncio
async def test_full_asset_lifecycle(auth_headers):
    """
    測試完整生命週期：
    1. 上傳 (fileupload)
    2. 查詢 (filedownload)
    3. 新增關聯檔案 (add-associated-files)
    4. 更新過期時間 (file-expiration)
    5. 封存 (filearchive)
    6. 刪除 (delfile)
    """
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=20.0) as client:
        
        # --- 1. 上傳資產 (POST /fileupload) ---
        # 準備檔案
        files = [
            ("primary_file", ("test_primary.txt", b"hello world v1", "text/plain")),
            ("associated_files", ("sub_file.txt", b"fake_image_data", "text/plain"))
        ]
        # 準備表單數據 (依據 cURL 範例)
        data = {
            "archive_date_or_ttl": "30d",
            "destroy_date_or_ttl": "1y"
        }

        response = await client.post("/fileupload", headers=auth_headers, data=data, files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        asset_info = response.json()
        asset_path = asset_info["asset_path"]
        version_id = asset_info["version_id"]

        print(f"\n[Step 1] Uploaded: {asset_path} (Version: {version_id})")

        # --- 2. 查詢資產 (GET /filedownload/{path}/{version}) ---
        res_get = await client.get(
            f"/filedownload/{asset_path}/{version_id}", 
            headers=auth_headers,
            params={"return_file_content": "false"}
        )
        assert res_get.status_code == 200
        assert res_get.json()["metadata"]["asset_path"] == asset_path
        print(f"[Step 2] Retrieve metadata success.")

        # --- 3. 新增關聯檔案 (POST /add-associated-files/{path}) ---
        new_assoc_files = [
            ("associated_files", ("extra.txt", b"extra content", "plain/text")),
        ]
        res_assoc = await client.post(
            f"/add-associated-files/{asset_path}",
            headers=auth_headers,
            files=new_assoc_files
        )
        assert res_assoc.status_code == 200
        # 檢查檔案清單是否增加
        filenames = [f[0] for f in res_assoc.json()["associated_filenames"]]
        assert "extra.txt" in filenames
        print(f"[Step 3] Added associated file.")

        # --- 4. 更新過期策略 (POST /file-expiration/{path}/{version}) ---
        exp_data = {
            "archive_date_or_ttl": "2026-06-01",
            "destroy_date_or_ttl": "2028-01-01"
        }
        res_exp = await client.post(
            f"/file-expiration/{asset_path}/{version_id}",
            headers=auth_headers,
            data=exp_data
        )
        assert res_exp.status_code == 200
        print(f"[Step 4] Expiration updated.")

        # --- 5. 封存資產 (POST /filearchive/{path}/{version}) ---
        res_arc = await client.post(
            f"/filearchive/{asset_path}/{version_id}",
            headers=auth_headers
        )
        assert res_arc.status_code == 200
        assert res_arc.json()["status"] == "archived"
        print(f"[Step 5] Asset archived.")

        # --- 6. 測試 Scenario 4: 重複上傳已封存的路徑 (應該報 400) ---
        # 使用相同的內容和路徑再次上傳
        res_reject = await client.post("/fileupload", headers=auth_headers, data=data, files=files)
        assert res_reject.status_code == 400
        assert "archived with identical content" in res_reject.json()["detail"]
        print(f"[Step 6] Scenario 4 check success (Locked).")

        # --- 7. 刪除資產 (POST /delfile/{path}/{version}) ---
        res_del = await client.post(
            f"/delfile/{asset_path}/{version_id}",
            headers=auth_headers
        )
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "destroyed"
        print(f"[Step 7] Asset destroyed.")

@pytest.mark.asyncio
async def test_list_versions_and_commits(auth_headers):
    """測試列表與歷史紀錄 API"""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 1. 測試列表提交歷史
        res_commits = await client.get(
            "/users/commits",
            headers=auth_headers,
            params={"page": 1, "page_size": 10}
        )
        assert res_commits.status_code == 200
        assert "commits" in res_commits.json()
        print("\n[History] Get user commits success.")

        # 2. 測試列出檔案版本 (假設已知一個路徑和檔案)
        # 注意：這裡使用剛剛生命週期測試可能留下的路徑
        test_path = "document/txt/test_primary"
        test_file = "test_primary.txt"
        res_ver = await client.get(
            f"/fileversions/{test_path}/{test_file}",
            headers=auth_headers
        )
        assert res_ver.status_code == 200

@pytest.mark.asyncio
async def test_path_traversal_protection(auth_headers):
    """測試路徑安全檢查"""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        bad_path = "reports/../../etc/passwd"
        # 編碼以確保繞過 Uvicorn/FastAPI 的預設規範化過程
        encoded_bad_path = urllib.parse.quote(bad_path, safe='')
        
        response = await client.get(
            f"/filedownload/{encoded_bad_path}/latest",
            headers=auth_headers
        )
        # 應該觸發你的 AssetManager._sanitize_path 並回傳 400
        assert response.status_code == 400
        assert "Path traversal" in response.json()["detail"]
        print("\n[Security] Path traversal attack blocked.")

@pytest.mark.asyncio
async def test_scenario_3_global_active_hit(auth_headers):
    """
    測試 Scenario 3: 全域內容重用 (Knowledge Hit)
    上傳相同的內容到不同的路徑，系統應回傳第一次上傳的資產資訊。
    """
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=20.0) as client:
        content = b"unique content for scenario 3"
        files_1 = [("primary_file", ("first.txt", content, "text/plain"))]
        files_2 = [("primary_file", ("second.txt", content, "text/plain"))]
        
        # 1. 第一次上傳
        res1 = await client.post("/fileupload", headers=auth_headers, files=files_1)
        path1 = res1.json()["asset_path"]
        ver1 = res1.json()["version_id"]

        # 2. 第二次上傳相同內容 (不同檔名/預期路徑也會不同)
        res2 = await client.post("/fileupload", headers=auth_headers, files=files_2)
        assert res2.status_code == 200
        
        # 驗證：應該回傳第一次的路徑與版本 (Scenario 3)
        assert res2.json()["asset_path"] == path1
        assert res2.json()["version_id"] == ver1
        assert res2.json()["existence_info"]["exists"] is True
        print("\n[Scenario 3] Knowledge Hit success: Content reused across different uploads.")

@pytest.mark.asyncio
async def test_scenario_5_cross_path_reuse_archived(auth_headers):
    """
    測試 Scenario 5: 跨路徑重用已封存內容 (Clone Point)
    當 A 路徑已封存，上傳相同內容到 B 路徑，系統應允許上傳並回傳重用 A 路徑的資產的 qdrant 資訊。
    """
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=20.0) as client:
        content = b"unique content for scenario 5"
        
        # 1. 上傳到路徑 A
        files_a = [("primary_file", ("path_a.txt", content, "text/plain"))]
        res_a = await client.post("/fileupload", headers=auth_headers, files=files_a)
        path_a = res_a.json()["asset_path"]
        ver_a = res_a.json()["version_id"]

        # 2. 封存路徑 A
        await client.post(f"/filearchive/{path_a}/{ver_a}", headers=auth_headers)

        # 3. 上傳相同內容到另一路徑 (改名以產生不同路徑)
        files_b = [("primary_file", ("path_b_different.txt", content, "text/plain"))]
        res_b = await client.post("/fileupload", headers=auth_headers, files=files_b)
        
        assert res_b.status_code == 200
        assert res_b.json()["asset_path"] != path_a  # 路徑應該不同
        assert res_b.json()["existence_info"]["exists"] is True
        assert "Identical content found in another asset path and is archived" in res_b.json()["existence_info"]["message"]
        print(f"\n[Scenario 5] Cross-path reuse success: Archived content cloned to {res_b.json()['asset_path']}.")

@pytest.mark.asyncio
async def test_invalid_expiration_logic(auth_headers):
    """測試非法的過期時間邏輯"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=20.0) as client:
        files = [("primary_file", ("test_date.txt", b"date content", "text/plain"))]
        
        # 1. 測試：封存時間大於刪除時間 (應報 400)
        data_bad = {
            "archive_date_or_ttl": "2029-01-01",
            "destroy_date_or_ttl": "2028-01-01" # 比 archive_date_or_ttl 還早
        }
        res = await client.post("/fileupload", headers=auth_headers, data=data_bad, files=files)
        assert res.status_code == 400
        assert "destroy_date must be later than archive_date" in res.json()["detail"]

        # 2. 測試：封存時間為過去 (應報 400)
        data_past = {
            "archive_date_or_ttl": "2000-01-01"
        }
        res_past = await client.post("/fileupload", headers=auth_headers, data=data_past, files=files)
        assert res_past.status_code == 400
        assert "must be in the future" in res_past.json()["detail"]
        print("\n[Validation] Invalid dates blocked correctly.")

@pytest.mark.asyncio
async def test_illegal_lifecycle_transitions(auth_headers):
    """測試非法的生命週期狀態轉換"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=20.0) as client:
        # 1. 先上傳一個資產 (狀態為 active)
        files = [("primary_file", ("lifecycle.txt", b"lifecycle test", "text/plain"))]
        res = await client.post("/fileupload", headers=auth_headers, files=files)
        asset_path = res.json()["asset_path"]
        version_id = res.json()["version_id"]

        # 2. 嘗試刪除 active 資產 (應報 400，因為必須先封存)
        res_del = await client.post(f"/delfile/{asset_path}/{version_id}", headers=auth_headers)
        assert res_del.status_code == 400
        assert "is not archived" in res_del.json()["detail"]

        # 3. 封存它
        await client.post(f"/filearchive/{asset_path}/{version_id}", headers=auth_headers)

        # 4. 嘗試再次封存已封存的資產 (應報 400)
        res_arc_again = await client.post(f"/filearchive/{asset_path}/{version_id}", headers=auth_headers)
        assert res_arc_again.status_code == 400
        assert "already archived" in res_arc_again.json()["detail"]
        print("\n[Lifecycle] Illegal transitions blocked correctly.")

@pytest.mark.asyncio
async def test_missing_auth_headers():
    """測試缺少 Header 的情況"""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 不帶 X-User-ID 或 X-Branch-ID
        res = await client.get("/users/commits")
        # 預期 FastAPI 會因為 Header 必填而報 422
        assert res.status_code == 422
        print("\n[Security] Missing headers blocked.")

@pytest.mark.asyncio
async def test_add_assoc_to_non_existent_asset(auth_headers):
    """測試新增關聯檔案到不存在的資產"""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        bad_path = "ghost/path/123"
        files = [("associated_files", ("ghost.txt", b"ghost", "text/plain"))]
        res = await client.post(f"/add-associated-files/{bad_path}", headers=auth_headers, files=files)
        assert res.status_code == 404
        print("\n[Error Handling] Add associated files to non-existent asset returns 404.")