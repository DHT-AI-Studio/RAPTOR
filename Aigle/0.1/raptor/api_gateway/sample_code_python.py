"""
RAPTOR API Gateway - Python Sample Code
========================================
這個檔案包含所有 API 端點的 Python 使用範例

API Base URL: http://raptor_open_0_1_api.dhtsolution.com:8012
API Docs: http://raptor_open_0_1_api.dhtsolution.com:8012/docs
"""

import requests
import json
from typing import Optional, List, Dict, Any


class RaptorAPIClient:
    """RAPTOR API Gateway 客戶端"""
    
    def __init__(self, base_url: str = "http://raptor_open_0_1_api.dhtsolution.com:8012"):
        self.base_url = base_url
        self.token: Optional[str] = None
    
    def _get_headers(self, include_auth: bool = True) -> Dict[str, str]:
        """取得請求標頭"""
        headers = {"Content-Type": "application/json"}
        if include_auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    # ==================== Authentication ====================
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """
        註冊新使用者
        
        Args:
            username: 使用者名稱
            email: 電子郵件
            password: 密碼
            
        Returns:
            註冊結果
        """
        url = f"{self.base_url}/api/v1/auth/register"
        payload = {
            "username": username,
            "email": email,
            "password": password
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def login(self, username: str, password: str) -> str:
        """
        登入並取得 JWT token
        
        Args:
            username: 使用者名稱
            password: 密碼
            
        Returns:
            JWT access token
        """
        url = f"{self.base_url}/api/v1/auth/login"
        # OAuth2 password flow 使用 form data
        data = {
            "username": username,
            "password": password
        }
        response = requests.post(url, data=data)
        response.raise_for_status()
        result = response.json()
        self.token = result.get("access_token")
        return self.token
    
    # ==================== Search ====================
    
    def video_search(
        self,
        query_text: str,
        embedding_type: str = "text",
        filename: Optional[List[str]] = None,
        speaker: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        影片相似度搜尋
        
        Args:
            query_text: 搜尋查詢文字
            embedding_type: 'text' 或 'summary'
            filename: 要搜尋的影片檔名列表
            speaker: 要篩選的說話者列表
            limit: 返回結果數量
            
        Returns:
            搜尋結果
        """
        url = f"{self.base_url}/api/v1/search/video_search"
        payload = {
            "query_text": query_text,
            "embedding_type": embedding_type,
            "limit": limit
        }
        if filename:
            payload["filename"] = filename
        if speaker:
            payload["speaker"] = speaker
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def audio_search(
        self,
        query_text: str,
        embedding_type: str = "text",
        filename: Optional[List[str]] = None,
        speaker: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        音訊相似度搜尋
        
        Args:
            query_text: 搜尋查詢文字
            embedding_type: 'text' 或 'summary'
            filename: 要搜尋的音訊檔名列表
            speaker: 要篩選的說話者列表
            limit: 返回結果數量
            
        Returns:
            搜尋結果
        """
        url = f"{self.base_url}/api/v1/search/audio_search"
        payload = {
            "query_text": query_text,
            "embedding_type": embedding_type,
            "limit": limit
        }
        if filename:
            payload["filename"] = filename
        if speaker:
            payload["speaker"] = speaker
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def document_search(
        self,
        query_text: str,
        embedding_type: str = "text",
        filename: Optional[List[str]] = None,
        source: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        文件相似度搜尋
        
        Args:
            query_text: 搜尋查詢文字
            embedding_type: 'text' 或 'summary'
            filename: 要搜尋的文件檔名列表
            source: 檔案類型 (如: csv, pdf, docx)
            limit: 返回結果數量
            
        Returns:
            搜尋結果
        """
        url = f"{self.base_url}/api/v1/search/document_search"
        payload = {
            "query_text": query_text,
            "embedding_type": embedding_type,
            "limit": limit
        }
        if filename:
            payload["filename"] = filename
        if source:
            payload["source"] = source
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def image_search(
        self,
        query_text: str,
        embedding_type: str = "text",
        filename: Optional[List[str]] = None,
        source: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        圖片相似度搜尋
        
        Args:
            query_text: 搜尋查詢文字
            embedding_type: 'text' 或 'summary'
            filename: 要搜尋的圖片檔名列表
            source: 副檔名 (如: jpg, png)
            limit: 返回結果數量
            
        Returns:
            搜尋結果
        """
        url = f"{self.base_url}/api/v1/search/image_search"
        payload = {
            "query_text": query_text,
            "embedding_type": embedding_type,
            "limit": limit
        }
        if filename:
            payload["filename"] = filename
        if source:
            payload["source"] = source
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def unified_search(
        self,
        query_text: str,
        embedding_type: str = "text",
        filters: Optional[Dict[str, Dict]] = None,
        limit_per_collection: int = 5,
        global_limit: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        跨集合統一搜尋
        
        Args:
            query_text: 搜尋查詢文字
            embedding_type: 'text' 或 'summary'
            filters: 每個集合的篩選條件
            limit_per_collection: 每個集合返回的結果數量 (最大 50)
            global_limit: 聚合後的最大結果總數 (最大 100)
            score_threshold: 最小分數閾值 (0.0-1.0)
            
        Returns:
            搜尋結果
        """
        url = f"{self.base_url}/api/v1/search/unified_search"
        payload = {
            "query_text": query_text,
            "embedding_type": embedding_type,
            "limit_per_collection": limit_per_collection
        }
        if filters:
            payload["filters"] = filters
        if global_limit:
            payload["global_limit"] = global_limit
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    # ==================== File Upload ====================
    
    def upload_file(
        self,
        file_path: str,
        archive_ttl: int = 30,
        destroy_ttl: int = 30
    ) -> Dict[str, Any]:
        """
        上傳單一檔案到儲存服務
        
        Args:
            file_path: 要上傳的檔案路徑
            archive_ttl: 封存時間 (天)
            destroy_ttl: 刪除時間 (天)
            
        Returns:
            上傳結果
        """
        url = f"{self.base_url}/api/v1/asset/fileupload"
        
        with open(file_path, 'rb') as f:
            files = {'primary_file': f}
            data = {
                'archive_ttl': archive_ttl,
                'destroy_ttl': destroy_ttl
            }
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            response = requests.post(url, files=files, data=data, headers=headers)
        
        response.raise_for_status()
        return response.json()
    
    def upload_files_batch(
        self,
        file_paths: List[str],
        archive_ttl: int = 30,
        destroy_ttl: int = 30,
        concurrency: int = 4
    ) -> Dict[str, Any]:
        """
        批次上傳多個檔案
        
        Args:
            file_paths: 要上傳的檔案路徑列表
            archive_ttl: 封存時間 (天)
            destroy_ttl: 刪除時間 (天)
            concurrency: 最大並行上傳數 (1-16)
            
        Returns:
            批次上傳結果
        """
        url = f"{self.base_url}/api/v1/asset/fileupload_batch"
        
        files = []
        try:
            for file_path in file_paths:
                files.append(('primary_files', open(file_path, 'rb')))
            
            data = {
                'archive_ttl': archive_ttl,
                'destroy_ttl': destroy_ttl,
                'concurrency': concurrency
            }
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
            return response.json()
        finally:
            for _, file_obj in files:
                file_obj.close()
    
    def upload_file_with_analysis(
        self,
        file_path: str,
        processing_mode: str = "default",
        archive_ttl: int = 30,
        destroy_ttl: int = 30
    ) -> Dict[str, Any]:
        """
        上傳檔案並自動進行分析
        
        Args:
            file_path: 要上傳的檔案路徑
            processing_mode: 處理模式
            archive_ttl: 封存時間 (天)
            destroy_ttl: 刪除時間 (天)
            
        Returns:
            上傳和分析結果
        """
        url = f"{self.base_url}/api/v1/asset/fileupload_analysis"
        
        with open(file_path, 'rb') as f:
            files = {'primary_file': f}
            data = {
                'processing_mode': processing_mode,
                'archive_ttl': archive_ttl,
                'destroy_ttl': destroy_ttl
            }
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            response = requests.post(url, files=files, data=data, headers=headers)
        
        response.raise_for_status()
        return response.json()
    
    def upload_files_batch_with_analysis(
        self,
        file_paths: List[str],
        processing_mode: str = "default",
        archive_ttl: int = 30,
        destroy_ttl: int = 30,
        concurrency: int = 4
    ) -> Dict[str, Any]:
        """
        批次上傳多個檔案並自動分析
        
        Args:
            file_paths: 要上傳的檔案路徑列表
            processing_mode: 處理模式
            archive_ttl: 封存時間 (天)
            destroy_ttl: 刪除時間 (天)
            concurrency: 最大並行上傳數 (1-16)
            
        Returns:
            批次上傳和分析結果
        """
        url = f"{self.base_url}/api/v1/asset/fileupload_analysis_batch"
        
        files = []
        try:
            for file_path in file_paths:
                files.append(('primary_files', open(file_path, 'rb')))
            
            data = {
                'processing_mode': processing_mode,
                'archive_ttl': archive_ttl,
                'destroy_ttl': destroy_ttl,
                'concurrency': concurrency
            }
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
            return response.json()
        finally:
            for _, file_obj in files:
                file_obj.close()
    
    # ==================== Asset Management ====================
    
    def list_file_versions(self, asset_path: str, filename: str) -> Dict[str, Any]:
        """
        列出檔案的所有版本
        
        Args:
            asset_path: 資產路徑識別碼
            filename: 檔名
            
        Returns:
            版本列表
        """
        url = f"{self.base_url}/api/v1/asset/fileversions"
        params = {
            "asset_path": asset_path,
            "filename": filename
        }
        response = requests.get(url, params=params, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def download_asset(
        self,
        asset_path: str,
        version_id: str,
        return_file_content: bool = True
    ) -> Any:
        """
        下載資產
        
        Args:
            asset_path: 資產路徑識別碼
            version_id: 版本 ID
            return_file_content: 是否返回檔案內容
            
        Returns:
            檔案內容或回應
        """
        url = f"{self.base_url}/api/v1/asset/filedownload"
        params = {
            "asset_path": asset_path,
            "version_id": version_id,
            "return_file_content": return_file_content
        }
        response = requests.get(url, params=params, headers=self._get_headers())
        response.raise_for_status()
        
        if return_file_content:
            return response.content
        else:
            return response.json()
    
    def archive_asset(self, asset_path: str, version_id: str) -> Dict[str, Any]:
        """
        封存資產
        
        Args:
            asset_path: 資產路徑識別碼
            version_id: 版本 ID
            
        Returns:
            封存結果
        """
        url = f"{self.base_url}/api/v1/asset/filearchive"
        params = {
            "asset_path": asset_path,
            "version_id": version_id
        }
        response = requests.post(url, params=params, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def delete_asset(self, asset_path: str, version_id: str) -> Dict[str, Any]:
        """
        刪除已封存的資產
        
        Args:
            asset_path: 資產路徑識別碼
            version_id: 版本 ID
            
        Returns:
            刪除結果
        """
        url = f"{self.base_url}/api/v1/asset/delfile"
        params = {
            "asset_path": asset_path,
            "version_id": version_id
        }
        response = requests.post(url, params=params, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    # ==================== Processing ====================
    
    def process_file(self, upload_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        處理已上傳的檔案
        
        Args:
            upload_result: 上傳結果資訊
            
        Returns:
            處理結果
        """
        url = f"{self.base_url}/api/v1/processing/process-file"
        payload = {"upload_result": upload_result}
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def get_cached_value(self, m_type: str, key: str) -> Dict[str, Any]:
        """
        從 Redis 取得快取值
        
        Args:
            m_type: 媒體類型 (document, video, image, audio)
            key: 快取鍵
            
        Returns:
            快取值
        """
        url = f"{self.base_url}/api/v1/processing/processing/cache/{m_type}/{key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_all_cache(self) -> Dict[str, Any]:
        """
        取得所有 Redis 快取
        
        Returns:
            所有快取鍵值對
        """
        url = f"{self.base_url}/api/v1/processing/cache/all"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    # ==================== Chat ====================
    
    def send_chat(
        self,
        user_id: str,
        message: str,
        search_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        發送聊天訊息
        
        Args:
            user_id: 使用者 ID
            message: 訊息內容
            search_results: 搜尋結果 (可選)
            
        Returns:
            聊天回應
        """
        url = f"{self.base_url}/api/v1/chat/chat"
        payload = {
            "user_id": user_id,
            "message": message
        }
        if search_results:
            payload["search_results"] = search_results
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def get_chat_memory(self, user_id: str) -> Dict[str, Any]:
        """
        取得使用者聊天記憶
        
        Args:
            user_id: 使用者 ID
            
        Returns:
            聊天記憶
        """
        url = f"{self.base_url}/api/v1/chat/memory/{user_id}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def clear_chat_memory(self, user_id: str) -> Dict[str, Any]:
        """
        清除使用者聊天記憶
        
        Args:
            user_id: 使用者 ID
            
        Returns:
            清除結果
        """
        url = f"{self.base_url}/api/v1/chat/memory/{user_id}"
        response = requests.delete(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    # ==================== Health ====================
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康檢查
        
        Returns:
            健康狀態
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


# ==================== 使用範例 ====================

def example_usage():
    """使用範例"""
    
    # 初始化客戶端
    client = RaptorAPIClient()
    
    # 1. 註冊使用者
    print("=== 註冊使用者 ===")
    try:
        result = client.register_user(
            username="test_user",
            email="test@example.com",
            password="secure_password"
        )
        print(f"註冊成功: {result}")
    except Exception as e:
        print(f"註冊失敗: {e}")
    
    # 2. 登入
    print("\n=== 登入 ===")
    try:
        token = client.login(username="test_user", password="secure_password")
        print(f"登入成功，Token: {token[:20]}...")
    except Exception as e:
        print(f"登入失敗: {e}")
        return
    
    # 3. 影片搜尋
    print("\n=== 影片搜尋 ===")
    try:
        result = client.video_search(
            query_text="機器學習",
            embedding_type="text",
            filename=["MV.mp4"],
            speaker=["SPEAKER_00"],
            limit=5
        )
        print(f"搜尋結果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"搜尋失敗: {e}")
    
    # 4. 文件搜尋
    print("\n=== 文件搜尋 ===")
    try:
        result = client.document_search(
            query_text="財務報告",
            embedding_type="text",
            filename=["EF25Y01.csv"],
            source="csv",
            limit=5
        )
        print(f"搜尋結果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"搜尋失敗: {e}")
    
    # 5. 統一搜尋
    print("\n=== 統一搜尋 ===")
    try:
        result = client.unified_search(
            query_text="機器學習和人工智慧",
            embedding_type="text",
            filters={
                "video": {"filename": ["MV.mp4"]},
                "document": {"source": "csv"}
            },
            limit_per_collection=3,
            global_limit=10,
            score_threshold=0.5
        )
        print(f"搜尋結果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"搜尋失敗: {e}")
    
    # 6. 上傳檔案 (假設有檔案)
    print("\n=== 上傳檔案 ===")
    try:
        # result = client.upload_file(
        #     file_path="path/to/your/file.pdf",
        #     archive_ttl=30,
        #     destroy_ttl=30
        # )
        # print(f"上傳成功: {result}")
        print("(跳過 - 需要實際檔案)")
    except Exception as e:
        print(f"上傳失敗: {e}")
    
    # 7. 聊天
    print("\n=== 聊天 ===")
    try:
        result = client.send_chat(
            user_id="test_user",
            message="請幫我總結一下機器學習的重點"
        )
        print(f"回應: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"聊天失敗: {e}")
    
    # 8. 取得聊天記憶
    print("\n=== 取得聊天記憶 ===")
    try:
        result = client.get_chat_memory(user_id="test_user")
        print(f"聊天記憶: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"取得失敗: {e}")
    
    # 9. 健康檢查
    print("\n=== 健康檢查 ===")
    try:
        result = client.health_check()
        print(f"健康狀態: {result}")
    except Exception as e:
        print(f"檢查失敗: {e}")


if __name__ == "__main__":
    example_usage()
