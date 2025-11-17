"""
RAPTOR API Gateway - Python Sample Code
========================================
This file contains Python usage examples for all API endpoints

API Base URL: http://raptor_open_0_1_api.dhtsolution.com:8012
API Docs: http://raptor_open_0_1_api.dhtsolution.com:8012/docs
"""

import requests
import json
from typing import Optional, List, Dict, Any


class RaptorAPIClient:
    """RAPTOR API Gateway Client"""

    def __init__(self, base_url: str = "http://raptor_open_0_1_api.dhtsolution.com:8012"):
        self.base_url = base_url
        self.token: Optional[str] = None

    def _get_headers(self, include_auth: bool = True) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if include_auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # ==================== Authentication ====================

    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """
        Register a new user

        Args:
            username: Username
            email: Email address
            password: Password

        Returns:
            Registration result
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
        Login and get JWT token

        Args:
            username: Username
            password: Password

        Returns:
            JWT access token
        """
        url = f"{self.base_url}/api/v1/auth/login"
        # OAuth2 password flow uses form data
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
        Video similarity search

        Args:
            query_text: Search query text
            embedding_type: 'text' or 'summary'
            filename: List of video filenames to search
            speaker: List of speakers to filter
            limit: Number of results to return

        Returns:
            Search results
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
        Audio similarity search

        Args:
            query_text: Search query text
            embedding_type: 'text' or 'summary'
            filename: List of audio filenames to search
            speaker: List of speakers to filter
            limit: Number of results to return

        Returns:
            Search results
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
        Document similarity search

        Args:
            query_text: Search query text
            embedding_type: 'text' or 'summary'
            filename: List of document filenames to search
            source: File type (e.g., csv, pdf, docx)
            limit: Number of results to return

        Returns:
            Search results
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
        Image similarity search

        Args:
            query_text: Search query text
            embedding_type: 'text' or 'summary'
            filename: List of image filenames to search
            source: File extension (e.g., jpg, png)
            limit: Number of results to return

        Returns:
            Search results
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
        Unified cross-collection search

        Args:
            query_text: Search query text
            embedding_type: 'text' or 'summary'
            filters: Filter conditions for each collection
            limit_per_collection: Number of results per collection (max 50)
            global_limit: Maximum total results after aggregation (max 100)
            score_threshold: Minimum score threshold (0.0-1.0)

        Returns:
            Search results
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
        Upload a single file to storage service

        Args:
            file_path: Path to the file to upload
            archive_ttl: Archive time (days)
            destroy_ttl: Destroy time (days)

        Returns:
            Upload result
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
        Batch upload multiple files

        Args:
            file_paths: List of file paths to upload
            archive_ttl: Archive time (days)
            destroy_ttl: Destroy time (days)
            concurrency: Maximum concurrent uploads (1-16)

        Returns:
            Batch upload result
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
        Upload file and automatically analyze

        Args:
            file_path: Path to the file to upload
            processing_mode: Processing mode
            archive_ttl: Archive time (days)
            destroy_ttl: Destroy time (days)

        Returns:
            Upload and analysis result
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
        Batch upload multiple files and automatically analyze

        Args:
            file_paths: List of file paths to upload
            processing_mode: Processing mode
            archive_ttl: Archive time (days)
            destroy_ttl: Destroy time (days)
            concurrency: Maximum concurrent uploads (1-16)

        Returns:
            Batch upload and analysis result
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
        List all versions of a file

        Args:
            asset_path: Asset path identifier
            filename: Filename

        Returns:
            List of versions
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
        Download asset

        Args:
            asset_path: Asset path identifier
            version_id: Version ID
            return_file_content: Whether to return file content

        Returns:
            File content or response
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
        Archive asset

        Args:
            asset_path: Asset path identifier
            version_id: Version ID

        Returns:
            Archive result
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
        Delete archived asset

        Args:
            asset_path: Asset path identifier
            version_id: Version ID

        Returns:
            Delete result
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
        Process uploaded file

        Args:
            upload_result: Upload result information

        Returns:
            Processing result
        """
        url = f"{self.base_url}/api/v1/processing/process-file"
        payload = {"upload_result": upload_result}
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def get_cached_value(self, m_type: str, key: str) -> Dict[str, Any]:
        """
        Get cached value from Redis

        Args:
            m_type: Media type (document, video, image, audio)
            key: Cache key

        Returns:
            Cached value
        """
        url = f"{self.base_url}/api/v1/processing/processing/cache/{m_type}/{key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_all_cache(self) -> Dict[str, Any]:
        """
        Get all Redis cache

        Returns:
            All cache key-value pairs
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
        Send chat message

        Args:
            user_id: User ID
            message: Message content
            search_results: Search results (optional)

        Returns:
            Chat response
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
        Get user chat memory

        Args:
            user_id: User ID

        Returns:
            Chat memory
        """
        url = f"{self.base_url}/api/v1/chat/memory/{user_id}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def clear_chat_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Clear user chat memory

        Args:
            user_id: User ID

        Returns:
            Clear result
        """
        url = f"{self.base_url}/api/v1/chat/memory/{user_id}"
        response = requests.delete(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    # ==================== Health ====================

    def health_check(self) -> Dict[str, Any]:
        """
        Health check

        Returns:
            Health status
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


# ==================== Usage Examples ====================

def example_usage():
    """Usage examples"""

    # Initialize client
    client = RaptorAPIClient()

    # 1. Register user
    print("=== Register User ===")
    try:
        result = client.register_user(
            username="test_user",
            email="test@example.com",
            password="secure_password"
        )
        print(f"Registration successful: {result}")
    except Exception as e:
        print(f"Registration failed: {e}")

    # 2. Login
    print("\n=== Login ===")
    try:
        token = client.login(username="test_user", password="secure_password")
        print(f"Login successful, Token: {token[:20]}...")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 3. Video search
    print("\n=== Video Search ===")
    try:
        result = client.video_search(
            query_text="machine learning",
            embedding_type="text",
            filename=["MV.mp4"],
            speaker=["SPEAKER_00"],
            limit=5
        )
        print(f"Search results: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Search failed: {e}")

    # 4. Document search
    print("\n=== Document Search ===")
    try:
        result = client.document_search(
            query_text="financial report",
            embedding_type="text",
            filename=["EF25Y01.csv"],
            source="csv",
            limit=5
        )
        print(f"Search results: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Search failed: {e}")

    # 5. Unified search
    print("\n=== Unified Search ===")
    try:
        result = client.unified_search(
            query_text="machine learning and artificial intelligence",
            embedding_type="text",
            filters={
                "video": {"filename": ["MV.mp4"]},
                "document": {"source": "csv"}
            },
            limit_per_collection=3,
            global_limit=10,
            score_threshold=0.5
        )
        print(f"Search results: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Search failed: {e}")

    # 6. Upload file (assuming file exists)
    print("\n=== Upload File ===")
    try:
        # result = client.upload_file(
        #     file_path="path/to/your/file.pdf",
        #     archive_ttl=30,
        #     destroy_ttl=30
        # )
        # print(f"Upload successful: {result}")
        print("(Skipped - requires actual file)")
    except Exception as e:
        print(f"Upload failed: {e}")

    # 7. Chat
    print("\n=== Chat ===")
    try:
        result = client.send_chat(
            user_id="test_user",
            message="Please summarize the key points of machine learning for me"
        )
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Chat failed: {e}")

    # 8. Get chat memory
    print("\n=== Get Chat Memory ===")
    try:
        result = client.get_chat_memory(user_id="test_user")
        print(f"Chat memory: {json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Get failed: {e}")

    # 9. Health check
    print("\n=== Health Check ===")
    try:
        result = client.health_check()
        print(f"Health status: {result}")
    except Exception as e:
        print(f"Check failed: {e}")


if __name__ == "__main__":
    example_usage()
