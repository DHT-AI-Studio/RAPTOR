"""
Raptor 0.3 Python Client
Usage:
    from raptor_client import RaptorClient
    client = RaptorClient("http://raptor_open_0_3_api.dhtsolution.com:8012", "user", "pass")
    client.login()
    result = client.upload_file("video.mp4")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from requests.exceptions import ConnectionError, Timeout


class RaptorAPIError(Exception):
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class RaptorClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.retries = retries
        self.retry_delay = retry_delay
        self._token: Optional[str] = None
        self._session = requests.Session()

    # ── Auth ──────────────────────────────────────────────────

    def login(self) -> str:
        resp = self._session.post(
            f"{self.base_url}/api/0.3/sso/login",
            data={"username": self.username, "password": self.password},
            timeout=30,
        )
        self._raise_for_status(resp)
        self._token = resp.json()["access_token"]
        return self._token

    def logout(self) -> Dict:
        resp = self._request("POST", "/api/0.3/sso/logout")
        self._token = None
        return resp.json()

    def _auth_headers(self) -> Dict[str, str]:
        if not self._token:
            self.login()
        return {"Authorization": f"Bearer {self._token}"}

    # ── Internal HTTP helpers ─────────────────────────────────

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RaptorAPIError(resp.status_code, detail)

    def _request(
        self,
        method: str,
        path: str,
        timeout: float = 30,
        **kwargs,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = {**self._auth_headers(), **kwargs.pop("headers", {})}
        last_exc: Exception = RuntimeError("no attempts made")

        for attempt in range(self.retries):
            try:
                resp = self._session.request(
                    method, url, headers=headers, timeout=timeout, **kwargs
                )
                self._raise_for_status(resp)
                return resp
            except (ConnectionError, Timeout) as exc:
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(self.retry_delay)
            except RaptorAPIError:
                raise

        raise last_exc

    # ── Asset ─────────────────────────────────────────────────

    def upload_file(
        self,
        file_path: str | Path,
        processing_mode: str = "default",
        archive_date_or_ttl: Optional[str] = None,
        destroy_date_or_ttl: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        data: Dict[str, str] = {"processing_mode": processing_mode}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)

        with path.open("rb") as f:
            resp = self._request(
                "POST",
                "/api/0.3/asset/fileupload_analysis",
                timeout=120,
                files={"primary_file": (path.name, f, "application/octet-stream")},
                data=data,
            )
        return resp.json()

    def upload_files_batch(
        self,
        file_paths: List[str | Path],
        processing_mode: str = "default",
        concurrency: int = 4,
        archive_date_or_ttl: Optional[str] = None,
        destroy_date_or_ttl: Optional[str] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, str] = {
            "processing_mode": processing_mode,
            "concurrency": str(concurrency),
        }
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)

        file_handles = [open(p, "rb") for p in file_paths]
        try:
            files = [
                ("primary_files", (Path(p).name, fh, "application/octet-stream"))
                for p, fh in zip(file_paths, file_handles)
            ]
            resp = self._request(
                "POST",
                "/api/0.3/asset/fileupload_analysis_batch",
                timeout=300,
                files=files,
                data=data,
            )
        finally:
            for fh in file_handles:
                fh.close()
        return resp.json()

    def download_asset(
        self,
        asset_path: str,
        version_id: str,
        return_file_content: bool = False,
    ) -> Dict[str, Any]:
        resp = self._request(
            "GET",
            f"/api/0.3/asset/filedownload/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
            params={"return_file_content": str(return_file_content).lower()},
        )
        return resp.json()

    def list_versions(self, asset_path: str, filename: str) -> Dict[str, Any]:
        resp = self._request(
            "GET",
            f"/api/0.3/asset/fileversions/{quote(asset_path, safe='')}/{quote(filename, safe='')}",
        )
        return resp.json()

    def get_user_commits(
        self,
        keyword: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        params = {"page": page, "page_size": page_size}
        if keyword:
            params["keyword"] = keyword
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        resp = self._request("GET", "/api/0.3/asset/users/commits", params=params)
        return resp.json()

    def archive_asset(self, asset_path: str, version_id: str) -> Dict[str, Any]:
        resp = self._request(
            "POST",
            f"/api/0.3/asset/filearchive/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
        )
        return resp.json()

    def delete_asset(self, asset_path: str, version_id: str) -> Dict[str, Any]:
        resp = self._request(
            "POST",
            f"/api/0.3/asset/delfile/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
        )
        return resp.json()

    def update_expiration(
        self,
        asset_path: str,
        version_id: str,
        archive_date_or_ttl: Optional[str] = None,
        destroy_date_or_ttl: Optional[str] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, str] = {}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)
        resp = self._request(
            "POST",
            f"/api/0.3/asset/file-expiration/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
            data=data,
        )
        return resp.json()

    # ── Processing status ─────────────────────────────────────

    def get_processing_status(self, m_type: str, correlation_id: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/api/0.3/processing/cache/{m_type}/{correlation_id}")
        return resp.json()

    def get_all_processing_status(self) -> Dict[str, Any]:
        resp = self._request("GET", "/api/0.3/processing/cache/all")
        return resp.json()

    def wait_for_processing(
        self,
        correlation_id: str,
        m_type: str = "video",
        timeout: float = 1800,
        interval: float = 15,
        on_step: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Poll until step == complete/failed or timeout.
        on_step(elapsed, step) is called whenever step changes.
        Returns the final status dict.
        """
        start = time.time()
        last_step = None

        while time.time() - start < timeout:
            elapsed = int(time.time() - start)
            try:
                data = self.get_processing_status(m_type, correlation_id)
            except RaptorAPIError as exc:
                if exc.status_code == 404:
                    time.sleep(interval)
                    continue
                raise

            step = data.get("value", {}).get("step") or data.get("step", "unknown")
            if step != last_step:
                if on_step:
                    on_step(elapsed, step)
                last_step = step

            if step in ("complete", "failed"):
                return data

            time.sleep(interval)

        raise TimeoutError(f"Processing did not complete within {timeout}s")

    # ── Search ────────────────────────────────────────────────

    def _search(self, endpoint: str, payload: Dict, timeout: float = 60) -> Dict[str, Any]:
        resp = self._request(
            "POST",
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        return resp.json()

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        payload_schema: str = "contextual",
        embedding_type: Optional[str] = None,
        type: Optional[str | List[str]] = None,
        filename: Optional[List[str]] = None,
        speaker: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"query": query, "top_k": top_k, "payload_schema": payload_schema}
        if embedding_type is not None:
            body["embedding_type"] = embedding_type
        if type is not None:
            body["type"] = type
        if filename is not None:
            body["filename"] = filename
        if speaker is not None:
            body["speaker"] = speaker
        if source is not None:
            body["source"] = source
        return self._search("/api/0.3/search/hybrid", body)

    def search_bm25(self, query: str, top_k: int = 10, **filters) -> Dict[str, Any]:
        return self._search("/api/0.3/search/bm25", {"query": query, "top_k": top_k, **filters})

    def search_vector(self, query: str, top_k: int = 10, **filters) -> Dict[str, Any]:
        return self._search("/api/0.3/search/vector", {"query": query, "top_k": top_k, **filters})

    def search_graphrag(
        self,
        query: str,
        max_depth: int = 2,
        limit: int = 50,
        score_threshold: float = 0.5,
        strategy: str = "hybrid",
    ) -> Dict[str, Any]:
        return self._search(
            "/api/0.3/search/graphrag",
            {"query": query, "max_depth": max_depth, "limit": limit,
             "score_threshold": score_threshold, "strategy": strategy},
        )

    def search_tkg(
        self,
        query: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        max_depth: int = 2,
        limit: int = 50,
        score_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "query": query, "max_depth": max_depth,
            "limit": limit, "score_threshold": score_threshold,
        }
        if time_start:
            body["time_start"] = time_start
        if time_end:
            body["time_end"] = time_end
        return self._search("/api/0.3/search/tkg", body)

    def video_search(
        self,
        query: str,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        score_threshold: float = 0.52,
    ) -> Dict[str, Any]:
        return self._search(
            "/api/0.3/search/video_search",
            {"query": query, "top_k": top_k,
             "candidate_multiplier": candidate_multiplier,
             "score_threshold": score_threshold},
        )

    # ── RAG ───────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "direct",
    ) -> Dict[str, Any]:
        return self._search(
            "/api/0.3/a2a/query",
            {"question": question, "top_k": top_k, "mode": mode},
            timeout=120,
        )

    # ── Convenience ───────────────────────────────────────────

    def upload_and_wait(
        self,
        file_path: str | Path,
        m_type: str = "video",
        processing_mode: str = "default",
        timeout: float = 1800,
        interval: float = 15,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Upload a file and wait for processing to complete. Returns processing status."""
        upload_data = self.upload_file(file_path, processing_mode=processing_mode)
        processing = upload_data["processing_result"]

        if processing.get("status") == "skipped":
            if verbose:
                print("[skipped] duplicate file — already processed")
            return {"status": "skipped", "upload_result": upload_data["upload_result"]}

        correlation_id = processing["correlation_id"]
        if verbose:
            print(f"[uploaded] correlation_id={correlation_id}")

        def _log(elapsed, step):
            if verbose:
                print(f"  [{elapsed:4d}s] {step}")

        result = self.wait_for_processing(
            correlation_id, m_type=m_type,
            timeout=timeout, interval=interval,
            on_step=_log,
        )
        return result


# ── CLI demo ──────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_URL = "http://raptor_open_0_3_api.dhtsolution.com:8012"
    VIDEO = "/opt/home/cing/cing_test/Raptor_0.2_Modular_TH_20260409/Raptor_0.3/用水沖一次就乾淨的衣服？_1080p.mp4"
    SEP = "─" * 60

    def section(title):
        print(f"\n{SEP}\n  {title}\n{SEP}")

    client = RaptorClient(BASE_URL, "test_basicuser", "12345")

    # 1. 登入
    section("1. 登入")
    token = client.login()
    print(f"[OK] token: {token[:40]}...")

    # 2. 上傳 + 追蹤進度
    section("2. 上傳並追蹤進度")
    upload_data = client.upload_file(VIDEO)
    upload_result = upload_data["upload_result"]
    processing = upload_data["processing_result"]

    print(f"[OK] asset_path : {upload_result['asset_path']}")
    print(f"     version_id : {upload_result['version_id'][:32]}...")
    exists = upload_result.get("existence_info", {})
    print(f"     exists     : {exists.get('exists')} — {exists.get('message','')[:60]}")

    if processing.get("status") == "skipped":
        print("[INFO] 重複檔案，跳過處理")
        correlation_id = "89de0bff-b371-484b-b254-99c7f36a183a"  # fallback
        result = client.get_processing_status("video", correlation_id)
        step = result.get("value", {}).get("step", "?")
        print(f"[INFO] 現有處理狀態: {step}")
        if step == "complete":
            summary = result.get("value", {}).get("summary", "")
            if summary:
                print(f"       摘要: {summary[:120]}...")
    else:
        correlation_id = processing["correlation_id"]
        print(f"[OK] correlation_id: {correlation_id}")

        def log_step(elapsed, step):
            print(f"     [{elapsed:4d}s] {step}")

        try:
            result = client.wait_for_processing(
                correlation_id, m_type="video", on_step=log_step
            )
            step = result.get("value", {}).get("step", "?")
            print(f"\n[OK] 處理完成 — step={step}")
            summary = result.get("value", {}).get("summary", "")
            if summary:
                print(f"     摘要: {summary[:120]}...")
        except TimeoutError:
            print("[WARN] 等待逾時，繼續執行搜尋...")

    # 3. video_search
    section("3. video_search")
    try:
        data = client.video_search("衣服清洗 用水沖洗", top_k=5)
        print(f"[OK] 找到 {data.get('total', 0)} 部影片")
        for v in data.get("results", []):
            print(f"     [{v['score']:.4f}] {v['filename']}")
            url = v.get("asset_url", "")
            if url:
                print(f"            asset_url: {url[:80]}...")
            for seg in v.get("segments", [])[:2]:
                print(f"            {seg['start_time']}s–{seg['end_time']}s  score={seg['score']:.4f}  來源={seg['sources']}")
    except RaptorAPIError as e:
        print(f"[FAIL] {e}")

    # 4. RAG query
    section("4. RAG query（direct）")
    try:
        data = client.query("這件衣服用水沖洗的原理是什麼？有什麼特殊材質或技術？")
        print(f"[OK] chunks_used: {data.get('chunks_used', 0)}")
        print(f"\n答案：\n{data.get('answer', '（無答案）')}")
        print("\n引用來源：")
        for src in data.get("sources", [])[:3]:
            meta = src.get("metadata", {})
            fname = meta.get("filename", "?")
            t0 = src.get("start_time")
            t1 = src.get("end_time")
            time_info = f" | {t0}s–{t1}s" if t0 is not None else ""
            print(f"  score={src['score']:.4f} | {fname}{time_info}")
    except RaptorAPIError as e:
        print(f"[FAIL] {e}")

    section("測試完成")
