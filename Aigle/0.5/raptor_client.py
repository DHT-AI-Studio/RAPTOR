"""
Raptor Python Client
Usage:
    from raptor_client import RaptorClient
    client = RaptorClient("http://raptor_open_0_5_api.dhtsolution.com:8012", "user", "pass")
    client.login()
    result = client.upload_file("video.mp4")
"""
from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
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
        realm_name: str = "dhtsolution",
        client_id: str = "raptor",
        retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.realm_name = realm_name
        self.client_id = client_id
        self.retries = retries
        self.retry_delay = retry_delay
        self._token: str | None = None
        self._session = requests.Session()

    # ── Auth ──────────────────────────────────────────────────

    def login(self) -> str:
        resp = self._session.post(
            f"{self.base_url}/api/0.5/sso/login",
            data={
                "username": self.username,
                "password": self.password,
                "realm_name": self.realm_name,
                "client_id": self.client_id,
            },
            timeout=30,
        )
        self._raise_for_status(resp)
        self._token = resp.json()["access_token"]
        return self._token

    def logout(self) -> dict:
        resp = self._request("POST", "/api/0.5/sso/logout")
        self._token = None
        return resp.json()

    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            self.login()
        return {"Authorization": f"Bearer {self._token}"}

    # ── Internal HTTP helpers ─────────────────────────────────

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except ValueError:
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
        archive_date_or_ttl: str | None = None,
        destroy_date_or_ttl: str | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        data: dict[str, str] = {"processing_mode": processing_mode}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)

        with path.open("rb") as f:
            resp = self._request(
                "POST",
                "/api/0.5/asset/fileupload_analysis",
                timeout=120,
                files={"primary_file": (path.name, f, "application/octet-stream")},
                data=data,
            )
        return resp.json()

    def upload_files_batch(
        self,
        file_paths: list[str | Path],
        processing_mode: str = "default",
        concurrency: int = 4,
        archive_date_or_ttl: str | None = None,
        destroy_date_or_ttl: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, str] = {
            "processing_mode": processing_mode,
            "concurrency": str(concurrency),
        }
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)

        with ExitStack() as stack:
            file_handles = [stack.enter_context(open(p, "rb")) for p in file_paths]
            files = [
                ("primary_files", (Path(p).name, fh, "application/octet-stream"))
                for p, fh in zip(file_paths, file_handles)
            ]
            resp = self._request(
                "POST",
                "/api/0.5/asset/fileupload_analysis_batch",
                timeout=300,
                files=files,
                data=data,
            )
        return resp.json()

    def download_asset(
        self,
        asset_path: str,
        version_id: str,
        return_file_content: bool = False,
    ) -> dict[str, Any]:
        resp = self._request(
            "GET",
            f"/api/0.5/asset/filedownload/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
            params={"return_file_content": str(return_file_content).lower()},
        )
        return resp.json()

    def list_versions(self, asset_path: str, filename: str) -> dict[str, Any]:
        resp = self._request(
            "GET",
            f"/api/0.5/asset/fileversions/{quote(asset_path, safe='')}/{quote(filename, safe='')}",
        )
        return resp.json()

    def get_user_commits(
        self,
        keyword: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        params = {"page": page, "page_size": page_size}
        if keyword:
            params["keyword"] = keyword
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        resp = self._request("GET", "/api/0.5/asset/users/commits", params=params)
        return resp.json()

    def archive_asset(self, asset_path: str, version_id: str) -> dict[str, Any]:
        resp = self._request(
            "POST",
            f"/api/0.5/asset/filearchive/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
        )
        return resp.json()

    def delete_asset(self, asset_path: str, version_id: str) -> dict[str, Any]:
        resp = self._request(
            "POST",
            f"/api/0.5/asset/delfile/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
        )
        return resp.json()

    def update_expiration(
        self,
        asset_path: str,
        version_id: str,
        archive_date_or_ttl: str | None = None,
        destroy_date_or_ttl: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, str] = {}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)
        resp = self._request(
            "POST",
            f"/api/0.5/asset/file-expiration/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
            data=data,
        )
        return resp.json()

    def upload_file_only(
        self,
        file_path: str | Path,
        archive_date_or_ttl: str | None = None,
        destroy_date_or_ttl: str | None = None,
    ) -> dict[str, Any]:
        """Decoupled upload -- no AI analysis triggered. Follow up with process_file()."""
        path = Path(file_path)
        data: dict[str, str] = {}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)
        with path.open("rb") as f:
            resp = self._request(
                "POST",
                "/api/0.5/asset/fileupload",
                timeout=120,
                files={"primary_file": (path.name, f, "application/octet-stream")},
                data=data,
            )
        return resp.json()

    def scan_files(self, files: list[dict[str, str]]) -> dict[str, Any]:
        """files: [{"path": ..., "checksum": ...}, ...] -- checks existence, uploads nothing."""
        resp = self._request(
            "POST", "/api/0.5/asset/scan",
            headers={"Content-Type": "application/json"},
            json={"files": files},
        )
        return resp.json()

    def unregister_asset(self, asset_path: str, version_id: str) -> dict[str, Any]:
        """Filter-only: removes an active asset from search results without deleting any
        indexed data. Reversible via process_file() with the same asset_path/version_id."""
        resp = self._request(
            "POST",
            f"/api/0.5/asset/unregister/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
        )
        return resp.json()

    def get_thumbnail(self, asset_path: str, version_id: str, second: float) -> bytes:
        resp = self._request(
            "GET",
            f"/api/0.5/asset/thumbnail/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
            params={"second": second},
        )
        return resp.content

    # ── Processing status ─────────────────────────────────────

    def process_file(
        self,
        asset_path: str,
        version_id: str,
        processing_mode: str | None = None,
    ) -> dict[str, Any]:
        """Register API -- triggers analysis for an asset uploaded via upload_file_only(),
        or reverses unregister_asset()."""
        body: dict[str, str] = {"asset_path": asset_path, "version_id": version_id}
        if processing_mode is not None:
            body["processing_mode"] = processing_mode
        return self._post_json("/api/0.5/processing/process-file", body)

    def get_processing_status(self, m_type: str, correlation_id: str) -> dict[str, Any]:
        resp = self._request("GET", f"/api/0.5/processing/cache/{m_type}/{correlation_id}")
        return resp.json()

    def get_all_processing_status(self) -> dict[str, Any]:
        resp = self._request("GET", "/api/0.5/processing/cache/all")
        return resp.json()

    def wait_for_processing(
        self,
        correlation_id: str,
        m_type: str = "video",
        timeout: float = 1800,
        interval: float = 15,
        on_step: callable | None = None,
    ) -> dict[str, Any]:
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

    def _search(self, endpoint: str, payload: dict, timeout: float = 60) -> dict[str, Any]:
        resp = self._request(
            "POST",
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        return resp.json()

    # ── Generic JSON helpers (used by the sections below) ──────

    def _get_json(self, path: str, params: dict | None = None, timeout: float = 30) -> Any:
        resp = self._request("GET", path, params=params, timeout=timeout)
        return resp.json()

    def _post_json(self, path: str, payload: dict | None = None, timeout: float = 30) -> Any:
        resp = self._request(
            "POST", path, headers={"Content-Type": "application/json"},
            json=payload or {}, timeout=timeout,
        )
        return resp.json()

    def _patch_json(self, path: str, payload: dict, timeout: float = 30) -> Any:
        resp = self._request(
            "PATCH", path, headers={"Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        return resp.json()

    def _put_json(self, path: str, payload: dict, timeout: float = 30) -> Any:
        resp = self._request(
            "PUT", path, headers={"Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        return resp.json()

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        payload_schema: str = "contextual",
        embedding_type: str | None = None,
        type: str | list[str] | None = None,
        filename: list[str] | None = None,
        speaker: list[str] | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "top_k": top_k, "payload_schema": payload_schema}
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
        return self._search("/api/0.5/search/hybrid", body)

    def search_bm25(self, query: str, top_k: int = 10, **filters) -> dict[str, Any]:
        return self._search("/api/0.5/search/bm25", {"query": query, "top_k": top_k, **filters})

    def search_vector(self, query: str, top_k: int = 10, **filters) -> dict[str, Any]:
        return self._search("/api/0.5/search/vector", {"query": query, "top_k": top_k, **filters})

    def search_graphrag(
        self,
        query: str,
        max_depth: int = 2,
        limit: int = 50,
        score_threshold: float = 0.5,
        strategy: str = "hybrid",
    ) -> dict[str, Any]:
        return self._search(
            "/api/0.5/search/graphrag",
            {"query": query, "max_depth": max_depth, "limit": limit,
             "score_threshold": score_threshold, "strategy": strategy},
        )

    def search_tkg(
        self,
        query: str,
        time_start: str | None = None,
        time_end: str | None = None,
        max_depth: int = 2,
        limit: int = 50,
        score_threshold: float = 0.3,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query, "max_depth": max_depth,
            "limit": limit, "score_threshold": score_threshold,
        }
        if time_start:
            body["time_start"] = time_start
        if time_end:
            body["time_end"] = time_end
        return self._search("/api/0.5/search/tkg", body)

    def video_search(
        self,
        query: str,
        top_k: int = 10,
        candidate_multiplier: int = 5,
        score_threshold: float = 0.52,
    ) -> dict[str, Any]:
        return self._search(
            "/api/0.5/search/video_search",
            {"query": query, "top_k": top_k,
             "candidate_multiplier": candidate_multiplier,
             "score_threshold": score_threshold},
        )

    def search_people(self, query: str, limit: int = 50) -> dict[str, Any]:
        """Requires the caller's face_detection GDPR consent."""
        return self._search("/api/0.5/search/people", {"query": query, "limit": limit})

    # ── RAG ───────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "direct",
    ) -> dict[str, Any]:
        return self._search(
            "/api/0.5/a2a/query",
            {"question": question, "top_k": top_k, "mode": mode},
            timeout=120,
        )

    # ── Chat ──────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"message": message}
        if history is not None:
            body["history"] = history
        if session_id is not None:
            body["session_id"] = session_id
        return self._post_json("/api/0.5/chat/chat", body, timeout=120)

    def get_chat_memory(self, session_id: str | None = None) -> dict[str, Any]:
        params = {"session_id": session_id} if session_id is not None else None
        return self._get_json("/api/0.5/chat/memory", params=params)

    def clear_chat_memory(self, session_id: str | None = None) -> int:
        """Returns the HTTP status code (204 on success) -- no response body."""
        params = {"session_id": session_id} if session_id is not None else None
        resp = self._request("DELETE", "/api/0.5/chat/memory", params=params)
        return resp.status_code

    def chat_completions(
        self,
        model: str,
        messages: list[dict[str, str]],
        engine: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if engine is not None:
            body["engine"] = engine
        return self._post_json("/api/0.5/chat/completions", body, timeout=120)

    # ── Memory (Module 26, long-term) ──────────────────────────

    def memory_store(
        self, text: str, frame_type: str = "fact", session_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, "frame_type": frame_type}
        if session_id is not None:
            body["session_id"] = session_id
        return self._post_json("/api/0.5/memory/store", body)

    def memory_retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self._get_json("/api/0.5/memory/retrieve", params={"query": query, "top_k": top_k})

    def memory_timeline(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        return self._get_json("/api/0.5/memory/timeline", params={"page": page, "page_size": page_size})

    def memory_compact(
        self,
        session_id: str | None = None,
        trigger: str = "auto",
        context_window: int = 128000,
        max_tokens: int | None = None,
        last_summarized_frame_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"trigger": trigger, "context_window": context_window}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if last_summarized_frame_id is not None:
            body["last_summarized_frame_id"] = last_summarized_frame_id
        resp = self._request(
            "POST", "/api/0.5/memory/compact",
            headers={"Content-Type": "application/json"},
            params={"session_id": session_id} if session_id is not None else None,
            json=body,
        )
        return resp.json()

    def memory_archive(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str | None = None,
        search_results: list | None = None,
        tool_calls: list | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"user_message": user_message, "assistant_response": assistant_response}
        if search_results is not None:
            body["search_results"] = search_results
        if tool_calls is not None:
            body["tool_calls"] = tool_calls
        resp = self._request(
            "POST", "/api/0.5/memory/archive",
            headers={"Content-Type": "application/json"},
            params={"session_id": session_id} if session_id is not None else None,
            json=body,
        )
        return resp.json()

    def memory_multimedia_search(
        self, query: str, top_k: int = 5, media_type: str | None = None,
    ) -> Any:
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if media_type is not None:
            body["media_type"] = media_type
        return self._post_json("/api/0.5/memory/multimedia/search", body)

    def memory_compact_evaluate(
        self, session_id: str | None = None, messages: list | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if session_id is not None:
            body["session_id"] = session_id
        if messages is not None:
            body["messages"] = messages
        return self._post_json("/api/0.5/memory/compact/evaluate", body)

    def memory_session_summaries(self, session_id: str) -> Any:
        return self._get_json(f"/api/0.5/memory/sessions/{quote(session_id, safe='')}/summaries")

    def delete_all_memory(self) -> int:
        """GDPR erasure of ALL Module 26 memory for this user -- irreversible. Returns status code."""
        resp = self._request("DELETE", "/api/0.5/memory")
        return resp.status_code

    # ── Benchmark (Module 22) ──────────────────────────────────

    def create_benchmark_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/api/0.5/benchmark/schemas", schema)

    def upload_benchmark_schema_file(self, file_path: str | Path) -> dict[str, Any]:
        """Same as create_benchmark_schema(), but from a .yaml/.yml/.json file."""
        path = Path(file_path)
        with path.open("rb") as f:
            resp = self._request(
                "POST", "/api/0.5/benchmark/schemas/upload",
                files={"file": (path.name, f, "application/octet-stream")},
            )
        return resp.json()

    def create_benchmark_run(self, schema_id: str) -> dict[str, Any]:
        return self._post_json("/api/0.5/benchmark/runs", {"schema_id": schema_id})

    def get_benchmark_run(self, run_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/benchmark/runs/{quote(run_id, safe='')}")

    def delete_benchmark_schema(self, schema_id: str) -> int:
        """Cascades: also deletes every run under this schema. Returns the HTTP status code."""
        resp = self._request("DELETE", f"/api/0.5/benchmark/schemas/{quote(schema_id, safe='')}")
        return resp.status_code

    def wait_for_benchmark_run(
        self, run_id: str, timeout: float = 120, interval: float = 3,
    ) -> dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            data = self.get_benchmark_run(run_id)
            if data.get("status") in ("completed", "failed"):
                return data
            time.sleep(interval)
        raise TimeoutError(f"Benchmark run {run_id} did not complete within {timeout}s")

    # ── Guardrail (Module 23) -- read-only; policy/system mutation is ─
    # ── intentionally not wrapped here (shared, global, affects every
    # ── user on the host) ───────────────────────────────────────

    def list_guardrail_policies(self) -> list[dict[str, Any]]:
        return self._get_json("/api/0.5/guardrail/policies")

    def get_guardrail_policy_template(self, format: str = "checker_rules") -> dict[str, Any]:
        return self._get_json("/api/0.5/guardrail/policies/template", params={"format": format})

    def get_active_guardrail_policy(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/guardrail/policies/active")

    def get_guardrail_policy(self, policy_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/guardrail/policies/{quote(policy_id, safe='')}")

    def get_guardrail_status(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/guardrail/system/status")

    # ── Compliance (Module 33, requires compliance-auditor role) ──

    def get_compliance_report(
        self, from_: str, to: str, format: str = "json", max_rows: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"from": from_, "to": to, "format": format}
        if max_rows is not None:
            params["max_rows"] = max_rows
        return self._get_json("/api/0.5/compliance/report", params=params)

    # ── GDPR / CCPA (Module 33) ─────────────────────────────────

    def get_consent(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/consent")

    def set_consent(self, purpose: str, granted: bool) -> dict[str, Any]:
        return self._put_json(f"/api/0.5/gdpr/consent/{quote(purpose, safe='')}", {"granted": granted})

    def get_dsar(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/dsar")

    def get_gdpr_audit_log_mine(self) -> Any:
        return self._get_json("/api/0.5/gdpr/audit-log/my")

    def request_gdpr_export(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/export")

    def get_gdpr_export_job(self, job_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/gdpr/export/jobs/{quote(job_id, safe='')}")

    def wait_for_gdpr_export(self, job_id: str, timeout: float = 120, interval: float = 3) -> dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            data = self.get_gdpr_export_job(job_id)
            if data.get("status") in ("completed", "failed"):
                return data
            time.sleep(interval)
        raise TimeoutError(f"GDPR export job {job_id} did not complete within {timeout}s")

    def get_retention_policy_dry_run(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/retention/policy/dry-run")

    # ── GDPR admin-only (requires the gdpr-admin realm role) ────

    def get_gdpr_audit_log(self, **params) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/audit-log", params=params or None)

    def get_gdpr_jobs(self, **params) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/jobs", params=params or None)

    def get_gdpr_jobs_summary(self, **params) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/jobs/summary", params=params or None)

    def get_retention_policy(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/retention/policy")

    def set_retention_policy(self, policies: list[dict[str, Any]]) -> Any:
        resp = self._request(
            "PUT", "/api/0.5/gdpr/retention/policy",
            headers={"Content-Type": "application/json"}, json=policies,
        )
        return resp.json()

    def get_processing_records(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/processing-records")

    def get_processing_records_ccpa(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/gdpr/processing-records/ccpa")

    def get_dsar_for_user(self, user_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/gdpr/dsar/{quote(user_id, safe='')}")

    def request_gdpr_export_for_user(self, user_id: str) -> dict[str, Any]:
        return self._post_json(f"/api/0.5/gdpr/export/{quote(user_id, safe='')}")

    # ── DocAgent (Module 29, via the Module 13 generic proxy) ──

    def docagent_formats(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/docagent/formats")

    def docagent_process(
        self,
        task: str,
        file_path: str | Path,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        data: dict[str, str] = {"task": task}
        if options is not None:
            data["options"] = json.dumps(options)
        with path.open("rb") as f:
            resp = self._request(
                "POST", "/api/0.5/docagent/process",
                timeout=120,
                files={"file": (path.name, f, "application/octet-stream")},
                data=data,
            )
        return resp.json()

    # ── Moderation (Module 23, requires moderation-admin role) ──

    def list_moderation_flags(self, **params) -> dict[str, Any]:
        return self._get_json("/api/0.5/moderation/flags", params=params or None)

    # ── Identity (Module 25 People Search) ──────────────────────

    def list_identities(self, **params) -> list[dict[str, Any]]:
        return self._get_json("/api/0.5/identity/identities", params=params or None)

    def upsert_identity(
        self,
        name: str,
        aliases: dict[str, str] | None = None,
        identity_type: str = "person",
        identity_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "identity_type": identity_type}
        if aliases is not None:
            body["aliases"] = aliases
        if identity_id is not None:
            body["identity_id"] = identity_id
        return self._post_json("/api/0.5/identity/upload", body)

    def get_identity(self, identity_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/identity/{quote(identity_id, safe='')}")

    def update_identity(self, identity_id: str, **fields) -> dict[str, Any]:
        return self._patch_json(f"/api/0.5/identity/{quote(identity_id, safe='')}", fields)

    def delete_identity(self, identity_id: str) -> dict[str, Any]:
        resp = self._request("DELETE", f"/api/0.5/identity/{quote(identity_id, safe='')}")
        return resp.json()

    def get_identity_traces(self, identity_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/identity/{quote(identity_id, safe='')}/traces")

    # ── Temporal (Module 25) ─────────────────────────────────────

    def temporal_capabilities(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/temporal/capabilities")

    def temporal_ingest(
        self,
        idempotency_key: str,
        content: str | None = None,
        content_type: str | None = None,
        asset: dict[str, str] | None = None,
        operation: str = "ingest",
        wait_seconds: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"idempotency_key": idempotency_key, "operation": operation}
        if content is not None:
            body["content"] = content
            body["content_type"] = content_type or "text/plain"
        if asset is not None:
            body["asset"] = asset
        if wait_seconds is not None:
            body["wait_seconds"] = wait_seconds
        return self._post_json("/api/0.5/temporal/ingest", body, timeout=120)

    def temporal_status(self, version_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/temporal/status/{quote(version_id, safe='')}")

    def wait_for_temporal_job(
        self, version_id: str, timeout: float = 60, interval: float = 3,
    ) -> dict[str, Any]:
        """The temporal job row for `version_id` only exists once the normal
        upload/analysis pipeline (parse -> chunk -> index) finishes and Module 25's
        consumer enqueues it -- a fresh ingest 404s here until then, not "pending"."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                return self.temporal_status(version_id)
            except RaptorAPIError as e:
                if e.status_code == 404:
                    time.sleep(interval)
                    continue
                raise
        raise TimeoutError(f"No temporal job for version {version_id} within {timeout}s")

    def temporal_reason(self, version_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/temporal/reason/{quote(version_id, safe='')}")

    def temporal_parse(
        self, text: str, timezone: str | None = None, anchor: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text}
        if timezone is not None:
            body["timezone"] = timezone
        if anchor is not None:
            body["anchor"] = anchor
        return self._post_json("/api/0.5/temporal/parse", body)

    def temporal_verify(
        self,
        claims: list[dict[str, Any]],
        evidence: list[str],
        complete: bool = False,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"claims": claims, "evidence": evidence, "complete": complete}
        if timezone is not None:
            body["timezone"] = timezone
        return self._post_json("/api/0.5/temporal/verify", body)

    def temporal_expiry(self, version_id: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/temporal/expiry/{quote(version_id, safe='')}")

    # ── Inference (Module 35, via the Module 13 declared-route proxy) ──

    def _inference_text(self, path: str, input_: str, model: str | None = None, **options) -> dict[str, Any]:
        body: dict[str, Any] = {"input": input_, "options": options}
        if model is not None:
            body["model"] = model
        return self._post_json(f"/api/0.5/inference{path}", body, timeout=120)

    def translate(self, input_: str, target_language: str, source_language: str | None = None,
                  model: str | None = None) -> dict[str, Any]:
        opts: dict[str, Any] = {"target_language": target_language}
        if source_language is not None:
            opts["source_language"] = source_language
        return self._inference_text("/text/translate", input_, model, **opts)

    def summarize_text(self, input_: str, style: str | None = None, max_sentences: int | None = None,
                        model: str | None = None) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if style is not None:
            opts["style"] = style
        if max_sentences is not None:
            opts["max_sentences"] = max_sentences
        return self._inference_text("/text/summarize", input_, model, **opts)

    def complete_text(self, input_: str, max_tokens: int | None = None, temperature: float | None = None,
                       model: str | None = None) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if max_tokens is not None:
            opts["max_tokens"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        return self._inference_text("/text/complete", input_, model, **opts)

    def transform_text(self, input_: str, instruction: str, model: str | None = None) -> dict[str, Any]:
        return self._inference_text("/text/transform", input_, model, instruction=instruction)

    def text_dispatch(self, task: str, input_: str, model: str | None = None, **options) -> dict[str, Any]:
        body: dict[str, Any] = {"task": task, "input": input_, "options": options}
        if model is not None:
            body["model"] = model
        return self._post_json("/api/0.5/inference/text", body, timeout=120)

    def complete_text_stream(self, input_: str, max_tokens: int | None = None,
                              temperature: float | None = None, model: str | None = None) -> requests.Response:
        """Caller iterates resp.iter_lines() for `data: {...}` SSE frames, ending in `data: [DONE]`."""
        opts: dict[str, Any] = {}
        if max_tokens is not None:
            opts["max_tokens"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        body: dict[str, Any] = {"input": input_, "options": opts}
        if model is not None:
            body["model"] = model
        return self._request(
            "POST", "/api/0.5/inference/text/complete/stream",
            headers={"Content-Type": "application/json"}, json=body, timeout=120, stream=True,
        )

    def diarize(self, audio_path: str | Path, model: str | None = None, num_speakers: int | None = None,
                language: str | None = None) -> dict[str, Any]:
        path = Path(audio_path)
        data: dict[str, str] = {}
        if model is not None:
            data["model"] = model
        if num_speakers is not None:
            data["num_speakers"] = str(num_speakers)
        if language is not None:
            data["language"] = language
        with path.open("rb") as f:
            resp = self._request(
                "POST", "/api/0.5/inference/voice/diarize", timeout=120,
                files={"audio": (path.name, f, "audio/wav")}, data=data,
            )
        return resp.json()

    def speech_to_text(self, audio_path: str | Path, language: str | None = None) -> dict[str, Any]:
        path = Path(audio_path)
        data: dict[str, str] = {}
        if language is not None:
            data["language"] = language
        with path.open("rb") as f:
            resp = self._request(
                "POST", "/api/0.5/inference/voice/stt", timeout=120,
                files={"audio": (path.name, f, "audio/wav")}, data=data,
            )
        return resp.json()

    def text_to_speech(self, text: str, language: str | None = None) -> bytes:
        data: dict[str, str] = {"text": text}
        if language is not None:
            data["language"] = language
        resp = self._request("POST", "/api/0.5/inference/voice/tts", timeout=120, data=data)
        return resp.content

    def clone_voice(
        self,
        reference_audio_path: str | Path,
        text: str,
        language: str,
        speaker_name: str | None = None,
        model: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        """Requires the caller's voice_clone GDPR consent. Returns (wav_bytes, {speaker_id, watermarked})."""
        path = Path(reference_audio_path)
        data: dict[str, str] = {"text": text, "language": language}
        if speaker_name is not None:
            data["speaker_name"] = speaker_name
        if model is not None:
            data["model"] = model
        with path.open("rb") as f:
            resp = self._request(
                "POST", "/api/0.5/inference/voice/clone", timeout=180,
                files={"reference_audio": (path.name, f, "audio/wav")}, data=data,
            )
        return resp.content, {
            "speaker_id": resp.headers.get("x-speaker-id", ""),
            "watermarked": resp.headers.get("x-watermarked", ""),
        }

    def list_clone_speakers(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/inference/voice/clone/speakers")

    def delete_clone_speaker(self, speaker_id: str) -> int:
        resp = self._request("DELETE", f"/api/0.5/inference/voice/clone/speakers/{quote(speaker_id, safe='')}")
        return resp.status_code

    def list_inference_models(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/inference/models")

    def get_default_models(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/inference/models/default")

    def set_default_model(self, task: str, model: str) -> int:
        """Persisted in Redis, applies platform-wide immediately -- not scoped to the
        caller. Not exercised by api_smoke_test.py for exactly that reason."""
        resp = self._request(
            "POST", "/api/0.5/inference/models/default",
            headers={"Content-Type": "application/json"}, json={"task": task, "model": model},
        )
        return resp.status_code

    def get_inference_model(self, name: str) -> dict[str, Any]:
        return self._get_json(f"/api/0.5/inference/models/{quote(name, safe='')}")

    def inference_health(self) -> dict[str, Any]:
        return self._get_json("/api/0.5/inference/health")

    # ── Convenience ───────────────────────────────────────────

    def upload_and_wait(
        self,
        file_path: str | Path,
        m_type: str = "video",
        processing_mode: str = "default",
        timeout: float = 1800,
        interval: float = 15,
        verbose: bool = True,
    ) -> dict[str, Any]:
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
    import os

    BASE_URL = os.environ.get("RAPTOR_BASE_URL", "http://raptor_open_0_5_api.dhtsolution.com:8012")
    VIDEO = os.environ["RAPTOR_TEST_VIDEO"]
    SEP = "─" * 60

    def section(title):
        print(f"\n{SEP}\n  {title}\n{SEP}")

    client = RaptorClient(BASE_URL, os.environ["RAPTOR_USERNAME"], os.environ["RAPTOR_PASSWORD"])

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
