import base64
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.datastructures import FileStorage

DEFAULT_BASE_URL = os.getenv(
    "RAPTOR_API_BASE_URL",
    "http://raptor_open_0_1_api.dhtsolution.com:8012",
)
SESSION_BASE_URL_KEY = "raptor_base_url"
SESSION_TOKEN_KEY = "raptor_token"
SESSION_ASSET_RECORDS_KEY = "raptor_asset_records"

PAGE_DEFINITIONS: List[Tuple[str, str]] = [
    ("settings", "環境設定"),
    ("auth", "帳號登入"),
    ("search", "搜尋功能"),
    ("uploads", "檔案上傳"),
    ("assets", "資產管理"),
    ("processing", "資料處理與快取"),
    ("chat", "聊天功能"),
]
DEFAULT_PAGE = "search"
VALID_PAGES = {key for key, _ in PAGE_DEFINITIONS}

AUTH_REQUIRED_ACTIONS = {
    "video_search",
    "audio_search",
    "document_search",
    "image_search",
    "unified_search",
    "upload_file",
    "upload_files_batch",
    "upload_file_with_analysis",
    "upload_files_batch_with_analysis",
    "list_file_versions",
    "download_asset",
    "archive_asset",
    "delete_asset",
    "process_file",
    "send_chat",
    "get_chat_memory",
    "clear_chat_memory",
}

ASSET_RESULT_ACTIONS = {
    "upload_file",
    "upload_files_batch",
    "upload_file_with_analysis",
    "upload_files_batch_with_analysis",
    "list_file_versions",
    "process_file",
}

ACTION_DISPLAY_NAMES = {
    "upload_file": "單檔上傳",
    "upload_files_batch": "批次上傳",
    "upload_file_with_analysis": "單檔上傳並分析",
    "upload_files_batch_with_analysis": "批次上傳並分析",
    "list_file_versions": "列出檔案版本",
    "process_file": "處理檔案",
}


def _load_sample_client_class():
    module_path = Path(__file__).resolve().parents[1] / "sample_code_python.py"
    spec = importlib.util.spec_from_file_location("raptor_sample_client", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Unable to load sample client module")
    spec.loader.exec_module(module)
    return module.RaptorAPIClient


RaptorAPIClient = _load_sample_client_class()

app = Flask(__name__)
app.secret_key = os.getenv("RAPTOR_WEB_SECRET", "raptor-demo-secret")


def _get_base_url() -> str:
    return session.get(SESSION_BASE_URL_KEY, DEFAULT_BASE_URL)


def _set_base_url(value: str) -> None:
    session[SESSION_BASE_URL_KEY] = value or DEFAULT_BASE_URL


def _get_token() -> Optional[str]:
    return session.get(SESSION_TOKEN_KEY)


def _set_token(value: Optional[str]) -> None:
    if value:
        session[SESSION_TOKEN_KEY] = value
    else:
        session.pop(SESSION_TOKEN_KEY, None)


def _get_asset_records() -> List[Dict[str, str]]:
    return session.get(SESSION_ASSET_RECORDS_KEY, [])


def _set_asset_records(records: List[Dict[str, str]]) -> None:
    session[SESSION_ASSET_RECORDS_KEY] = records


def _make_client() -> Any:
    client = RaptorAPIClient(base_url=_get_base_url())
    token = _get_token()
    if token:
        client.token = token
    return client


def _parse_comma_separated(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    parts = [item.strip() for item in raw.replace("\r", "").replace("\n", ",").split(",")]
    cleaned = [item for item in parts if item]
    return cleaned or None


def _parse_optional_int(raw: str) -> Optional[int]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return int(raw)


def _parse_optional_float(raw: str) -> Optional[float]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return float(raw)


def _sanitize_upload_name(filename: Optional[str], fallback_stem: str) -> str:
    name = (filename or "").strip()
    if not name:
        name = f"{fallback_stem}.bin"
    return Path(name).name or f"{fallback_stem}.bin"


def _run_with_temp_file(file_storage: FileStorage, runner: Callable[[str], Any]) -> Any:
    if not file_storage or not file_storage.filename:
        raise ValueError("請選擇要上傳的檔案")
    safe_name = _sanitize_upload_name(file_storage.filename, "upload")
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / safe_name
        file_storage.save(target_path)
        return runner(str(target_path))


def _run_with_temp_files(file_storages: Iterable[FileStorage], runner: Callable[[List[str]], Any]) -> Any:
    stored_paths: List[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for index, file_storage in enumerate(file_storages):
            if not file_storage or not file_storage.filename:
                continue
            safe_name = _sanitize_upload_name(file_storage.filename, f"upload_{index}")
            target_path = Path(tmp_dir) / safe_name
            file_storage.save(target_path)
            stored_paths.append(str(target_path))
        if not stored_paths:
            raise ValueError("請至少選擇一個檔案")
        return runner(stored_paths)


def _format_response(data: Any, action: str) -> Dict[str, Any]:
    if isinstance(data, (dict, list)):
        return {"type": "json", "value": json.dumps(data, indent=2, ensure_ascii=False)}
    if isinstance(data, bytes):
        encoded = base64.b64encode(data).decode()
        return {
            "type": "binary",
            "value": encoded,
            "note": "已將檔案內容轉為 Base64 文字，請自行另存為檔案。",
        }
    if data is None:
        return {"type": "text", "value": "成功執行"}
    return {"type": "text", "value": str(data)}


def _format_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            payload = exc.response.json()
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except ValueError:
            return exc.response.text or str(exc)
    return str(exc)


def _extract_asset_entries(data: Any) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            lowered = {key.lower(): value[key] for key in value}
            asset_path = lowered.get("asset_path") or lowered.get("assetpath")
            version_id = (
                lowered.get("version_id")
                or lowered.get("versionid")
                or lowered.get("version")
            )
            filename = (
                lowered.get("filename")
                or lowered.get("file_name")
                or lowered.get("name")
                or lowered.get("primary_filename")
                or lowered.get("uploaded_file")
            )
            if asset_path and version_id:
                results.append(
                    {
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "filename": filename,
                    }
                )
            for nested in value.values():
                _walk(nested)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(data)
    return results


def _register_asset_records(records: Iterable[Dict[str, Any]], source: str) -> None:
    if not records:
        return

    existing = _get_asset_records()
    updated: List[Dict[str, str]] = existing[:]

    for record in records:
        asset_path = str(record.get("asset_path", "") or "").strip()
        version_id = str(record.get("version_id", "") or "").strip()
        filename = str(record.get("filename", "") or "").strip()
        if not asset_path or not version_id:
            continue

        entry = {
            "asset_path": asset_path,
            "version_id": version_id,
            "filename": filename,
            "source": ACTION_DISPLAY_NAMES.get(source, source),
        }

        updated = [item for item in updated if not (
            item.get("asset_path") == asset_path and item.get("version_id") == version_id
        )]
        updated.insert(0, entry)

    if len(updated) > 20:
        updated = updated[:20]

    _set_asset_records(updated)


def _prepare_asset_records_for_view() -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for record in _get_asset_records():
        safe_record = {
            "asset_path": str(record.get("asset_path", "")),
            "version_id": str(record.get("version_id", "")),
            "filename": str(record.get("filename", "")),
            "source": str(record.get("source", "")),
        }
        encoded = base64.b64encode(
            json.dumps(safe_record, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        prepared.append({**safe_record, "encoded": encoded})
    return prepared


@app.route("/")
def root() -> Any:
    return redirect(url_for("dashboard", page=DEFAULT_PAGE))


@app.route("/dashboard")
def dashboard_redirect() -> Any:
    return redirect(url_for("dashboard", page=DEFAULT_PAGE))


@app.route("/dashboard/<page>", methods=["GET", "POST"])
def dashboard(page: str) -> Any:
    if page not in VALID_PAGES:
        abort(404)

    context: Dict[str, Any] = {
        "pages": PAGE_DEFINITIONS,
        "active_page": page,
        "base_url": _get_base_url(),
        "token": _get_token(),
        "last_action": None,
        "result": None,
        "error": None,
        "requires_auth_actions": AUTH_REQUIRED_ACTIONS,
        "asset_records": _prepare_asset_records_for_view(),
    }

    if request.method == "POST":
        action = request.form.get("action", "").strip()
        context["last_action"] = action
        base_url = request.form.get("base_url") or DEFAULT_BASE_URL
        _set_base_url(base_url)
        context["base_url"] = _get_base_url()

        try:
            if action == "update_base_url":
                context["result"] = {"type": "text", "value": "已更新 API Base URL"}
                return render_template("index.html", **context)

            if action == "clear_asset_history":
                _set_asset_records([])
                context["asset_records"] = _prepare_asset_records_for_view()
                context["result"] = {"type": "text", "value": "已清除最近資產紀錄"}
                return render_template("index.html", **context)

            client = _make_client()

            if action == "logout":
                _set_token(None)
                context["token"] = None
                context["result"] = {"type": "text", "value": "已清除登入狀態"}
                return render_template("index.html", **context)

            if action == "register":
                payload = client.register_user(
                    username=request.form.get("username", "").strip(),
                    email=request.form.get("email", "").strip(),
                    password=request.form.get("password", ""),
                )
                context["result"] = _format_response(payload, action)
                if isinstance(payload, (dict, list)) and action in ASSET_RESULT_ACTIONS:
                    _register_asset_records(_extract_asset_entries(payload), action)
                    context["asset_records"] = _prepare_asset_records_for_view()
                return render_template("index.html", **context)

            if action == "login":
                token = client.login(
                    username=request.form.get("username", "").strip(),
                    password=request.form.get("password", ""),
                )
                _set_token(token)
                context["token"] = token
                context["result"] = _format_response({"access_token": token}, action)
                return render_template("index.html", **context)

            if action in AUTH_REQUIRED_ACTIONS and not _get_token():
                raise ValueError("此操作需要先登入，請先取得 Token。")

            if action == "video_search":
                payload = client.video_search(
                    query_text=request.form.get("query_text", ""),
                    embedding_type=request.form.get("embedding_type", "text"),
                    filename=_parse_comma_separated(request.form.get("filename", "")),
                    speaker=_parse_comma_separated(request.form.get("speaker", "")),
                    limit=int(request.form.get("limit", 5)),
                )
                context["result"] = _format_response(payload, action)

            elif action == "audio_search":
                payload = client.audio_search(
                    query_text=request.form.get("query_text", ""),
                    embedding_type=request.form.get("embedding_type", "text"),
                    filename=_parse_comma_separated(request.form.get("filename", "")),
                    speaker=_parse_comma_separated(request.form.get("speaker", "")),
                    limit=int(request.form.get("limit", 5)),
                )
                context["result"] = _format_response(payload, action)

            elif action == "document_search":
                payload = client.document_search(
                    query_text=request.form.get("query_text", ""),
                    embedding_type=request.form.get("embedding_type", "text"),
                    filename=_parse_comma_separated(request.form.get("filename", "")),
                    source=request.form.get("source") or None,
                    limit=int(request.form.get("limit", 5)),
                )
                context["result"] = _format_response(payload, action)

            elif action == "image_search":
                payload = client.image_search(
                    query_text=request.form.get("query_text", ""),
                    embedding_type=request.form.get("embedding_type", "text"),
                    filename=_parse_comma_separated(request.form.get("filename", "")),
                    source=request.form.get("source") or None,
                    limit=int(request.form.get("limit", 5)),
                )
                context["result"] = _format_response(payload, action)

            elif action == "unified_search":
                filters_raw = request.form.get("filters")
                filters_value = None
                if filters_raw:
                    filters_value = json.loads(filters_raw)
                payload = client.unified_search(
                    query_text=request.form.get("query_text", ""),
                    embedding_type=request.form.get("embedding_type", "text"),
                    filters=filters_value,
                    limit_per_collection=int(request.form.get("limit_per_collection", 5)),
                    global_limit=_parse_optional_int(request.form.get("global_limit")),
                    score_threshold=_parse_optional_float(request.form.get("score_threshold")),
                )
                context["result"] = _format_response(payload, action)

            elif action == "upload_file":
                archive_ttl = int(request.form.get("archive_ttl", 30))
                destroy_ttl = int(request.form.get("destroy_ttl", 30))
                file_storage = request.files.get("primary_file")

                payload = _run_with_temp_file(
                    file_storage,
                    lambda path: client.upload_file(
                        file_path=path,
                        archive_ttl=archive_ttl,
                        destroy_ttl=destroy_ttl,
                    ),
                )
                context["result"] = _format_response(payload, action)

            elif action == "upload_files_batch":
                archive_ttl = int(request.form.get("archive_ttl", 30))
                destroy_ttl = int(request.form.get("destroy_ttl", 30))
                concurrency = int(request.form.get("concurrency", 4))
                payload = _run_with_temp_files(
                    request.files.getlist("primary_files"),
                    lambda paths: client.upload_files_batch(
                        file_paths=paths,
                        archive_ttl=archive_ttl,
                        destroy_ttl=destroy_ttl,
                        concurrency=concurrency,
                    ),
                )
                context["result"] = _format_response(payload, action)

            elif action == "upload_file_with_analysis":
                archive_ttl = int(request.form.get("archive_ttl", 30))
                destroy_ttl = int(request.form.get("destroy_ttl", 30))
                processing_mode = request.form.get("processing_mode", "default")
                payload = _run_with_temp_file(
                    request.files.get("primary_file"),
                    lambda path: client.upload_file_with_analysis(
                        file_path=path,
                        processing_mode=processing_mode,
                        archive_ttl=archive_ttl,
                        destroy_ttl=destroy_ttl,
                    ),
                )
                context["result"] = _format_response(payload, action)

            elif action == "upload_files_batch_with_analysis":
                archive_ttl = int(request.form.get("archive_ttl", 30))
                destroy_ttl = int(request.form.get("destroy_ttl", 30))
                concurrency = int(request.form.get("concurrency", 4))
                processing_mode = request.form.get("processing_mode", "default")
                payload = _run_with_temp_files(
                    request.files.getlist("primary_files"),
                    lambda paths: client.upload_files_batch_with_analysis(
                        file_paths=paths,
                        processing_mode=processing_mode,
                        archive_ttl=archive_ttl,
                        destroy_ttl=destroy_ttl,
                        concurrency=concurrency,
                    ),
                )
                context["result"] = _format_response(payload, action)

            elif action == "list_file_versions":
                payload = client.list_file_versions(
                    asset_path=request.form.get("asset_path", ""),
                    filename=request.form.get("filename", ""),
                )
                context["result"] = _format_response(payload, action)

            elif action == "download_asset":
                return_content = request.form.get("return_file_content") == "on"
                data = client.download_asset(
                    asset_path=request.form.get("asset_path", ""),
                    version_id=request.form.get("version_id", ""),
                    return_file_content=return_content,
                )
                context["result"] = _format_response(data, action)

            elif action == "archive_asset":
                payload = client.archive_asset(
                    asset_path=request.form.get("asset_path", ""),
                    version_id=request.form.get("version_id", ""),
                )
                context["result"] = _format_response(payload, action)

            elif action == "delete_asset":
                payload = client.delete_asset(
                    asset_path=request.form.get("asset_path", ""),
                    version_id=request.form.get("version_id", ""),
                )
                context["result"] = _format_response(payload, action)

            elif action == "process_file":
                raw = request.form.get("upload_result", "")
                if not raw:
                    raise ValueError("請提供 upload_result JSON 資料")
                parsed = json.loads(raw)
                payload = client.process_file(upload_result=parsed)
                context["result"] = _format_response(payload, action)

            elif action == "get_cached_value":
                payload = client.get_cached_value(
                    m_type=request.form.get("m_type", ""),
                    key=request.form.get("cache_key", ""),
                )
                context["result"] = _format_response(payload, action)

            elif action == "get_all_cache":
                payload = client.get_all_cache()
                context["result"] = _format_response(payload, action)

            elif action == "send_chat":
                search_results_raw = request.form.get("search_results")
                search_results_value = None
                if search_results_raw:
                    search_results_value = json.loads(search_results_raw)
                payload = client.send_chat(
                    user_id=request.form.get("user_id", ""),
                    message=request.form.get("message", ""),
                    search_results=search_results_value,
                )
                context["result"] = _format_response(payload, action)

            elif action == "get_chat_memory":
                payload = client.get_chat_memory(user_id=request.form.get("user_id", ""))
                context["result"] = _format_response(payload, action)

            elif action == "clear_chat_memory":
                payload = client.clear_chat_memory(user_id=request.form.get("user_id", ""))
                context["result"] = _format_response(payload, action)

            elif action == "health_check":
                payload = client.health_check()
                context["result"] = _format_response(payload, action)

            else:
                raise ValueError("未知的操作")

            if action in ASSET_RESULT_ACTIONS and isinstance(context["result"], dict) and context["result"].get("type") == "json":
                raw_json = json.loads(context["result"]["value"])
                _register_asset_records(_extract_asset_entries(raw_json), action)
                context["asset_records"] = _prepare_asset_records_for_view()

        except Exception as exc:  # pylint: disable=broad-except
            context["error"] = _format_error(exc)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8013, debug=True)
