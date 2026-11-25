"""
Unit tests for LegacyApiAliasMiddleware (app/middlewares/legacy_api_alias.py).

Covers a real 500 found live on a deployed host: a /api/0.3/asset/filedownload/...
request whose asset_path contains non-ASCII characters (e.g. a CJK filename,
decoded by the ASGI server before this middleware sees scope["path"]) crashed
every such request. The middleware put that decoded, non-ASCII path straight
into the "Link" response header; MutableHeaders.append() encodes header
values as latin-1, so any character outside latin-1 raised UnicodeEncodeError
for every response -- not a 400/404 for that one asset, every aliased
request whose path needed any non-ASCII percent-decoding.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_legacy_api_alias.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.middlewares.legacy_api_alias import LegacyApiAliasMiddleware
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app(seen_paths: list) -> FastAPI:
    app = FastAPI()
    app.add_middleware(LegacyApiAliasMiddleware, canonical_prefix="/api/0.5/")

    @app.get("/api/0.5/asset/filedownload/{asset_path:path}/{version_id}")
    def download(asset_path: str, version_id: str):
        seen_paths.append(asset_path)
        return {"ok": True}

    return app


def test_ascii_legacy_path_is_aliased_with_deprecation_headers():
    seen: list = []
    res = TestClient(_app(seen)).get("/api/0.3/asset/filedownload/video/mp4/plain_name/abc123")
    assert res.status_code == 200 and seen == ["video/mp4/plain_name"]
    assert res.headers["deprecation"] == "true"
    assert res.headers["link"] == '</api/0.5/asset/filedownload/video/mp4/plain_name/abc123>; rel="successor-version"'


def test_non_ascii_legacy_path_does_not_crash_the_response():
    """The real bug: a CJK filename used to raise UnicodeEncodeError inside
    the Link header, turning every such request into a 500."""
    seen: list = []
    res = TestClient(_app(seen)).get("/api/0.3/asset/filedownload/video/mp4/%E8%BC%9D%E9%81%94/abc123")
    assert res.status_code == 200 and seen == ["video/mp4/輝達"]
    assert res.headers["link"] == '</api/0.5/asset/filedownload/video/mp4/%E8%BC%9D%E9%81%94/abc123>; rel="successor-version"'


def test_unaliased_group_passes_through_unchanged():
    seen: list = []
    app = _app(seen)

    @app.get("/api/0.3/other/{rest:path}")
    def other(rest: str):
        return {"rest": rest}

    res = TestClient(app).get("/api/0.3/other/x")
    assert res.status_code == 200 and "link" not in res.headers
