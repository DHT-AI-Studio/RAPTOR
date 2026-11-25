"""resolve_asset needs a version: without one the asset service answers a presign request with a
redirect (HTTP 307), which reached callers as an unexplained INVALID_ARGUMENT."""
from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException
from interceptors.errors import ValidationError
from services import asset_resolver
from services.asset_resolver import resolve_asset


class _Storage:
    def __init__(self):
        self.calls = 0

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        self.calls += 1
        return {"metadata": {"primary_filename": "clip.mp4"}}

    async def get_presign_url(self, key, version_id, user):
        self.calls += 1
        return f"http://store/{key}?v={version_id}"


@pytest.mark.asyncio
async def test_without_a_version_id_the_caller_is_told_what_is_missing_and_nothing_is_asked_downstream():
    storage = _Storage()
    with pytest.raises(ValidationError, match="version_id is required"):
        await resolve_asset(storage, "video/mp4/clip", "", object())
    assert storage.calls == 0


@pytest.mark.asyncio
async def test_with_a_version_id_the_platform_asset_path_resolves():
    resolved = await resolve_asset(_Storage(), "video/mp4/clip", "v1", object())
    assert (resolved.url, resolved.filename) == ("http://store/video/mp4/clip/clip.mp4?v=v1", "clip.mp4")


class _Flaky(_Storage):
    """Fails `failures` times with `error` on every call kind, then answers."""

    def __init__(self, failures, error):
        super().__init__()
        self.failures, self.error = failures, error

    async def get_presign_url(self, key, version_id, user):
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return f"http://store/{key}?v={version_id}"


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    monkeypatch.setattr(asset_resolver, "_RETRY_DELAYS_S", (0, 0))


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [HTTPException(503, "x"), HTTPException(502, "x"), HTTPException(504, "x"), httpx.ConnectError("x")])
async def test_a_transient_failure_is_retried_and_the_caller_never_sees_it(error):
    storage = _Flaky(2, error)
    resolved = await resolve_asset(storage, "video/mp4/clip", "v1", object())
    assert resolved.filename == "clip.mp4"


@pytest.mark.asyncio
async def test_a_failure_that_keeps_happening_still_reaches_the_caller_after_three_tries():
    storage = _Flaky(99, HTTPException(503, "x"))
    with pytest.raises(HTTPException):
        await resolve_asset(storage, "video/mp4/clip", "v1", object())
    assert storage.calls == 1 + 3  # one metadata read, then three presign attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 409])
async def test_a_client_error_is_never_retried(status):
    storage = _Flaky(99, HTTPException(status, "x"))
    with pytest.raises(HTTPException):
        await resolve_asset(storage, "video/mp4/clip", "v1", object())
    assert storage.calls == 1 + 1


@pytest.mark.asyncio
async def test_a_missing_asset_is_still_not_found_and_not_retried():
    class _Gone(_Storage):
        async def download_asset(self, *a, **k):
            self.calls += 1
            raise HTTPException(404, "no")

        async def get_presign_url(self, *a, **k):
            self.calls += 1
            raise HTTPException(404, "no")

    from interceptors.errors import NotFoundError

    storage = _Gone()
    with pytest.raises(NotFoundError):
        await resolve_asset(storage, "video/mp4/x", "v1", object())
    assert storage.calls == 2  # metadata + the one key spelling; no retries
