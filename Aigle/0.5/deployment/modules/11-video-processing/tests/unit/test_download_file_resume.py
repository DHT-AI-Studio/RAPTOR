"""SeaweedFSClient.download_file() -- resumable download. A mid-stream
failure retries from the byte offset already on disk via an HTTP Range
request, instead of restarting the whole file from scratch.

Run from the 11-video-processing/ directory:
    pytest tests/unit/test_download_file_resume.py -v
"""

import aiohttp
import api_client
import pytest


class _FakeContent:
    def __init__(self, chunks, fail_after=None):
        self._chunks = chunks
        self._fail_after = fail_after

    async def iter_chunked(self, n):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise aiohttp.ClientPayloadError("simulated mid-stream failure")
            yield chunk


class _FakeResp:
    def __init__(self, status, json_body=None, chunks=None, fail_after=None):
        self.status = status
        self._json_body = json_body
        self.content = _FakeContent(chunks or [], fail_after=fail_after)

    async def json(self):
        return self._json_body

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class _FakeSession:
    """Base fake -- subclasses override get() for each test's scripted
    sequence of responses. Also doubles as its own async context manager,
    matching `async with aiohttp.ClientSession(...) as session`."""

    def __init__(self, metadata_url, presigned_url):
        self.metadata_url = metadata_url
        self.presigned_url = presigned_url
        self.presigned_get_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    def get(self, url, headers=None, params=None):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_download_file_resumes_after_mid_stream_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(api_client, "TEMP_FILE_DIR", str(tmp_path))
    chunk1, chunk2 = b"first-half-", b"second-half"

    class _Session(_FakeSession):
        def get(self, url, headers=None, params=None):
            if url == self.metadata_url:
                return _FakeResp(200, json_body={"primary_file": {"url": self.presigned_url}})
            self.presigned_get_calls.append(headers or {})
            if len(self.presigned_get_calls) == 1:
                # dies right after the first chunk -- never reaches chunk2
                return _FakeResp(200, chunks=[chunk1, chunk2], fail_after=1)
            # second attempt: Range honored, server returns only the remainder
            return _FakeResp(206, chunks=[chunk2])

    fake_session = _Session(metadata_url="http://asset/filedownload/x/v1", presigned_url="http://presigned/x")
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    client = api_client.SeaweedFSClient()
    result_path = await client.download_file(
        download_url="http://asset/filedownload/x/v1",
        download_headers={}, download_params={}, filename="video.mp4",
    )

    with open(result_path, "rb") as f:  # noqa: ASYNC230 -- test helper reads a small tmp file, no event loop contention
        assert f.read() == chunk1 + chunk2
    # Second attempt's Range header asks for exactly what's missing --
    # len(chunk1) bytes already on disk from the failed first attempt.
    assert fake_session.presigned_get_calls[1] == {"Range": f"bytes={len(chunk1)}-"}


@pytest.mark.asyncio
async def test_download_file_restarts_clean_when_range_not_honored(tmp_path, monkeypatch):
    """Server ignores the Range request and returns 200 (full content)
    instead of 206 -- must discard the partial file and write the fresh
    response, not append it onto the stale partial bytes."""
    monkeypatch.setattr(api_client, "TEMP_FILE_DIR", str(tmp_path))
    chunk1, full = b"partial-", b"the-whole-file-from-scratch"

    class _Session(_FakeSession):
        def get(self, url, headers=None, params=None):
            if url == self.metadata_url:
                return _FakeResp(200, json_body={"primary_file": {"url": self.presigned_url}})
            self.presigned_get_calls.append(headers or {})
            if len(self.presigned_get_calls) == 1:
                return _FakeResp(200, chunks=[chunk1], fail_after=0)  # fails before writing anything
            return _FakeResp(200, chunks=[full])  # Range ignored -- full file, every time

    fake_session = _Session(metadata_url="http://asset/filedownload/x/v1", presigned_url="http://presigned/x")
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    client = api_client.SeaweedFSClient()
    result_path = await client.download_file(
        download_url="http://asset/filedownload/x/v1",
        download_headers={}, download_params={}, filename="video.mp4",
    )
    with open(result_path, "rb") as f:  # noqa: ASYNC230 -- test helper reads a small tmp file, no event loop contention
        assert f.read() == full


@pytest.mark.asyncio
async def test_download_file_cleans_up_and_raises_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(api_client, "TEMP_FILE_DIR", str(tmp_path))

    async def _no_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(api_client.asyncio, "sleep", _no_sleep)

    class _Session(_FakeSession):
        def get(self, url, headers=None, params=None):
            if url == self.metadata_url:
                return _FakeResp(200, json_body={"primary_file": {"url": self.presigned_url}})
            self.presigned_get_calls.append(headers or {})
            return _FakeResp(500)

    fake_session = _Session(metadata_url="http://asset/filedownload/x/v1", presigned_url="http://presigned/x")
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    client = api_client.SeaweedFSClient()
    with pytest.raises(Exception, match="File download returned 500"):
        await client.download_file(
            download_url="http://asset/filedownload/x/v1",
            download_headers={}, download_params={}, filename="video.mp4",
            max_retries=3,
        )
    assert len(fake_session.presigned_get_calls) == 3
    assert list(tmp_path.iterdir()) == []  # partial file cleaned up, not left behind
