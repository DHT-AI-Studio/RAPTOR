"""thumbnail_worker/main.py: NVDEC decode must fall back to software decode
on failure (confirmed live: the GPU can genuinely be out of memory under
contention from other services sharing it) rather than 502ing outright."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thumbnail_worker.main import _extract_frame_nvdec


def _fake_proc(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_nvdec_success_never_tries_cpu_fallback():
    with patch(
        "thumbnail_worker.main.asyncio.create_subprocess_exec",
        AsyncMock(return_value=_fake_proc(0, stdout=b"\xff\xd8jpeg\xff\xd9")),
    ) as mock_exec:
        result = await _extract_frame_nvdec("http://example/video.mp4", 1.0)
    assert result == b"\xff\xd8jpeg\xff\xd9"
    assert mock_exec.await_count == 1
    assert "-hwaccel" in mock_exec.await_args.args


@pytest.mark.asyncio
async def test_nvdec_failure_falls_back_to_cpu_and_succeeds():
    procs = [
        _fake_proc(1, stderr=b"CUDA_ERROR_OUT_OF_MEMORY"),
        _fake_proc(0, stdout=b"\xff\xd8jpeg\xff\xd9"),
    ]
    with patch(
        "thumbnail_worker.main.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=procs),
    ) as mock_exec:
        result = await _extract_frame_nvdec("http://example/video.mp4", 1.0)
    assert result == b"\xff\xd8jpeg\xff\xd9"
    assert mock_exec.await_count == 2
    first_cmd, second_cmd = mock_exec.await_args_list[0].args, mock_exec.await_args_list[1].args
    assert "-hwaccel" in first_cmd
    assert "-hwaccel" not in second_cmd


@pytest.mark.asyncio
async def test_both_nvdec_and_cpu_fallback_fail_raises_502():
    procs = [
        _fake_proc(1, stderr=b"CUDA_ERROR_OUT_OF_MEMORY"),
        _fake_proc(1, stderr=b"some other ffmpeg error"),
    ]
    with patch(
        "thumbnail_worker.main.asyncio.create_subprocess_exec",
        AsyncMock(side_effect=procs),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _extract_frame_nvdec("http://example/video.mp4", 1.0)
    assert exc_info.value.status_code == 502
